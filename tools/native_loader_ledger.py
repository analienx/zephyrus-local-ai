"""Static ExLlamaV3 loading scenarios from pinned tensor headers; no torch/CUDA.

Does NOT certify model fit: allocator, graph, transient lifetimes and display
reservations must be validated only in the final hardware phase.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from tools.static_memory_budget import MIB, estimate

EMBED = "model.language_model.embed_tokens.weight"
COMPONENTS = {"text", "output_head", "mtp", "vision"}
RECONSTRUCT_SLICE_N = 32768  # pinned exllamav3/modules/quant/exl3.py


def inventory(path: Path, config: dict) -> dict:
    groups = {name: 0 for name in COMPONENTS}
    reconstruct = {name: 0 for name in COMPONENTS}
    seen = set()
    embedding_bytes = None
    text_cfg = config.get("text_config", config)
    with Path(path).open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            name, component, size = row["tensor"], row["component"], int(row["bytes"])
            if name in seen or component not in COMPONENTS or size < 0:
                raise ValueError(f"Invalid or repeated inventory tensor: {name}")
            seen.add(name)
            groups[component] += size
            if name == EMBED:
                dims = [int(n) for n in row["shape"].split("x")]
                expected = [int(text_cfg["vocab_size"]), int(text_cfg["hidden_size"])]
                if component != "text" or row["dtype"] != "BF16" or dims != expected:
                    raise ValueError("Embedding source does not match native CPU-preferred loader")
                embedding_bytes = size
            if name.endswith(".trellis"):
                if row["quant_format"] not in ("exl3", "exl3_header_inferred"):
                    raise ValueError(f"Unknown trellis format: {name}")
                ni, no = int(row["in_features"]), int(row["out_features"])
                if ni < 1 or no < 1:
                    raise ValueError(f"Unknown EXL3 matrix dimensions: {name}")
                # reconstruct_hgemm allocates at most a 32768-column fp16 slice.
                reconstruct[component] = max(reconstruct[component], 2 * ni * min(no, RECONSTRUCT_SLICE_N))
    if embedding_bytes is None or not seen:
        raise ValueError("Required text embedding missing from checkpoint inventory")
    if not 0 < embedding_bytes <= groups["text"]:
        raise ValueError("Invalid embedding component accounting")
    return {"serialized": groups, "cpu_embedding": embedding_bytes,
            "max_reconstructed_matrix": reconstruct, "tensor_count": len(seen)}


def scenario(config: dict, inv: dict, *, context: int,
             mtp: bool = True, vision: bool = False, staging: int = 0,
             history: int = 4, slots: int = 1, k_bits: int = 6, v_bits: int = 5) -> dict:
    if staging not in (0, 1):
        raise ValueError("Only native quant-direct staging modes 0 and 1 are modeled")
    components = ("text", "output_head") + (("mtp",) if mtp else ()) + (("vision",) if vision else ())
    disk = sum(inv["serialized"][name] for name in components)
    gpu_inputs = disk - inv["cpu_embedding"]
    cache = estimate(config, context=context, k_bits=k_bits, v_bits=v_bits,
                     mtp=mtp, history=history, slots=slots)
    # Source: triton_paged.py stages one fp16 K and V window, rounded up
    # in page count to a power of two, for sufficiently large causal prefill.
    heads, dim = cache["geometry"]["heads"], cache["geometry"]["dim"]
    pages = cache["context_reserved"] // 256
    rounded_pages = 1 << (pages - 1).bit_length()
    staging_ceiling = (rounded_pages * 256 * heads * dim * 2 * 2) if staging == 1 else 0
    max_recon = max(inv["max_reconstructed_matrix"][name] for name in components)
    non_head_recon = max(inv["max_reconstructed_matrix"][name] for name in components if name != "output_head")
    # Individual potential peaks are NOT an additive measured upper bound:
    # actual live overlap, output activations and allocator caching are unknown.
    return {"components": components, "disk_payload_bytes": disk,
            "cpu_embedding_bytes": inv["cpu_embedding"],
            "gpu_serialized_weight_input_bytes": gpu_inputs,
            "kv_and_recurrent_bytes": cache["known_resident_bytes"],
            "fp16_reconstruct_single_matrix_bytes": max_recon,
            "fp16_reconstruct_non_output_head_bytes": non_head_recon,
            "fp16_staged_single_attention_window_bytes": staging_ceiling,
            "fp16_fallback_single_layer_pool_bytes": cache["context_reserved"] * heads * dim * 2 * 2,
            "context_reserved": cache["context_reserved"],
            "optimistic_known_gpu_inputs": gpu_inputs + cache["known_resident_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nominal-vram-mib", type=int, default=16303)
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    parser.add_argument("--include-vision", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--staging", type=int, choices=(0, 1), default=0)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()
    if args.nominal_vram_mib < 1:
        parser.error("Nominal VRAM must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inv = inventory(args.manifest, config)
    print(f"Tensor inventory: {inv['tensor_count']} payloads; CPU embedding: "
          f"{inv['cpu_embedding']/MIB:.2f} MiB (native source-preferred residency)")
    for context in args.contexts:
        row = scenario(config, inv, context=context, mtp=not args.no_mtp,
                       vision=args.include_vision, staging=args.staging,
                       history=args.history, slots=args.slots)
        remaining = args.nominal_vram_mib * MIB - row["optimistic_known_gpu_inputs"]
        print(f"ctx={row['context_reserved']} disk={row['disk_payload_bytes']/MIB:.2f}MiB "
              f"gpu_weight_input={row['gpu_serialized_weight_input_bytes']/MIB:.2f}MiB "
              f"kv+recurrent={row['kv_and_recurrent_bytes']/MIB:.2f}MiB "
              f"unbudgeted={remaining/MIB:.2f}MiB "
              f"single_matrix_scratch={row['fp16_reconstruct_single_matrix_bytes']/MIB:.2f}MiB "
              f"non_output_head_scratch={row['fp16_reconstruct_non_output_head_bytes']/MIB:.2f}MiB "
              f"single_attn_staging_ceiling={row['fp16_staged_single_attention_window_bytes']/MIB:.2f}MiB "
              f"fallback_single_layer_pool={row['fp16_fallback_single_layer_pool_bytes']/MIB:.2f}MiB")
    print("NO FIT CLAIM: unknown allocator/slab/graph/activation/loader duplication, "
          "display use, peak overlap, per-device free VRAM or runtime dispatch.")


if __name__ == "__main__":
    main()
