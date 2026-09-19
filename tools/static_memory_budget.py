"""Source-derived Qwen hybrid-cache allocation model. No torch, CUDA or inference.

Read a checkpoint config.json; report only allocations whose shapes are
explicit in pinned ExLlamaV3 source. NOT a VRAM-fit or throughput certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIB = 1024 ** 2
PAGE_SIZE = 256  # exllamav3/constants.py at pinned reference revision


def geometry(config: dict) -> dict:
    t = config.get("text_config", config)
    n = int(t["num_hidden_layers"])
    types = t.get("layer_types")
    if types is None:
        interval = int(t["full_attention_interval"])
        types = ["full_attention" if (i + 1) % interval == 0
                 else "linear_attention" for i in range(n)]
    if len(types) != n or set(types) - {"linear_attention", "full_attention"}:
        raise ValueError("Unexpected or incomplete model layer_types")
    return {"full": types.count("full_attention"),
            "gdn": types.count("linear_attention"), "heads": int(t["num_key_value_heads"]),
            "dim": int(t["head_dim"]), "gk": int(t["linear_num_key_heads"]),
            "gv": int(t["linear_num_value_heads"]),
            "dk": int(t["linear_key_head_dim"]), "dv": int(t["linear_value_head_dim"]),
            "conv": int(t["linear_conv_kernel_dim"]),
            "mtp_layers": int(t.get("mtp_num_hidden_layers", 0))}


def estimate(config: dict, *, context: int = 32768, k_bits: int = 6,
             v_bits: int = 5, mtp: bool = True, history: int = 4,
             slots: int = 1) -> dict:
    if context < 1 or history < 0 or slots < 1 or not (2 <= k_bits <= 8 and 2 <= v_bits <= 8):
        raise ValueError("Invalid context, history, slots or cache bitrate")
    g = geometry(config)
    values = g["heads"] * g["dim"]
    if values % 32:
        raise ValueError("Native cache format requires token width divisible by 32")
    tokens_alloc = ((context + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    per_layer = values * (k_bits + v_bits) // 8 + 2 * (values // 32) * 2
    # MTP is a separate draft component with its own one-layer attention cache.
    draft_layers = g["mtp_layers"] if mtp else 0
    cache_target = tokens_alloc * per_layer * g["full"]
    cache_draft = tokens_alloc * per_layer * draft_layers
    matrix_per_state = g["gv"] * g["dk"] * g["dv"] * 4
    conv_channels = 2 * g["gk"] * g["dk"] + g["gv"] * g["dv"]
    matrix = slots * (history + 1) * g["gdn"] * matrix_per_state
    conv = slots * g["gdn"] * conv_channels * (g["conv"] + history) * 2
    return {"geometry": g, "context_requested": context, "context_reserved": tokens_alloc,
            "kv_bytes_per_token_per_attention_layer": per_layer,
            "kv_target_bytes": cache_target, "kv_draft_bytes": cache_draft,
            "recurrent_matrix_bytes": matrix, "recurrent_conv_bytes": conv,
            "known_resident_bytes": cache_target + cache_draft + matrix + conv,
            "recurrent_checkpoint_payload_bytes": g["gdn"] *
                (matrix_per_state + conv_channels * g["conv"] * 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Official or converted model config.json")
    parser.add_argument("--context", type=int, default=32768, help="Reserved cache tokens, not prompt length")
    parser.add_argument("--kv-k-bits", type=int, default=6)
    parser.add_argument("--kv-v-bits", type=int, default=5)
    parser.add_argument("--history", type=int, default=4, help="Recurrent speculative history depth")
    parser.add_argument("--slots", type=int, default=1, help="Reserved concurrent recurrent slots")
    parser.add_argument("--no-mtp", action="store_true", help="Exclude separate MTP draft cache")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)
    out = estimate(config, context=args.context, k_bits=args.kv_k_bits,
                   v_bits=args.kv_v_bits, mtp=not args.no_mtp,
                   history=args.history, slots=args.slots)
    if args.json:
        print(json.dumps(out, sort_keys=True, indent=2))
        return
    print("Static allocation model (no weights or GPU allocations measured)")
    print("Geometry:", out["geometry"])
    print("Reserved cache tokens:", out["context_reserved"])
    for label, key in [("Target KV", "kv_target_bytes"),
                       ("MTP draft KV", "kv_draft_bytes"),
                       ("Recurrent FP32 matrices", "recurrent_matrix_bytes"),
                       ("Recurrent BF16 conv", "recurrent_conv_bytes"),
                       ("Known allocation subtotal", "known_resident_bytes"),
                       ("Single recurrent checkpoint payload", "recurrent_checkpoint_payload_bytes")]:
        print(f"{label}: {out[key]:,} bytes ({out[key] / MIB:.2f} MiB)")
    print("NOT INCLUDED: any model weights, tokenizer, vision model, CUDA graphs, "
          "attention/reconstruction scratch, driver/display reservations, allocator "
          "fragmentation, checkpoint stash or additional cache instances. "
          "A file size is not evidence that the model fits in available VRAM.")


if __name__ == "__main__":
    main()
