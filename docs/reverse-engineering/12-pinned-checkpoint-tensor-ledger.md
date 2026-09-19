# Pinned EXL3 3.0-bpw checkpoint: complete tensor ledger, not a load benchmark

Audit on 2026-09-19 of public Hugging Face `turboderp/Qwen3.8-27B-exl3`, revision `004a887127d8304ca2d5475d3a3c41f1761fdd27`, branch `SC_3.00bpw_H4_V4`. Source: https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/004a887127d8304ca2d5475d3a3c41f1761fdd27 . No model weight bytes were downloaded, no model was loaded, no GPU inference or benchmarking was run. Two HTTP Range requests per shard retrieved only each public safetensors header. Never substitute another revision's metadata without rerunning this audit.

## Exact evidence and reconciliation

| Object | SHA-256 of downloaded metadata-only file |
|---|---|
| `config.json` | `d71f9f2cdd6638987601e1aaca1d89916733c442ff8dec23daffeac03bdeb6a8` |
| `quantization_config.json` | `e7c692d6330f098f6e5c4f002e88e686cbd7500df5ab5cbf48d44583b194133c` |
| `model.safetensors.index.json` | `7f2014881e099a91eec7aa650d8384943d32e9f419838f36b0b784b291c4e3d2` |
| shard 1 header (170,968 bytes) | `b2211d90601395b4ee93c08cd8c1562b5d8dc956c9cf0a4aed4aebec12ffa377` |
| shard 2 header (191,536 bytes) | `c97163318635bad51f3cf440a2436da15a8735a278fc726cb72e35d2d08a0930` |

Index has **3,080** stored tensors; safetensors headers have exactly these 3,080 unique names in the expected shards. All dtype×shape products match each tensor's `data_offsets`; their sum matches `metadata.total_size`: **12,982,006,772 bytes**. Shard lengths are **8,583,890,388** and **4,398,478,904** bytes; subtracting 8-byte prefixes and exact header lengths leaves precisely the max declared tensor end offset in each shard. No unexplained payload gap or header/index contradiction was found. This does not prove that values inside those byte ranges are valid model weights.

`quantization_config.json` describes **707 logical groups / 1,910 stored tensors**, with 12,363,094,596 described bytes. Another **1,170 tensors / 618,912,176 bytes** lack that description, despite appearing in both shard headers and file index. The missing entries are vision, MTP, and 144 text GDN housekeeping tensors. The header parser covers them all; it does not fabricate missing quantization calibration metadata.

## Actual component budget (serialized tensor bytes, not observed resident VRAM)

| Component | Tensor count | Bytes | MiB |
|---|---:|---:|---:|
| `model.language_model.*` | 2,050 | 11,730,829,888 | 11,187.39 |
| `lm_head.*` | 4 | 636,206,084 | 606.73 |
| `mtp.*` | 39 | 159,552,544 | 152.16 |
| `model.visual.*` | 987 | 455,418,256 | 434.32 |
| **All four** | **3,080** | **12,982,006,772** | **12,380.60** |

The language-model token embedding alone is BF16 `[248320, 5120]` = **2,542,796,800 bytes**. The LM output head is **4-bit EXL3**, not a separate full BF16 projection: its trellis occupies 635,699,200 bytes and remaining side tensors 506,884 bytes. The inspected text trellis groups divide into **297×3-bit, 65×4-bit, 34×2-bit and 4×5-bit**, plus one 4-bit output head. Consequently, a blanket 3-bpw kernel or precision policy is incorrect. From header shapes only, the unlisted vision `trellis` entries are **164×4-bit** and the unlisted MTP entries **8×3-bit**; these are **shape-inferred**, not quantizer-calibration declarations.

Every inventory row records shard, actual dtype, shape, payload bytes, component, optional logical quant group, declared or inferred bitrate, projection shape, and **hard eligibility only** for specialized native small-m kernels. Actual kernel selection also depends on architecture, co-residency, m, n, k, codebook and autotune; the CSV does not claim measured execution dispatch.

## Loading semantics and implications

Pinned native ExLlamaV3: [`Model.from_config`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/model/model.py#L158) defaults to `component="text"`. [`model_init.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/model_init.py#L195) constructs `component="mtp"` separately when `--mtp` selects the same directory; [`generator.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/generator/generator.py#L246) attaches that draft to its target. [`qwen3_5_mtp.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/architecture/qwen3_5_mtp.py#L164) references target embedding and target output head rather than constructing dedicated duplicates. These are source properties; we have not traced actual GPU tensor residency or every vision loader entry point.

Text+head+MTP serialized tensors = **11,946.29 MiB**. Vision is optional for the proposed text/coding lane, but costs at least its **434.32 MiB serialized payload** plus unknown processor/activation workspaces when enabled. Do not infer that skipping vision is valid for an image request, or that it can be switched on during a running text-only session without extra allocation and model loading.

**Native-loader correction (later source audit):** The table below is a deliberately conservative *serialized-all-weights-on-GPU* scenario, **not** the native ExLlamaV3 residency path. The native text embedding is CPU-preferred, removing 2,425 MiB from GPU weight inputs. For the corrected 16-GB candidate screen, peak-scratch separation, and actual load behavior see [13 — native allocation lifecycle](13-native-loader-allocation-lifecycle.md) and [`tools/native_loader_ledger.py`](../../tools/native_loader_ledger.py). The older 32K provisional choice below is superseded as a capacity conclusion; the production context remains undecided until source-correct implementation and final hardware validation.

## Paper capacity check — do not treat positive remainder as 'fits'

For `text+output_head+mtp`, one native recurrent slot, MTP-4 state history, and 6/5-bit compressed K/V, [the static planner](../../tools/model_residency_plan.py) combines **serialized bytes** with code-derived native cache tensor allocations. Against **16,303 MiB physical GPU memory** previously read from the Zephyrus (not currently free VRAM):

| Reserved tokens | Serialized weights MiB | KV+GDN state MiB | Physical total minus accounted MiB |
|---:|---:|---:|---:|
| 16,384 | 11,946.29 | 1,135.50 | 3,221.21 |
| 32,768 | 11,946.29 | 1,543.50 | 2,813.21 |
| 65,536 | 11,946.29 | 2,359.50 | 1,997.21 |
| 131,072 | 11,946.29 | 3,991.50 | 365.21 |

**Not included:** CUDA/runtime graphs, driver/Windows desktop, allocator fragmentation, per-layer scratch during load and prefill, FP16 attention-window staging, GPU cache checkpoints, any weight-format expansion/duplicated modules or multimodal preprocessing. Host checkpoint stashes separately consume system RAM. Because the residual at 128K is just 365 MiB *before* these costs, do not set it as an unconditionally advertised context capacity. A 32K provisional profile limits reservation while retaining an option for larger contexts after correctness and eventual final hardware validation. This is a *design envelope*, not a benchmark-driven runtime recommendation.

## Reproduce with metadata only

```powershell
# Download only the three small public JSON metadata files to a directory of your choosing:
# config.json, quantization_config.json, model.safetensors.index.json
# Then request exact 206 byte ranges for the two safetensors HEADERS (never the weight payload):
python tools/safetensors_header_audit.py --metadata-dir C:\path\to\metadata --fetch-headers --output-csv data\qwen38-27b-exl3-3bpw-tensor-inventory.csv
python -m tools.model_residency_plan --config C:\path\to\metadata\config.json --manifest data\qwen38-27b-exl3-3bpw-tensor-inventory.csv --nominal-vram-mib 16303
python -m unittest discover -s tests -v
```

**Safety:** `--fetch-headers` requires HTTP 206 with the exact requested range and aborts before reading response bytes on HTTP 200. Header size is capped at 2 MiB, names are validated, metadata/header shape and byte counts must match, and the shard total must equal prefix+header+data. The public [generated 3,080-row CSV](../../data/qwen38-27b-exl3-3bpw-tensor-inventory.csv) contains no model weight values; the private working directory holds the small downloaded source metadata and headers. If the model revision changes, rerun the audit and keep the old CSV labeled with its original revision.
