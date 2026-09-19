# Reproducible static allocation ledger (no model inference)

Script: [`tools/static_memory_budget.py`](../../tools/static_memory_budget.py); arithmetic tests: [`tests/test_static_memory_budget.py`](../../tests/test_static_memory_budget.py). This is a **lower-bound model of explicitly allocated cache/state tensors** in pinned ExLlamaV3 `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`, not a GPU benchmark and not a VRAM-fit declaration. The 16-GB laptop was not used for model inference.

Model source: [Qwen/Qwen3.8-27B config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json); locally fetched reference SHA-256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` on 2026-09-19. This identifies the *configuration bytes checked*, not a pinned model-weight revision. The local copy is kept outside the public project; no checkpoint weights were downloaded.

```
python tools/static_memory_budget.py PATH_TO_CONFIG_JSON --context 32768 --kv-k-bits 6 --kv-v-bits 5 --history 4 --slots 1
python tools/static_memory_budget.py PATH_TO_CONFIG_JSON --context 65536 --history 3 --json
```

It checks explicit layer types, full-attention/GDN counts, KV head geometry, page rounding (native `PAGE_SIZE=256`), FP16 per-32-value cache scales, the optional one-layer MTP draft cache, GDN FP32 matrix history, BF16 convolution history and batch-slot multiplicity. The `--no-mtp` flag excludes the separate draft attention cache, **not** trunk weights or recurrent state. It never accesses CUDA and does not import PyTorch.

## Official configuration: 6/5 KV, MTP enabled, one slot, depth four

| Cache token reservation | Target KV | MTP KV | FP32 recurrence + BF16 conv | Known subtotal |
|---:|---:|---:|---:|---:|
| 32,768 | 768 MiB | 48 MiB | 727.5 MiB | 1,543.5 MiB |
| 65,536 | 1,536 MiB | 96 MiB | 727.5 MiB | 2,359.5 MiB |
| 131,072 | 3,072 MiB | 192 MiB | 727.5 MiB | 3,991.5 MiB |

At 32,768, selecting three rather than four history steps (only if the drafter's protocol and actual max history permit it) changes recurrence from 727.5 MiB to 582.5625 MiB, a modeled **144.9375 MiB** decrease. Do not confuse reducing the draft depth with a free speedup or apply it without updating the cache allocation and verification logic.

## What this tool cannot prove

The native loader places individual layers on devices, but this script deliberately treats all modeled target and draft caches and recurrent slots as if allocated on **one GPU**. A future split/offload case needs a per-device allocation map. Unlike a real VRAM trace, a static subtotal does not account for (1) actual checkpoint tensor files, mixed quant metadata and embedding/head sharing; (2) transient reconstructed FP16 matrices (`reconstruct_hgemm` may allocate up to a 5120×32768×2 B = 320 MiB output-head slice); (3) quantized-KV prefill staging and runtime-dependent dispatch; (4) model/GPU runtime persistent buffers and CUDA graphs; (5) allocator fragmentation and Windows display processes; (6) serialized GDN checkpoints in host RAM; (7) concurrent jobs, background GPU services and differing draft-cache geometry.

An unmodified official text configuration is **not** proof that a particular converted quant has the same tensor set, selected MTP head or memory organization. The script uses the official architecture as a hypothetical *geometry reference* and requires precise data from the actual quantization manifest before making a specific fit recommendation. Source-level arithmetic proves these tensor shape totals conditional on the stated runtime, not the available GPU headroom.

**Verification stage:** the five CPU-only tests passed on 2026-09-19. They check the official geometry fixture, 1536 B/token/layer, 720 MiB FP32 recurrent matrices at history four, 7.5 MiB conv, target/draft cache totals, page rounding, one-slot/two-slot scaling, and fail-closed malformed geometry. Do not describe them as numerical model-quality tests, kernel parity, local inference or GPU benchmarks.
