# Native EXL3 CUDA dispatch: a 3-bpw Blackwell-specific audit

Static inspection of `turboderp-org/exllamav3@02aef45cd681b960a00afcd0749a4ab99e6c1bfe`, no model load or GPU benchmark. Sources: [`modules/quant/exl3.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/modules/quant/exl3.py), [`quant/exl3_gemm.cu`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/exllamav3_ext/quant/exl3_gemm.cu), [`quant/exl3_gemv.cu`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/exllamav3_ext/quant/exl3_gemv.cu), [`quant/exl3_gemv_kernel.cuh`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh).

## Exact decision tree (default settings)

`LinearEXL3.forward`: if input row count `m <= 144`, or no-reconstruct explicitly requested, call direct `BC_LinearEXL3.run_alloc`; otherwise call `reconstruct_hgemm`. For a direct call, `exl3_gemm_gr` first offers the QTIP-inspired GEMV specialization if no forced shape/SM override; if it declines, select/autotune the ordinary EXL3 cooperative GEMM. **Direct does not imply the specialized GEMV always executes.**

The specialized `exl3_gemv_cfg` requires `2 <= bits <= 4`, Hadamard side tensors, `m <= EXL3_GEMV_MAX_M` (source comments characterize the useful range as `m <= 8`), K and N multiples of 128, and a codebook-compatible implementation. At 3 bits on a Blackwell device, the automatic path picks the narrow kernel only when `N/32` fits one co-resident block wave, or when `K <= 2048` and `N <= 8192`; otherwise it returns `-1`, allowing ordinary cooperative GEMM. The unconditional broad 3-bpw rule in source applies to `cc == CC_ADA`, **not SM120**. A forced debug option can select a wider path but is not a production recommendation.

At >=5 bits (e.g. a protected EXL3_6 output head), the QTIP GEMV specialization is hard-ineligible. A 3-bpw model is *mixed*: layer projections and output head can take different paths, which is why average file bitrate is insufficient to predict the effective kernel mix. The RTX 5080's compute capability 12.0 was read from device metadata; no native model was launched or dispatch trace captured.

## Distinct prefill and temporary-weight paths

For `m > 144`, native `reconstruct_hgemm` reconstructs FP16 weights per matrix, slicing output dimensions at most 32768 columns. It can fuse Hadamard transforms into reconstruction only for larger row counts (`m >= 1024` with dimensions divisible by 128 and fused path enabled); this is a source threshold, **not** a measured cross-over on the laptop. The GGUF EXL3 port reconstructs for widths above 16. The target output head is a particularly large temporary matrix: one native reconstructed 5120×32768 FP16 slice is **320 MiB**; this is an upper slice *size*, not a guarantee of concurrent live allocation or the actual peak for a specific prompt.
## What to implement first and what *not* to assume

1. Build a *static tensor dispatch inventory* from actual EXL3 tensor shapes, bit widths and codebook metadata. Label specialized-GEMV candidates, conditional co-resident-wave candidates, ordinary direct-GEMM candidates, and reconstructed-prefill candidates. Occupancy-dependent branches are unknown until source-specific kernels and device properties are resolved; a static tool must say `conditional` rather than claim a kernel runs.
2. For K=3 and M<=8 on SM120, source-level options are (i) improve the existing regular GEMM; (ii) widen the narrow GEMV's profitable envelope after inspecting cooperative residency; or (iii) reuse existing kernels unchanged. None is an established performance winner without last-stage laptop measurements.
3. At intermediate M=9..144, native already uses direct cooperative GEMM: do not port the GGUF path's T<=16 limitation into the native runtime. At larger M, inspect per-matrix reconstruction size, chunking and Hadamard-fusion thresholds before inventing persistent fp16 weight caching.
4. Performance numbers from an offline SM120 EXL3 **quantizer** are not inference-kernel data. A compile-capable CUDA build and shape-specific correctness tests are prerequisites to any changes; actual GPU inference and performance work remain explicitly deferred until the last project stage.

## Reference-only shape examples from official text config

- `hidden_size=5120`, `intermediate_size=17408`, `vocab_size=248320`, `num_kv_heads=4`, `head_dim=256`; full-attention K/V projection output width `1024` each. The quantizer may fuse neighboring matrices, so the on-disk tensor's actual N, bitrate and codebook must be read from its manifest rather than inferred from an unfused architecture name.
- At a 5120-input × 32768-output reconstructed FP16 **slice**, scratch is `5120*32768*2 = 335,544,320 B = 320 MiB`. The native output-head slice cap is 32768; this example excludes transforms, activations, allocator pooling and overlap with unrelated tensors.
- Record *physical* tensor formats: GGUF EXL3 `mul1` only versus native EXL3 `mul1`/other codebooks; the tiny-GEMV eligibility check treats these differently. A source-compatible model format is not automatic kernel compatibility.
