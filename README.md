# Zephyrus Local AI

**Research-first, reproducible, hardware-specific LLM inference for the NVIDIA RTX 5080 Laptop GPU (16 GB).** Optimize *usable model quality, response latency, sustained throughput and practical context*, not a peak token/s claim. Qwen3.8-27B is the provisional model, not a hardwired architectural dependency.

> **Status · 2026-09-19:** substantial upstream source-level audit and independent CPU-only speculative-decoding math tests. No Zephyrus GPU result or custom Blackwell kernel is yet claimed. Third-party RTX 3090 Ti benchmarks are identified as such. The linked X article was not directly retrievable; the author's own GitHub-hosted article, runbook, code, and subsequent fixes are the primary references.

[Source audit (5 parts)](docs/reverse-engineering/README.md) · [Speculative math CI](https://github.com/analienx/zephyrus-local-ai/actions/workflows/spec-math.yml) · [Architecture](docs/architecture.md) · [Benchmark/quality protocol](docs/benchmark-protocol.md) · [Research issues](https://github.com/analienx/zephyrus-local-ai/issues)

## Why this project exists

[Jake ATX's llamAmpere](https://github.com/JakeATX/llamAmpere) combines model-specific quantization, MTP speculative decoding, compressed KV caches, custom attention/quant kernels and runtime changes for **24 GB Ampere SM86**. Its [v0.3 measured release](https://github.com/JakeATX/llamAmpere/blob/36a6bca81/docs/llamampere-v0.3/ARTICLE.md) and the later [v0.3.1 GGUF-native EXL3 work](https://github.com/JakeATX/llamAmpere/blob/c7a3a742f55d76e46a7c4ccfc40bf01a8be318ee/docs/exl3.md) concern *different source revisions*. Recompiling SM86 flags or copying a 24 GB context layout into a 16 GB Blackwell laptop is not a hardware-specific port. We are reconstructing the actual operations and compatibility conditions first.

## Completed source-level investigations

- [01 — Reference decode path](docs/reverse-engineering/01-reference-decode-path.md): end-to-end draft/target verification round, author's measured bottleneck census, shortlist and attention fast-path dispatch, CUDA graphs and unsuccessful experiments.
- [02 — EXL3 and physical GPU memory](docs/reverse-engineering/02-exl3-and-memory.md): trellis/16×16 tiles, codebook, Hadamard and scale companions, small-width direct GEMV, invalid fallback risks and per-token exact cache arithmetic.
- [03 — p/q verifier and recurrent state](docs/reverse-engineering/03-target-verifier-and-state.md): actual coupled target-draw/rejection/residual implementation, fallback conditions, sampler correctness and September 18 GDN rollback/replay fixes.
- [04 — Native ExLlamaV3 comparison](docs/reverse-engineering/04-native-exllamav3-comparison.md): shared target output head, **argmax+ID-match** versus Jake's p/q MTP, native paged 6/5-KV physical layout and distinct recurrent-state transaction.
- [05 — EXL3 prefill dispatch](docs/reverse-engineering/05-exl3-prefill-kernel-dispatch.md): native direct quant GEMM through 144 activation rows versus the GGUF EXL3 branch reconstructing weights above 16; native CUDA kernel/autotuner and remaining wide-prefill limitations.

Additional research inventory: [mechanism ledger](docs/article-reverse-engineering.md) and [earlier preliminary source map](docs/source-audit.md). Each source claim points to the applicable pinned upstream revision; mathematical deductions and remaining open questions are labeled separately.

**Independent executable groundwork:** [`tools/spec_math.py`](tools/spec_math.py) and [`tests/test_spec_math.py`](tests/test_spec_math.py) check the one-position exact p/q output law, conditional acceptance, restricted vocabularies and randomized distributions without the GPU or model weights. The public CPU [GitHub Actions run passed](https://github.com/analienx/zephyrus-local-ai/actions/runs/35426542090). This does **not** establish correctness of upstream C++/CUDA or performance on the laptop.

## Practical findings already established from source

The native Qwen MTP path borrows the **target** output head; the published GGUF 64K draft-vocabulary shortlist indexes the existing full head and is a computational optimization, not a corresponding reduction in model-weight VRAM. Native EXL3 proposals use argmax with ID-match at verification, whereas Jake's latest p/q path samples draft proposals and uses a residual law. Engine results cannot be attributed entirely to CUDA or quantization when their effective speculative algorithms differ.

Native 6/5-bit KV plus one identically compressed draft-attention layer is modeled at **26,112 bytes/token**, based on actual native `int32` packed data and fp16 scale arrays for 16 target attention layers, compared with the author's **25,984 bytes/token** at q8 K / turbo3 V / q8 drafter. This near equality **does not prove either engine can fit a given maximum context**. The author's reported 3.0-bpw mixed-precision GGUF model is 10.91 GiB on disk and approached the 16 GiB whole-card limit in a ~65.5K-prompt MTP4 test on a desktop GPU. Count resident model, reserved KV pages, recurrent state, transient activations, graphics and driver usage before adopting a configuration.

For prefill, native ExLlamaV3 already implements a direct quantized GEMM for up to **144 activation rows**, while the audited GGUF-native EXL3 port reconstructs f16 weights for `T>16`. Native also reconstructs at `T>144` by default, with more sophisticated large-output slicing and optional transform fusion at high row counts. Prioritize investigating these *existing* implementation boundaries, source-level correctness and actual Blackwell dispatch before designing any new CUDA kernel.

## Architecture and implementation order

Keep the OpenAI-compatible localhost API adapter thin, provide separate pinned runtime/format/model configurations, do not vendor gigabytes of models or private machine artifacts, and preserve licenses/attribution for adapted source. The initial candidate lanes are native ExLlamaV3 EXL3, upstream llama.cpp low-bit GGUF, and a verified SM120-safe build of Jake's later GGUF EXL3 port. A different model may replace Qwen3.8-27B when objective quality-adjusted performance supports the change.

**Research gate:** finish exact native/ggml GPU kernel dispatch and loader audits, including Qwen3.8 variant compatibility, attention fallback and state correctness; build source-independent correctness fixtures. Only then capture the actual Zephyrus GPU/VRAM/power/driver inventory and reproduce equal-workload quality/speed baselines. Finally implement one narrowly scoped, attributable, testable optimization against an observed bottleneck. [P0 hardware](https://github.com/analienx/zephyrus-local-ai/issues/1) · [P1 deep source audit](https://github.com/analienx/zephyrus-local-ai/issues/2) · [P2 model/engine comparison](https://github.com/analienx/zephyrus-local-ai/issues/3).

## GitHub About description

`Research-driven, Blackwell-optimized local LLM inference for 16 GB RTX 5080 laptops: quantization, MTP, KV cache, kernels and reproducible quality/speed benchmarks.`

## Attribution

Author's performance research: [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere); model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); reference engine: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); EXL3 format and native engine: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3). Preserve applicable license text for any incorporated code. This project's research is neither affiliation with, nor a claim of authorship of, their implementations.
