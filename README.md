# Zephyrus Local AI

**Research-first, reproducible, laptop-specific LLM inference engineering for a 16 GB NVIDIA RTX 5080 Laptop GPU.** We optimize *usable answer quality, latency, sustained throughput and practical context*, not isolated peak token/s. The target model is provisional: Qwen3.8-27B; smaller, better-quantized, or different models may win under the actual 16 GB and power limits.

> **Status (2026-09-19): design/reverse-engineering stage.** No GPU benchmark or claimed speedup has yet been reproduced on the owner's Zephyrus. No custom kernel has been validated here. Third-party benchmark results must never be presented as measurements on this laptop.

## The problem

Jake ATX's [llamAmpere work](https://github.com/JakeATX/llamAmpere) combines a model-specific mixed quant, native multi-token prediction (MTP), compressed KV caches, exact p/q speculation, fused attention, efficient quantized matrix-vector verification, and GPU-memory/graph changes. His results target 24 GB SM86 Ampere; simply recompiling with `-DCMAKE_CUDA_ARCHITECTURES=120` or copying his launch flags is **not** an SM120 port and does not solve our 16 GB memory constraint. Our job is to audit the source changes and rebuild only demonstrably useful, correct techniques for the Blackwell laptop.

## Design, research and evidence

- [Architecture decision and execution plan](docs/architecture.md): boundaries, candidate engines, hardware prerequisites, performance/quality gates and stages.
- [Jake article/source reverse-engineering ledger](docs/article-reverse-engineering.md): individual mechanisms, code locations, memory trade-offs, portability classification and experiments needed before implementation.
- [Benchmark and quality protocol](docs/benchmark-protocol.md): paired measurements, exactness tests, quality regression gates and report schema.
- [Public issue tracker](../../issues): implementation work must point to a hypothesis and a test.

## Candidate stack (not yet a verdict)

1. **ExLlamaV3 / TabbyAPI, EXL3 around 3.0 bpw, KV 6/5, native MTP** — GPU-resident candidate with strong third-party 16 GB results; verify exact model + runtime support and actual memory use locally.
2. **Upstream llama.cpp, low-bit GGUF + native MTP** — portable and well-instrumented reference/control; investigate quant-specific Blackwell decode kernels and low-bit alternatives.
3. **Jake's pinned llamAmpere release** — a **source/experiment reference**, never our production baseline until an SM120-safe clean build is verified. Do not enable SM86-only fused paths on Blackwell.
4. **2-bit/weight and alternative model candidates** — include only with independently measured coding/agent quality, actual prompt/decode performance and context; model quality cannot be inferred from bpw or headline speed alone.

Keep the API adapter small (loopback OpenAI-compatible endpoint where supported), model formats and upstream engine forks external, and source modifications in explicitly versioned patch series. Do **not** vendor downloaded model weights, proprietary code, machine identifiers, prompts, credentials, user paths or raw private workstation logs in this public repo.

## Next milestone

Capture a sanitized GPU/OS/driver/CUDA/thermal/power/VRAM inventory on the Zephyrus and freeze reproducible source revisions and models. Run stock no-MTP and MTP baselines over real coding/agent, reasoning and retrieval fixtures **before** any CUDA optimization. The machine was unavailable at project initialization; hardware experiments are pending. GitHub-hosted CI may test CPU-only parsers, benchmarking math and documentation; ordinary hosted runners are **not** stand-ins for a local RTX 5080 Laptop GPU.

## Repository description (for GitHub About)

`Research-driven, Blackwell-optimized local LLM inference for 16 GB RTX 5080 laptops: quantization, MTP, KV cache, kernels and reproducible quality/speed benchmarks.`

## Acknowledgements and upstream

Original performance research: [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere); model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); baseline engine: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); EXL3 engine: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3). Changes copied from upstream must preserve their applicable licenses and attribution; do not imply affiliation or ownership of their results.
