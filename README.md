# Zephyrus Local AI

**Research-first, reproducible, laptop-specific LLM inference engineering for a 16 GB NVIDIA RTX 5080 Laptop GPU.** Optimize *usable answer quality, latency, sustained throughput and practical context*, not isolated peak tokens/s. The target model is provisional: Qwen3.8-27B; alternative model/quant/engine combinations remain open.

> **Status · 2026-09-19:** source reverse engineering and CPU-only correctness tests. No GPU speedup has been reproduced on the owner's Zephyrus. No custom CUDA kernel has been implemented here. Third-party performance reports must not be presented as Zephyrus measurements.

[Speculative math CI](https://github.com/analienx/zephyrus-local-ai/actions/workflows/spec-math.yml) · [Research issues](https://github.com/analienx/zephyrus-local-ai/issues)

## Why this project exists

Jake ATX's [llamAmpere](https://github.com/JakeATX/llamAmpere) optimizes Qwen3.8-27B for a **24 GB SM86 RTX 3090/3090 Ti** by combining a model-specific quant, MTP and exact p/q verification, a draft vocabulary shortlist, compressed attention and CUDA/runtime changes. Its [v0.3 measurements](https://github.com/JakeATX/llamAmpere/blob/36a6bca81/docs/llamampere-v0.3/ARTICLE.md) and later [v0.3.1 EXL3 work](https://github.com/JakeATX/llamAmpere/blob/c7a3a742f55d76e46a7c4ccfc40bf01a8be318ee/docs/exl3.md) describe **different revisions and workloads**. Recompiling SM86 kernels for Blackwell or transplanting a 24 GB configuration does not establish a correct, efficient 16 GB system.

The linked X article itself did not expose its full text through the research interface; we examined the author's independently hosted article, runbook, code and subsequent corrections. Explicit evidence tags in the documents distinguish source observations, author measurements, theoretical deductions and untested proposals.

## Reverse-engineering documents (read before implementation)

| Document | What it resolves |
|---|---|
| [01 · Reference decode path](docs/reverse-engineering/01-reference-decode-path.md) | A full speculative round, p/q mechanism, target/draft output-head split, shortlisting fast-path predicates, attention dispatch and author-reported bottleneck census. |
| [02 · EXL3 and 16 GiB memory](docs/reverse-engineering/02-exl3-and-memory.md) | The real trellis/Hadamard/scaling operator, 16×16 tile warp decode, width-based GEMV/GEMM fallback, unsafe incomplete-operator fallback, per-token KV byte arithmetic and actual published 3-bpw memory bounds. |
| [03 · Target verifier and GDN state](docs/reverse-engineering/03-target-verifier-and-state.md) | Precise eligibility gates, coupled target-draw p/q verifier, residual law, complexity caveats, draft replay, recurrent-state rollback and the September 18 upstream correction. |
| [Source audit — earlier snapshot](docs/source-audit.md) | Initial source map. Superseded by the above on target-verifier detail and new GDN replay repair. |
| [Architecture](docs/architecture.md) | Engine abstraction, durable evidence and implementation boundaries. |
| [Mechanism ledger](docs/article-reverse-engineering.md) | Audit inventory with unknowns explicitly tracked. |
| [Benchmark and quality protocol](docs/benchmark-protocol.md) | Fair comparisons and correctness/quality/thermal gates once hardware is available. |

**Executable groundwork:** [`tools/spec_math.py`](tools/spec_math.py) is a CPU-only, source-independent algebraic reference for exact one-position p/q speculative decoding, including shortlist support and the conditional target-draw accept rule; [`tests/test_spec_math.py`](tests/test_spec_math.py) checks normalization, identity, restricted support, adversarial cases and seeded random distributions. These tests do **not** validate Jake's C++/CUDA runtime or Zephyrus results. They run on a public CPU GitHub Actions runner with path-filtered triggers. The separate hardware-specific workload and profiler harness will be designed after the static source audit, not represented by this math test.

## Key source-derived constraints

**Weight format is not a speed guarantee.** Jake's mixed EXL3 3.0-bpw model occupies 10.91 GiB on disk, but his published short-context MTP-4 run peaks at 14,403 MiB whole-card, and a 65,531-token prompt with continuation peaks at 15,688 MiB on a 24 GB 3090 Ti. The same implementation reconstructs full f16 weights for activation width >16, creating a serious prefill cost; 3.5-bpw and 4.0-bpw settings can exceed a safe Windows 16 GB budget. This is not evidence that any specific context fits our laptop.

**The KV calculation has an exact source basis.** For the author's `q8_0-K/turbo3-V + q8_0/q8_0 drafter` setup: target K 17,408 B/token + target V 6,400 B/token + drafter K/V 2,176 B/token = **25,984 B/token**, excluding recurrent state, model, workspace, allocator and desktop. Do not confuse `-c`, actual populated tokens and peak whole-card memory.

**p/q may silently fall back.** Grammar, reasoning budget, greedy/unsupported sampling, backend-drafted tokens without matching probability arrays or unavailable CPU target distributions can turn off exact p/q at a request or position level. A credible result records the effective route and source revision, not just the CLI flags.

**Recurrent-state restoration is correctness-critical.** The original v0.3 article left GDN replay unresolved; [the author's September 18 PR #8](https://github.com/JakeATX/llamAmpere/pull/8) repairs replay/checkpoint, wildcard sequence removal and short-ubatch snapshots. Any port must include equivalent state transition tests; next-token greedy hash alone is insufficient.

## Architecture decision, not an engine selection

Inspect three independent execution lanes: native ExLlamaV3 EXL3 with supported compressed KV/MTP; upstream llama.cpp low-bit GGUF; and Jake's later GGUF-native EXL3 implementation after verifying its non-standard graph sources, numerical parity and valid SM120 dispatch. A separate smaller/better-quantized model can replace Qwen3.8-27B if quality-adjusted throughput warrants it. Keep an OpenAI-compatible localhost API adapter thin and version any upstream engine changes as attributed, reviewable patches. Never vendor weights, model-license-prohibited content, credentials, private workstation traces or machine-specific paths into this public repository.

**Work order:** complete source-path, fallback, GDN and EXL3 prefill audits; establish reusable correctness/quality fixtures and sanitizer-safe traces; obtain actual Zephyrus GPU inventory; then run a paired engine/model comparison; finally implement a single isolated CUDA optimization against the demonstrated bottleneck. [P0 inventory](https://github.com/analienx/zephyrus-local-ai/issues/1), [P1 source audit](https://github.com/analienx/zephyrus-local-ai/issues/2), [P2 comparison](https://github.com/analienx/zephyrus-local-ai/issues/3).

## GitHub About description

`Research-driven, Blackwell-optimized local LLM inference for 16 GB RTX 5080 laptops: quantization, MTP, KV cache, kernels and reproducible quality/speed benchmarks.`

## Upstream attribution

Performance research: [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere); model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); reference runtime: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); EXL3 format and engine: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3). Preserve applicable license texts and attribution for any incorporated implementation; source audits are not claims of authorship or affiliation.
