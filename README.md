# Zephyrus Local AI

**Research-first, reproducible, hardware-specific LLM inference for the NVIDIA RTX 5080 Laptop GPU (16 GB).** Optimize usable model quality, response latency, sustained throughput and practical context, not a peak token/s claim. Qwen3.8-27B is the provisional model, not a permanent dependency.

> **Status · 2026-09-19:** complete 3,080-tensor header inventory for a pinned public mixed EXL3 checkpoint, executable metadata-only residency planner, source audits and CPU-only correctness tests. No Zephyrus GPU result or custom Blackwell kernel is claimed. The original linked X article was not directly retrievable; the author's own GitHub-hosted article, runbook, code and subsequent fixes are the research sources.

[Source-level reverse-engineering index](docs/reverse-engineering/README.md) · [CPU speculative-math CI](https://github.com/analienx/zephyrus-local-ai/actions/workflows/spec-math.yml) · [Architecture](docs/architecture.md) · [Benchmark/quality protocol](docs/benchmark-protocol.md) · [Research issues](https://github.com/analienx/zephyrus-local-ai/issues)

## Why this project exists

[Jake ATX's llamAmpere](https://github.com/JakeATX/llamAmpere) combines model-specific quantization, MTP, compressed KV caches, attention/quant kernels and runtime changes for **24 GB SM86 Ampere**. Its [v0.3 measurements](https://github.com/JakeATX/llamAmpere/blob/36a6bca81/docs/llamampere-v0.3/ARTICLE.md) and later [v0.3.1 GGUF-native EXL3 implementation](https://github.com/JakeATX/llamAmpere/blob/c7a3a742f55d76e46a7c4ccfc40bf01a8be318ee/docs/exl3.md) come from different software revisions. Recompiling SM86 flags or copying a 24 GB layout into a 16 GB Blackwell laptop is not a port; reconstruct the executed operations and their correctness conditions first.

## Completed source-level investigations

- [01 — Full reference inference round](docs/reverse-engineering/01-reference-decode-path.md): draft/target scheduling, author's historical kernel and host cost census, shortlist and attention dispatch, graph and negative experiments.
- [02 — EXL3 and physical GPU memory](docs/reverse-engineering/02-exl3-and-memory.md): bitstream/codebook, Hadamard and scales, GEMV, mixed tensor precision, whole-card memory bounds and unsafe incomplete-operator fallback.
- [03 — Target verifier and recurrent state](docs/reverse-engineering/03-target-verifier-and-state.md): actual coupled p/q verification/residual, effective fallback, replay and September 18 GDN repairs.
- [04 — Native ExLlamaV3 comparison](docs/reverse-engineering/04-native-exllamav3-comparison.md): borrowed target output head, **argmax/ID-match vs sampled q/p-q**, native 6/5 paged-KV physical layout and independent recurrent transaction.
- [05 — EXL3 prefill kernel dispatch](docs/reverse-engineering/05-exl3-prefill-kernel-dispatch.md): native direct-quantized GEMM through 144 activation rows vs GGUF EXL3 reconstruction above 16; autotuning and large-prefill fallback.
- [06 — Blackwell engineering priorities](docs/reverse-engineering/06-blackwell-engineering-priorities.md): separate **offline SM120 quantization** optimizations from actual inference; kernel/attention dispatch, non-portable launch assumptions and scoped implementation hypotheses.

The [mechanism ledger](docs/article-reverse-engineering.md) and [earlier source audit](docs/source-audit.md) remain for cross-reference. Each newer document identifies the source revision and separates source-derived facts, external benchmark results, mathematical deductions and untested proposals.

**Executable correctness groundwork:** [`tools/spec_math.py`](tools/spec_math.py) and [`tests/test_spec_math.py`](tests/test_spec_math.py) independently check the exact one-position p/q output distribution, conditional acceptance, zero-support vocab and randomized probabilities. The public CPU [GitHub Actions run succeeded](https://github.com/analienx/zephyrus-local-ai/actions/runs/35426542090); this does not test upstream C++/CUDA, model quality or the Zephyrus.

## Decisions available from code before any benchmarking

Native Qwen-family MTP shares the target output head and proposes argmax; the GGUF 64K draft shortlist indexes rows of the already resident full head, reducing projection computation **without** reducing full-head VRAM. Jake's later p/q proposal/acceptance policy is different, so engine comparison must separate accepted tokens per round from time per round and control sampling conditions.

From the actual native cache representation, 6/5-bit K/V across 16 target attention layers plus one similarly compressed draft layer models to **26,112 B/token** before page padding, versus the author's **25,984 B/token** q8 K/turbo3 V/q8 drafter. That near equality does not prove a given context fits. The author's 3.0-bpw mixed GGUF EXL3 file is 10.91 GiB; a 65.5K-token prompt plus generated continuation peaked at ~15,688 MiB whole-card on a desktop 3090 Ti. Resident weights, graph allocations, page reserves, recurrent state, reconstruction scratch and desktop graphics remain separate constraints.

Native ExLlamaV3 already provides direct EXL3 GEMM through **144 activation rows**, while the audited GGUF EXL3 port reconstructs fp16 weights above 16 rows. Native also reconstructs beyond 144 rows by default with output slicing and optional high-row transform fusion. Inspect existing native CUDA tiling and SM120 dispatch before attempting a new Blackwell kernel; don't mistake faster offline EXL3 quantization for faster inference.

## Executable local-AI fitness harness (research-only)

The [offline fitness evaluator, balanced A/B/B/A campaign planner, guarded local API streaming collector, optional whole-GPU telemetry and container-only code behavioral-test grader](bench/README.md) turn configuration research into a repeatable **quality × task-completion-time × context × memory** workflow. The included fast-but-wrong candidate is SIMULATED and explicitly fails its tool-call quality gate; none of its numbers are Zephyrus measurements. The live collector refuses execution until separately authorized final-stage validation and an armed static profile. **No live model or GPU benchmark has been run.**

## Architecture and execution gates

Keep the OpenAI-compatible localhost API layer thin and keep source/model revisions and format candidates pinned; preserve attribution and compatible licenses. Do not publish model weights, credentials, private prompts, machine identifiers or raw workstation logs. Candidate lanes are native ExLlamaV3 EXL3, upstream llama.cpp low-bit GGUF and a demonstrably correct SM120-capable build of Jake's newer GGUF EXL3. A different model can replace Qwen3.8-27B when reproducible quality-adjusted throughput supports it.

Finish exact GPU kernel/attention fallback, target sampler and Qwen3.8-specific model-loader audits and build standalone correctness fixtures **before** hardware-specific performance tuning. Then inventory Zephyrus GPU/driver/VRAM/power; compare identical active contexts, quant quality and real workloads; implement one narrow improvement against a proven kernel/host bottleneck. [P0 hardware](https://github.com/analienx/zephyrus-local-ai/issues/1) · [P1 source audit](https://github.com/analienx/zephyrus-local-ai/issues/2) · [P2 engine/model comparison](https://github.com/analienx/zephyrus-local-ai/issues/3).

## Proposed GitHub About description

`Research-driven, Blackwell-optimized local LLM inference for 16 GB RTX 5080 laptops: quantization, MTP, KV cache, kernels and reproducible quality/speed benchmarks.`

## Attribution

[Original research: JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere) · [Model: Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [Reference engine: ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) · [EXL3 format and native engine: turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3). Preserve all applicable license texts for any incorporated implementation; no affiliation or ownership of upstream work is implied.

## September 19 source-audit update (no benchmarks)

The Zephyrus is online, with RTX 5080 Laptop compute capability 12.0 and reported 16,303 MiB GPU memory. Upstream source mirrors are pinned separately from this repository. The revised [reverse-engineering index](docs/reverse-engineering/README.md) now includes actual GDN allocation/replay analysis, Blackwell 3-bpw kernel dispatch, and an [executable source-derived static memory model](tools/static_memory_budget.py) with [CPU-only tests](tests/test_static_memory_budget.py). The static model uses the *official Qwen3.8 geometry*, not inferred generic defaults; it is deliberately not a VRAM-fit certificate.

**Project execution boundary:** source inspection, design, safe offline analysis and CPU-only correctness tests are permitted; **do not launch models, run GPU kernels, inference benchmarks or performance sweeps on the Zephyrus until the implementation and research phases are finished**. The measured-performance phase is last, per owner instruction. Public GitHub-hosted CPU checks do not replace the final Zephyrus validation.

## Native-loader source correction (no GPU benchmarking)

The [pinned loader allocation audit](docs/reverse-engineering/13-native-loader-allocation-lifecycle.md) identifies the text embedding as CPU-resident in native ExLlamaV3, so the earlier serialized-all-weights-on-GPU screen is too pessimistic. The [native loader ledger](tools/native_loader_ledger.py) separates CPU embeddings, GPU weight inputs, KV/recurrent reservations, EXL3 reconstruction, full-cache fallback and optional prefill staging. **Positive headroom is not a model-fit claim.** Loading/autosplit itself executes model forward passes; do not invoke it until final authorized GPU validation.

## First reversible runtime intervention (prepared; no model execution)

[Strict packed-KV fallback guard](patches/exllamav3/0001-opt-in-strict-quantized-cache.patch) + [pinned text+MTP profile](profiles/qwen38-27b-exl3-3bpw-text-mtp.json) + [CPU-only source/metadata preflight](tools/serving_preflight.py) are ready for static review. The baseline upstream worktree remains untouched; an isolated source worktree confirms the patch applies, reverses cleanly and passes syntax compilation. The opt-in guard prevents silent FP16 full-cache expansion on the eager quantized-KV dispatcher; it neither guarantees total VRAM fit nor proves quality or speed. [Full audit and reproduction steps](docs/reverse-engineering/14-strict-quant-cache-profile.md). GPU model loading, inference and benchmarking remain prohibited until the final validation stage.
