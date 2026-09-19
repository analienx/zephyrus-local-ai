# Architecture and implementation decision (ADR-0001)

**State:** research architecture accepted as initial plan; engine/model and numerical targets are *not* selected or verified. **Date:** 2026-09-19. Scope: a single locally owned Zephyrus G16 with nominal RTX 5080 Laptop 16 GB. ASUS advertises up to 120 W graphics power for some G16 5080 configurations; treat actual SKU, power mode, clocks, GPU memory bandwidth, RAM and driver as unknown until inventoried. The desktop RTX 5080 is not a comparable power/thermal reference.

## Product contract

Private, local-only model serving for code editing, agents, long-file reasoning and optional image understanding, usable with clients speaking an OpenAI-compatible API. Optimize the Pareto frontier of (a) task-success/answer quality, (b) end-to-end latency including reasoning and tool turns, (c) sustained output tokens/s and prefill, (d) usable active context, (e) stability and power/temperature. No speed-only winner can displace a model that reliably completes more tasks, and a nominal 262K model limit is *not* a claim of usable 262K resident context.

## Deployment boundary

```text
Local client / coding agent / test harness
    -> loopback-only API adapter (auth, health, streaming, request instrumentation)
    -> profile manager (explicit, reversible selection; no opaque per-token routing)
    -> pinned external inference engine (ExLlamaV3/TabbyAPI OR llama.cpp)
    -> immutable model/quant artifact + tokenizer/chat template + MTP head
    -> CUDA sm_120 kernels, target/draft caches and recurrent-state manager
    -> RTX 5080 Laptop GPU + Windows/WSL or native Linux (chosen after measurements)

Experimental branch: isolated source patchset -> correctness harness -> A/B report
                                     -> only then production profile promotion.
```

Keep the server interface independent of model formats. The adapter never rewrites chat templates or tool-call semantics on its own. Preserve complete reasoning/tool responses as permitted by the runtime, and avoid unverified format conversions. One GPU process and one interactive request slot at baseline; enable concurrent sessions only after memory and latency qualification. Endpoint binds to `127.0.0.1` by default; do not expose to LAN without explicit configuration and authentication. Do not commit user prompts, personal telemetry, secrets or raw client logs.

## Why an engine-selection gate precedes our own CUDA fork

- **ExLlamaV3 native EXL3**: strong research baseline for 2.5–3.5 bpw dense weights under 16 GB, with configurable KV cache and MTP where supported. Its existing Blackwell kernels may already outperform a GGUF fork. Test text, tool calls, native Qwen recurrent state, draft head and optional multimodal support; no engine-specific capability is assumed until measured.
- **Upstream llama.cpp / GGUF**: auditable C++/CUDA baseline with MTP and fine-grained KV controls; supports reproducible model/quant controls and source-level kernel instrumentation. For a 16 GB resident model, compare IQ3, selective mixed precision and MTP-head quantization, but never silently use CPU offload in a GPU-resident comparison.
- **Jake's llamAmpere**: code and performance hypothesis donor, *not* a wholesale dependency. Current published v0.3/v0.3.1 numbers target SM86 and 24 GB. A clean SM120 build, memory fit, backend-op tests and matched source/effective-launch audits are mandatory before cross-engine benchmarking.
- **Alternate low-bit / FP8-NVFP4 backends**: evaluate truly resident weight+KV+draft configurations and model-task accuracy first. INT2 marketing alone does not show hardware-native instructions or a quality-preserving 2-bit quant. FP8 weights of a dense 27B model are not automatically a 16 GB solution.

Avoid two engine forks and a giant orchestrator. One selected production runtime and one pinned control runtime are enough initially. Store source pin + narrow patch series, not copied upstream source trees or model weights. New CUDA kernels require profiler evidence that an already supported sm_120 kernel is the bottleneck and cannot be fixed through configuration or an upstream contribution.

## 16 GB allocation ledger — measure, never guess

`free at idle = physical VRAM - desktop/display/other allocations`. `available serving = free at idle - safety margin`. Total serving includes resident target weights, vision tower/projector if on GPU, embeddings/output head, target KV, **separate MTP draft KV**, GDN/recurrent states and rollback snapshots, graph-capture pools, activation workspace, temporary buffers, allocator fragmentation, batch allocations and other CUDA processes. For comparison, publish peak **allocated** and **reserved** GPU memory, model load overhead, exact OOM position and sustained residency. Advertised 16 GB is not 16 GiB free to a serving process. Leave an empirically measured safety margin and fail closed (reduce context or change profile) rather than letting an OOM or OS paging occur.

Use workload profiles rather than one magical universal quant: `quality` (largest model/quant that fits usable context), `balanced` (high-quality 3-bpw target if verified), `long-context` (lower-weight or paged-KV design if recall tests pass), and `vision` (account separately for projector and image tokens). Profiles are configuration manifests, **not** separate permanently loaded model servers. Never switch quant within a conversation without explicitly invalidating/rebuilding the relevant state.

## Research and implementation sequence

### Gate 0 — sanitized hardware inventory

Record GPU name/PCI ID/compute capability, total/free VRAM, actual power limit and mode, sustained clocks and temperature, driver/CUDA toolkit, CPU/RAM and memory bandwidth, Windows/WSL/native Linux version, engine version, filesystem/cache placement and other GPU consumers. Check driver-visible CUDA in WSL independently. A restart or AC power mode difference invalidates an otherwise paired A/B. Save only non-identifying summaries in Git; local full inventory stays ignored.

### Gate 1 — immutable model and engine controls

Pin exact source commits, CUDA/PyTorch version, model revision and SHA-256, quant calibration data, MTP-head type, tokenizer, sampler defaults, chat template, tool-call adapter, context, memory cache and output length. Compare the same model at multiple quantizations; compare other models on the **same tasks** without claiming byte-identical text. Confirm legal redistribution/licensing before public code inclusion.

### Gate 2 — capacity and quality frontier

At 8/16/32/64/100K active tokens as feasible, compare native 3.0 bpw EXL3 + 6/5 KV + k=2 MTP, 3.5 bpw where it fits, low-bit GGUF + supported MTP, and a 2.0–2.5-bpw candidate only if quality passes. Include no-spec controls, prompt processing, time-to-first-token, decode, warm/cold starts, grammar/tool use, structured data, long-context recall, coding correctness and actual VRAM peak. Include a 20–30 minute plugged-in sustained test to detect thermal throttling. Make a decision based on task success **and** speed at the user's active context, not a desktop 512-output-token headline.

### Gate 3 — profiler-driven micro-optimization

Extract GPU kernel breakdown using Nsight Systems/Compute or an equivalent compatible profiler; classify target matvec, quant-unpack, target/draft attention, GDN, sampled verifier, host/sync, graphs and cache transfer. Profile verification widths separately and record achieved memory bandwidth, occupancy, register/spill and shared-memory pressure. Check source-path selection: the fast kernel may be compiled but never called. Pick exactly one largest measurable cost; compare an upstream implementation or launch tuning before writing custom CUDA.

### Gate 4 — isolated implementation and correctness

Small branch per hypothesis, opt-in flag + kill switch + architecture check, deterministic short test and full workflow gates. For changes to numerical kernels use fixed-token logit deltas, perplexity/KL, greedy token comparison, sanitizer and tool-task accuracy; for speculation use exact p/q distributional equivalence at sampled temperatures, grammar/penalty fallback and recurrent rollback canaries. For model quantization use task metrics against high-precision references; never describe it as lossless. Reject or revert if quality or sustained thermal stability degrades despite higher tok/s.

### Gate 5 — local product and publication

Ship one selected server profile with a reproducible launcher, startup GPU/free-VRAM preflight, local health endpoint, safe shutdown, model SHA checks, structured opt-in performance logs and test manifests. Portable examples for Windows/WSL/Linux follow after a verified local path. Publish plots/results only with real provenance (machine, source SHAs, weights, exact flags, context, prompt, sampler, output and confidence intervals). No unattended GPU benchmark or auto-merge from generic cloud CI.

## Repository topology after the research gates

`docs/` — design, source-review ledger, licensing, decision records and test protocol; `profiles/` — schema-checked versioned runtime/model configurations; `bench/` — fixture definitions, low-overhead benchmark runner and exact raw-result schema; `scripts/` — inventory, build pinning, launch/preflight; `patches/` — narrowly scoped upstream patches only after need is proven; `tests/` — CPU semantics, GPU backend and task-based quality checks; `.github/workflows/` — low-cost CPU unit/lint/license checks, **no** hosted-GPU benchmark claims. Keep generated binary/model/trace directories untracked.

## Architectural stop rules

Stop optimization if its claimed gain is smaller than noise, derived from a loop/invalid output, limited to a different GPU/quant/workload, paid for by CPU spill or repeated downloads, based on an unverified CLI flag, or violates a quality gate. If the unmodified engine meets the target, remain upstream; the best maintained optimization may be **no fork**.

## References

[ASUS 2025 Zephyrus G16 specifications](https://rog.asus.com/cz/laptops/rog-zephyrus/rog-zephyrus-g16-2025-gu605/spec/) · [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) · [llamAmpere article](https://github.com/JakeATX/llamAmpere/blob/main/docs/llamampere-v0.3/ARTICLE.md) · [Blackwell hardware capabilities](https://developer.nvidia.com/cuda/gpus) · [ExLlamaV3](https://github.com/turboderp-org/exllamav3) · [llama.cpp](https://github.com/ggml-org/llama.cpp).
