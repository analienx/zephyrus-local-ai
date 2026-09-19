# Zephyrus inference qualification protocol v0

**Protocol first, numbers second.** No measured Zephyrus results are present as of 2026-09-19. This document is a specification; records must use `measured` / `estimated` / `external` provenance labels, never mix them into one ranking.

## Comparable experiment manifest

Each run must record: UTC timestamp, sanitized hardware ID (GPU SKU, power limit, total/free MiB VRAM, GPU compute capability, OS/WSL, driver, CUDA runtime/toolkit, AC/battery and power mode), engine repository/commit/build flags, CUDA graph/cache flags and **effective** kernel dispatch, model/revision/SHA-256/quantization/calibration/vision placement/MTP-head quant, template/tokenizer versions, API flags, active and reserved context, exact prompt fixture SHA, actual input/output token counts, seed, sampling chain, thinking mode, MTP depth/draft cache/type and speculative acceptance, target cache format, concurrency, fan/thermal mode, execution process IDs and any background GPU workload. Record output text hash and semantic validity in protected local artifacts; do not publish private prompts or machine paths.

Do not make a comparison if source/weights/config changed outside the single stated experiment variable. If two engines require different KV formats or templates, label that comparison `system-level` rather than attributing it to one kernel. Never compare decode with prefill+decode or speculative aggregate speed with non-speculative per-step speed. `GB`, `GiB`, `MB` and `MiB` are not interchangeable.

## Workload matrix

| Dimension | Required cells |
| --- | --- |
| Active context | Short 2–4K; 16K, 32K, 64K, 100K if supported; report real active tokens and whether context allocation is lazy or pre-reserved. |
| Output | 256/1024 for interactive latency, >=2048 for stable decode and an extended multi-turn coding/agent session. |
| Sampler | Greedy for exact replay and separate sampled T=1/top-k=20/top-p=.95 (or published equivalent) for user-visible performance; preserve request settings. |
| Speculation | OFF; MTP k=1,2,3,4 as runtime permits, with draft acceptances, draft overhead, verification widths and separate draft KV memory. |
| Task types | Code creation + tests; code edit + regression tests; tool-call/JSON validity; agentic multi-turn repo work; general reasoning; multilingual; retrieval/needle at varied positions; prompt-cache reuse; optional vision with a separately measured GPU footprint. |
| Profile | First warmed AC steady-state, cold load/first request separately; 20–30 minute continuous plugged-in run at a fixed ambient/fan/power mode, no thermal/power-limit oscillation hidden from report. |

## Metrics

**Quality (first-class):** unit/integration test pass rate for code, exact tool-schema validity, task completion within tool-turn budget, deterministic factual/retrieval recall at early/middle/late positions, malformed/repetitive output detection, robust reasoning result correctness and sampled multi-seed stability. Report both per-task and aggregate results alongside reference high-precision model and the same engine at best feasible quant. Quant quality is not proven by greedy hash equality across *different quantizations*; MTP correctness is a separate property.

**Latency/throughput:** TTFT including prompt ingestion; prefill prompt tokens/s and total first-token wall time; median/p95 intertoken latency, decode accepted/output tokens/s excluding prefill, end-to-end completed-task seconds, tool-call turnaround. Track GPU kernel time, host/sync time, tokens/pass, passes/s, accepted draft positions, rejection count and total MTP overhead where exposed. At fixed draft depth and compatible accounting: `passes = predicted_n - draft_n_accepted`, `tok/pass = predicted_n/passes`, `passes/s = passes/decode_seconds`, `tok/s = tok/pass * passes/s`; validate engine-specific counter semantics before relying on that identity.

**Memory:** GPU driver free/used before load, after weights, after KV/graph allocation, after prompt population, while decoding and peak; engine allocated/reserved; per-component budget (weights, target KV, draft KV, GDN state, graphs, activations); host RAM and disk paging; repeat after ten request/cancel/reuse cycles. OOM is a failed cell, not something to hide by offloading the largest layers to CPU; CPU-offloaded results belong in a clearly labelled separate tier.

**Power/stability:** GPU utilization/clocks, actual power, throttle flags and temperature throughout long runs, sustained tok/s ratio relative to the warm burst, process crashes, numerical errors, intermittent incorrect output and model-load reliability.

## Paired A/B decision rule

Pin all controls; warm both arms under equivalent conditions; alternate A/B/B/A (or a balanced randomized order) for at least five pairs, collect whole-run raw records and note source and output hash. Report paired median, spread and confidence interval, with *all* cells (including OOM, loops, errors) and a predeclared acceptance gate. Prefer practical gains above timing noise and consistent real-task benefit; a microkernel win without end-to-end win does not ship. Different generated text after temperature >0 may create workload confounds; replay identical fixed tokens for low-level attribution, then independently test sampled natural continuations and distributional behavior. Do not delete an unfavourable seed or compare a contaminated baseline. A repeating/degenerate answer is a **failed quality cell** and cannot be used as a high-speed fixture.

## Speculative decoding correctness

For exact p/q verification, a draft token d from distribution q must be accepted with probability `min(1,p(d)/q(d))`, with rejected draws sampled from normalized `max(0,p-q)`; ensure that the selected top-k/top-p/min-p/temperature/bias/grammar operations are accounted for in p and q. Test greedy byte identity vs non-MTP target, sampled output distribution over many seeds, rejected-token rollback at each draft position, MTP-state recomputation, pending tool call, cancellation, prompt-cache reuse and long-context recurrent state. A 1:1 greedy match does **not** establish sampled equivalence; speculative decoding does not repair quantization loss.

## Artifacts and CI

Public `results/` holds only explicitly sanitized fixture manifest, source/revision hashes, numeric summaries, hardware SKU and aggregate correctness metrics. Ignore raw prompts, profiler traces, memory dumps, personal file paths, large binary/model weights, local client tokens and logit dumps with sensitive data. GitHub Actions may run low-cost CPU schema and math tests and documentation checks, but its host does not have our 16 GB mobile GPU; actual performance labels require locally captured receipts. Keep cloud jobs manual or narrowly scoped when appropriate, avoiding expensive unbounded matrices.

## Final-stage execution order (only after the pre-GPU implementation and explicit authorization)

1. Hardware inventory and free-VRAM receipt; test clean CUDA backend ops, validate engine actually uses sm_120 paths.
2. Match model/checksum and quality harness; stock no-MTP baseline in an in-VRAM profile.
3. MTP off/on and k/cache quant A/B with memory and correctness receipts.
4. 3.0 vs 3.5 bpw EXL3, practical GGUF IQ3/IQ4 and a 2-bit candidate on *the same task suite*; compare speed versus quality/context.
5. Only after bottleneck attribution, microbenchmark one promising upstream/Blackwell kernel intervention, test exactness and full workload regression.

## Implemented evaluation tool and current boundary

[bench/README.md](../bench/README.md) documents the current transport-neutral trial format, offline deterministic judges, A/B/B/A schedule, loopback-only future SSE collector and optional final-stage code/GPU-telemetry receipts. The bundled four synthetic smoke fixtures are **harness validation only**, not the eventual coding-agent task suite or measured model comparison. Before the final stage prepare a disposable project with real code-edit and tool-loop tests, verify tokenizer/template hashes and require model/engine dispatch receipts. The collector is dry-run by default and rejects live execution without separate final-stage authorization.
