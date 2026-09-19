# Pinned source audit — preliminary, not a validated port

**2026-09-19.** Original text of the user-linked X article is inaccessible through the current retrieval interface. The author-maintained v0.3 article, runbook, experiment handover and source repository are directly inspectable. The following are *direct source observations* against JakeATX/llamAmpere commit [`36a6bca81`](https://github.com/JakeATX/llamAmpere/tree/36a6bca81), not claims that the code executed on the Zephyrus.

## Finding S01: MTP drafting has a distribution-dependent fallback

Read [`common/speculative.cpp` at pinned commit, lines 2170–2320](https://github.com/JakeATX/llamAmpere/blob/36a6bca81/common/speculative.cpp#L2170-L2320). The draft loop resets draft-region KV for chained heads while retaining the prompt prefix and switches head per iteration. It chooses the exact p/q branch only when `pq_enabled`, sampling params and result-q storage are available **and** `pq_params_ok` is true. The draft token is sampled using the request-derived sampler and the full candidates are saved in `result_q`. If the backend already sampled a token and cannot supply the matching distribution, p/q is disabled for that position; a non-p/q position instead selects the top candidate. Confidence threshold `p_min` can terminate drafting, while chained-head and shared-memory draft families use different batching rules.

**Blackwell implication:** measure *effective* verifier mode and the actual request sampler, not merely whether `LLAMA_SPEC_PQ=1` or `--spec-draft-p-min 0` is present. Instrument exact/fallback position counts and verify partial rollback across multiple head layers, request-specific top-k/top-p and backend sampling. A generic 'MTP enabled' benchmark masks qualitatively different algorithms.

## Finding S02: fused attention is shape- and capability-dispatched

Read [`ggml/src/ggml-cuda/fattn.cu` lines 1–130](https://github.com/JakeATX/llamAmpere/blob/36a6bca81/ggml/src/ggml-cuda/fattn.cu#L1-L130). The source includes both `fattn-mma-turbo.cuh` and generic `fattn-mma-f16.cuh`/vector alternatives. A built-in optional `GGML_FATTN_PATH_STATS` census reports path, K/V dtype, head dimension D, query width `n_q`, GQA ratio, packed `ncols2` and KV depth bucket. The generic switch checks `turing_mma_available(cc)`, query shape and GQA compatibility; it conditionally excludes misaligned or unsupported mask/bias cases from GQA optimization.

**Blackwell implication:** enable the dispatch census first. Confirm which path *actually runs* at MTP k=1–4, target versus drafter, short versus long KV, and Blackwell-compatible attention shapes. Compiling sm_120 or enabling the fused flag does not prove an optimized MMA path was selected. Establish fallback behavior and correctness before microbenchmarks.

## Finding S03: do not misattribute the v0.3 speedup

The [v0.3 article](https://github.com/JakeATX/llamAmpere/blob/main/docs/llamampere-v0.3/ARTICLE.md) explicitly decomposes `tok/s = tok/pass × passes/s` on clean fixtures and attributes most relative improvement to per-pass kernel throughput, not improved MTP acceptance alone. Its stock-vs-fork comparison has controlled source SHAs and excludes looping outputs. The article's P5b/P5c and P6 quantify target and drafter attention separately; DF3 wide verification is built but **not enabled by default** and DFlash2 remained slower in those tests. The [handover](https://github.com/JakeATX/llamAmpere/blob/main/QWEN38_SM86_FRONTIER_HANDOVER.md) records multiple rejected candidate kernels; copying rejected experiments would repeat avoidable work.

**Blackwell implication:** the next audit must extract the exact dispatch tree for compressed attention, verify-width MMVQ/MMQ, quant unpack and GPU sampler. Profiling must rank real kernel time under the **selected 16 GB quant**, not under Jake's ~4.56-bpw 24 GB GGUF.

## Evidence remaining before code

- Diff `36a6bca81` against pinned upstream and TurboQuant, extract touched CUDA functions, host allocation changes and architecture guards. Source audit must include `fattn-mma-turbo.cuh`, quant/MMVQ dispatch and target/draft cache builders.
- Audit `common/speculative.cpp` *verification* side, not only drafting: residual distribution construction, q=0 handling, precision, grammar/bias/penalty eligibility, RNG and rollback. Add independent mathematical tests without copying upstream's assumptions.
- Inspect 2026-09-18 v0.3.1 EXL3-in-GGUF and low-bit/ternary branches as **separate versions**; v0.3 benchmarks do not establish their throughput. For each quant engine, establish how packed tensor bytes, dequant scales/codebooks, MTP head and transient workspace fit in 16 GB.
- Inspect ExLlamaV3 source and pinned release for sm_120 EXL3 decode kernels, target/draft KV and native MTP, then compare supported *effective* settings with llama.cpp. Test a 2-bit candidate's quality against higher-precision reference and 3-bpw alternative before any claims.
- Audit Gated DeltaNet FP32 state and snapshot code on speculative rejection, eviction, prompt reuse and mixed ubatch shapes. The author reports `LLAMA_GDN_REPLAY=1` as a remaining pre-existing issue in v0.3; reproduce/resolve correctness before changes to that path.
- Verify exact Zephyrus GPU architecture, driver, CUDA, total/free VRAM and sustained AC power locally. No measurements or patches can clear the hardware gate while remote access is offline.

**Promotion criterion:** each candidate's source commit, affected function, hypothesis, expected bottleneck, required correctness test, measurable speed+VRAM outcome and rollback flag are linked in a single issue/PR. Preserve upstream license and attribution; do not publish raw private workstation traces.
