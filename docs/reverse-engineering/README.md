# Source-level reverse-engineering index

**Static audit snapshot 2026-09-19.** The linked X article is not directly retrievable here, so primary references are the author's GitHub-hosted article/runbook, pinned source and subsequent fixes. Upstream source observations, author-measured numbers, mathematical deductions, independently run CPU tests and real Zephyrus GPU measurements are distinct evidence classes. **No Zephyrus GPU measurement or SM120 custom kernel is claimed.**

1. [Reference decode path](01-reference-decode-path.md) — speculative round, measured original bottleneck census, draft head/shortlist, attention dispatch and negative results.
2. [EXL3 format and 16 GiB memory](02-exl3-and-memory.md) — codebook, Hadamard/scales, 16×16 tiles, tensor/fallback correctness, GEMV, full-weight prefill reconstruction and KV memory arithmetic.
3. [Actual target verifier and GDN state](03-target-verifier-and-state.md) — coupled p/q implementation and residual law, fallback eligibility, recurrent state and September 18 repair.
4. [Native ExLlamaV3 comparison](04-native-exllamav3-comparison.md) — shared target head, **argmax/ID-match vs sampled q/p-q**, paged 6/5-bit KV layout and independent recurrent rollback.
5. [EXL3 prefill kernel dispatch](05-exl3-prefill-kernel-dispatch.md) — native direct trellis GEMM through 144 activation rows versus GGUF EXL3 reconstruction above 16, cooperative autotuner and wide-prefill fallback.
6. [Blackwell engineering priorities](06-blackwell-engineering-priorities.md) — separate offline quantization from inference; trace compressed attention and kernel-shape portability; scoped implementation designs and explicit unverified assumptions.

## Decisions established without accessing the laptop

**Weight and cache:** 3.0-bpw is a provisional mixed-precision reference, *not* every tensor at 3 bits. The author's v0.3.1 mixed GGUF EXL3 file is 10.91 GiB, with whole-card peaks ~14,403 MiB in a short-context MTP-4 run and ~15,688 MiB after a 65.5K-token prompt plus continuation on a 3090 Ti. Do not extrapolate fit or speed to a 16 GiB Windows laptop. Native 6/5 K/V plus one comparable draft attention layer has modeled 26,112 B/token from actual arrays, almost equal to author's 25,984 B/token q8-K/turbo3-V/draft. Different runtime memory allocation and native paged reserves can matter more than 128 B/token.

**MTP:** native ExLlamaV3's Qwen-family draft path aliases target lm_head, proposes argmax and verifies by target ID-match; Jake's p/q mode proposes sampled q, can accept beyond ID-match and uses target-minus-q residual. Equal output quality or temperature does not imply equal acceptance cost. Jake's q list scan has O(|p|×|q|) worst-case structure, but a top-k20 proposal makes q small in the published setup; do not call this a major CPU bottleneck without representative sizes.

**Prefill:** the audited GGUF-native EXL3 path lacks direct quantized GEMM above 16 activation rows, whereas native ExLlamaV3 includes one through 144 rows with shape/compute-capability/SM-count autotuning. Native also reconstructs above 144 by default, with selective transform fusion and chunking. Full persistent reconstructed 27B fp16 weights are incompatible with this 16 GiB target. The required EXL3 Hadamard and scale transforms cannot be silently discarded by fallback.

**Correctness:** quantized K/V is useful only if the selected attention kernel consumes it in compressed form. GDN rollback, exact sampler conditions, full target vocabulary, draft token remapping, EXL3 side transforms and unsupported-backend fail-closed behavior are part of the end-to-end model. The public CPU [speculative-math tests](../../tests/test_spec_math.py) passed [GitHub Actions](https://github.com/analienx/zephyrus-local-ai/actions/runs/35426542090); these do not execute the model or GPU.

## Remaining static research gates before implementation

- Pin and diff relevant upstream kernel revisions, inspect native GEMM/attention compressed-KV executed path and Qwen3.8-specific model registration, graph and converter.
- Add target-sampler C++ mode/fallback and recurrent checkpoint/rollback fixtures, including constrained outputs, EOS and request interruption.
- Trace shared output head, draft cache, model tensor precision, GDN buffers, paged KV reservation, graph and reconstructed weight scratch without assuming a selected `bpw` implies resident bytes.
- Do not count the native repository's **sm_120 offline quantizer** kernel optimization as a decode tok/s improvement. Any new CUDA kernel must beat a correct existing runtime under exact, attribution-safe, effective-dispatch controlled conditions on appropriate hardware.

## Additional source audits and executable ledgers

7. [Open source-parity and safety questions](07-source-audit-open-questions.md) — explicit unresolved correctness and source-dispatch tests, separated from proven work.

8. [Native compressed-KV dispatch and bounded prefill staging](08-native-quant-cache-dispatch.md) — precise Qwen fast-path eligibility, distinct decode/prefill kernels, Hadamard-rotated packing, `EXL3_QC_STAGING` allocation trade-off, fail-closed fallback and no-benchmark implementation constraints. This is a pinned-source trace, **not** a Zephyrus performance claim.

9. [Actual Qwen3.8 recurrent-state allocation and optional ingredient replay](09-qwen38-recurrent-memory-and-replay.md) — official 48-value-head geometry, exact native FP32 snapshot/conv memory, speculative depth and safe replay-state transaction.
10. [Blackwell EXL3 direct kernel dispatch](10-blackwell-exl3-kernel-dispatch.md) — native 3-bpw small-m GEMV eligibility, conditional SM120 shape envelope, general GEMM fallback and reconstructed-prefill scratch.
11. [Static memory-allocation ledger and CPU-only calculator](11-static-allocation-ledger.md) — source-derived page-rounded KV, MTP and recurrent allocations validated against the official Qwen3.8 configuration without loading a model or running a GPU benchmark.

12. [Pinned checkpoint tensor ledger](12-pinned-checkpoint-tensor-ledger.md) — reconciled all 3,080 public safetensors-header entries, actual mixed bitrates, optional vision/MTP costs and 16 GiB static budgets. [Header-only verifier](../../tools/safetensors_header_audit.py), [full per-tensor CSV](../../data/qwen38-27b-exl3-3bpw-tensor-inventory.csv), [residency planner](../../tools/model_residency_plan.py). NO WEIGHTS DOWNLOADED, NO MODEL LOADS OR GPU BENCHMARKS.

13. [Native loader allocation lifecycle and corrected capacity screen](13-native-loader-allocation-lifecycle.md) — CPU-resident text embedding, MTP weight aliases, deferred-loader copies and slabs, synthetic GPU forwards in autosplit, compressed-cache FP16 fallback, staged-prefill scratch and reconstructed EXL3 weight slices. The [native loader ledger](../../tools/native_loader_ledger.py) replaces the all-serialized-weights-on-GPU screen with a **source-classified but still unmeasured** candidate profile. GPU model loading and performance benchmarking remain deferred to the final validation stage.
