# Exact source-parity and safety gates still open

This is a **live static-research gap report**, not a claim that the entire original article has already been reimplemented. It specifies concrete source-level evidence still needed *before* an SM120 optimized runtime is shipped.

## Proposal, verification and distribution

- Audit `common/speculative.cpp`'s `pq_params_ok`, mirrored draft sampler composition, backend-sampling token ID remapping, and actual stored q candidate-list length for target top-k/top-p/min-p and penalties. Exactness follows only if q represents the distribution actually used to sample that token.
- Audit `common/sampling.cpp` verifier `common_sampler_sample_and_accept_n_pq`, including fallback when grammar/reasoning budget or temperature 0 makes p/q ineligible, replay's last-token exception, and cases where q_d is numerically zero despite a proposed d. A mathematical identity in the CPU tests does not prove implementation correctness under all these branches.
- For native ExLlamaV3, record the Qwen-family `sample_from_state(argmax)` producer and generator's sampled-target ID-match consumer as **different proposal law**; do not compare against Jake's p/q implementation as if only kernels differ.
- Test stop-token handling and constrained output, custom sampling chains, prompt cache replay, task cancellation and non-greedy sampling under each engine's supported semantics. Compare *distributions* rather than requiring identical random token sequences from two valid stochastic algorithms.

## Kernel dispatch and tensor semantics

- Source-trace `ggml_cuda_exl3_gemv_supported(T)` and native `LinearEXL3.forward` down to actual kernel specialization for T=1,2,3,4,5,16,17,32,64,144,145,512,1024. The source establishes GGUF reconstruction beyond 16 and native beyond 144 **by default**, not whether their compiled branch executes faster on SM120.
- Audit GGUF EXL3's extra sources `suh`/`svh` and the graph allocator, backend `supports_op`, CPU/other-device fallback; any incomplete transform must fail closed. Separate fp16-vs-fp32 accumulation changes and near-tie argmax drift from outright lost transforms.
- Map actual compressed attention kernels in native 6/5 and author q8-K/turbo3-V for singleton, draft singleton and verifier widths; confirm whether native `get_qkv` direct path is used or `get_kv` expands full f16 arrays, with explicit dtype, GQA and SM120 predicates.
- Verify CUDA compute capability is used in the **inference** GEMM/attention kernel selector rather than assuming the native offline `quantize_tiles_optimized` sm_120 path accelerates token generation. Record PTX/SASS/instantiated template and effective launch shape when hardware is accessible.

## Resident-memory accounting and quality

- Confirm exact Qwen3.8-27B HF/EXL3/GGUF model variant, real tokenizer hash, output head sharing and tensor count. Protect high-impact tensors according to verified KL/task regressions, not a blanket 2/3 bpw rule.
- Reconcile mmap file size and actual resident GPU bytes, stage-by-stage per-layer reconstructed matrices, native KV page pools/scale blocks and draft cache, GDN checkpoints and host RAM, CUDA graphs, workspace fragmentation and Windows desktop allocations. `n_ctx` max and populated prompt tokens are different quantities.
- Validate both coding-agent quality and attention/GDN long-context behavior under lower cache precision; speculative decoding correctness does not guarantee quality retention from weight or KV quantization.

## Recommended first implementation boundary

The **most specific source-supported candidate** is a safe, correct, bounded direct-trellis GEMM for GGUF EXL3 activation widths 17–144, informed by native ExLlamaV3's existing quantized GEMM and autotuner. It is contingent on numerical parity, attribution/license compatibility and actual GLM/GDN/Qwen layer-layout support. It remains an unimplemented research direction; adapting existing native runtime may avoid the work entirely. No GPU optimization should be called successful before on-device verification.
