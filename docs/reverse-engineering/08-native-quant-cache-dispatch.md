# Native ExLlamaV3 quantized-cache dispatch: source-level audit

Static source audit, 2026-09-19. No model load, GPU inference or benchmark was run. Source checkout: `turboderp-org/exllamav3@02aef45cd681b960a00afcd0749a4ab99e6c1bfe`; local mirror is outside this project and is not vendored.

## Cache allocation and memory representation

[`exllamav3/cache/quant.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/cache/quant.py) stores K and V as separate packed `int32` bit streams, each with an FP16 scale per 32 scalar values. The constructor supports K/V widths 2–8 and requires the reserved token count to be a page multiple. `get_qkv()` returns packed tensors and scales **without allocating an FP16 cache**. `get_kv()` instead creates two full cache-shaped FP16 arrays and dequantizes them; do not equate selecting the quantized cache class with taking the fast compressed-attention path.

For the documented Qwen target geometry (16 attention layers, 4 KV heads, head dim 256), a 6/5 cache consumes `(6+5)*1024/8 + 2*1024/32*2 = 1536` bytes/token/layer, or 24,576 bytes/token target. One same-geometry draft attention layer adds 1,536 bytes/token: 26,112 bytes/token total before page rounding, metadata, transient staging, model weights and recurrent state. Verify actual MTP cache head geometry and total reserved pages from the selected loaded checkpoint, not a generic configuration label.

## Dispatch is a capability check, not a model setting

[`attention_fn/dispatch.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/modules/attention_fn/dispatch.py) performs an **up-front packed-cache write** (`update_kv_direct`) and passes `get_qkv()` only if `_qc_staging < 2`, the cache is `CacheLayer_quant`, `compand_a == 0`, Q is FP16, `dim <= 512`, `dim % 32 == 0`, and `cu_seqlens is None`. Otherwise it calls `get_kv()` and dispatches the conventional path, potentially allocating cache-sized FP16 temporaries. We must audit the selected Qwen3.8 path for all these conditions.

Compressed queries are restricted to `_fns_qc`: `fn_triton_paged_attn_decode_qc` for `q_len <= 16` with no non-causal spans, and `fn_triton_paged_attn_prefill_qc` for longer or span queries; both require FP16, `dim <= 512`, multiple-of-32 dimensions. The dispatcher maintains distinct `fn_qc` and ordinary `fn` hints. If compressed dispatch has no match it raises instead of silently interpreting `k_cache=None,v_cache=None` as uncached attention. This fail-closed property must survive any port.

## What the compressed attention kernel actually does

[`attention_fn/triton_paged.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/modules/attention_fn/triton_paged.py) stores quantized K/V in per-32-value Hadamard-rotated groups, with per-group scales. For attention, the Q representation is rotated once, in-loop K/V unpack-and-scale stays in the rotated basis, and the output is inverse-rotated; no per-KV-tile Hadamard is needed. It implements bit-plane unpacking for widths 2–8 and shape-dependent GQA tiles. A new GPU path must preserve the orthonormal scale, offset/midpoint, masking, causal positions, scale application and inverse rotation—not merely dequantize raw words and apply standard dot products.

`EXL3_QC_STAGING`: `0` keeps decode **and prefill** in packed-cache online-dequant mode with minimal staging VRAM; `1` (default) retains online decode but may stage prefill; `2` forces the conventional full FP16-cache path for comparison/debug. The default prefill-stage threshold is `q_len >= 256`, and only the qualifying causal/no-new-KV path stages. The staging array is sized to the *referenced* and optionally bounded KV window, rounded to a power of two in pages—not a permanent whole-pool FP16 copy. It is still a transient VRAM peak that must be counted against a 16 GiB Windows desktop.

Existing Blackwell attention policy is **already conditional on device capability**; for padded head dim <=128 it selects narrower KV tiles than on older GPUs, with comments documenting desktop GPU measurements. Do not transplant an SM86 tile geometry or interpret the desktop result as a measured laptop result. The existing code also distinguishes quantized decode from prefill and handles q_len up to 16 in the former.

## Design implications before GPU work

1. Keep the native quant-direct dispatcher intact as a reference. Any novel implementation needs explicit per-call path accounting: `qc_decode`, `qc_prefill_online`, `qc_prefill_staged`, `fp16_fallback`, and `unsupported`.
2. Default to no hidden expansion of full cache arrays when calculating a 16 GiB fit; use online mode when capacity is tight, and allow staged prefill only if a preallocated peak-memory cap permits it. A setup-time nominal cache size is insufficient.
3. Maintain independent numerical tests for rotated-cache attention (K/V per-group scales, exact and partial pages, GQA heads, causal mask, bounded/sliding cache, queries 1/2/5/16/17/256, recurrent rollback and graph-address stability). Tests that execute GPU kernels belong to the **last-stage hardware validation**, per project instruction.
4. Verify the concrete Qwen3.8 architecture and selected checkpoint satisfy the fast-path predicates, rather than assuming that configuring 6/5 cache globally guarantees online decompression.

**Evidence boundary:** This is a source trace, not a Blackwell benchmark, memory-allocation measurement or proof of numerically correct inference on the Zephyrus. Source commit and exact dispatch predicates are recorded to make those later tests falsifiable.
