# Opt-in packed-KV fail-closed profile: implementation and CPU-only evidence

2026-09-19. Pinned native source: [`exllamav3@02aef45`](https://github.com/turboderp-org/exllamav3/tree/02aef45cd681b960a00afcd0749a4ab99e6c1bfe). Pinned EXL3 checkpoint: `turboderp/Qwen3.8-27B-exl3@004a887127d8304ca2d5475d3a3c41f1761fdd27`, `SC_3.00bpw_H4_V4`. Only metadata and source were inspected. **No weight load, GPU kernel, inference, numerical-parity or performance test was performed.**

## Exact failure mechanism

The native [`dispatch.py`](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/modules/attention_fn/dispatch.py) calls `CacheLayer_quant.get_qkv()` only when staging mode <2, companding is zero, query dtype is FP16, head dimension <=512 and divisible by 32, and `cu_seqlens` is absent. Any failing prerequisite takes `get_kv()`, which creates TWO full-pool FP16 tensors for the current attention layer, not merely the referenced KV prefix. At 32,768 reserved tokens and 4x256 KV geometry that pool is **128 MiB per attention call**, in addition to the packed cache and other scratch.

The default staging=1 also conditionally creates a separate referenced-window FP16 K/V pair for longer prefill; `EXL3_QC_STAGING=0` disables this *prefill* staging but does **not** prohibit the `get_kv()` fallback. BC-attention has an independent packed-cache eligibility gate (`bc_attn.py`); when it declines, eager dispatch must still respect strictness. The upstream packed-cache candidate list already refuses generic cache-less kernels, preventing an ignored-context silent result.

## Scoped runtime change

[`patches/exllamav3/0001-opt-in-strict-quantized-cache.patch`](../../patches/exllamav3/0001-opt-in-strict-quantized-cache.patch) adds `EXL3_QC_STRICT=1` to the **eager** dispatcher. It factors the original direct-quant eligibility predicate into one variable, then, BEFORE `get_kv()` or cache update, raises an explicit error if a quantized layer would fall back to a full-pool FP16 allocation. Default behavior is unchanged when the environment variable is absent. This is a protection against that allocation path, **not** an assurance of correct/fast packed attention for every call or of a total-VRAM ceiling.

The patch generator [`tools/build_quant_cache_guard_patch.py`](../../tools/build_quant_cache_guard_patch.py) requires exact matching upstream source anchors; it fails on changed source rather than creating a speculative diff. Its output passes `git apply --check` on the pinned untouched checkout. A separate detached worktree was patched and verified with `git apply --reverse --check` and Python source compilation. The pinned reference worktree remains unmodified; the experimental worktree contains only the dispatcher modification.

## Reproducible profile contract

[`profiles/qwen38-27b-exl3-3bpw-text-mtp.json`](../../profiles/qwen38-27b-exl3-3bpw-text-mtp.json) pins the checkpoint and native source revisions, config/manifest/patch SHA-256, text+MTP without vision, 6/5-bit quant cache, compand=0, 32K reserved tokens, history=4, slots=1, strict-mode requirement and online prefill (`EXL3_QC_STAGING=0`). The profile is a **static candidate**, not a production-ready runtime configuration. CPU-only [`tools/serving_preflight.py`](../../tools/serving_preflight.py) verifies all these metadata and geometry conditions and checks the pinned engine code for the audited eligibility predicates.

`--phase prepared` accepts only a clean, unpatched upstream checkout with an applicable patch. `--phase armed` accepts only a checkout where the exact patch has been applied and can be cleanly reversed. An **unpatched** checkout is rejected in armed mode. Neither phase imports `torch`, runs the model, sets environment variables, starts a server, or verifies actual query dtype, runtime dispatch, GPU fit or numerical output. Runtime `EXL3_QC_STRICT=1`, `EXL3_QC_STAGING=0`, and `EXL3_BC_ATTN=1` must be configured **before** importing the inference runtime, not inferred from the profile file automatically.

The 32K screen models ~9,521.29 MiB GPU-weight inputs plus 1,543.5 MiB target/draft KV and recurrent state; ~5,238.21 MiB of the reported 16,303 MiB physical capacity remains **unbudgeted**, not free or proven available. The configured 2,048 MiB *static residual floor* is only a reject threshold for a paper plan, NOT a runtime reserve or safe-memory guarantee. CPU embedding and optional vision are accounted separately. A staged or full-cache expansion, allocator fragmentation, graph capture, reconstructed matrices, device graphics and activation overlap can still exhaust memory.

## Static validation command (never launches a model)

```powershell
$repo = 'C:\Workspace\repos\zephyrus-local-ai'
$src = 'C:\Workspace\repos\zephyrus-local-ai-sources'
Set-Location $repo
python -m tools.serving_preflight --profile profiles/qwen38-27b-exl3-3bpw-text-mtp.json `
  --metadata-dir "$src\metadata-3bpw" `
  --manifest data/qwen38-27b-exl3-3bpw-tensor-inventory.csv `
  --upstream-root "$src\exllamav3" `
  --patch patches/exllamav3/0001-opt-in-strict-quantized-cache.patch --phase prepared
python -m unittest discover -s tests -v
```
