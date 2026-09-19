# Qwen3.8-27B recurrent memory: actual allocations and replay design

Static source audit (2026-09-19); no model load, CUDA kernel run, benchmark or GPU allocation. Primary configuration: [Qwen/Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json). Pinned code: [native GDN state](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/modules/gated_delta_net.py), [native cache](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/cache/cache.py), [native loader](https://github.com/turboderp-org/exllamav3/blob/02aef45cd681b960a00afcd0749a4ab99e6c1bfe/exllamav3/model_init.py), [Jake recurrent memory](https://github.com/JakeATX/llamAmpere/blob/c7a3a742f55d76e46a7c4ccfc40bf01a8be318ee/src/llama-memory-recurrent.cpp).

## A. The actual model geometry

The official text_config explicitly has 64 layers: the fourth layer of every four is full attention (16 total); the other 48 are linear GDN. For GDN: `linear_num_key_heads=16`, `linear_num_value_heads=48`, both head dimensions 128, `linear_conv_kernel_dim=4`. For conventional attention: 4 KV heads of dimension 256. `mtp_num_hidden_layers=1`. `vocab_size=248320`, `hidden_size=5120`. The public configuration identifies a conditional-generation / vision-capable model, but ExLlamaV3 `Model.from_config` defaults to component `text`; count vision weights only if the vision component is actually loaded.

The *native source default* of 32 GDN value heads in the config reader is overridden by the official 48. A static estimate that inherits 32 undercounts recurrent-state memory by 33% of the actual total; never use the generic default in a Qwen3.8 memory report.

## B. Native ExLlamaV3 allocation formula

`GDNLayerState` allocates `conv_state[batch_slots,fdim_qkv,conv_kernel_dim+max_history]` in BF16 and `recurrent_state[batch_slots,max_history+1,n_v_heads,k_head_dim,v_head_dim]` in FP32. The module defines `fdim_qkv=2*n_k_heads*k_head_dim+n_v_heads*v_head_dim`, so Qwen has `2*16*128+48*128=10240` conv channels. **These are resident allocated dimensions**, not solely the checkpoint payload.

One recurrent matrix, all 48 GDN layers: `48 layers * 48 v-heads * 128 * 128 * 4 bytes = 150,994,944 B = 144 MiB`. One state checkpoint additionally holds `48 * 10240 * 4 * 2 = 3,932,160 B = 3.75 MiB` of convolution state (see `get_checkpoint_size()`); checkpoint payload is **147.75 MiB**, not the same as active per-slot allocation.
With `max_history=4`, batch_slots=1, active matrices are `5*144 = 720 MiB`; conv buffers `48*10240*(4+4)*2 = 7.5 MiB`; total **727.5 MiB resident**, excluding other caches and allocator overhead. `max_history=3` yields **582.5625 MiB**. Every additional configured recurrent batch slot duplicates these buffers. Native `model_init.py` derives `max_history` from the draft model's default size, explicit draft-token count and n-gram settings; Qwen3.5 MTP defaults to four draft tokens. The CLI's normal default autosplit max batch size is 1, but the general `Cache` constructor defaults to 16: identify the actual caller and its arguments, not the constructor default alone.

## C. GGUF experiment: ingredient replay is distinct from normal snapshots

In Jake's `src/llama-memory-recurrent.cpp` / `.h`, normal state storage uses `(1+n_rs_seq)` full-state groups. Opt-in `LLAMA_GDN_REPLAY=1` instead keeps an authoritative full state `s_l`, another full `s_ckpt_l`, and a per-token chronological ingredient ring `ingr_l`. `llama_hparams::n_embd_s_ingredient()` supplies `4*ssm_d_inner` scalar elements per layer and token for GDN (`k,v,g,beta` rows padded to a common value-head width). For this model, `ssm_d_inner = 48*128 = 6144`; at 4-byte state type each ingredient is `4*6144*4 = 98,304 B` per GDN layer. Across 48 layers that is **4.5 MiB per speculative history token**, compared with **144 MiB for a full matrix snapshot**.

At four speculative steps, the source-representation model is `2*144 + 4*4.5 = 306 MiB`, versus `5*144 = 720 MiB`: **414 MiB theoretical matrix/history difference** for one sequence, before conv history, other recurrent formats, intermediate replay graph buffers, per-cell padding and backend allocation. This is NOT a current native ExLlamaV3 option and cannot be credited as real savings in an EXL3 native-memory fit. The author's opt-in implementation and its September 18 rollback/session-restore repair are correctness *references*, not a proven SM120 production path.

## D. Required behavior to preserve if replay is ported

The logical state is the checkpoint plus a chronological accepted-prefix ring; the optimistic forward state may include rejected proposals. Maintain ring span and pending replay length across reject, full acceptance, EOS, serialization, session restore, sequence cloning, relocating memory cells and wildcard sequence removal. A ring that stores just the generated tail without checkpoint identity or accepted-prefix mapping is mathematically insufficient. Test distinct stop/grammar/temperature conditions, repeated partial rollbacks and checkpoint eviction before considering an optional replay mode.

**Design:** keep the native full-snapshot path as the baseline and initially reserve its exact 727.5 MiB for a one-slot, depth-four configuration. Ingredient replay is a separate later-stage memory optimization gated by source parity, numerical correctness and an allocation-aware reason to need the freed capacity. No inference, benchmarking or GPU profiling has been performed here.
