# DFlash / DFlash2 architecture notes (from reading z-lab/dflash `dflash/model.py`, MIT)

## DFlash draft (checkpoint z-lab/Qwen3-4B-DFlash-b16)
- 5 Qwen3-style decoder layers, hidden 2560, 32 heads / 8 KV heads, head_dim 128, MLP 9728, tied embeddings (uses the target's embedding and LM head).
- Target features: hidden states of target layers [1, 9, 17, 25, 33] concatenated -> `fc` (5*H -> H) -> `hidden_norm`.
- Every draft layer attends over K/V = [proj(target_hidden) ; proj(block hidden)], queries from block positions only
  (the "KV injection"). Bidirectional within the block (is_causal False for full_attention layers).
- Block input = embeddings of [anchor token, MASK x15]; block_size 16, mask_token_id 151669. One forward predicts 15 tokens.
- Greedy: draft_tokens = argmax(target_lm_head(draft_hidden)). Verify: one target pass over the 16 tokens; accept longest matching prefix, plus a bonus token.
- Target hidden features for the accepted prefix are kept as the next draft's context (`target_hidden[:, :produced]`).

## DFlash2 additions
- `GroupedDynamicCausalConv` before attention and before the MLP in every layer: a depthwise causal conv whose per-group taps are
  a static base kernel plus a kernel predicted from the hidden state (`kernel_projection`). Config: conv_kernel_size, conv_group_size.
  It `prepare`s (conv on input, emit dynamic kernel) and `finish`es (conv on output with the same dynamic kernel).
- `CandidateSelector`: keep top-k logits per position (`selector_top_k`), then pick one path with a left-to-right bigram score:
  score(c_t | prev) = logit_t(c) + <E_pred[prev] * W h_t , E_succ[c]> with rank-`selector_rank` codebooks. Greedy argmax or sampled
  (q rows returned for lossless rejection sampling). The path is sequential over positions in the reference code (python loop) -> a latency cost to fuse.
- Sampling mode uses draft probs/indices in `_rejection_sample` to stay lossless.

## Implications for recursive training
- The draft's context is the *target's* accepted hidden features, so on-policy data needs target hidden states along accepted prefixes.
- Selector path adds within-block dependency, which is what keeps acceptance from decaying at later positions; at block 64+ this matters more.
## Released DFlash2 hyperparameters for Qwen3.8-27B (read from config.json, 2026-11-08)
Checkpoint `z-lab/Qwen3.8-27B-DFlash2` (identical-size mirror `incoai/Qwen3.8-27B-DFlash2`; both 3,848,817,896 B):
- `DFlash2DraftModel`, model_type qwen3, 5 layers, hidden 5120, 32 heads / 8 KV, head_dim 128, MLP 17408, vocab 248320, untied (target embed/head are used).
- `dflash_config`: **block_size 8**, **selector_top_k 16**, **selector_rank 256**, **conv_kernel_size 2**, **conv_group_size 16**, mask_token_id 248070, target_layer_ids [5, 19, 33, 47, 61] (target has 64 layers).
- All 5 layers `sliding_attention`, sliding_window 2048, `is_causal: false` (bidirectional in block), rope_theta 1e7, bf16.
- Target Qwen3.8-27B: 64 layers, 48 GatedDeltaNet (linear_attention) + 16 full-attention (every 4th), hidden 5120, 24 q-heads / 4 KV, head_dim 256, one MTP head (mtp_num_hidden_layers 1).
- Contrast, DSpine-b16 for the same target: block 16, target_layer_ids [1,16,31,46,61], sliding_window 8192, block-causal (`draft_block_causal`), code_bridge rank 512 / code_size 1024 (see docs/research/dspine.md).
