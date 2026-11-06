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
- Open: DFlash2 hyperparameters (top-k, rank, kernel size) are only in the released checkpoint config; read them from `config.json` of z-lab/Muse-Glimmer-30B-DFlash2.
