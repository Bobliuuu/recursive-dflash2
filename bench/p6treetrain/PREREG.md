# Phase 6 preregistration: tree-aware DSpine drafter (kill test)

Written and committed **before** any variant number was seen (the official-weights proxy validation is reported in the
write-up and is allowed to be seen first; it is the baseline, not a variant).

## Question
Can a DSpine drafter trained for *tree* acceptance beat the official `tencent/Qwen3-8B-DSpine-b16` weights under the
Phase 2 best-first tree (48 nodes, depth cap 31, Qwen3-8B target, T=0)?

## Data
- Prompts: `train/prep_prompts.py` (open-perfectblend, eval-overlap filtered, as Phase 3).
- Responses: Qwen3-8B, thinking off, **greedy (T=0)**, max 1536 new tokens, SGLang AR (`train/gen_responses.py --temp 0`).
  Greedy because the tree is a T=0 method and the proxy needs the target's greedy path.
- Held-out: the last 256 completed samples (never trained on). Train: the rest.

## Proxy metric (`train/train_treeaware.py`, `proxy_eval`)
Teacher-forced on the 256 held-out sequences x 128 random anchors per sequence (fixed seeds, identical anchors for every
variant). Drafter run at block 32 (anchor + 31 positions; positions 1..15 are identical to a block-16 run because the
drafter is block-causal). Truth = the target's teacher-forced argmax at each position, truncated at the first position
where the data token differs from it (so the conditioning stays on the greedy path) or the sequence ends.
- Candidates = top-16 of the draft's marginal logits; table [31,16,16] = SpecForge `DSpineDraftModel.transition_scores`
  (exact last-write refinement conditioned on each predecessor candidate; row 0 on the anchor).
- **proxy chain tau** = 1 + accepted length of DSpine's greedy refined chain (`select_cached`) over the first 15 positions.
- **proxy tree tau** = 1 + accepted length of the best-first tree: N = 48 nodes incl. the anchor (47 draft nodes),
  node score = product of softmax(table row) along the path (temperature 1), 8 children pushed per pop, depth <= 31.
  (Same policy as `bench/p2_e1_tree.py` / the engine kernel.) Also logged: N = 16/32/64, backbone-argmax chain.
- Only ratios across variants on the same proxy are used for decisions.

## Variants (all warm-started from the official b16 weights)
Common recipe: block 32, LR 2e-4 cosine, 4% warmup, correct-predecessor curriculum off (replace-prob 0), 64 anchors per
sequence, 32 sequences per optimizer step, **925 optimizer steps** (same as Phase 3), seed 42, bf16 + fp32 master AdamW.
Tree loss L_tree = sum_i w_i [ CE_refined(a_i | true predecessor, softmax over the 16 candidates)
                                + 0.5 * CE_backbone_fullvocab(a_i) ],  w_i = exp(-(i-1)/14),
where a_i is the target's argmax at position i. The refined term is the per-step factor of the joint path probability that
the best-first tree ranks by, so it pushes up the joint probability of the target's greedy path (listwise over the 16
siblings of each true node); the backbone term keeps the greedy token inside the top-16 candidate set.
- **V1 `mix1`**: L = L_DSpine + 1.0 * L_tree
- **V2 `mix4`**: L = L_DSpine + 4.0 * L_tree (tree-dominant dose)
- Control `ctl` (DSpine loss only, same data/steps): run **only if V1 or V2 passes**, to attribute the gain to the loss
  vs the greedy data. It does not change the pass/kill decision.

## Bar (decision)
A variant PASSES if, on the held-out proxy, at its final step:
- proxy tree tau (N=48) >= **1.05 x** the official weights' proxy tree tau (N=48), and
- proxy chain tau (15 tokens) >= **0.98 x** the official weights' proxy chain tau.
No variant passes -> **KILL** (no engine runs). Intermediate evals (every 300 steps) are logged but the decision uses the
final checkpoint only.

## Only if a variant passes
Convert the checkpoint (Tencent `selected_prefix_bridge.*` format), serve with the Phase 2 tree path (N=48, D=31) and chain
b16 on the Phase 2 protocol (5 datasets x 40 prompts, T=0, c=1, max_new 2048), official weights re-run on the same box as
the control, plus the teacher-forced losslessness check (`bench/p2_tf_check.py`).

## Budget
One Vast A100-80GB (label `rdf2-treetrain`), hard cap 20 USD total for the phase.
