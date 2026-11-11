# Phase 9 preregistration: deletion tuning of the DSpine drafter (kill test)

Written and committed **before** any mask was trained or evaluated. The official-weights proxy numbers are the baseline,
not a variant.

## Question
Can a learned deletion mask over the official `tencent/Qwen3-8B-DSpine-b16` drafter's MLP `gate_proj` weights (weights
otherwise frozen; Qwen3-8B target untouched, so verification stays lossless) raise mean acceptance length on Qwen3-8B,
either in general or for one domain (code) without hurting the others?

## Method (Lu et al., arXiv 2611.10866; reference code read for this phase)
- Masked: the 5 drafter `layers.*.mlp.gate_proj.weight` matrices (5 x 12288 x 4096 = 251.7M weights). Nothing else.
- One fp32 logit per weight, N(0, 0.01^2), then the Wanda prior: lowest `|W| * ||x||` 1/2048 of each matrix shifted
  by -3 std, the rest by +3 std (N(-+0.03, 0.01^2)). Wanda input norms from the drafter's own training forward on 64
  training sequences.
- Forward W * sigmoid(s * G), s linear 100 -> 1000 over the steps. Loss = DSpine objective (SpecForge b15ad38, the
  Phase 3/6 trainer path, in-process target, curriculum off) - 1e-5 * (1 - step/steps) * sum ||W * mask||^2.
- AdamW on the logits only, LR 5e-4 constant, weight decay 0, clip 1.0. 300 optimizer steps x 16 sequences x 64
  anchors, block 16. Deploy: keep G > 0, zero the rest.
- One configuration only; no hyperparameter search after any proxy number is seen.

## Data
- Training prompts: `train/prep_prompts.py` (open-perfectblend, eval-overlap filtered), 10,000 prompts.
- Responses: Qwen3-8B itself, thinking off, greedy (T=0), max 1536 new tokens, SGLang AR (`train/gen_responses.py`).
- Masks: **gen** (all general training samples except the last 256) and **code** (general samples whose source names
  code + MBPP train/validation/prompt-split responses).
- Proxy eval sets (target greedy responses, same generation settings), disjoint from the frozen benchmark prompts
  (all 100 ids of each `bench/data/prompts/*.jsonl` excluded): GSM8K test, MATH-500, HumanEval, MBPP test (64 prompts
  each, formatted as `bench/prep_prompts.py`), **chat** = held-out open-perfectblend samples with a non-math, non-code
  source (stand-in for MT-Bench, whose 80 prompts are all frozen), and **general** = the 256 held-out samples (64 used).

## Proxy
Teacher-forced, `train/train_deletion.py`: 64 sequences x 128 seeded anchors per domain, identical blocks for every
variant. Truth = the target's greedy argmax, cut where the data leaves the greedy path. Primary metric **chain15** =
1 + accepted length of DSpine's refined greedy chain over 15 drafts (the served chain b16). Secondary: tree48 (depth
<= 15), backbone-argmax chain, top-16 candidate coverage. 95% CIs: paired bootstrap over sequences of the ratio
variant / official.

Variants: official; `gen`, `code`; their Wanda-init masks (`*_wanda`, step 0); matched-count random controls
(`*_rand`: per matrix, the same number of deleted weights chosen uniformly at random).

## Bar (decision, per mask)
A mask PASSES if, on the proxy chain15 ratio to official:
1. on at least one domain d of {gsm8k, math500, humaneval, mbpp, chat}: ratio >= **1.03** and the 95% CI lower bound > 1.00;
2. on every one of those five domains: ratio >= **0.99**;
3. on d: ratio(mask) - ratio(matched random control) >= **0.02**.

No mask passes -> **KILL**: no served runs. If a mask passes: one served benchmark, SGLang DSpine chain block 16,
batch 1, T=0, `bench/run_cell.py`, 40 prompts per dataset, masked draft dir (`train/deletion_mask.py apply`) vs the
official weights, same container and build.

## Budget
Modal, A100-80GB, hard cap 25 USD for the phase, logged in the write-up.
