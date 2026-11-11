# Phase 7 preregistration: DSpine drafter trained from scratch (Qwen3-8B)

Written and committed before any training number exists. The official-weights proxy numbers (phase 6 baseline) are the reference, not a variant.

## Question
Every earlier training run (Phases 3 and 6) was a ~925-step warm start from `tencent/Qwen3-8B-DSpine-b16`. Does a DSpine drafter trained
**from random init**, at increasing compute, reach or beat the official weights under the same proxy, and under the Phase 4 tree?

## Fixed setup
- Target: Qwen3-8B bf16, thinking off. Drafter: DSpine, official architecture (`train/dspine_sf_config.json`), random init, **no `--init-official`**.
- Data: `train/prep_prompts.py` (open-perfectblend, eval-overlap filtered), responses from Qwen3-8B at T=0.7, max 1536 new tokens
  (`train/gen_responses.py`), as Phase 3. Held-out: last 256 samples, never trained on.
- Recipe: `train/train_dspine.py`, block 16, lr 6e-4 cosine, warmup 4%, gamma 7, replacement-probability curriculum **on** (0.5 peak),
  seed 42, bf16 + fp32 master AdamW. 32 sequences per optimizer step.
- Metric: the Phase 6 proxy (`train/train_treeaware.py proxy_eval`): 256 held-out sequences x 128 fixed anchors, proxy chain tau and proxy tree tau
  at N=48 nodes, depth 31. The official b16 weights are measured on the same proxy first and are the reference (`ref`).

## Tiers and budget caps (user-approved 2026-11-10; each tier gated on the previous result)
| Tier | Anchors/seq | Epochs | Sequences | Hard cap |
|---|---|---|---|---|
| A sanity | 128 | 1 | ~68K | $40 (data generation included) |
| B mid | 256 | 3 | ~68K | $150 |
| C paper recipe | 512 | 6 | ~68K | $300 |
Spend is the sum of Vast charges for the tier's boxes. A box is destroyed when its tier ends or the cap is hit, whichever is first.

## Decision rules (proxy ratios vs `ref`, held-out 256)
- **Tier A gate:** proceed to B only if proxy chain tau >= 0.85 x ref. Below 0.70 x ref: the pipeline or recipe is suspect; stop, debug, do not buy B.
- **Tier B gate:** proceed to C only if proxy tree tau >= 0.95 x ref AND improved >= 5% over tier A.
- **Success (any tier):** proxy tree tau >= 1.05 x ref and proxy chain tau >= ref. Then run the engine (`bench/p4_serve.sh fast 48 31`, 5 datasets x 40 prompts) and
  report tok/s and x AR. Anything short of this is reported as "from-scratch training at this budget does not beat the official drafter".
- Negative or partial results are published in docs/ with the numbers; README only states the final conclusion in one line.

## Reporting
Per tier: GPU, hours, dollars, step count, final loss, proxy chain/tree tau at N=16/32/48/64, ratios vs ref. Raw JSON in `bench/results/p7scratch/`,
logs in `logs/p7scratch/`, pushed to the repo after every checkpoint and eval.
