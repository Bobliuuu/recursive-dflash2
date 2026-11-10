# Phase 6: training DSpine for tree acceptance (kill test; Qwen3-8B, 1x A100-80GB)

**Verdict: KILL.** Neither tree-aware fine-tune reaches the preregistered bar (proxy tree tau at 48 nodes >= +5% over the
official `tencent/Qwen3-8B-DSpine-b16` weights, chain not worse than -2%):

- mix1 gives **+2.6%** tree tau, chain -1.5%.
- mix4 gives **+2.8%** tree tau, chain -1.7%.

So no engine runs were done (the preregistration does not allow them without a pass). The Phase 2 setup (official b16
weights, best-first 48-node tree, engine tau 11.29, 627 tok/s) remains the best configuration.

Preregistration: [`bench/p6treetrain/PREREG.md`](../bench/p6treetrain/PREREG.md), committed in `b545798` before any number.
Code: `train/train_treeaware.py` (trainer + proxy), `bench/p6treetrain/box_*.sh`. Raw JSON: `bench/results/p6treetrain/`.

## Question and hypothesis
DSpine was trained for chain acceptance, and its loss is mostly L1 to the target's T=1 distribution (alpha=0.1 CE + 0.9 L1).
The best-first tree ranks nodes by the joint probability of the drafter's conditional q along a path. For greedy verification
the ideal ranking is P(target argmax = c | prefix), not p itself. The hypothesis was that training q towards this (CE on the
target's argmax, conditioned on the true predecessor, over the 16 candidates) raises tree tau at a fixed node budget.

## Data
- **Prompts:** 16,000 from `train/prep_prompts.py` (open-perfectblend; eval-set overlap filtered, 638 dropped).
- **Responses:** Qwen3-8B, thinking off, **greedy (T=0)**, max 1536 new tokens, SGLang AR. 15,993 completed, mean 550
  response tokens, 6.2% truncated. Generation took about 25 minutes.
- **Split:** the last 256 completed samples are held out; the other 15,675 are training data.
- **Training passes:** 925 steps x 32 sequences is about 1.9 epochs.
- **Departure from the plan:** the plan had 24K prompts. The count was cut to 16K before any number was seen, to fit the
  cap. The preregistration does not fix the count.

## Proxy and its validation
The proxy is teacher-forced on the 256 held-out sequences x 128 seeded anchors (32,768 blocks), with identical anchors for
every model. The drafter is run at block 32.
- **Candidates and table:** top-16 candidates of the draft marginal per position, and the exact [31,16,16] last-write
  refinement table (SpecForge `transition_scores`).
- **Truth:** the target's teacher-forced argmax, cut where the data leaves the greedy path. The mean usable truth length
  is 27.9 of 31.
- **Chain:** DSpine's refined greedy chain (`select_cached`) over 15 drafts, bonus counted.
- **Tree:** the engine policy (softmax at temperature 1, joint probability, 8 children per pop, N-1 pops, depth <= 31).

Official b16 weights (`proxy_official.json`):

| metric | proxy | engine (Phase 2) |
|---|---|---|
| chain, 15 drafts | 8.24 | 7.99 (chain b16) |
| tree N=48, D=31 | 11.62 | 11.29 |
| **tree48 / chain15** | **1.411** | **1.413** |
| tree N=16 / 32 / 64 | 8.58 / 11.07 / 11.91 | 11.47 at N=64 |
| backbone-argmax chain, 15 | 7.60 | |
| candidate prefix coverage, first 15 | 9.86 | 9.9 (Phase 2 offline) |

The proxy's tree-over-chain ratio matches the engine ratio to 0.1%. Absolute values are about 3% above the engine. So the
proxy ranks tree and chain configurations the way the engine does. As preregistered, only ratios across variants are used.

## Variants (as preregistered; warm start from the official weights)
Common recipe:
- block 32, 925 optimizer steps x 32 sequences x 64 anchors;
- LR 2e-4 cosine with 4% warmup, clip 1.0, curriculum off, seed 42;
- bf16 weights with an fp32 master copy for AdamW. About 12 s/step, 3.1 h per variant.

The tree loss is:

`L_tree = sum_i w_i [CE_refined(argmax_i | true predecessor, over the 16 candidates) + 0.5 CE_backbone(argmax_i)]`,
with `w_i = exp(-(i-1)/14)`.

- **mix1:** L_DSpine + 1.0 L_tree
- **mix4:** L_DSpine + 4.0 L_tree

## Results (held-out proxy; ratio to the official weights in parentheses)

| model @ step | chain15 | chain31 | tree16 | tree32 | **tree48** | tree64 | tree48, depth <= 15 | cand. coverage 15 |
|---|---|---|---|---|---|---|---|---|
| official | 8.237 | 10.092 | 8.577 | 11.071 | **11.622** | 11.910 | 9.595 | 9.862 |
| mix1 @300 | 8.022 (0.974) | 9.850 | 8.478 | 11.035 | 11.640 (1.002) | 11.961 | 9.539 | 9.761 |
| mix1 @600 | 8.073 (0.980) | 10.022 | 8.531 | 11.200 | 11.834 (1.018) | 12.160 | 9.594 | 9.819 |
| **mix1 @925 (final)** | **8.114 (0.985)** | 10.120 (1.003) | 8.566 | 11.284 | **11.919 (1.026)** | 12.239 (1.028) | 9.623 (1.003) | 9.843 |
| mix4 @300 | 8.000 (0.971) | 9.810 | 8.494 | 11.048 | 11.663 (1.004) | 11.981 | 9.560 | 9.760 |
| mix4 @600 | 8.044 (0.977) | 9.987 | 8.537 | 11.226 | 11.858 (1.020) | 12.174 | 9.609 | 9.819 |
| **mix4 @925 (final)** | **8.093 (0.983)** | 10.100 (1.001) | 8.584 | 11.310 | **11.947 (1.028)** | 12.267 (1.030) | 9.639 (1.005) | 9.844 |

Decision (`summary.json`):

| variant | tree48 ratio (bar >= 1.05) | chain15 ratio (bar >= 0.98) | result |
|---|---|---|---|
| mix1 | 1.026 | 0.985 | fail (tree) |
| mix4 | 1.028 | 0.983 | fail (tree) |

Reading:
- **The gain is half the bar and comes almost entirely from depths 16-31.**
  - When the tree is capped at depth 15, the trained models are +0.3% / +0.5%. At N=16 they are flat.
  - The candidate coverage of the first 15 positions drops slightly (0.998).
  - Within the first 15 positions, then, the conditionals are not ranked better, which is what the hypothesis needed. What
    improved is the block-32 positions beyond 16, which the official b16 checkpoint only extrapolates.
- **Dose barely matters.** A 4x heavier tree loss adds 0.2 points over mix1 (+2.8% vs +2.6%) and costs 0.2 more points of
  chain. The ranking signal the tree uses is already close to what the drafter can learn in this budget.
- **Chain drops early, then mostly recovers.**
  - Chain falls 2.6-2.9% at step 300, at peak LR, from the warm-start perturbation.
  - It recovers to -1.5% / -1.7% by the end, inside the -2% guard.
  - The tree curve rose from step 600 to step 900 but was flat from 900 to 925 (LR near 0). A longer cosine schedule might
    add a little, but not the missing +2.4 points at the same rate.
- **Attribution is open.** The preregistered DSpine-loss-only control (same greedy data, block 32) was to run only after a
  pass, so it was not run. The +2.6% therefore cannot be split between the tree loss and plain block-32 training on greedy
  data. Phase 3's block-32 fine-tune (T=0.7 data, DSpine loss) was neutral in the engine: tree 11.15 vs 11.29. That hints
  the greedy data and/or the loss do something, but it is a different measurement.
- **The engine is unlikely to show more.** Even if the proxy gain carried over one to one, about +2.7% tau at equal verify
  cost would be about +2-2.5% tok/s (about 640-645 vs 627 tok/s). That is near the 1-2% run-to-run noise and well short of
  what the bar asked for.

## Why this probably does not work (link to Phase 4)
Phase 4's coverage study found that two thirds of 48-node rounds end on a "budget miss": the true token was a candidate,
usually ranked 2nd or 3rd in its row, but its node was not selected. The hope here was that a ranking-targeted loss would
move those tokens to rank 1. In about 1.9 epochs of warm-started fine-tuning it did not, for the first 15 positions. Their
CE terms were already part of DSpine's objective (10% weight), and the drafter's ranking errors seem to come from what its
parallel MASK formulation can infer (as Phase 3 concluded), not from the loss weighting. A from-scratch run with the tree
loss at the full DSpine budget (about 100x this compute) is the only version not ruled out here.

## Procedural notes
- **Normalisation bug, fixed before any result.** The first mix1 launch divided the loss by the supervised-row count a
  second time. Phase 3's trainer needed that division, but the SpecForge version used here (`b15ad38`) already normalises.
  The bug showed as gradient norm 0.00. That run was killed at step 20 and restarted after the fix (commit `3610baa`). No
  evaluation came from it.
- SpecForge was pinned at `b15ad38` and installed `--no-deps` into the DSpine venv (torch 2.14.1+cu130). The official
  weights load through the Tencent <-> SpecForge mapping with no missing or unexpected keys. Step 0 of both runs reproduces
  the official proxy exactly.
- No checkpoints were kept, because the bar was not passed. The box was destroyed.

## Cost (Vast, label `rdf2-treetrain`, cap 20 USD)

| box | GPU | price | used | cost |
|---|---|---|---|---|
| 60014918 | A100-SXM4-80GB | 1.42 USD/h | about 0.4 h (setup, smoke test, start of data generation; stopped early) | about 0.6 USD |
| 60018370 | A100-SXM4-80GB | 1.39 USD/h | 7.45 h (setup 0.3, data 0.5, official proxy, mix1 3.2, mix4 3.2) | about 10.4 USD |

The total is **about 11 USD** (wall clock x listed price, not the invoice). Both boxes are destroyed.

## Reproduction
```
bash bench/setup_box.sh                      # on a fresh A100 box, repo at /workspace/recursive-dflash2
bash bench/p6treetrain/box_all.sh            # SpecForge b15ad38, 16K greedy samples, official proxy, mix1, mix4
# results: /workspace/p6/{official,mix1,mix4}/proxy.json
```
