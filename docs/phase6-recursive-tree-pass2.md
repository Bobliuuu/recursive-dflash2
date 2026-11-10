# Phase 6: a recursive pass 2 on pass-1 features, scored as a tree (Qwen3-8B, A100-80GB): killed

Short answer: **no.** I tested the most promising items on Phase 3's untried list: a pass 2 that takes pass-1 hidden **features** (with a lookahead over later block positions that pass 1 never sees) and is scored by the **best-first tree** it produces, plus a **small, cheap pass-2 head**. Both variants are on-policy w.r.t. pass 1 and warm-started from the official weights. Neither clears the preregistered bar (`bench/p6refine/PREREG.md`, committed before training). The full 5-layer pass 2 lifts proxy tree tau by +4.6% (95% CI 4.1-5.2%, just under the +5% bar), but pays ~3.3 ms more per round and would need about +18% to break even. The 2-layer pass 2 is cheap (+1.0-1.9 ms) and gives nothing (+0.1%). No engine runs were made.

## Setup
- Baseline: official `tencent/Qwen3-8B-DSpine-b16` with the Phase 2 in-engine tree, N=48 nodes, depth cap 31. Engine numbers: tau 11.29, 627 tok/s, round time 18.0 ms. Engine chain b16: tau 7.99.
- Data, regenerated on the box (Phase 3's data was lost):
  - Prompts: 10,600 from `mlabonne/open-perfectblend` via `train/prep_prompts.py`, which drops prompts that overlap the eval sets.
  - Training: 10,344 responses from Qwen3-8B at T=0.7, thinking off, max 1536 new tokens.
  - Held-out eval: 256 disjoint prompts with responses at **T=0 (greedy)**. The data path is therefore the target's greedy path.
- Proxy (`train/p6_pass2_tree.py`): teacher-forced on 256 sequences x 128 seeded anchors, giving 31,587 blocks with identical anchors for every model. Per block:
  - top-16 candidates per position;
  - the DSpine transition table `[31 x 16 x 16]`, computed with SpecForge `transition_scores`, the same math as the engine's `serial_rs_table`;
  - **chain tau**: DSpine's refined greedy chain over 15 drafts;
  - **tree tau**: a batched torch re-implementation of the engine's `tree_build` rule (joint probability, top-8 expansion, N-1 pops, depth cap 31), walked along the data path. The bonus token is counted.
  - Bootstrap CIs resample the 256 sequences, paired against the official model.
- Cost (`bench/p6refine/timing.py`): a CUDA-graph replica of one draft round at bs=1. The replica's pass 1 takes 3.13 ms, against 3.15 ms measured in the engine. The engine draft time is scaled by t(pass 2) / t(pass 1).

## Proxy validation (official weights, `bench/p6refine/eval_step0.json`)
| | proxy | engine |
|---|---|---|
| chain b16 (15 drafts, refined) | 9.71 | 7.99 |
| tree N=48, D=31 | 14.17 | 11.29 |
| tree / chain | **1.460** | **1.413** |

- The ratio is within 3.3% of the engine's, inside the preregistered +-15% band, so the proxy is accepted as a ranking tool.
- Absolute proxy taus are about 1.2-1.25x the engine's. The held-out prompts are training-like, and teacher forcing has no trajectory effects.
- The proxy overstates depth gains. Its chain over 31 drafts versus 15 is 12.49 / 9.71 = 1.29, while the engine's b32 / b16 chain is 9.03 / 7.99 = 1.13. Gains concentrated at positions 16-31 should therefore be discounted further.
- Other N (proxy): tree16 9.90, tree32 13.39, tree64 14.55. The candidate prefix coverage is 16.4 positions.
- Warm-start identity check: at step 0 both variants reproduce the official proxy (tree48 ratio 1.0000, CI [0.9998, 1.0003]).

## Variants
Both variants were trained jointly on the same micro-batches: 300 steps x 32 sequences x 64 anchors, standard DSpine objective, gamma 7, AdamW with fp32 master weights, cosine LR. This took about 20 s/step on one A100.

- **V1, full pass 2**:
  - A trainable copy of the official 5-layer drafter at block 32.
  - Block input at position t is the MASK embedding plus `sum_{o=0..4} W_o h1[t+o]`, where h1 is pass 1's normed final hidden state. The W_o are zero-initialised.
  - Pass 1 is block-causal, so this is new information: pass 2 sees pass 1's beliefs about the next four positions.
  - LR 1e-4.
- **V2, small pass 2**:
  - Two DSpine layers on pass 1's last pre-injection residual, plus `sum_{o=1..4} W_o h1[t+o]`.
  - Initialised from official layers 3 and 4, with `o_proj` / `down_proj` and the first injection write zeroed, so step 0 is exactly pass 1.
  - The context projection is shared with pass 1.
  - LR 2e-4.
  - Two tree builds: (a) its own candidates; (b) pass 1's candidates rescored by its own table, which saves the second lm_head.

## Results (proxy, 31,587 blocks; `bench/p6refine/eval_step{0,150,300}.json`)
| model (step) | chain b16 | chain 31 | tree16 | tree32 | **tree48** | tree64 | coverage | tree48 / official (95% CI) |
|---|---|---|---|---|---|---|---|---|
| official | 9.71 | 12.49 | 9.90 | 13.39 | **14.17** | 14.55 | 16.39 | 1 |
| V1 (150) | 9.65 | 12.95 | 9.88 | 13.82 | 14.73 | 15.18 | 17.18 | 1.039 [1.034, 1.045] |
| V2 own cand (150) | 9.68 | 12.44 | 9.90 | 13.38 | 14.16 | 14.55 | 16.39 | 0.999 [0.997, 1.001] |
| V2 pass-1 cand (150) | 9.69 | 12.46 | 9.90 | 13.37 | 14.16 | 14.54 | | 0.999 [0.997, 1.000] |
| V1 (300) | 9.67 | 13.07 | 9.89 | 13.90 | **14.83** | 15.28 | 17.29 | **1.046 [1.041, 1.052]** |
| V2 own cand (300) | 9.71 | 12.51 | 9.92 | 13.42 | **14.19** | 14.58 | 16.41 | **1.001 [1.000, 1.003]** |
| V2 pass-1 cand (300) | 9.72 | 12.51 | 9.92 | 13.41 | **14.18** | 14.57 | | **1.001 [0.9995, 1.002]** |

## Decision (preregistered bar: tree48 >= +5% AND estimated tok/s > 627)
| pair | dT per round (est.) | break-even tree ratio | proxy tree ratio (step 300) | est. tok/s | verdict |
|---|---|---|---|---|---|
| V1 full pass 2 | +3.28 ms | 1.182 | 1.046 (CI 1.041-1.052) | 627 x 1.046 x 18.0/21.28 = **555** | **KILL** (misses +5% point estimate, -11% tok/s) |
| V2, own candidates | +1.92 ms | 1.106 | 1.001 | **567** | **KILL** |
| V2, pass-1 candidates | +1.01 ms | 1.056 | 1.001 | **594** | **KILL** |

Estimated tok/s = 627 x ratio x 18.0 / (18.0 + dT), with dT = 3.15 ms x t(pass 2) / t(pass 1) from `bench/p6refine/timing.json`. The pass-1 replica measured 3.13 ms, V1's pass 2 3.26 ms, V2 with its own candidates 1.90 ms, and V2 with pass-1 candidates 1.01 ms.

## Reading
- **Lookahead features help a little, and only deep in the block.** V1's gain comes from positions 16-31: its 31-draft chain gains about +0.6 (13.07 vs 12.49) while its 15-draft chain does not move or drops slightly, and its candidate coverage rises from 16.4 to 17.2 positions.
  - This is where the proxy overstates the engine (the proxy depth ratio is 1.29 against the engine's 1.13).
  - V1 also trains positions 16-31 at block 32, which the official b16 checkpoint never did. Part or all of the gain may be ordinary b32 training and not the recursion.
  - In Phase 3, b32 training did not beat the free extrapolation in the engine.
  - No "V1 without pass-1 features" control was run. V1 fails the speed bar even if every bit of its gain is credited to the recursion, so the control could not change the decision.
- **A cheap pass 2 learns nothing.** Two extra layers with the same lookahead stay at 1.00x. The information pass 1 lacks is not recoverable with a thin head on its residual stream.
  - That is the only form of pass 2 cheap enough to pay off on the round: at +1.0 ms it needs +5.6%, and at +1.9 ms +10.6%.
- **Economics, as Phase 3 predicted.** On an A100 the draft pass is about 17% of the tree round, so a full second pass needs +18% tau just to break even. A small pass 2 breaks even only above +5.6-10.6%. The best measured gain, +4.6% from the full pass 2, is below every one of these thresholds. And because the proxy over-weights deep positions, the true engine gain is likely smaller still.
- Not tried: from-scratch pass-2 training with a large budget, more on-policy rounds (pass 1 is frozen here, so the second round is identical), and a bidirectional pass-2 mask. None looks able to close an 18% gap for the full pass 2. For the small pass 2, the evidence is a flat 1.00x after 300 steps.

## Cost (Vast, label `rdf2-refine2`, cap 20 USD)
| box | GPU | price | used | cost |
|---|---|---|---|---|
| 60015178 (rdf2-refine2) | A100-SXM4-80GB | 1.152 USD/h incl. disk | 2.5 h (setup 0.3 h, data generation 0.4 h, timing, smoke test, 300-step joint training + 3 evals 1.75 h) | ~2.9 USD |

Total about 3 USD of the 20 USD cap. Estimated as wall-clock x listed price, not from the invoice. Box destroyed and confirmed gone from `vastai show instances`. No Modal use.

## Reproduction
`bash bench/setup_box.sh` and then `bash bench/p6refine/run_box.sh`. The stages are: install SpecForge at b15ad386; build prompts; generate data (eval at T=0, train at T=0.7); time the round; and train/evaluate with `train/p6_pass2_tree.py`, which evaluates at steps 0, 150 and 300. Raw outputs are `bench/p6refine/eval_step*.json`, `timing.json` and `train.log`. Weights are not committed.
