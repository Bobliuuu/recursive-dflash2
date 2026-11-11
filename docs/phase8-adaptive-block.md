# Phase 8: content-adaptive block length (offline study, no GPU)

**Question.** A community report (a tweet by the porter of NVFP4 DSpine to NInfer, RTX 5090, Nov 9 2026; unverified) compares DSpine at block K11 with DFlash2 at K9 at 2K context: **Python/code +22% (1 stream) / +7% (8 streams) for DSpine, English prose -12% / -16%** (absolute: Python 554.2 / 1674.8 tok/s, English 179.5 / 684.4). If the best drafter and block length depend on content, a fixed K leaves speed on the table. Hypothesis: choosing K (or the drafter) per request, from a prompt-domain label or an online running acceptance estimate, beats every fixed K on a mixed workload.

**Answer on our data: no, or not by enough to measure.** On the committed runs, choosing the block length or drafter per domain gains **-0.2% to +2%** over the best fixed setting (within the 1-2% run noise). A request-level oracle gains +3.5..5% in simulation. The best online policy gains +1.3%, and that is with in-sample tuning. Every number is below the 5% gate set for a GPU validation, so **no GPU run was made** (SPEND below). The reason is structural. DSpine's drafter is block-causal, so acceptance can only rise with K. At batch 1 the A100 round time is almost flat in K (15.2 ms at 16, 16.0 ms at 32), so a long block is nearly always worth it even on prose. At concurrency 8 cost is steep, but K16 is already best on every domain except a tie on prose.

Code: `bench/adaptive_k.py` (policies), `bench/test_adaptive_k.py` (CPU tests), `bench/p8_adaptive_block.py` (analysis), `bench/patches/apply_adaptive_k_hook.py` (shadow-mode SGLang hook, default off). Output: `bench/results/p8_adaptive_block.json`.

## Domains and data
- Domains: code = HumanEval + MBPP, math = GSM8K + MATH-500, prose = MT-Bench (first turn). "Mixed" = all 5 datasets, 40 prompts each at c=1 and 80 at c=8, weighted by their actual output tokens (tok/s = total tokens / total time). "code+prose" = HumanEval + MBPP + MT-Bench, the split the tweet uses.
- What is committed: per-request **histograms** of accepted drafts per round (`spec_hist`), cell makespans, and per-round timers (`*timing*.jsonl`; these hold no accept counts). **No per-round acceptance sequences are committed** (the phase-2 dump `dump.bin` is not in the repo). So the online-policy results below come from a simulator, not from replaying real traces.

## Part A: measured cells (no model)
Each cell is a real engine run. "Per-domain" picks the best config separately for code / math / prose, which is the ceiling for any prompt classifier because it assumes perfect labels. "Per-dataset" picks per dataset, so it is more optimistic and includes selection on noise.

| family (all T=0 unless noted) | best fixed (mixed tok/s) | per-domain choice | per-domain gain | per-dataset gain | code+prose: per-domain gain |
|---|---|---|---|---|---|
| A100 8B DSpine chain c=1, K in {16,24,32,48,64} | K32 (526.0) | code K24, math K32, prose K32 | +0.8% | +1.6% | +1.9% |
| A100 8B DSpine chain c=8, K in {8,12,16,32} | K16 (2549) | code K16, math K16, prose K12 | -0.2% | -0.2% | -0.1% |
| A100 8B chain K16/K32 + tree N32..64, c=1 | tree N48 (595.1) | N48 everywhere | +0.0% | +0.5% | +0.0% |
| H100 8B chain 16/32/48 + tree 32..96, c=1 | tree64 (926.4) | code 64, math 48, prose 64 | +0.3% | +0.9% | +0.0% |
| H100 27B drafters c=1 T=0 (DFlash2-b8, DSpine-16, DSpine+SPS, DSpine w8, DFlash2@16) | DSpine-16 (225.3) | code DSpine+SPS, else DSpine-16 | +0.6% | +1.0% | +0.7% |
| same, c=1 T=1 | DSpine+SPS (193.1) | code DSpine+SPS, math DFlash2@16, prose DFlash2 | +1.1% | +2.2% | +0.4% |
| same, c=8 T=0 (64 prompts) | DSpine+SPS (1121.7) | code DSpine-16, else DSpine+SPS | +0.7% | +0.7% | +1.0% |
| same, c=8 T=1 (64 prompts) | DSpine+SPS (949.5) | math DFlash2, else DSpine+SPS | +1.6% | +1.6% | +0.0% |

Measured per-dataset tok/s, A100 8B DSpine chain (gsm8k / math500 / humaneval / mbpp / mtbench):

| K | c=1 | c=8 |
|---|---|---|
| 8 | - | 2509 / 2589 / 2441 / 2249 / 1527 |
| 12 | - | 2789 / 3007 / 2607 / 2530 / 1537 |
| 16 | 672 / 647 / 517 / 490 / 292 | 3181 / 3360 / 2771 / 2609 / 1525 |
| 24 | 748 / 678 / 524 / 524 / 270 | - |
| 32 | 766 / 691 / 548 / 483 / 304 | 2635 / 2620 / 2123 / 1978 / 1065 |
| 48 | 760 / 671 / 540 / 476 / 300 | - |
| 64 | 722 / 680 / 497 / 494 / 270 | - |

Prose does prefer shorter blocks, but only at c=8 and only as a tie (MT-Bench 1527 / 1537 / 1525 for K8 / 12 / 16). At c=1 prose is best at K32, like code and math. The c=1 per-domain "code K24" pick comes from one MBPP cell (524 vs 483 at K32). The K24 MT-Bench cell is below K16, which acceptance monotonicity rules out for the same text. So these differences are run-to-run output divergence (bf16 shape noise changes the generated text), not a domain effect.

**The tweet's code-vs-prose flip does not reproduce on Qwen3.8-27B / H100.** At native widths, DSpine-16 vs DFlash2-b8 at c=1 T=0 is +9% / +18% on HumanEval / MBPP and **+11% on MT-Bench**; at c=8 T=0 it is +18% / +17% and +7%. The only prose deficit is at c=1 T=1: DSpine-16 164 vs DFlash2 166 tok/s (-1%), with code +7..13%. At matched width 8 (DSpine w8 vs DFlash2-b8) DSpine is behind or tied everywhere (c=8 T=0: code -11%, prose -7%), which matches the README's "slightly behind at matched verify width". Different model, engine, GPU, quantisation and widths (K11 vs K9) from the tweet, and only 8 prompts per 27B c=1 cell, so this neither confirms nor refutes their numbers. The NInfer fork was not found: a search for the porter's handle found no match, only the upstream NInfer project and unrelated forks.

## Part B: simulation, A100 Qwen3-8B DSpine chain, c=1
**Model.**
- *Rounds.* For each request, rounds are drawn i.i.d. from that request's accepted-draft histogram in the block-48 chain run (`p2_blk48_*_T0_c1.json`). DSpine is block-causal, so a width-K verify accepts `min(a, K-1)` and emits that plus one bonus token.
- *Calibration.* Pure truncation underestimates measured small-K acceptance by 3-8%. After a round truncated at K, the next round, drafted from a nearer anchor, does better than a fresh draw. A segment-replay model is worse still. So emitted tokens are scaled by a per-(dataset, K) factor = measured tau(K) / truncated tau(K). K = 16/24/32/48 use the c=1 cells. K = 8/12 have no c=1 run and use the c=8 runs' ratio against their own block-32 histograms. The factors are 0.90-1.10 (`calibration` in the JSON). By construction, every fixed-K simulation reproduces the measured tau.
- *Round time.* Per dataset, measured makespan / rounds at K = 16..48. K = 8/12 are K16 minus the CUDA-graph verify increments (-0.62 / -0.31 ms, `p2_verify_cost_8b_ctx1536.json`).
- *Policies.* They see only the censored count `min(a, K-1)` and a single dataset-agnostic cost curve (`A100_8B_C1_ROUND_MS`). They never see the dataset label. Online results are the mean of 5 seeds; the seed spread on the mixed number is about +-0.3%.

Simulated tok/s, fixed K (mixed = all 5):

| K | code | math | prose | mixed | code+prose |
|---|---|---|---|---|---|
| 8 | 396 | 456 | 290 | 399 | 356 |
| 12 | 472 | 573 | 320 | 477 | 412 |
| 16 | 508 | 662 | 308 | 514 | 424 |
| 24 | 526 | 698 | 274 | 514 | 411 |
| **32** | 513 | **710** | 306 | **528** | **425** |
| 48 | 506 | 692 | 301 | 518 | 419 |

Policies vs the best fixed K (K32, 527.7 tok/s mixed):

| menu | policy | code | math | prose | mixed | vs K32 |
|---|---|---|---|---|---|---|
| {8,16,32} | per-domain oracle (code 32, math 32, prose 16) | 513 | 710 | 308 | 529.0 | +0.2% |
| {8,16,32} | per-request oracle | 525 | 716 | 335 | 546.8 | +3.6% |
| {8,16,32} | online EMA, ratio 2.0 | 503 | 700 | 325 | 529.7 | +0.4% |
| {8,16,32} | online EMA, ratio 3.0 (best online) | 512 | 708 | 323 | 534.5 | **+1.3%** |
| {8,16,32} | online hazard, decay 0.9 | 512 | 709 | 304 | 526.4 | -0.3% (always K32) |
| {8,16,32} | per-round clairvoyant (reference, not a policy) | 577 | 777 | 347 | 589.0 | +11.6% |
| {16,24,32,48} | per-domain oracle (code 24, math 32, prose 16) | 526 | 710 | 308 | 533.8 | +1.2% |
| {16,24,32,48} | per-request oracle | 539 | 717 | 320 | 546.0 | +3.5% |
| {16,24,32,48} | online EMA 3.0 | 519 | 702 | 312 | 530.6 | +0.6% |
| {8,12,16,24,32,48} | per-domain oracle (code 24, math 32, prose 12) | 526 | 710 | 320 | 539.1 | +2.2% |
| {8,12,16,24,32,48} | per-request oracle | 539 | 717 | 340 | 554.6 | +5.1% |
| {8,12,16,24,32,48} | online EMA 3.0 | 514 | 698 | 326 | 533.7 | +1.1% |
| {8,12,16,24,32,48} | online EMA 1.5 | 466 | 646 | 314 | 495.5 | -6.1% |

Reading:
- Adaptivity helps prose (304 -> 323..340) and costs code and math a little, because shrinking K there gives up long full-accept runs. On the mixed workload the two cancel to about +1%.
- EMA thresholds are fragile: ratio 1.5 loses 3-6%. The hazard policy is censoring-aware and uses the prior and cost curve. It correctly learns that on this cost curve the long tail of prose acceptance still pays for K32, so it converges to the best fixed K and gains nothing.
- The per-request oracle (+3.5..5%) is the ceiling for any policy that picks one K per request. It is optimistic: it knows each request's whole acceptance distribution in advance, and its K8/12 points rest on the c=8-derived calibration.
- The per-round clairvoyant (+8..13%) needs the round's outcome before choosing the width, so no policy can reach it. It only shows how much the round-to-round spread is worth.

## Part C: simulation, c=8 (per-request widths inside one batched verify)
Setup:
- Eight always-busy slots draw requests from the shuffled mixed pool.
- Rounds come from the block-32 c=8 histograms (`p2_blk32_*_T0_c8.json`), with the same kind of calibration (c=8 cells, K in {8,12,16,32}).
- Step time is a function of total verify tokens, from the measured c=8 cells: 64 tokens 20.2 ms, 96 22.6, 128 23.2, 256 33.3.
- `allocate_batch` greedily widens the request whose extra width adds the most expected tokens per ms. This is the same idea as DSpine's SPS `batch8_tokens` budget.

| policy | mixed tok/s | vs K16 |
|---|---|---|
| fixed K8 / K12 / K16 / K32 | 2247 / 2419 / **2547** / 1896 | |
| per-domain (code 16, math 16, prose 12; in-sample search) | 2574 | +1.0% |
| online hazard + allocate_batch (decay 0.8 / 0.9) | 2531 / 2537 | -0.6% / -0.4% |

The measured equivalent agrees. On 27B/H100, DSpine's own confidence-driven SPS (dynamic verify width) vs full-16 verify is -0.9% at c=1 T=0 (223.2 vs 225.3 mixed), +0.9% at c=8 T=0 (1121.7 vs 1111.7), and +2.8% at c=1 T=1 (193.1 vs 187.9).

## Implementation
- `bench/adaptive_k.py` is pure Python and needs no torch. It contains:
  - `FixedK`, `EmaThresholdK` and `HazardK`. HazardK keeps Kaplan-Meier-style per-position hazards with forgetting and a prior from the phase-0/2 conditional acceptance, and chooses argmax E[emitted(K)] / cost(K). Both online policies probe one step wider after a full-block (censored) round.
  - `allocate_batch` for batched verify, `AdaptiveKController` (per-request registry), and `make_policy` / `parse_menu` for env strings.
  - `ShadowLogger`, used by the hook.
- `bench/patches/apply_adaptive_k_hook.py` patches SGLang 0.5.22 `dflash_worker_v2.py` right after the accept step. It works with or without the p2/p4 tree patch; both layouts were checked against a stub file, not a real SGLang install. It is active only with `RDF2_ADAPTIVE_K_LOG=<file.jsonl>` and only at bs=1. Each round it logs `{seq, k, accepted, next_k}`. **Shadow mode only: it does not change the served K.** Real per-round switching needs verify (and draft) CUDA graphs captured at several widths, plus matching KV / position handling. That is a substantial SGLang change, and the gains above do not justify it. Its main use is to record the per-round acceptance order that this study had to simulate. It assumes `accept_len` counts accepted drafts excluding the bonus; check the first logged run against the cell's `spec_hist`.

## Caveats
- Simulation, not engine runs: Parts B and C are models calibrated to measured cells. Part A cells are real but differ by 1-2% run to run, and more on MT-Bench (output divergence). Per-dataset and per-request "oracles" select on that noise and are optimistic.
- I.i.d. rounds within a request remove temporal structure, such as code blocks vs explanation inside one answer, that an online policy could exploit. Real traces from the shadow hook would settle this; the per-round clairvoyant bounds what is at stake.
- Truncation source: the block-48 chain (c=1) and block-32 chain (c=8) runs. Histograms cap at 47 and 31 accepted drafts, so K > 48 (c=1) and K > 32 (c=8) cannot be simulated. 64 was measured in Part A only.
- Calibration factors absorb the anchor-shift effect and run noise per (dataset, K). They are applied as a dataset-level constant regardless of the policy's recent K history.
- c=1 has no engine run at K = 8/12. Their round time is extrapolated from HF-graph increments and their acceptance factor comes from c=8 runs on a different prompt sample.
- T=0, Qwen3-8B, A100 only for the simulation. The 27B drafter-choice table has 8 prompts per c=1 cell. Domain labels are assumed perfect: real prompt classifiers would do worse than the per-domain oracle, and our GSM8K / MATH prompts carry a templated suffix that makes them trivially classifiable.
- Tweet numbers are the porter's claims (NVFP4, RTX 5090, NInfer, K11 vs K9). We did not run NInfer, an NVFP4 DSpine or K11.

## SPEND / RUNS
GPU validation was gated on a simulated mixed-workload gain >= 5% for a realisable policy. The best realisable policy gave +1.3% and the per-domain oracle +0.2..2.2%. **No GPU run, no Modal app, spend 0 USD.**

## Reproduce (CPU, ~1 min)
```bash
python -m pytest bench/test_adaptive_k.py -q        # 16 tests
python bench/p8_adaptive_block.py --seeds 5          # writes bench/results/p8_adaptive_block.json, prints the tables
# on a GPU box (optional, logs per-round acceptance + shadow choices, served K unchanged):
python bench/patches/apply_adaptive_k_hook.py        # --revert undoes it
RDF2_ADAPTIVE_K_LOG=/tmp/ak.jsonl RDF2_ADAPTIVE_K_POLICY=hazard bash bench/p4_serve.sh chain
```
