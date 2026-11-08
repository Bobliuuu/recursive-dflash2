# Phase 1: AR vs EAGLE-3 vs DFlash vs DSpine on Qwen3-8B (1x A100-80GB, SGLang 0.5.22)

All numbers from one engine build, one GPU, one frozen prompt set. Raw per-request JSON (with output text and per-round acceptance
histograms): `bench/results/p1_<method>_<dataset>_T<temp>_c<conc>.json`; aggregates `bench/results/phase1_summary.json`
(`bench/analyze_phase1.py`); full analysis printout `logs/phase1_analysis.txt`; server/client logs in `logs/`.
Setup details (venv, nvcc fix, flashinfer backend, prompt templates, thinking OFF) are in `docs/step0-findings.md`.

## Protocol actually run
- Methods: AR; EAGLE-3 = `RedHatAI/Qwen3-8B-speculator.eagle3`, **chain** with 3 draft steps (verify width 4); DFlash = `z-lab/Qwen3-8B-DFlash-b16`
  (verify width 16); DSpine = `tencent/Qwen3-8B-DSpine-b16` (verify width 16, default full verify, no SPS table).
- T in {0, 1} (T=1: top_p=1, top_k off, plain temperature 1, one seed, `sampling_seed` fixed per prompt index); concurrency in {1, 8} (closed loop,
  `ThreadPoolExecutor(conc)` over /generate); max_new_tokens 2048; thinking OFF; radix cache disabled.
- Datasets: GSM8K, MATH-500, HumanEval, MBPP, MT-Bench first turn. **Prompts per cell: 40 at c=1, 80 at c=8** (seeded subsample, same ids across methods;
  MT-Bench has 80 total). Not the full sets, so absolute values carry sampling noise of a few percent (T=1 more). LCB / Arena-Hard not run.
- tok/s = total generated tokens / makespan of the cell (includes prefill and any queueing), after 4 discarded warmup requests. Speedup = same-cell ratio to AR
  (same dataset, T, concurrency); the summary rows average the five per-dataset ratios. tau = emitted tokens per verify round (bonus counted), pooled over rounds.
- Round-time split (draft vs verify) is in step 0 only (bs=1, DFlash/DSpine).

## Headline (mean over the 5 datasets)
| T | conc | method | tau (mean of datasets) | tok/s (mean) | speedup vs AR (mean of per-dataset ratios) | datasets |
|---|---|---|---|---|---|---|
| 0 | 1 | ar | 1.00 | 91 | 1.00x | 5 |
| 0 | 1 | eagle3 | 2.56 | 184 | 2.03x | 5 |
| 0 | 1 | dflash | 6.03 | 407 | 4.50x | 5 |
| 0 | 1 | dspine | 7.99 | 521 | 5.75x | 5 |
| 0 | 8 | ar | 1.00 | 632 | 1.00x | 5 |
| 0 | 8 | eagle3 | 2.58 | 1149 | 1.82x | 5 |
| 0 | 8 | dflash | 6.02 | 2149 | 3.40x | 5 |
| 0 | 8 | dspine | 7.86 | 2600 | 4.11x | 5 |
| 1 | 1 | ar | 1.00 | 89 | 1.00x | 5 |
| 1 | 1 | eagle3 | 2.42 | 174 | 1.95x | 5 |
| 1 | 1 | dflash | 4.74 | 319 | 3.57x | 5 |
| 1 | 1 | dspine | 6.92 | 433 | 4.85x | 5 |
| 1 | 8 | ar | 1.00 | 638 | 1.00x | 5 |
| 1 | 8 | eagle3 | 2.43 | 1085 | 1.70x | 5 |
| 1 | 8 | dflash | 4.83 | 1756 | 2.76x | 5 |
| 1 | 8 | dspine | 6.90 | 2182 | 3.42x | 5 |


(`tau` for AR is 1 by definition. EAGLE-3 tau is for a 3-step chain.)

Reading it: at c=1, T=0 DSpine is 5.75x AR, DFlash 4.50x, EAGLE-3 (chain) 2.03x, i.e. DSpine/DFlash = 1.28x in speed and 1.32x in tau. At c=8 (AR itself is
already 7x more throughput, 632 tok/s, because decoding one token per request is bandwidth-bound) the speedups shrink to 4.11x / 3.40x / 1.82x; DSpine keeps its
~1.2x edge over DFlash. T=1 costs DSpine ~13% of tau (7.99 to 6.92) and DFlash ~21% (6.03 to 4.74; DFlash proposes greedily, DSpine samples proposals), so the gap widens at T=1 (1.46x tau, 1.36x speed at c=1).
MT-Bench (open chat) is the weak domain for every drafter (DSpine 3.2x, DFlash 2.5x, EAGLE-3 1.8x at c=1).

## Per dataset: tau / tok/s / speedup vs AR
Per-dataset tau / tok/s / speedup
| T | conc | method | gsm8k | math500 | humaneval | mbpp | mtbench |
|---|---|---|---|---|---|---|---|
| 0 | 1 | ar | 1.00 / 91 | 1.00 / 90 | 1.00 / 90 | 1.00 / 91 | 1.00 / 91 |
| 0 | 1 | eagle3 | 2.49 / 178 / 1.97x | 2.59 / 188 / 2.08x | 2.73 / 195 / 2.17x | 2.73 / 196 / 2.16x | 2.26 / 162 / 1.79x |
| 0 | 1 | dflash | 6.50 / 432 / 4.76x | 7.99 / 544 / 6.02x | 6.30 / 423 / 4.69x | 6.09 / 412 / 4.55x | 3.28 / 225 / 2.48x |
| 0 | 1 | dspine | 10.61 / 665 / 7.32x | 9.69 / 645 / 7.13x | 7.89 / 515 / 5.71x | 7.41 / 489 / 5.39x | 4.37 / 291 / 3.21x |
| 0 | 8 | ar | 1.00 / 645 | 1.00 / 649 | 1.00 / 667 | 1.00 / 572 | 1.00 / 625 |
| 0 | 8 | eagle3 | 2.53 / 1129 / 1.75x | 2.54 / 1152 / 1.78x | 2.81 / 1315 / 1.97x | 2.70 / 1124 / 1.96x | 2.30 / 1028 / 1.65x |
| 0 | 8 | dflash | 6.41 / 2180 / 3.38x | 7.70 / 2854 / 4.40x | 6.56 / 2374 / 3.56x | 6.17 / 2158 / 3.77x | 3.27 / 1179 / 1.89x |
| 0 | 8 | dspine | 10.10 / 2878 / 4.46x | 9.40 / 3326 / 5.13x | 8.11 / 2742 / 4.11x | 7.59 / 2619 / 4.58x | 4.12 / 1438 / 2.30x |
| 1 | 1 | ar | 1.00 / 90 | 1.00 / 89 | 1.00 / 89 | 1.00 / 89 | 1.00 / 89 |
| 1 | 1 | eagle3 | 2.45 / 175 / 1.94x | 2.36 / 170 / 1.91x | 2.61 / 187 / 2.10x | 2.56 / 183 / 2.06x | 2.14 / 154 / 1.73x |
| 1 | 1 | dflash | 5.60 / 371 / 4.12x | 5.49 / 373 / 4.19x | 5.14 / 344 / 3.87x | 4.62 / 313 / 3.52x | 2.83 / 193 / 2.17x |
| 1 | 1 | dspine | 9.75 / 590 / 6.55x | 7.60 / 484 / 5.43x | 6.99 / 438 / 4.92x | 6.29 / 398 / 4.48x | 3.99 / 254 / 2.86x |
| 1 | 8 | ar | 1.00 / 651 | 1.00 / 629 | 1.00 / 614 | 1.00 / 653 | 1.00 / 641 |
| 1 | 8 | eagle3 | 2.47 / 1138 / 1.75x | 2.38 / 1087 / 1.73x | 2.67 / 1242 / 2.02x | 2.47 / 1019 / 1.56x | 2.17 / 937 / 1.46x |
| 1 | 8 | dflash | 5.76 / 2014 / 3.09x | 5.51 / 2079 / 3.31x | 5.29 / 1921 / 3.13x | 4.75 / 1728 / 2.64x | 2.85 / 1040 / 1.62x |
| 1 | 8 | dspine | 9.39 / 2838 / 4.36x | 7.48 / 2456 / 3.91x | 7.15 / 2264 / 3.69x | 6.42 / 2051 / 3.14x | 4.05 / 1303 / 2.03x |


## Losslessness check (T=0)
Losslessness vs AR (T=0, exact text match on identical prompts; bf16 batch-shape noise expected)
| conc | method | dataset | byte-identical / compared | mean common-prefix fraction | median |
|---|---|---|---|---|---|
| 1 | eagle3 | gsm8k | 21 / 40 | 0.63 | 1.00 |
| 1 | eagle3 | math500 | 8 / 40 | 0.35 | 0.20 |
| 1 | eagle3 | humaneval | 16 / 40 | 0.62 | 0.74 |
| 1 | eagle3 | mbpp | 9 / 40 | 0.45 | 0.28 |
| 1 | eagle3 | mtbench | 14 / 40 | 0.56 | 0.65 |
| 1 | dflash | gsm8k | 14 / 40 | 0.55 | 0.56 |
| 1 | dflash | math500 | 9 / 40 | 0.43 | 0.29 |
| 1 | dflash | humaneval | 10 / 40 | 0.51 | 0.49 |
| 1 | dflash | mbpp | 8 / 40 | 0.45 | 0.28 |
| 1 | dflash | mtbench | 13 / 40 | 0.55 | 0.51 |
| 1 | dspine | gsm8k | 14 / 40 | 0.55 | 0.56 |
| 1 | dspine | math500 | 9 / 40 | 0.43 | 0.29 |
| 1 | dspine | humaneval | 10 / 40 | 0.51 | 0.49 |
| 1 | dspine | mbpp | 8 / 40 | 0.45 | 0.28 |
| 1 | dspine | mtbench | 13 / 40 | 0.55 | 0.51 |
| 8 | eagle3 | gsm8k | 37 / 80 | 0.64 | 0.85 |
| 8 | eagle3 | math500 | 17 / 80 | 0.44 | 0.33 |
| 8 | eagle3 | humaneval | 13 / 80 | 0.45 | 0.42 |
| 8 | eagle3 | mbpp | 25 / 80 | 0.49 | 0.39 |
| 8 | eagle3 | mtbench | 22 / 80 | 0.43 | 0.25 |
| 8 | dflash | gsm8k | 33 / 80 | 0.59 | 0.66 |
| 8 | dflash | math500 | 19 / 80 | 0.45 | 0.32 |
| 8 | dflash | humaneval | 15 / 80 | 0.47 | 0.50 |
| 8 | dflash | mbpp | 17 / 80 | 0.46 | 0.35 |
| 8 | dflash | mtbench | 27 / 80 | 0.53 | 0.46 |
| 8 | dspine | gsm8k | 27 / 80 | 0.54 | 0.44 |
| 8 | dspine | math500 | 20 / 80 | 0.44 | 0.30 |
| 8 | dspine | humaneval | 16 / 80 | 0.50 | 0.53 |
| 8 | dspine | mbpp | 17 / 80 | 0.45 | 0.37 |
| 8 | dspine | mtbench | 26 / 80 | 0.50 | 0.45 |

DFlash vs DSpine text identity (T=0)
  c=1 gsm8k: 40 / 40 identical
  c=1 math500: 40 / 40 identical
  c=1 humaneval: 40 / 40 identical
  c=1 mbpp: 40 / 40 identical
  c=1 mtbench: 40 / 40 identical
  c=8 gsm8k: 50 / 80 identical
  c=8 math500: 17 / 80 identical
  c=8 humaneval: 21 / 80 identical
  c=8 mbpp: 23 / 80 identical
  c=8 mtbench: 35 / 80 identical
AR-vs-AR repeat floor (gsm8k, T=0, c=1): 40 / 40

AR-vs-AR repeat (GSM8K, T=0, c=1, separate server start): **40 / 40 byte-identical**, so AR is deterministic and any mismatch below is caused by the speculative path.

Findings (honest reading):
- **Speculative outputs are NOT byte-identical to AR** here: only 20-50% of responses match exactly (GSM8K 35-55%, MATH-500 ~20-25%, since long outputs give more chances to diverge),
  mean common-prefix fraction 0.35-0.64. This is bf16 numerics, not a drafter bug: the target forward over a 16-token (DFlash/DSpine) or 4-token (EAGLE-3) verify block uses
  different kernels/shapes than the 1-token decode of AR, so near-tie argmaxes flip, after which the continuations diverge.
- **DFlash and DSpine emit byte-identical text at c=1 on all 200 prompts (5 datasets x 40)**, although their drafts and acceptance lengths differ a lot. So at T=0 the output is
  determined by the target and the verify shape, not by the drafter: that is the expected behaviour of a correct lossless verifier. (At c=8 they differ on many prompts, 17-50 of 80 identical,
  because batch composition then changes the kernels; not a drafter effect either.)
- GSM8K accuracy (boxed answer vs gold) is equal across methods within noise: T=0 c=1 0.875 for all four (n=40); T=0 c=8 0.838/0.825/0.838/0.825 (AR/EAGLE-3/DFlash/DSpine, n=80);
  T=1 c=1 0.900/0.875/0.875/0.900; T=1 c=8 0.825/0.838/0.825/0.825. (Table in `logs/phase1_analysis.txt`.)
- NOT done: a distributional test at T=1 (first-token / n-gram histograms vs AR with chi-square), so T=1 losslessness is only supported by the accuracy parity above.
  A strict AR-equivalence test would need AR run with the same 16-wide verify shape (e.g. a forced-acceptance-0 drafter); not built.

## Comparability caveats / failures
- **EAGLE-3 is the weakest-configured baseline.** The RedHat speculators checkpoint does not load in SGLang 0.5.22 as shipped (`Eagle3Speculator`, no `model_type`). I converted it
  (`bench/convert_eagle3_redhat.py`: `layers.0.*` -> `midlayer.*`, embed dropped, AngelSlim's config) AND patched SGLang so `LlamaForCausalLMEagle3` honours `norm_before_residual=True`
  (`bench/patches/apply_eagle3_normres_patch.py`). Without the norm patch tau was 1.19; with it 2.5. AngelSlim's `Qwen3-8B_eagle3` (native SGLang format) gave tau 2.31 in a quick 8-prompt check, so RedHat is the stronger of the two.
  Config is a plain chain (3 steps, topk 1, 4 verified tokens), not the tree (e.g. 16/60 verified tokens) used in the DFlash paper's EAGLE-3 numbers; a tree would raise tau and speed. So EAGLE-3 here is "chain EAGLE-3".
- Block/verify width is each method's default (EAGLE-3 4, DFlash 16, DSpine 16); no matched-budget sweep was run.
- A100 uses flashinfer, not FA3 (Hopper only). Ratios between methods may shift on H100/B200 (T_verify smaller, T_draft share larger).
- T=1 is one seed per cell, n=40 (c=1) / 80 (c=8): treat differences under ~5% as noise. Speedup rows average per-dataset ratios, so they differ slightly from ratio-of-mean tok/s.
- DSpine in this run uses the default full verify16 (no `--speculative-dspark-sps-table-path`), since SPS tuning is documented for the 27B setup; a tuned width might help c=8.
- Cap hits: MATH-500 responses hit the 2048 cap in ~13% of requests (all methods generate until the cap in those, which inflates nothing but lowers tau slightly for them).
- Step-0 cells (n=100) and phase-1 c=1 cells (n=40) overlap but are separate runs: DSpine T=0 tau is 7.97 (step 0, n=100) vs 7.99 (phase 1), DFlash 6.08 vs 6.03.
- No run failed outright; the only restarts were harness bugs (driver liveness race, histogram padding), none changed reported numbers.
