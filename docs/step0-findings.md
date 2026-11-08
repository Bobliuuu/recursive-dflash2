# Step 0 findings: DSpine and DFlash on Qwen3-8B, A100-80GB, SGLang 0.5.22

Raw data: `bench/results/step0_{dflash,dspine}_<dataset>_T0_c1.json` (per-request records incl. acceptance histograms and
output text), `bench/results/timing/{dflash,dspine}_step0.jsonl` (per-round draft/verify GPU times),
`bench/results/step0_summary.json` (aggregates, produced by `bench/analyze_step0.py`). Logs in `logs/`.

## Setup (what was actually run)
- Hardware: 1x A100-SXM4-80GB (Vast), driver 580, torch 2.14.1+cu130, SGLang 0.5.22, `sglang-dspine` 0.2.0 (pip -e from Tencent/DSpine),
  flashinfer 0.7.1. One venv serves AR, EAGLE-3, DFlash (built into SGLang 0.5.22) and DSpine (plugin). Nothing needed a second build.
- Attention backend: **flashinfer** (FA3 is Hopper-only, not available on A100). Required a CUDA 13.0 nvcc from pip
  (`nvidia-cuda-nvcc==13.0.*`) because the box nvcc 12.4 cannot JIT flashinfer 0.7.1 (`--compress-mode`).
- Models: `Qwen/Qwen3-8B` bf16; drafters `z-lab/Qwen3-8B-DFlash-b16`, `tencent/Qwen3-8B-DSpine-b16` (block 16, verify width 16, no
  DSpark-style verify caps: default full verify16). `--page-size 1 --mem-fraction-static 0.65 --disable-radix-cache`, CUDA graphs bs 1..64.
- Protocol: **thinking OFF** (`enable_thinking=False` in the chat template, no system prompt), T=0, concurrency 1, `max_new_tokens=2048`,
  native `/generate` endpoint. 4 warmup requests discarded. Prompts: seeded 100-prompt subsample per dataset (MT-Bench: all 80, first turn only),
  frozen in `bench/data/prompts/` (templates in `bench/prep_prompts.py`; GSM8K/MATH get a "reason step by step, final answer in \boxed{}" suffix).
  LiveCodeBench and Arena-Hard were not run (not cheap here); so the 5-dataset means below are not the papers' 7-dataset means.
- Metrics: acceptance length tau = emitted tokens per verify round (bonus token included). Per-position stats come from SGLang's per-request
  `spec_correct_drafts_histogram` (rounds by number of accepted draft tokens, 0..15). **Timing**: not an engine metric; I patched
  `dflash_worker_v2.py` (`bench/patches/apply_timing_patch.py`, active only with `SPEC_TIMING_FILE`) to record CUDA events around
  (a) the draft stage = draft-model forward + draft sampling + verify-input prep and (b) the target verify forward over the 16-token block.
  Everything else (accept kernel, scheduler) is "other". Events measure GPU-timeline time, so CPU launch gaps are included.
  Round time ~14.0 ms; tau/round time reproduces the measured tok/s within a few percent, so "other" is small.

## Results (T=0, c=1, A100)
| dataset (n) | DFlash tau | DSpine tau | DSpine/DFlash | DFlash tok/s | DSpine tok/s | DFlash full-block | DSpine full-block |
|---|---|---|---|---|---|---|---|
| GSM8K (100) | 6.37 | 10.18 | 1.60 | 423 | 643 | 0.092 | 0.358 |
| MATH-500 (100) | 7.97 | 9.62 | 1.21 | 541 | 637 | 0.210 | 0.324 |
| HumanEval (100) | 6.49 | 8.15 | 1.26 | 435 | 530 | 0.127 | 0.215 |
| MBPP (100) | 6.32 | 7.63 | 1.21 | 426 | 500 | 0.129 | 0.200 |
| MT-Bench t1 (80) | 3.27 | 4.29 | 1.31 | 224 | 286 | 0.019 | 0.042 |
| mean of 5 | 6.08 | 7.97 | 1.31 | 410 | 519 | 0.116 | 0.228 |

"Full-block" = fraction of verify rounds in which all 15 draft tokens were accepted (16 tokens emitted). 13% of MATH-500 responses hit the
2048 cap (2% MBPP, 1% MT-Bench).

Per-position conditional acceptance P(position i accepted | i-1 accepted), i = 1..15, pooled over the 5 datasets (subject to selection bias: late positions are only reached in easy contexts, so the rising trend is partly composition):
- DFlash: 0.84 0.81 0.81 0.83 0.85 0.86 0.87 0.88 0.89 0.89 0.89 0.90 0.89 0.89 0.89. Unconditional survival at 16: 0.12.
- DSpine: 0.88 0.87 0.87 0.88 0.89 0.90 0.91 0.91 0.91 0.91 0.91 0.92 0.92 0.91 0.92. Unconditional survival at 16: 0.21.
Per dataset tables are in `step0_summary.json` (`cond_accept`, `survival`, `hist_pooled`). GSM8K DSpine is flat at 0.92-0.95 over all positions.

## Timing (B=16, bs=1, A100)
| | T_draft (median) | T_verify (median) | T_d/T_v | round |
|---|---|---|---|---|
| DFlash | 2.71 ms | 11.06 ms | 0.245 | 13.8 ms |
| DSpine | 2.97 ms | 11.05 ms | 0.269 | 14.0 ms |
Means are within 1% of medians. 34.6k (DSpine) and 44.1k (DFlash) rounds. Not split per dataset (contexts here are short, <3k tokens).
T_verify = 11 ms for a 16-token verify of an 8B model on A100 (weights alone ~16 GB / 2 TB/s = 8 ms), so verification is ~79% of the round.

## Kill-test verdicts
1. **T_d/T_v < 0.2 kills draft-ahead/stacking.** Measured 0.245 (DFlash) and 0.269 (DSpine) on A100: *not* killed by the letter of the test.
   But the naive cap 1 + T_d/T_v = 1.25 / 1.27 is only reachable if every round is a full accept, because the next block can only be
   pre-drafted behind the verify if the whole block is accepted (and the guessed bonus token also has to match). Weighting by the full-block
   fraction, the upper bound on the draft-ahead speedup is 1/(1 - P_full * T_d/(T_d+T_v)): DSpine 1.08 (GSM8K), 1.07 (MATH), 1.05 (HumanEval),
   1.04 (MBPP), 1.01 (MT-Bench); DFlash 1.02-1.04 on math/code, 1.00 on MT-Bench. So single-path draft-ahead is worth <= ~5-8% for
   DSpine on math/code and ~nothing on chat. Verdict: the T_d/T_v test passes, the *effective* gain does not justify it unless branching over
   several bonus guesses is added (SSD-style, which needs a drafter on separate hardware; here it would compete for the same GPU). Caveat:
   H100/B200 shift T_v down and the ratio up (the published ratios came from those GPUs), so re-measure there.
2. **Full-block-accept fraction < 10% kills long blocks.** Pooled: DSpine 0.21 (pass), DFlash 0.116 (borderline pass). Per dataset it splits
   cleanly: DSpine math/code 0.20-0.36 (pass), chat (MT-Bench) 0.04 (kill); DFlash GSM8K 0.092 and MT-Bench 0.019 are below the line.
   Conditional acceptance at late positions stays ~0.9 (DSpine) on math and code, so a block longer than 16 has headroom *on those domains*;
   on open-ended chat it does not. Long blocks therefore need an adaptive length (or confidence gating), not a fixed 32/64.
3. **Paper vs README tau discrepancy (DSpine paper 4.82 vs DFlash 3.77; README 7.83 vs 5.92): our run matches the README.**

   | | GSM8K | MATH | HumanEval | MBPP | MT-Bench | mean (these 5) |
   |---|---|---|---|---|---|---|
   | DFlash ours | 6.37 | 7.97 | 6.49 | 6.32 | 3.27 | 6.08 |
   | DFlash README | 6.32 | 7.99 | 6.56 | 6.29 | 4.26 | 6.28 |
   | DSpine ours | 10.18 | 9.62 | 8.15 | 7.63 | 4.29 | 7.97 |
   | DSpine README | 10.55 | 10.30 | 8.27 | 7.66 | 5.63 | 8.48 |
   | DSpine paper | 5.67 | 5.49 | 4.94 | 4.98 | 4.11 | 5.04 |

   The released checkpoints reproduce the README within 1-6% on GSM8K/MATH/HumanEval/MBPP (DFlash within 1%); the gap to the paper is 1.6-1.9x
   on those. MT-Bench is 20-24% below the README in both methods (3.27 vs 4.26, 4.29 vs 5.63) while the DSpine/DFlash ratio (1.31) matches
   the README's (1.32); probably prompt handling (we use first turn only, no system prompt) but I did not verify. Why the paper is lower: the
   paper's table was produced with drafters *retrained by the authors on ShareGPT* (their stated protocol; also DFlash 3.77 there, vs
   the official z-lab checkpoint, trained on ~800K Nemotron+CodeAlpaca, which gives 6.1 here). The released DSpine checkpoint is the
   Open-PerfectBlend-trained one the README table describes. So the two tables are different models, not different measurement conventions: our
   DFlash number (official checkpoint) is 1.6x the paper's DFlash, which pins the difference on training data/recipe rather than on how tau
   is counted (we count the bonus token; if the paper did not, it would only lower its numbers by ~1, not 2x). I could not retrain on ShareGPT
   here to confirm. The paper's *relative* claim holds up: DSpine/DFlash is 1.31 here vs 1.28 in the paper (4.82/3.77) and 1.32 in the README.

## Caveats
- Single run per cell, n=100 prompts (MT-Bench 80), so GSM8K/HumanEval tau has ~+-3% sampling noise; T=0 only.
- Prompt templates are mine (documented above) and differ from the papers', which affects absolute numbers (MT-Bench most).
- Timing is for bs=1 on A100 with flashinfer; not comparable to the papers' H100/B200 + FA3 timings.
