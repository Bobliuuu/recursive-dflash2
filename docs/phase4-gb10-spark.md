# Phase 4: speculative decoding on a DGX Spark (GB10), Qwen3-8B, batch 1, T=0

Question: GB10 has ~273 GB/s of LPDDR5x unified memory (A100: ~2 TB/s of HBM) and a lot of compute per byte, so AR decode of an 8B bf16
model is slow and a wide verify should cost about the same as a 1-token step. Are DFlash / DSpine / DSpine-tree speedups on GB10 larger
than on A100 (hypothesis: tree >= 8-10x)?

**Answer: same-config speedups are 4-10% larger than on A100, not 8-10x for bf16; the big wins come from quantized targets.**
- bf16 target: best is a **32-node DSpine tree: tau 10.46, 84.1 tok/s, 7.80x AR** (A100 best: 48-node tree, 7.20x). DSpine b16 chain 6.11x
  (A100 5.75x), DFlash b16 4.71x (A100 4.50x). The A100's 48-node optimum moves down to 32 nodes on GB10 because cuBLAS bf16 GEMMs on
  sm_121 get slower above 32 rows (verify +2% at 32 tokens, +16% at 48), not wider.
- Why not 8-10x: tau is the same on both GPUs, and the round time relative to one AR step barely changes. The DSpine draft streams 3.7 GB per
  round, 22% of a target step on GB10 (27% on A100); only the fixed per-round overhead shrinks (A100 ~12% of a step, GB10 ~6%).
- Quantized target, same bf16 drafter, same 32-node tree: **FP8 126 tok/s (11.7x bf16 AR, 6.5x FP8 AR)**; **NVFP4 157 tok/s (14.6x bf16 AR,
  5.3x NVFP4 AR)**, on the first 20 prompts per dataset. gsm8k reaches 175 (FP8) and 204 (NVFP4) tok/s.
- Losslessness: teacher-forced checks pass for every reported tree config (mismatch rates and max logit margins at the level of the b16 chain,
  i.e. bf16 near-ties only); details at the end.

## Setup (what was actually run)
- Box: spark2 (DGX Spark GB10), aarch64, sm_121, 121 GB unified memory, CUDA 13.0 driver. Shared box: every timing run held an exclusive
  GPU lock, and the SGLang server used `--mem-fraction-static 0.25` (about 30 GB).
- Stack: the same engine as Phase 1/2: SGLang 0.5.22, `sglang-dspine` 0.2.0 (pip -e Tencent/DSpine), torch 2.14.1+cu130, flashinfer 0.7.1.
  All have aarch64 wheels; flashinfer JIT-compiles its kernels for sm_121a on first use (~3 min). No fallback harness was needed: every
  number here comes from SGLang. Two GB10-specific changes: a uv-managed Python (the system python3.12 has no headers, so Triton could not
  build its driver stub) and `--disable-prefill-cuda-graph` (SGLang 0.5.22 otherwise captures 58 prefill shapes at ~100 s each on GB10; prefill
  graphs do not touch the decode rounds). Setup: `bench/gb10/setup_spark.sh`; runner: `bench/gb10/run_suite.sh`; queues: `bench/gb10/queue_*.sh`.
- Protocol: identical to Phase 1/2 (`bench/run_cell.py`): frozen 5-dataset prompt set (`bench/data/prompts`), thinking off, max_new 2048, c=1,
  T=0, radix cache off, page size 1, flashinfer attention, the same checkpoints (`Qwen/Qwen3-8B`, `tencent/Qwen3-8B-DSpine-b16`,
  `z-lab/Qwen3-8B-DFlash-b16`). Headline cells use 40 prompts per dataset (as on A100); sweep cells use 20 (the first half of the same 40), and
  are compared against the other cells restricted to the same 20 prompts (`NFIRST=20 bench/gb10/summarize.py`). AR uses 8 prompts per dataset:
  AR speed is flat (10.8 tok/s on every dataset, 92.8 ms per token). Speedup = same-dataset ratio to GB10 AR, averaged over the 5 datasets;
  tau = emitted tokens per verify round (bonus included).
- Timers: CUDA-event round timer (`bench/patches/apply_timing_patch.py`, draft = d0->v0, verify = v0->v1) plus the Phase 4 per-segment timer
  (`bench/patches/apply_timing_patch_p4.py`). "other" = wall time per round (cell makespan / rounds) minus median draft minus median verify, so
  it includes prefill, HTTP and the draft-KV append.
- Tree code: `tree` = Phase 2 glue (`bench/p2tree`); `treefast` (tags `f_*`) = the Phase 4 fast path from `docs/phase4-tree-overhead.md`
  (factored table and best-first selection inside the draft CUDA graph, one pack kernel, FlashInfer custom-mask prefix-tile skip).
- Raw per-request JSON: `bench/results/gb10/<tag>_<dataset>_T0_c1.json`; summaries `bench/results/gb10/summary*.json`; timers and logs
  `logs/gb10/`.

## Headline (40 prompts per dataset, mean of 5 datasets, c=1, T=0)

| config | tau | tok/s | x AR (GB10) | x AR (A100, same config) | draft ms | verify ms | other ms | round ms |
|---|---|---|---|---|---|---|---|---|
| AR | 1.00 | 10.8 | 1.00x | 1.00x (90.6 tok/s) | - | 92.8 / token | - | 92.8 |
| DFlash b16 | 5.92 | 50.7 | 4.71x | 4.50x | 19.1 | 92.3 | 4.7 | 116.1 |
| DSpine b16 chain | 7.87 | 65.9 | 6.11x | 5.75x | 20.1 | 92.5 | 5.4 | 118.1 |
| DSpine chain, block 32 (same weights) | 8.95 | 73.2 | 6.79x | 6.17x | 20.3 | 94.2 | 5.9 | 120.5 |
| DSpine tree 48 nodes, depth 31 (Phase 2 glue) | 11.12 | 78.2 | 7.26x | 6.92x | 24.8 | 107.4 | 7.6 | 139.9 |
| DSpine tree 48 nodes, depth 31 (Phase 4 fast path) | 11.12 | 79.0 | 7.33x | 7.20x | 23.6 | 107.3 | 7.7 | 138.6 |
| **DSpine tree 32 nodes, depth 31 (fast path)** | **10.46** | **84.1** | **7.80x** | 6.55x (Phase 2 glue) | 21.0 | 94.6 | 6.9 | 122.5 |

Per dataset x AR (gsm8k / math500 / humaneval / mbpp / mtbench):
- DSpine b16 chain: 7.97 / 7.44 / 6.04 / 5.83 / 3.28 (A100: 7.32 / 7.13 / 5.71 / 5.39 / 3.21)
- tree 48 (fast): 9.92 / 9.01 / 6.93 / 6.82 / 3.99 (A100 fast path: 9.69 / 8.92 / 6.76 / 6.74 / 3.87)
- **tree 32 (fast): 10.51 / 9.86 / 7.33 / 7.25 / 4.07** (113 / 106 / 79 / 78 / 44 tok/s)

A100 columns are the Phase 1/2/4 cells (`p1_*`, `p2_blk32`, `p2_tree{32,48}_d31`, `p4_tree48_factored`), same prompts and protocol.
tau is a property of the drafter and target, not of the GPU: GB10 and A100 agree within bf16 noise (b16 chain 7.87 vs 7.99, tree48 11.12 vs
11.29, tree32 10.46 vs 10.43). Draft / verify / other are medians of the per-round CUDA-event timer; round = cell wall time / rounds.

## Roofline: A100 vs GB10 for Qwen3-8B bf16 at batch 1

Per decode step the target streams ~15.2 GB of weights (8.2 B params x 2 bytes, minus the 1.2 GB input embedding, which is only gathered).

| | A100-SXM4-80GB | GB10 (DGX Spark) | GB10 / A100 |
|---|---|---|---|
| memory bandwidth, spec | 2039 GB/s HBM2e | 273 GB/s LPDDR5x | 0.134 |
| measured: device copy / weight-streaming GEMM | - | 218 / 175 GB/s (`roofline_gb10.json`) | |
| measured: AR step (SGLang, CUDA graph) | 11.1 ms (1.37 TB/s effective) | 92.8 ms (164 GB/s effective, 60% of spec) | 8.4x slower |
| bf16 dense GEMM, measured 8192^3 | (312 TFLOP/s spec) | 32.5 TFLOP/s; fp8 64.2 | |
| ridge (bf16 weights: verify rows before compute-bound) | ~150 (spec) to ~230 (effective BW) | ~185 (measured) | similar |
| verify 16 / 32 / 48 / 64 tokens (SGLang, ms) | 11.05 / 11.70 / 11.91 / 12.10 (+0 / 6 / 8 / 10%) | 92.5 / 94.2 / 107.3 / 112.5 (+0 / 2 / 16 / 22%) | |
| DSpine b16 draft (drafter 2.5 GB + target lm_head 1.2 GB) | 2.97 ms (0.27 of verify) | 20.1 ms (0.22 of verify) | |
| other per round (scheduler, accept, KV append, HTTP/prefill share) | ~1.3 ms (8% of round) | 5.4 ms (4.6% of round) | |

Speedup = tau x T_AR / T_round, and tau is the same on both GPUs. So the only way GB10 can beat A100 is a smaller round-time overhead
relative to one AR step: T_round / T_AR = (verify + draft + other) / T_AR. For the b16 chain that is 1.27 on GB10 vs 1.37 on A100 (6.11x vs 5.75x).
The ratio cannot approach 1, because both GPUs are bandwidth bound in the same way for the drafter as for the target: the DSpine draft
streams 3.7 GB per round (22-27% of a target step on either GPU). What GB10 removes is most of the fixed per-round cost (A100 ~1.3 ms of
scheduler/launch time is 12% of an 11 ms step; GB10's 5 ms is 6% of a 93 ms step).

The hypothesis that GB10 has a much wider "free verify" range is wrong: the ridge point (FLOP/s / bytes/s) is about the same (~185 rows on GB10
from the measured 32.5 TFLOP/s / 175 GB/s, ~150-230 on A100), and in practice the cuBLAS bf16 kernels on sm_121 fall off a cliff above 32 rows
(next section). On A100 verify cost grows slowly and smoothly with width (+8% at 48); on GB10 it is flat to 32 (+2%) and then jumps (+16% at 48).

## Why 32 nodes beats 48 on GB10: a cuBLAS cliff above 32 rows

`bench/gb10/linear_sweep.py` and `gemm_variants.py` time the real Qwen3-8B GEMM shapes (`F.linear`, [out, in] bf16 weights, 36 layers of
qkv / o / gate_up / down plus the lm_head) as SGLang calls them (`linear_sweep.json`, `gemm_variants.json`):

| rows M | 1 | 16 | 32 | 40 | 48 | 64 | 96 | 112 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|---|---|
| GEMM stack, cuBLAS (ms) | 85.7 | 83.9 | 85.2 | 94.8 | 96.9 | 101.2 | 105.2 | 90.7 | 91.0 | 142.0 |
| same, rows zero-padded to 128 | | | | 93.4 | 94.3 | 93.5 | 93.8 | | | |
| cuBLASLt preferred | | 84.9 | 86.0 | 99.7 | 101.7 | 108.4 | 99.9 | 90.7 | 92.0 | |

So 1-32 rows cost the same (pure weight streaming at ~175 GB/s), 33-111 rows pick slower kernels (qkv 0.29 -> 0.36 ms, gate_up 1.11 -> 1.33 ms per
layer at 48 rows), and 112-128 rows are fast again. The engine's verify follows the same shape: 92.5 ms at 16 tokens, 94.5 at 24 and 32, 107.3
at 48, 112.5 at 64. A tree that stays within 32 nodes keeps verify at +2%; 48 nodes cost +16% verify for +6% tau.

Workaround tried: `bench/gb10/apply_gemm_pad_patch.py` (env `RDF2_GEMM_PAD=128`) zero-pads every bf16 GEMM input with 32 < M < 128 rows to 128
rows inside SGLang's `UnquantizedLinearMethod`. In the engine it recovered only part of the micro-benchmark gain (48 nodes: verify 107.3 ->
106.9 ms; 64 nodes: 112.5 -> 108.7 ms; 96 nodes: 112.0 ms), so padded 64/96-node trees (7.23x / 7.13x on 20 prompts) still lose to the unpadded
32-node tree (7.42x on the same 20 prompts). The rest of the 48-row verify overhead is outside the GEMMs (attention, norms and the larger
logits/argmax over 48 rows; not broken down further).

## Tree width x depth sweep (20 prompts per dataset, paired; fast path unless noted)

| config | tau | tok/s | x AR | gsm8k / math500 / humaneval / mbpp / mtbench tok/s | draft ms | verify ms | other ms |
|---|---|---|---|---|---|---|---|
| DSpine b16 chain | 7.62 | 63.8 | 5.92x | 86 / 80 / 65 / 61 / 27 | 20.1 | 92.5 | 5.3 |
| DSpine b32 chain | 8.50 | 69.7 | 6.47x | 98 / 91 / 66 / 67 / 26 | 20.3 | 94.2 | 5.7 |
| tree 16, depth 15 | 8.20 | 68.1 | 6.32x | 89 / 85 / 69 / 67 / 31 | 20.5 | 92.6 | 5.9 |
| tree 24, depth 23 | 9.39 | 76.0 | 7.05x | 104 / 94 / 75 / 74 / 33 | 20.8 | 94.5 | 6.5 |
| tree 32, depth 15 | 9.14 | 74.4 | 6.91x | 99 / 92 / 74 / 74 / 34 | 20.7 | 94.5 | 6.2 |
| tree 32, depth 23 | 9.84 | 79.4 | 7.36x | 111 / 98 / 78 / 77 / 33 | 20.8 | 94.5 | 6.7 |
| **tree 32, depth 31** | **9.92** | **80.0** | **7.42x** | 112 / 101 / 76 / 77 / 33 | 20.8 | 94.5 | 6.6 |
| tree 48, depth 31 (Phase 2 glue) | 10.63 | 74.7 | 6.93x | 107 / 95 / 71 / 70 / 31 | 24.8 | 107.4 | 7.4 |
| tree 48, depth 31 | 10.63 | 75.7 | 7.02x | 109 / 96 / 71 / 71 / 31 | 23.6 | 107.3 | 7.3 |
| tree 48, depth 31, rows padded to 128 | 10.63 | 75.3 | 6.98x | 105 / 92 / 74 / 73 / 31 | 23.8 | 106.9 | 7.7 |
| tree 64, depth 31 | 10.94 | 74.4 | 6.90x | 108 / 92 / 71 / 71 / 30 | 24.3 | 112.5 | 7.6 |
| tree 64, depth 31, padded | 11.10 | 77.9 | 7.23x | 112 / 99 / 73 / 74 / 32 | 23.9 | 108.7 | 7.4 |
| tree 96, depth 31, padded | 11.37 | 76.9 | 7.13x | 110 / 98 / 73 / 73 / 30 | 24.9 | 112.0 | 8.1 |

(The 32-node cell on these 20 prompts was run twice, in separate servers: 80.0 and 79.9 tok/s.) The 20 prompts are the first half of each
40-prompt set, so absolute values are a bit lower than the headline (mtbench's first 20 prompts are harder), but every row uses the same 20.
Draft time is flat to 32 nodes (the in-graph table and selection add ~0.5 ms over the b32 chain) and +3-4 ms above that (the DSpine draft itself runs
over N block rows and hits the same GEMM cliff).

## Quantized target weights (stretch): FP8

Bytes dominate on GB10, so the target was also served as `Qwen/Qwen3-8B-FP8` (Qwen's block-128 FP8 checkpoint; SGLang picks its SM120 CUTLASS
FP8 GEMM). The DSpine drafter stays bf16 (the plugin refuses a quantized draft: `ValueError: The DSPINE draft does not support quantized weights`).
20 prompts per dataset for the spec cells, 8 for AR, paired with the bf16 cells on the same 20 prompts:

| config (20 prompts) | tau | tok/s | x bf16 AR | x FP8 AR | draft ms | verify ms | round ms (GPU) |
|---|---|---|---|---|---|---|---|
| bf16 AR | 1 | 10.8 | 1.00x | 0.56x | - | 92.8 / token | |
| FP8 AR | 1 | 19.4 | 1.80x | 1.00x | - | 51.5 / token | |
| bf16 target, DSpine b16 chain | 7.62 | 63.8 | 5.92x | 3.29x | 20.1 | 92.5 | |
| bf16 target, tree 32 | 9.92 | 79.9 | 7.41x | 4.12x | 21.0 | 94.6 | 122 |
| FP8 target, DSpine b16 chain | 7.55 | 98.5 | 9.14x | 5.08x | 19.9 | 51.6 | 73.4 |
| **FP8 target, tree 32** | **10.17** | **126.4** | **11.73x** | **6.52x** | 20.9 | 53.2 | 76.1 |
| FP8 target, tree 48 | 10.92 | 125.8 | 11.67x | 6.49x | 23.5 | 56.3 | 81.9 |

Per dataset (FP8 target, tree 32): 175 / 164 / 122 / 119 / 53 tok/s (gsm8k / math500 / humaneval / mbpp / mtbench).
- Acceptance does not drop with the FP8 target (tau 10.17 vs 9.92 for the bf16 target on the same prompts): the bf16 DSpine drafter predicts the
  FP8 model's greedy tokens as well as the bf16 model's.
- FP8 verify cost is flatter in width than bf16 (51.6 / 53.2 / 56.3 ms at 16 / 32 / 48 tokens), so 32 and 48 nodes tie.
- With the FP8 target the bf16 drafter is 27% of the round (20.9 of 76 ms), so the speedup over FP8 AR (6.5x) is lower than over bf16 AR.
  A quantized drafter would be the next lever (see the experimental patch below).
- "Lossless" here means lossless with respect to the FP8 target (teacher-forced check below), not byte-identical to the bf16 model.

## Quantized target weights (stretch): NVFP4

`nvidia/Qwen3-8B-NVFP4` (ModelOpt NVFP4 weights, group 16, FP8 KV cache, bf16 lm_head), served with `--quantization modelopt_fp4` (FlashInfer
FP4 GEMM, autotuned on first start). An explicit `--quantization` flag is also applied to the draft, which the DSpine plugin refuses, so the
spec cells add `--speculative-draft-model-quantization unquant` (bf16 drafter, as for FP8). 20 prompts per dataset (8 for AR):

| config (20 prompts) | tau | tok/s | x bf16 AR | x NVFP4 AR | draft ms | verify ms | round ms (GPU) |
|---|---|---|---|---|---|---|---|
| NVFP4 AR | 1 | 29.5 | 2.74x | 1.00x | - | 33.9 / token | |
| NVFP4 target, DSpine b16 chain | 7.25 | 123.0 | 11.42x | 4.17x | 19.8 | 35.0 | 56.5 |
| **NVFP4 target, tree 32** | **9.68** | **156.8** | **14.55x** | **5.31x** | 20.6 | 36.4 | 58.9 |
| NVFP4 target, tree 48 | 10.21 | 149.5 | 13.87x | 5.06x | 23.4 | 39.3 | 64.6 |

Per dataset (NVFP4 target, tree 32): 204 / 191 / 159 / 160 / 70 tok/s. Acceptance drops a little (tree 32 tau 9.68 vs 9.92 with the bf16 target,
b16 chain 7.25 vs 7.62): the bf16-trained drafter agrees slightly less with the 4-bit model. With a 36 ms verify the bf16 drafter is now 35% of the
round, so a quantized drafter (or a drafter trained on the NVFP4 model) is where the remaining speed is.

All three targets at their best config (first 20 prompts per dataset):

| target | AR tok/s | best spec config | spec tok/s | x bf16 AR | x own AR |
|---|---|---|---|---|---|
| bf16 | 10.8 | tree 32 | 79.9 | 7.41x | 7.41x |
| FP8 (block 128) | 19.4 | tree 32 | 126.4 | 11.73x | 6.52x |
| NVFP4 | 29.5 | tree 32 | 156.8 | 14.55x | 5.31x |

## Losslessness

`bench/p2_tf_check.py` (teacher-forced: run the HF bf16 target over prompt + engine output and compare every generated token with its argmax;
first 8 outputs per cell, up to 1500 tokens each). Mismatches with a small logit margin are bf16 near-ties from the different verify shapes,
as on A100 (Phase 2: tree 0.33-0.53%, chain 0.36-0.60%, margins <= 0.5). Results `bench/results/gb10/tf_check_{a,e,sweep}.json`:

| cell | mismatch rate gsm8k / math500 / humaneval / mbpp / mtbench (%) | max margin (logit) | mismatches with margin > 0.5 |
|---|---|---|---|
| DSpine b16 chain (reference) | 0.64 / - / - / - / 0.98 | 0.25 | 0 |
| DSpine b32 chain, DFlash b16 | 0.51, 0.56 (gsm8k) | 0.25 | 0 |
| **tree 32, depth 31 (headline)** | 0.56 / 0.36 / 0.44 / 0.48 / 1.40 | 0.5 | 0 |
| tree 48, depth 31, Phase 2 glue | 0.42 / 0.35 / 0.56 / 0.54 / 1.24 | 0.5 | 0 |
| tree 48, fast path | 0.42 / - / - / - / 1.24 (identical to Phase 2 glue) | 0.25 | 0 |
| tree 64, padded GEMMs | 0.47 / - / - / - / 1.07 | 0.25 | 0 |

## Other things tried
- **FP8 DSpine drafter.** The plugin refuses a quantized draft; `bench/gb10/apply_dspine_draftquant_patch.py` (env `RDF2_DSPINE_DRAFT_QUANT=1`)
  lifts the check so `--speculative-draft-model-quantization fp8` loads. It changed nothing: same tau and the same draft time (tree 32:
  20.9 vs 20.8 ms with the bf16 target, 20.9 vs 21.0 ms with the FP8 target; `q_tree32_d31_fp8draft`, `fp8_q_tree32_d31_fp8draft`), so SGLang's
  online FP8 path evidently does not reach the DSpine draft's layers. A real quantized drafter needs plugin work and is the main remaining lever
  for the FP8/NVFP4 targets (draft = 27-35% of their round).
- **cuBLASLt instead of cuBLAS** for the bf16 GEMMs: no better at 33-111 rows (table above).
- **GEMM row padding** to 128 rows: see above; helps 64/96-node trees a little, not enough to beat 32 nodes.

## What would move GB10 further
1. A drafter whose bytes match the target's: quantize or distill the 2.5 GB DSpine draft (plus the 1.2 GB lm_head read per round). With the NVFP4
   target, halving the draft's 20.6 ms would take the round from 59 to ~49 ms (+20% tok/s at the same tau).
2. A bf16 GEMM kernel for 33-128 rows that stays at the weight-streaming rate (e.g. a split-K / TGV-style GEMV kernel like SGLang's
   `cutedsl` backend, which is SM10x-only today). That would make 48-64-node trees pay off on GB10 as on A100 (+7-10% tau over 32 nodes).
3. A drafter trained against the quantized target (NVFP4 loses 2-5% tau with the bf16-trained drafter).

## Caveats
- One box, one run per cell (the 32-node tree was run twice on the 20-prompt subset: 80.0 / 79.9 tok/s). The box is shared; all timing runs held
  an exclusive GPU lock, but other users' CPU activity cannot be excluded. AR is 8 prompts per dataset; its speed is flat (10.8 tok/s on all five).
- Quantized-target cells use 20 prompts per dataset and are compared against the bf16 cells on the same 20 prompts.
- Batch 1, T=0 only, as in Phase 2. Thinking off. Not measured: concurrency > 1, T=1, Qwen3.8-27B (the 27B bf16 model alone is ~55 GB, above the
  ~45 GB memory budget this shared box allowed; an FP8 or NVFP4 27B would fit, not attempted for time).

## Reproduction (on the box, repo at `~/rdf2/recursive-dflash2`)
1. `bash bench/gb10/setup_spark.sh` (no root; everything under `~/rdf2`), then add `RDF2_PORT=<free local port>` to `~/rdf2/env.sh`.
2. One cell: `flock ~/.gpu.lock bash bench/gb10/run_suite.sh treefast f40_tree32_d31 "gsm8k math500 humaneval mbpp mtbench" 40 32 31`
   (modes: `ar`, `dflash`, `dspine`, `dspine_blk <N>`, `tree <N> <D>`, `treefast <N> <D>`; env `TARGET`, `EXTRA`, `RDF2_GEMM_PAD`).
   The queues actually run are `bench/gb10/queue_{a,b,b2,c,d,e,m,n,f,g}.sh` (b/c/e/m were partly superseded after lock timeouts; see `logs/gb10/`).
3. Summaries: `python bench/gb10/summarize.py ar <tags>` (`NFIRST=20` for the paired subset), `python bench/gb10/summarize_quant.py`.
4. Micro-benchmarks: `bench/gb10/roofline_micro.py`, `linear_sweep.py`, `gemm_variants.py`. Teacher-forced checks: `bench/p2_tf_check.py`
   (HF; `TF_MODEL=Qwen/Qwen3-8B-FP8` for FP8), `bench/gb10/sgl_tf_run.sh` + `sgl_tf_check.py` (SGLang prefill, for NVFP4).

Quantized targets cannot be checked with HF here (HF's FP8 loader needs `accelerate`, which is not in the venv; HF has no ModelOpt NVFP4
path). `bench/gb10/sgl_tf_check.py` does the same teacher-forced test with SGLang's own prefill of the same target (`input_top_logprobs`,
alignment self-check: 0 misaligned tokens everywhere), and the plain AR output of the same target is checked the same way as the noise floor.
Calibration on bf16 (`sgltf_bf16.json`): tree 32 0.39 / 0.38 / 0.51 / 0.36 / 1.11%, AR 0.43% (gsm8k) and 1.25% (mtbench), all margins <= 0.25, so
this check agrees with the HF one.

| target (SGLang prefill reference) | tree 32: mismatch % gsm8k / math500 / humaneval / mbpp / mtbench, max margin | same target's AR: gsm8k / mtbench, max margin |
|---|---|---|
| FP8 (`sgltf_fp8.json`) | 0.74 / 0.69 / 0.77 / 1.07 / 2.94, 0.75 | 0.88 / 2.52, 0.75 |
| NVFP4 (`sgltf_nvfp4.json`) | 2.6 / 2.3 / 3.3 / 3.8 / 11.7, 7.1 | 3.0 / 11.0, 6.7 |

FP8: the tree is at the FP8 AR noise floor (tree 48 0.71%, b16 chain 0.71% on gsm8k). NVFP4: SGLang's NVFP4 prefill and decode paths disagree
with each other far more than bf16 does (even plain NVFP4 AR output mismatches its own prefill on 3% / 11% of tokens, with margins up to 6.7 logits;
the FP8 KV cache of this checkpoint is the likely cause), so this check cannot certify token-level losslessness for NVFP4. What it shows is that the
tree (and the b16 chain: 2.6% on gsm8k) adds no disagreement beyond what NVFP4 AR itself has. Treat the NVFP4 cells as "same quality as NVFP4 AR
in SGLang", not as bit-exact greedy.
