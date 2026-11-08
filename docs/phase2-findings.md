# Phase 2: improvements on top of DSpine (Qwen3-8B, 1x A100-80GB, SGLang 0.5.22)

Baseline = Phase 1 DSpine-b16, T=0, c=1, thinking off, same prompt set (5 datasets x 40 prompts, `bench/data/prompts`), max_new 2048.
The b16 control was re-run in this session (`p2_blk16_*`) and reproduces Phase 1 exactly (tau 7.99, 524 tok/s mean, 5.79x AR).
All raw JSON is in `bench/results/p2_*.json`; code in `bench/p2_*.py`, `bench/p2tree/`; logs in `logs/`. Nothing here retrains any weights.

## Headline
| Idea | Verdict | Measured (c=1, T=0, mean of 5 datasets) |
|---|---|---|
| E1 tree verification (best-first tree over DSpine's exact predecessor-conditioned table) | **WINS, implemented end to end** | tree N=48: tau 11.29, **627 tok/s = 6.92x AR, +19.7% vs chain b16** (524 tok/s, 5.79x) |
| E2 longer block with the unchanged b16 checkpoint (block 32) | **WINS, free** (config change only) | b32 chain: tau 9.03, 558 tok/s, +6.5% (b24 +4.8%, b48 +4.8%, b64 +1.7%) |
| E2 second pass conditioned on pass-1 tokens (D-Loop style) | Skipped (not meaningful without training), see below | n/a |
| E3 confidence-gated verify width | **Killed** | AUROC 0.91 per position, 0.95 for full block, but projected gain +0.9% (c=1) / +3.9% (c=8) |
| E4 n-gram / prompt-lookup tail after a full 16-block | **Killed** | +0.26..0.59 tokens per round, projected +0.8..1.9% speed |

Tree and longer-block gains exist **only at c=1** (verify is weight-bandwidth bound there). At c=8 the same changes lose: block 32 at c=8 is -22% tok/s
(2084 vs 2689), and a 48-node tree verifies 384 tokens per step, so it would lose too (not measured, see "Concurrency").
The tree path is implemented for greedy (T=0), batch size 1 only. T=1 would need multi-draft rejection sampling (not done).

## Setup facts that make the experiments cheap
DSpine's SGLang plugin computes, per draft round, K=16 candidate ids per position and a refinement table `serial_rs_table` of shape [15, 16, 16]:
`table[i, j, c]` is the logit of candidate `c` at position `i` given that position `i-1` chose candidate `j` (row 0 is conditioned on the anchor).
So DSpine is a real predecessor-conditioned model over a 16-wide candidate set, not just per-position marginals, and a tree can be built with exact
conditional probabilities. A one-line hook (`bench/patches/apply_p2_dump_hook.py`, env `RDF2_DUMP`) dumps anchor, greedy chain, candidates and this table
for every bs=1 round. 200 requests (40 per dataset) gave 12,884 rounds. Because T=0 verified output equals the target's greedy continuation, the
logged output is the "oracle" continuation, so any tree can be replayed offline at the logged anchors (exact for each logged round; the tree's own
trajectory would visit different anchors, so tau is a paired per-round estimate). The offline numbers matched the engine to within a few percent (below).

## E1: tree verification
### Offline replay (b16 rollouts, chain emitted tokens/round = 7.62; `p2_e1_*.json`)
| verify nodes N (incl. anchor) | best-first tree | static rank-trie | beam (W=2/3/4 -> 31/46/61 nodes) |
|---|---|---|---|
| 16 | 8.01 (+5.1%) | 7.62 | |
| 24 | 8.57 | 7.99 | |
| 32 | 8.86 (+16.3%) | 8.19 (+7.5%) | W2 (31): 8.56 |
| 48 | 9.18 (+20.5%) | 8.47 | W3 (46): 8.93 |
| 64 | 9.36 (+22.9%) | 8.66 (+13.7%) | W4 (61): 9.17 |
Node probabilities = softmax(table row) with temperature 1.0 (0.5 and 2.0 were worse: N=32 8.67 / 8.59). The static trie (no per-round build, chosen on half of
the requests, tested on the other half) loses about half of the dynamic gain, and a fixed-width beam is clearly worse than best-first at equal node count,
so the engine uses a per-round best-first build. Only 9.9 of the first 15 true tokens are inside the 16-candidate sets on average (coverage ceiling).
Rounds with all 15 drafts accepted: 21.5% (chain), 29% (tree N=64): the 16-token block is a hard cap on tau.

### Verify cost vs width (real engine, `p2_width_cost_timing_b*.jsonl`, `p2_widthcost.sh`)
DSpine checkpoint run at block size N (garbage-free draft positions, real target verify, SGLang CUDA graphs, SPEC timer):
T_verify = 11.05 (16), 11.57 (24), 11.70 (32), 11.91 (48), 12.10 ms (64); T_draft 2.97 -> 3.15 ms. So 48 tokens cost only +0.86 ms (+8% of a 14.0 ms round).
(The HF-graph sweeps `p2_verify_cost_8b_ctx*.json` give the same increments but a higher absolute level, 17.7 ms at B=1; only the increments were used.)
Predicted speed from these numbers: +11..14% with zero tree overhead, +7.6..10.7% with 0.5 ms, +6% with 1 ms. The measured engine result is above this
because the longer block (E2) also lifts tau (see below).

### Engine implementation (works, bs=1 greedy)
`bench/p2tree/` + `bench/p2_tree_run.sh`:
- `tree_kernels.py`: three CUDA kernels (torch `load_inline`): `tree_build` (one block: best-first selection of N-1 nodes by joint probability from the
  [D,16,16] table, writes tokens/parents/depths/positions and the tree-attention mask directly **bit-packed into FlashInfer's custom-mask buffer**),
  `tree_accept` (walk the tree against the target's per-node argmax, emit tokens/bonus/new seq len, build the KV move list), `kv_move` (two-stage copy of the
  accepted path's K/V rows into the first slots of the verify window, all 36 layers, so the committed positions are contiguous).
- `glue.py` + `apply_tree_patch.py` patch SGLang's `dflash_worker_v2.py` (build tree after the draft graph, use tree positions and mask in the verify
  input, tree accept instead of chain accept, reorder target hidden rows to path order before the draft-KV append, chain positions for the append),
  `dflash_utils.py` (capture the verify graph with a custom-mask wrapper) and `flashinfer_backend.py` (allow SGLang's sync-free `fast_prefill_plan` with a
  custom mask; the stock path blocks on 3 D2H copies per replay, which cost ~3 ms/round in the first version `v1`).
- Draft model = the unchanged b16 checkpoint with `block_size` set to N in a local config copy; tree depth is capped at D=31 (positions beyond 31 are poor, below).
- Verification of correctness: kernels vs the Python reference (`test_kernels.py`, 0 mismatches over 400 rounds incl. the packed mask); teacher-forced
  check of engine outputs against the HF target (`p2_tf_check.py`, `p2_tf_check.json`): tree mismatch rate 0.33-0.53% with max margin 0.5 logit, chain b16 baseline
  0.36-0.60%, i.e. bf16 near-ties only. Byte-identical-to-AR fraction (this noisy metric): chain b16 54/200, b32 63/200, tree48 67/200.

### Engine results (`p2_tree*_d31_*_T0_c1.json`, `p2_e1_tree_engine_summary.json`)
| config | tau (mean of datasets) | tok/s (mean) | speedup vs AR | vs chain b16 | gsm8k / math500 / humaneval / mbpp / mtbench tok/s |
|---|---|---|---|---|---|
| chain b16 (control) | 7.99 | 524 | 5.79x | 1.000 | 672 / 647 / 517 / 490 / 292 |
| chain b32 | 9.03 | 558 | 6.17x | 1.066 | 766 / 691 / 548 / 483 / 304 |
| tree N=32 | 10.43 | 593 | 6.55x | 1.132 | 799 / 731 / 562 / 553 / 318 |
| tree N=40 | 10.90 | 611 | 6.75x | 1.166 | 827 / 746 / 604 / 558 / 318 |
| **tree N=48** | **11.29** | **627** | **6.92x** | **1.197** | **846 / 775 / 586 / 588 / 339** |
| tree N=56 | 11.18 | 614 | 6.78x | 1.173 | 833 / 763 / 578 / 572 / 324 |
| tree N=64 | 11.47 | 616 | 6.80x | 1.176 | 829 / 768 / 581 / 565 / 335 |
| tree N=48, first version (stock sync-heavy plan) | 11.29 | 556 | 6.14x | 1.061 | |
N=48 is the plateau (differences between 48 and 64 are within run noise, about 1-2%). Gain by dataset at N=48: gsm8k +26%, math500 +20%, humaneval +13%, mbpp +20%, mtbench +16%.
Timer per round at N=48: draft+tree-build 3.69 ms (chain draft 2.97), verify 12.51 ms (chain 11.05; the custom-mask prefill is ~0.5 ms slower than the
causal path), remaining ~2.6 ms scheduler/commit/draft-KV append (chain: 1.8). N=32/64 runs used a version of the glue that computes the table for all
block rows (0.17 ms slower); N=40/48/56 use the trimmed version. Run-to-run noise is about 1% (the N=48 cell gave 621 and 627 tok/s with the two glue versions).

Why the tree gain is larger than predicted from the b16 offline table: the tree is run on a block-48 draft, whose positions 17..31 come from the same weights
(see E2), so full-block-cap rounds can continue past 16. Offline on b32 rollouts (`p2_e1_*_on_b32.json`): chain 8.36, best-first N=32 9.50, N=48 10.04, N=64 10.34,
N=96 10.67; beam W=2/3/4 (63/94/125 nodes; the `nodes_incl_anchor` labels in `p2_e1c_beam_tree_on_b32.json` are for 16-token blocks and wrong there) 9.59/10.11/10.42.

## E2: adaptive / longer block
(a) **Longer block with the unchanged b16 checkpoint.** SGLang builds the draft block from `config.block_size`; setting it to N (local config copy, weights untouched)
makes DSpine propose N-1 tokens. Conditional acceptance of the extra positions (b32 rollouts, `p2_e3_confidence_on_b32.json`): positions 16-18 stay at 0.88-0.92,
then decay to 0.85 (22-24), 0.78 (25-28), 0.65-0.71 (29-30). Real engine tok/s at c=1 (`p2_e2b_blocksweep_T0_c1.json`):
| block | tau | tok/s | speedup vs AR | vs b16 |
|---|---|---|---|---|
| 16 | 7.99 | 524 | 5.79x | 1.000 |
| 24 | 8.77 | 549 | 6.06x | 1.048 |
| 32 | 9.03 | 558 | 6.17x | 1.066 |
| 48 | 9.03 | 549 | 6.07x | 1.049 |
| 64 | 8.92 | 533 | 5.88x | 1.017 |
By dataset b32/b16 tok/s: gsm8k 1.14, math500 1.07, humaneval 1.06, mbpp 0.99, mtbench 1.04. A drafter trained at longer blocks (the DBloom point in the prior-art note) should do better
than this free extrapolation. At c=8 (`p2_e2b_blocksweep_T0_c8.json`, 80 prompts/cell) b32 is 2084 tok/s vs 2689 for b16 (-22%); there the optimum is b16 (b12 2494, b8 2263).
(b) **Second pass conditioned on pass-1 tokens: skipped.** DSpine already runs a serial last-write refinement conditioned on the predecessor token (K=16 exact-sender),
which is the cheap part of what a second pass would give. A full pass 2 with this checkpoint feeds real tokens into a drafter that was trained only on mask-token inputs; that is
out of distribution and D-Loop (arXiv 2610.06011) trains for it. Re-running the drafter at the anchor "within the same verify" is not possible, because the hidden states of the
block needed for the next draft only exist after the verify. So no meaningful no-training test exists; it belongs to a training phase.

## E3: confidence gating (`p2_e3_confidence.json`)
- Predictive power is high: AUROC of conditional max-prob for "draft token accepted given the prefix was accepted" = 0.909 (neg-entropy 0.907; 0.89-0.94 by position);
  cumulative-survival product predicts "full 15-block accepted" with AUROC 0.952 (predicted mean survival 0.269 vs actual 0.215, mildly overconfident).
- Truncating the verify width by cumulative survival (width = 1 + #positions with survival >= theta): theta=0.1 gives mean width 12.6 and loses only 0.8% of tau.
  Round-time model uses the **measured** curves: c=1 from the engine timer (+HF increments below 16), c=8 from the real fixed-block runs (round = tau*8/tok_s:
  8: 20.2 ms, 12: 22.7 ms, 16: 23.3 ms; 32: 33.6 ms). Projected speed: c=1 +0.9% (best theta=0.05), c=8 +3.9% (best theta=0.15), with the optimistic per-round accounting.
  Fixed widths are clearly negative (c=8 real: b12 -7%, b8 -16%). Cost is nearly flat below 16 tokens on A100/8B, so there is nothing to save. Killed.
- Chain-vs-tree gating: confidence only weakly predicts where the tree helps (AUROC 0.64 for "tree improves this round"); gating by survival < 0.2 uses the tree on 64% of rounds and keeps
  66% of the tree gain, but a static-shape graph pays the wide verify either way. Killed. (Confidence is still what the tree builder uses implicitly, as joint probability.)
- Earlier draft of E3 used a linear per-token cost model that predicted +16% at c=8; the measured c=8 fixed-width runs disprove it (the cost per token is not constant: ~0.02 ms/token from 12 to 16 tokens/request).

## E4: n-gram / prompt-lookup tail (`p2_e4_ngram_tail.json`)
Only rounds with all 15 drafts accepted can use a tail: 21.5% of rounds. Lookup of the last n-gram (n=2,3,4) over prompt + generated text + accepted block, proposing m=4/8/16 tokens
verified after the block (first tail token must equal the bonus): match found in 51-73% of those rounds, mean extra accepted tokens per full-accept round 1.2 (m=4), 1.8 (m=8), 2.3-2.7 (m=16),
i.e. +0.26..0.59 emitted tokens per round (+3.4..7.7% tau), but with the 8B verify-width cost the projected speed gain is only +0.8..1.9% (best: n=4,m=16: +1.9%). Killed.
The tail idea is subsumed by longer blocks (E2) and trees (E1), which already extend full-accept rounds.

## Concurrency, sampling, caveats
- c=8: measured only for chain widths (above). A 48-node tree verifies 8x48=384 tokens per step; from the measured b32 point (256 tokens, 33.6 ms vs 23.3 ms) it would need roughly
  40+ ms vs tau +41%, i.e. a loss of order 15%. Not measured. Tree verify should be a bs=1 (latency) mode only.
- T=1: not implemented (tree needs multi-path rejection sampling). The offline conditional tables exist, so a lossless sampler is a well-defined follow-up.
- Offline tau estimates are paired per logged chain anchor (trajectory differences ignored); engine runs are the ground truth.
- The tree path asserts bs=1, all-greedy and a graph-captured draft; other cases are not supported in this experimental patch (non-tree rounds in a tree server would use a stale mask).
- 40 prompts per cell, one seed (T=0 deterministic up to bf16 shape noise); differences below ~2% are noise.

## Reproduction (box: `source /workspace/env.sh`)
1. `python bench/patches/apply_p2_dump_hook.py <DSpine>/sglang_dspine/runtime/worker.py`; `bench/p2_serve_dump.sh dump.bin`; `python bench/p2_collect.py ...` then `p2_e1_tree.py`, `p2_e1b_static.py`, `p2_e1c_beam.py`, `p2_e3_conf.py`, `p2_e4_ngram.py` (set `P2_T1=block-1` for non-16 blocks).
2. Block sweep: `bench/p2_blocksweep.sh "16 24 32 48 64" 1 0` (needs `/workspace/dspine_b<N>` dirs = b16 weights + config with block_size N), summary `p2_summarize_blk.py`.
3. Tree verify: `bench/p2_tree_sweep.sh 48:31` (applies `bench/p2tree/apply_tree_patch.py`, which edits the installed SGLang; `--revert` restores), summary `p2_summarize_tree.py`.
