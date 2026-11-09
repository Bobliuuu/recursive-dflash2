# Phase 4, experiment 4: raising draft candidate coverage (Qwen3-8B, 1x A100-80GB, SGLang 0.5.22, batch 1, T=0)

Question: the DSpine tree (48 nodes, depth 31) offers 16 candidates per position, and only about 10 of the first 15 true tokens fall
inside them. If each position offered more candidates (K = 32 or 64), would tau rise enough to reach >= 670 tok/s?

**Answer: no.** Coverage is not what limits the 48-node tree. The limit is the node budget, and that comes down to how the draft ranks
its candidates.
- K=64 raises the share of rounds that run to depth 31 inside the candidate sets a lot (16-candidate prefix coverage 13.4 -> 17.3 tokens).
- But the offline tree tau at N=48 rises only **+1.7%** (10.78 -> 10.97). The target was +8%.
- In the engine, K=64 gives **tau 11.44 vs 11.29 (+1.3%)**, but the draft costs 0.57 ms more per round. The result is **632 tok/s vs
  645 tok/s** for K=16 on the same box.
- K=32 gives tau 11.27 and 634 tok/s.
- No score policy over the same tables (temperature, per-depth temperature, calibration, blending in the marginal logit) gains more than +0.5%.
- Outputs are lossless (teacher-forced check passes). The fine-tune step was not run, because its precondition ("coverage is the cap") is false.

All runs used one Vast A100-SXM4-80GB (label `rdf2-coverage`) set up with `bench/setup_box.sh`, the Phase 1/2 prompt set (5 datasets x 40
prompts, thinking off, max_new 2048, c=1, T=0) and the Phase 4 headline path (factored in-graph table, FlashInfer mask tile skip).

## Method

- DSpine's candidate width is a checkpoint config value (`dflash_config.selected_prefix_last_write_refine.head_topk`, a power of two from
  2 to 64). Raising it to 64 needs no retraining.
  - Candidates are the top-K of the draft's marginal logits.
  - The exact refinement table `serial_rs_table` gives the logit of candidate c given the predecessor's candidate j. Each entry depends
    only on its (j, c) pair.
  - So one K=64 dump contains the K=16 and K=32 tables as sub-blocks (top-16/32 by marginal logit).
- Dump (`bench/p4cov/apply_cov_dump_hook.py`, `run_dump.sh`, `collect.py`):
  - stock block-48 DSpine chain at head_topk 64, 200 requests, 11,874 rounds;
  - per round: the 64 candidates, their marginal logits and the [31,64,64] table, stored as float32.
  - The K=64 chain has the same tau as the K=16 chain (9.03, the dataset mean of output tokens per verify), so the trajectory is the Phase 2 one.
- Offline replay (`bench/p4cov/sim.py`): best-first tree by joint probability at the logged chain anchors, replayed against the target's
  greedy output, as in Phase 2 (`p2_e1_tree.py`).
  - This is a paired estimate. It underestimates the engine tau by about 5% at every K (K=16: offline 10.78, engine 11.29), so only the ratios are used.
  - Loss decomposition (`bench/p4cov/analyze.py`).
- Engine (`bench/p4cov/tree_run.sh`, `p4tree_k/`, `tree_kernels_k.py`): the Phase 4 glue, unchanged, with its selection kernel swapped
  for a K-generic one.
  - The kernel is two in-graph launches: a warp per (depth, predecessor) row computes the softmax and top-E, then one warp does the pops.
  - `test_k.py`: the trees are identical to the Phase 4 kernel at K=16 (300/300) and to the offline reference at K=32/64 (200/200 each).

## Coverage (offline, mean of 5 datasets)

| K | P(true token in set) at depth 1 / 5 / 10 / 15 / 20 / 25 / 31 | true tokens in set, first 15 | consecutive prefix covered (cap 31) | same, cap 15 |
|---|---|---|---|---|
| 16 | 0.99 / 0.87 / 0.75 / 0.68 / 0.61 / 0.54 / 0.47 | 12.2 | 13.4 | 10.0 |
| 32 | 0.99 / 0.90 / 0.82 / 0.76 / 0.71 / 0.65 / 0.59 | 12.9 | 15.3 | 10.8 |
| 64 | 1.00 / 0.93 / 0.87 / 0.82 / 0.78 / 0.74 / 0.70 | 13.5 | 17.3 | 11.6 |

The "consecutive prefix covered" column is the tau of an unlimited tree inside the candidate sets, minus one. K=64 raises it by 3.8 tokens.

## Offline tree tau (emitted tokens per round, mean of 5 datasets; chain on the same rounds 9.03)

| policy | N=48 | N=64 | N=96 |
|---|---|---|---|
| K=16, softmax T=1 (engine policy) | 10.78 | 11.12 | 11.50 |
| K=32, T=1 | 10.90 (+1.1%) | 11.26 | 11.69 |
| K=64, T=1 | **10.97 (+1.7%)** | 11.34 | 11.78 (+9.3%) |
| K=32 / K=64, 16 children per node instead of 8 | 10.90 / 10.97 | 11.26 / 11.34 | 11.69 / 11.78 |
| K=16 / K=64, T=0.7 | 10.53 / 10.72 | | |
| K=16 / K=64, T=1.5 | 10.79 / 10.85 | | |
| K=16 / K=64, calibrated by (depth, prob) bins, fit on even requests\* | 10.77 / 10.85 | | |
| K=16 / K=64, per-depth temperature 1 -> 1.5 (pooled over rounds; T=1 pooled: 9.98 / 10.17) | 10.05 / 10.22 (+0.7% / +2.4% vs pooled K=16 T=1) | | |
| K=16 / K=64, + 0.25 x marginal logit (pooled) | 9.89 / 10.07 | | |

\* On the held-out odd requests the calibrated tree is +0.5% (K=16) and -0.6% (K=64) against T=1 (`odd` block of `sim_k64.json`).

The verify cost of N=96 is about twice that of N=48. Phase 2 found no engine gain from N=48 to N=64, so the N=96 column is not a usable lever.

## Why more coverage does not help at N=48 (`analyze_k64.json`, 11,874 rounds, pooled)

Each round's loss is classified at the first true token the 48-node tree misses:

| K | true path fully in tree (depth 31) | coverage miss: true token not a candidate | budget miss: candidate, but its node not selected | budget misses where the true token was rank 1 / 2 / 3 in its row |
|---|---|---|---|---|
| 16 | 3.5% | 33.2% | 63.2% | 12% / 39% / 18% |
| 64 | 3.7% | 12.8% | 83.6% | 12% / 32% / 16% |

- With 16 candidates, two thirds of rounds already end on a token that was a candidate. The 47 nodes run out first.
- K=64 turns most coverage misses into budget misses, at about the same depth: mean loss depth 8.7 vs 8.9.
- The tree can afford about 1.5 branches per depth. The missed true token is usually the 2nd or 3rd choice of its row.
- So the lever is the draft's ranking (sharper conditionals), not a wider candidate set.
- A warm-start fine-tune for top-K recall targets the wrong thing. Phase 3 also showed that a cheap fine-tune barely moves the
  conditionals. It was not run.

## Engine results (same box, mean of 5 datasets)

| config | tau | tok/s | x AR | draft ms | verify ms | round ms | gsm8k / math500 / humaneval / mbpp / mtbench tok/s |
|---|---|---|---|---|---|---|---|
| K=16, K-generic glue (`p4cov_tree48_k16`) | 11.294 | 645.4 | 7.13x | 3.523 | 12.361 | 16.43 | 870 / 799 / 604 / 605 / 348 |
| K=32 (`p4cov_tree48_k32`) | 11.267 | 634.4 | 7.01x | 3.743 | 12.363 | | 864 / 769 / 605 / 588 / 346 |
| K=64 (`p4cov_tree48_k64`) | **11.442** | **632.3** | 6.98x | 4.088 | 12.361 | 16.98 | 855 / 776 / 599 / 601 / 331 |
| Phase 4 headline path, re-run on this box (`p4_cov_p4ctl`) | 11.294 | 647.2 | 7.15x | 3.509 | 12.367 | | 872 / 801 / 605 / 608 / 349 |
| Phase 4 headline, as published (other box) | 11.294 | 651.5 | 7.20x | 3.365 | 12.340 | 16.29 | 880 / 807 / 609 / 611 / 352 |

- This box is about 0.7% slower than the Phase 4 box: the re-run Phase 4 path gives 647.2 vs 651.5 tok/s, with 200/200 identical outputs.
  The K-generic glue at K=16 costs another 0.3% (645.4; two selection launches instead of one).
- K=16 through the K-generic glue gives outputs byte-identical to the published Phase 4 headline (200/200, `p4_compare_text.py`) at
  the same tau.
- K=64 changes the tree, so 75/200 outputs are byte-identical to K=16 and the rest differ only through bf16 near-ties (lossless check below).
- Extra draft time at K=64 (+0.57 ms per round) comes from DSpine's own top-64, its K^2 greedy-chain table (`serial_table`, which still
  runs although the tree does not use its chain), the [31,64,64] factored table and the 4x larger selection.
- Bound: with zero extra draft cost, K=64's +1.3% tau would give about 656 tok/s on this box (647.2 x 11.442 / 11.294), or 660 at the
  published 651.5. That is still short of 670.

## Lossless check

`bench/p2_tf_check.py` (teacher-forced HF Qwen3-8B argmax vs engine output, 8 requests per dataset, `bench/results/p4cov/tf_check.json`):
- `p4cov_tree48_k64`: mismatch rate 0.38% / 0.39% / 0.35% / 0.42% / 0.96% (gsm8k / math500 / humaneval / mbpp / mtbench).
  Max margin 0.25 logit, none above 0.5.
- `p4cov_tree48_k16`: gsm8k 0.34%, mtbench 1.35%, max margin 0.25.
- Same as the Phase 4 cells (bf16 near-ties). **Pass.**

## Cost

- One Vast A100-SXM4-80GB at 1.17 USD/h for about 0.9 h: **about 1.05 USD** (wall clock x listed price, not the invoice). Box destroyed.

## Reproduction

```
bash bench/setup_box.sh
bash bench/p4cov/run_dump.sh 40                                # K=64 dump of the b48 chain -> /workspace/p4cov/{dump_k64.bin,meta_k64.json} (6.3 GB)
python bench/p4cov/sim.py /workspace/p4cov/dump_k64.bin /workspace/p4cov/meta_k64.json sim_k64.json
python bench/p4cov/analyze.py /workspace/p4cov/dump_k64.bin /workspace/p4cov/meta_k64.json analyze_k64.json   # run from bench/p4cov
python bench/p4cov/test_k.py
RDF2_TABLE=factored RDF2_TREE_K=64 bash bench/p4cov/tree_run.sh tree48_k64 skip 48:31    # also K=32, K=16
python bench/p4cov/summarize.py p4cov_tree48_k64=/workspace/p4/timing_tree48_k64.jsonl ...
python bench/p2_tf_check.py results/p4cov/tf_check.json p4cov_tree48_k64:gsm8k:8 ...     # server stopped
```
