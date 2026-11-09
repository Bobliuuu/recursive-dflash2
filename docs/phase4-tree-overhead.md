# Phase 4, experiment 1: cutting tree-verify overhead (Qwen3-8B, 1x A100-80GB, SGLang 0.5.22, batch 1, T=0)

Question: the Phase 2 DSpine tree (48 nodes, depth 31) gets tau 11.29 and 627 tok/s, but each round costs several ms more than the b16 chain.
Hypothesis: if that overhead is removed, the tree reaches >= 690 tok/s (>= 7.6x AR) at the same tau.

**Answer: no.** Cutting the overhead gave **651.5 tok/s, 7.20x AR, at the same tau (11.294)**. That is +3.4% over the Phase 2 path
re-measured on the same box (630.3 tok/s, 6.96x). Output is lossless: all 200 outputs are byte-identical to the Phase 2 tree's, and the
teacher-forced check passes. 690 cannot be reached by removing overhead. A tree with zero overhead over a 48-token causal chain verify
would get about 674 tok/s on these prompts, and 690 needs a round 0.4 ms shorter than the chain's own round (see "Why 690 is out of reach").

All runs used one fresh Vast A100-SXM4-80GB (label `rdf2-tree-ovh`) with the Phase 1/2 harness (`bench/run_cell.py`): 5 datasets x 40 prompts,
thinking off, max_new 2048, c=1, T=0. The x AR column uses the Phase 1 AR cells, as in Phase 1/2. Raw JSON is in `bench/results/p4_*`, the
per-round timers in `bench/results/p4_timing/`, the summary in `bench/results/p4_summary.json` (`bench/p4_summarize.py`) and the logs in `logs/p4/`.

## Results (mean of 5 datasets)

| config | tau | tok/s | x AR | gsm8k / math500 / humaneval / mbpp / mtbench tok/s |
|---|---|---|---|---|
| Phase 2 tree48 as published (`p2_tree48_d31`, Phase 2 box) | 11.294 | 626.8 | 6.92x | 846 / 775 / 586 / 588 / 339 |
| Phase 2 tree48 re-run here, old timer (`p2_p4base_tree48`)\* | 11.294 | 631.9 | 6.98x | 854 / 780 / 591 / 592 / 341 |
| **before**: Phase 2 tree48, new timer (`p4_tree48_base`) | 11.294 | 630.3 | 6.96x | 852 / 779 / 589 / 591 / 340 |
| + FlashInfer custom-mask tile skip (`p4_tree48_p2glue_fimask`) | 11.294 | 638.2 | 7.05x | 863 / 789 / 597 / 599 / 344 |
| + table and selection in the draft graph, exact table (`p4_tree48_exact`) | 11.294 | 642.2 | 7.09x | 868 / 794 / 601 / 602 / 346 |
| + same, DSpine bf16 table rescaled (`p4_tree48_rescaled`, dropped) | 11.175 | 646.7 | 7.14x | 882 / 792 / 617 / 614 / 329 |
| **after**: + same, factored exact table (`p4_tree48_factored`) | **11.294** | **651.5** | **7.20x** | **880 / 807 / 609 / 611 / 352** |
| reference: chain b48, causal verify (`p4_chain48`) | 9.033 | 546.7 | 6.04x | 758 / 667 / 538 / 473 / 298 |

\* Raw JSON for this first reproduction run was not copied off the box before it was destroyed. The numbers come from the session's summary output.

Run-to-run noise is about 1% (the same Phase 2 path gave 630.3 and 631.9 on this box). The +3.4% total is above that. Each step on its own
is +0.6% to +1.3%, close to the noise, but the per-round timer below shows each saving directly.

## Where the round time goes (median ms per decode round, GPU timeline)

A new timer (`bench/patches/apply_timing_patch_p4.py`) records CUDA events at 7 points in the DFlash/DSpine v2 round, plus host timestamps.
It writes rounds without a host sync. With the Phase 2 path the two timers agree: verify 12.55 ms with either, and 630.3 vs 631.9 tok/s, within noise.

| segment | before (Phase 2 path) | after (factored) | chain b48 | what it is |
|---|---|---|---|---|
| draft graph | 3.160 | 3.365 | 3.149 | DSpine draft forward (+ tree table and selection in the after column) |
| tree prep | 0.577 | 0.014 | 0.009 | Phase 2: eager fp32 table (~25 torch ops) + 220 us build kernel; after: one pack kernel |
| verify prep | 0.038 | 0.038 | 0.041 | verify batch / FlashInfer plan |
| verify | 12.553 | 12.340 | 11.992 | target forward over 48 tokens |
| accept | 0.067 | 0.067 | 0.043 | tree accept walk + 2-stage KV compaction (chain: chain accept) |
| draft-KV append | 0.207 | 0.208 | 0.210 | target hidden -> draft KV |
| tail | 0.268 | 0.268 | 0.260 | until the next round's draft starts |
| **round** | **16.850** | **16.285** | **15.696** | |
| tree overhead vs chain b48 | 1.154 | 0.589 | 0 | |

Draft + tree build fell from 3.74 to 3.38 ms (chain b48: 3.16), and verify from 12.55 to 12.34 ms (chain b48: 11.99). "Other" (everything
outside draft and verify) was 0.66 ms per round before and after (chain: 0.55). The host is not the bottleneck at bs=1. Host work is about
2 ms per round, and the host waits about 14 ms per round on the GPU (`cpu_*` fields in the timing files). So the cuts had to come off the GPU timeline.

The task brief puts the tree's "other" at 2.6 ms and the chain's at 1.8 ms. Those figures are effective round time (tau / tok/s) minus draft
minus verify, so they include per-request prefill and HTTP overhead, about 1 ms per round, which is the same for chain and tree. Measured per
round, accept and KV compaction cost only 0.024 ms more than the chain, and the draft-KV append costs the same. Fixing KV compaction would buy nothing.

## What was changed

1. **Tree table and best-first selection inside the DSpine draft CUDA graph.** Files: `bench/p4tree/glue_fast.py`,
   `tree_kernels_fast.py`, `apply_tree_patch_fast.py`. DSpine already runs `DSpineDraftSampler.propose` inside the draft graph. A hook on it
   builds the tree on every bs=1 round. Before verify, the host now launches one `tree_pack` kernel, which writes the packed FlashInfer mask
   for the current prefix length, the positions, the commit positions and the verify tokens.
   - New selection kernel (45 us, was 220 us). It computes all row softmaxes and top-8 orders in parallel, then runs the 47 pops in one warp.
     Each expanded node pushes its children in non-increasing score order, so only one "head" per node has to be compared. Tokens, parents,
     depths and the packed mask are bit-identical to the Phase 2 kernel on 600 random tables (`test_fast.py`).
   - Moving the Phase 2 table into the graph saves launches but little GPU time (`exact` row: draft 3.61 ms). The table is about 0.4 ms of
     true-fp32 GEMM and elementwise work on [496, 4096] fp32 intermediates, with TF32 off as DSpine requires.
   - **Factored table (headline).** Same fp32 math, re-associated:
     logit[j,c] = rho_j * ((x*g).h_c + s * m_j . TGf[c]).
     - TGf[v] = W^T (h_v*g) and the fp32 codebook projections [G_p; A] sqrt(C) code_v are precomputed once (~0.9 GB of GPU memory).
     - rho_j = rsqrt(mean(refined_j^2) + eps), where ||refined_j||^2 = ||x||^2 + 2s (W^T x).m_j + s^2 m_j^T (W^T W) m_j.
     - There is no [496, 4096] fp32 intermediate. In-graph cost is ~0.2 ms instead of ~0.45 ms.
     - Against the eager exact table (debug run, 1000 rounds): max |logit diff| 3e-5, identical tree in 99.6% of rounds, same tau,
       200/200 identical outputs (`p4_microbench/table_debug_factored.json`).
   - Tried and dropped:
     - `rescaled`: DSpine's own greedy-chain Triton table times rho_j (`dspine_table.py`, which adds a store of the [n,16,16] table it
       already computes). It is the fastest table, but its bf16 factor tables move tail probabilities enough to change 86% of trees. Tau
       fell to 11.18 (mtbench 5.92 -> 5.53), for 646.7 tok/s overall.
     - Raw DSpine table without rho: its logits lack the RMSNorm factor (log-prob MAE 3230), and tau collapses (gsm8k 15.9 -> 13.0).
2. **FlashInfer custom-mask prefill skips the mask on full-prefix KV tiles** (`bench/p4tree/apply_fi_mask_patch.py`). This patches the
   installed `prefill.cuh`, and the JIT rebuilds the kernel. Stock FlashInfer evaluates the custom mask on every KV tile (a global byte load
   and a bit test per score). Causal mode does that only on diagonal tiles. Now a tile that lies fully inside the prefix and inside its
   split-KV chunk skips the mask. This assumes the custom mask is all-ones over the prefix, which is true for tree verify.
   - Verify went from 12.553 to 12.342 ms. Microbenchmark, 36 layers (`p4_mask_bench.py`, `p4_microbench/mask_bench_*.json`): custom-mask
     prefill cost +0.58..1.03 ms over causal before the patch and +0.07..0.48 ms after, for P = 256..3072.
   - Checked against an fp32 reference at 13 prefix lengths x 3 split settings (`test_fi_mask.py`). The error matches stock FlashInfer.
     That test caught a first version that ignored chunk ends; it was wrong under auto split-KV (error 0.036 vs 0.0009).
   - The remaining custom-mask cost is about 0.35 ms per verify. A torch-profiler trace (`p4_profile.sh`, `p4_trace_kernels.py`) shows the
     same kernel config and grid as causal (CTA_TILE_Q 128, grid 27x8), but 28.9 us vs 19.6 us per layer. Both kernels sit at the
     255-register cap, so this is register pressure in the custom-mask variant, not mask loads. Two more fixes were tried and dropped:
     skipping the bit lookup for prefix columns inside `LogitsMask` (no gain, `mask_bench_patched2.json`), and running the tree on the
     causal kernel with tree bits applied on the diagonal tiles. The second was correct, but the extra code slows the causal kernel itself
     about as much as it saves (`mask_bench_treecausal.json`).
   - Gotcha: SGLang keeps its own FlashInfer JIT cache in `~/.cache/sglang/.cache/flashinfer`, which ninja rebuilds by header mtime. A
     revert by `os.replace(.orig)` restores an older mtime, so ninja kept the patched kernel. The first "stock" baseline in this session
     was therefore not stock. It was discarded and re-run after the script was fixed (it now touches reverted headers).
3. **KV compaction and append: measured, left as is.** Accept plus the 2-stage KV move take 0.067 ms. The draft-KV append is the same as the chain's.

## Why 690 is out of reach

- Tree overhead per round is now 0.59 ms over a 48-token causal chain verify: 0.35 ms custom-mask attention, 0.22 ms table and
  selection, 0.02 ms accept.
- Bound: remove all of it per dataset (effective round tau / tok/s minus 0.59 ms) and the tree would reach about **674 tok/s
  (7.45x)**. Applying the same calculation to the Phase 2 path gives the same bound.
- Reaching 690 at tau 11.29 needs a median round of about 15.3 ms. That is 0.97 ms below what we have now and 0.38 ms below the 48-token
  chain's own round (15.70 ms).
- So 690 needs a cheaper draft or verify than the chain itself, not less tree overhead. Candidate levers, none of them "tree overhead":
  - draft only the 31 positions the tree uses (block 32 instead of 48), roughly -0.1 ms;
  - a custom-mask attention kernel without the register-pressure penalty, up to -0.35 ms;
  - a faster small-M GEMM for the 48-row verify. Dense bf16 GEMMs dominate both the draft and the verify (profile above).

## Lossless check

- `bench/p2_tf_check.py`: teacher-forced HF Qwen3-8B argmax vs engine output, 8 requests per dataset, results in `bench/results/p4_tf_check.json`.
  - `p4_tree48_factored`: mismatch rate 0.34% / 0.52% / 0.33% / 0.53% / 1.35% (gsm8k / math500 / humaneval / mbpp / mtbench).
    Max margin 0.5 logit, no mismatch with margin > 0.5.
  - `p4_tree48_base`: identical numbers, because its outputs are identical.
  - `p4_chain48`: 0.54% / 0.39% / 0.53% / 0.35% / 1.14%, max margin 0.5.
  - These are bf16 near-ties, as in Phase 2. **Pass.**
- `bench/p4_compare_text.py`: `p4_tree48_factored` vs `p4_tree48_base` is 200/200 byte-identical. vs the Phase 2 published `p2_tree48_d31`
  it is also 200/200. `p4_tree48_exact` vs base is 200/200.

## Cost

- One Vast A100-SXM4-80GB at 1.17 USD/h for 1.66 h: **about 1.95 USD** (wall clock x listed price, not the invoice). Box destroyed.
- `bench/setup_box.sh` rebuilds the stack from a bare `nvidia/cuda:12.4.1-devel-ubuntu22.04` image in about 15 minutes:
  - uv venv, DSpine `pip -e` (which pulls SGLang 0.5.22, torch 2.14.1+cu130 and flashinfer 0.7.1);
  - CUDA 13.0 nvcc / nvvm / crt pins, CUDA_HOME with lib64 symlinks;
  - both models, `/workspace/dspine_b{16,32,48}`, `/workspace/env.sh`.

## Reproduction

```
bash bench/setup_box.sh                                    # fresh box (repo at /workspace/recursive-dflash2)
RDF2_TABLE=factored bash bench/p4_tree_run.sh tree48_factored p4tree skip 48:31   # headline
bash bench/p4_tree_run.sh tree48_base p2tree stock 48:31   # Phase 2 path, same timer
bash bench/p4_tree_run.sh tree48_p2glue_fimask p2tree skip 48:31
RDF2_TABLE=exact bash bench/p4_tree_run.sh tree48_exact p4tree skip 48:31
bash bench/p4_chain_run.sh chain48 48                      # 48-token causal chain (overhead floor)
python bench/p4_summarize.py p4_tree48_factored=/workspace/p4/timing_tree48_factored.jsonl ...
python bench/p2_tf_check.py results/p4_tf_check.json p4_tree48_factored:gsm8k:8 ...    # server stopped
python bench/p4tree/test_fast.py; python bench/p4tree/test_fi_mask.py; python bench/p4_mask_bench.py
```
Debug comparison of a table mode against the exact table: `RDF2_TABLE=<mode> RDF2_TABLE_DEBUG=<out.json> bash bench/p4_tree_run.sh ...`.
In `bench/results/p4_*`, the `abl_*`, `dbg_*` and `smoke_*` cells are 6-10 prompt exploratory runs; their names follow the renamed table
modes (`exact` = in-graph Phase 2 table). Inside the JSONs, `tree48_rescaled` and `tree48_exact` carry their run-time method names
`dspine_tree48_fast` and `dspine_tree48_fastexact`.
