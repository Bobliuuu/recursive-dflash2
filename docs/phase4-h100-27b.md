# Phase 4, experiment 3: the DSpine tree on H100, then on Qwen3.8-27B

Date 2026-11-09. One Vast H100 80GB HBM3 (SXM, label `rdf2-h100-tree`), SGLang 0.5.22 + `sglang-dspine` (DSpine HEAD 321b0fc), set up with
`bench/setup_box.sh` (arch 9.0). Batch 1, T=0 throughout.

Questions:
1. Does the Qwen3-8B tree gain hold on an H100, and which node count is best there? Success bar: tree >= +20% over the chain.
2. Can the tree run on the hybrid Qwen3.8-27B (48 GatedDeltaNet + 16 attention layers) DSpine, losslessly? Success bar: >= 5.2x AR at c=1 T=0.

**Answers.**
1. **Yes.** Best is **N=64 nodes (depth 31): 970 tok/s, 6.49x AR, tau 11.38**. That is **+27% over the stock b16 chain** (762 tok/s, 5.09x)
   and +18% over the best chain (block 32, 821 tok/s, 5.49x). The sweep is flat between 48 and 96 nodes (959 / 970 / 957 tok/s).
   Lossless (teacher-forced check passes for every reported config).
2. **The port works and is lossless, but it does not reach 5.2x.** Best tree: **244.1 tok/s, 4.95x AR** (same at 24, 32 and 48 nodes;
   tau 7.1-8.0) vs 231.0 tok/s, 4.68x for the published DSpine chain config on the same box: **+5.7%**. The hybrid model's verify pays
   ~0.15 ms per tree node for per-node fp32 GDN state snapshots (151 MB each), which eats most of the +25-41% tau gain.

## 1. Qwen3-8B on H100

Harness as Phases 1/2/4 (`bench/run_cell.py`, 5 datasets x 40 prompts, thinking off, max_new 2048). Runner `bench/p5h100/run8b.sh`
(same server flags as `bench/p4_tree_run.sh`; tree = the Phase 4 fast path, `RDF2_TABLE=factored` + FlashInfer mask tile skip).
x AR uses the AR cells measured on this same H100 (`ar_*`, 149.7 tok/s). Raw JSON: `bench/results/p5h100/`, summary
`bench/results/p5h100/summary8b.json` (`bench/p5h100/summarize8b.py`), per-round timers `bench/results/p5h100/timing/`, logs `logs/p5h100/`.

| config | tau | tok/s | x AR | gsm8k / math500 / humaneval / mbpp / mtbench tok/s | draft ms | verify ms | round ms |
|---|---|---|---|---|---|---|---|
| AR | 1 | 149.7 | 1.00x | 149 / 150 / 150 / 150 / 150 | | | |
| chain b16 (stock DSpine) | 7.84 | 762.2 | 5.09x | 989 / 934 / 753 / 723 / 412 | 2.18 | 7.10 | 9.83 |
| chain b32 | 8.75 | 820.7 | 5.49x | 1148 / 1015 / 793 / 736 / 412 | 2.22 | 7.33 | 10.12 |
| chain b48 | 8.75 | 803.9 | 5.37x | 1121 / 996 / 777 / 720 / 405 | 2.26 | 7.52 | 10.34 |
| tree 32 / d31 | 10.48 | 938.8 | 6.27x | 1242 / 1131 / 894 / 935 / 493 | 2.34 | 7.61 | 10.53 |
| tree 48 / d31 | 11.06 | 959.6 | 6.41x | 1295 / 1203 / 915 / 888 / 497 | 2.41 | 7.80 | 10.83 |
| **tree 64 / d31** | **11.38** | **970.3** | **6.49x** | 1322 / 1175 / 931 / 915 / 509 | 2.44 | 7.97 | 11.00 |
| tree 96 / d31 | 11.84 | 957.3 | 6.40x | 1314 / 1190 / 906 / 889 / 488 | 2.55 | 8.40 | 11.58 |

(timer columns: median ms per decode round, GPU timeline, `bench/patches/apply_timing_patch_p4.py`.)

- H100 vs A100: everything is about 1.5x faster (AR 149.7 vs 90.5 tok/s, tree48 959.6 vs 651.5), and the speedups over AR shrink a little
  (tree48 6.41x vs 7.20x, chain b16 5.09x vs 5.78x): AR is closer to the bandwidth roofline than the spec rounds are.
- Verify cost curve: 7.10 ms at 16 tokens, 7.80 at 48, 7.97 at 64, 8.40 at 96, i.e. about 0.012-0.016 ms per extra node. On the A100 it was
  11.05 ms at 16 and 11.91 at 48. In relative terms the H100 curve is a bit steeper (+9.9% from 16 to 48 vs +7.8%), but still flat enough
  that the optimum moves up from 48 to 64 nodes. Past 64 the tau gain (+0.46 at 96) no longer pays for the extra draft and verify time.
- tau is slightly lower than on the A100 for the same config (chain b16 7.84 vs 7.99, tree48 11.06 vs 11.29): different bf16 kernels change
  some greedy near-ties and so the trajectories.
- Lossless: `bench/p2_tf_check.py` (HF Qwen3-8B teacher-forced argmax, 8 requests per dataset; `bench/results/p5h100/tf_check_8b.json`).
  Mismatch rates 0.2-1.3% for chain16, tree48, tree64 and tree96, all with logit margin <= 0.5 (chain16 has the same profile). **Pass.**

## 2. Qwen3.8-27B (hybrid GDN/attention) on the same H100

Protocol as Phase 1 27B (`docs/phase1-qwen3.8-27b.md`): thinking on, `reasoning_effort=xhigh`, max_new 2048, the first 8 shuffled prompts
(seed 42) of each of the 5 datasets, c=1, T=0, ctx 8192, `--mamba-ssm-dtype float32`. Client `bench/p5h100/client27b.py` (native `/generate`
with the chat template applied client-side, so the output ids are saved; the published `dspine16` rows reproduce token-for-token on gsm8k
lengths). Runner `bench/p5h100/run27b.sh`, summary `bench/results/p5h100/q27b/summary27b.json` (`summarize27b.py`), raw JSON
`bench/results/p5h100/q27b/`, timers `bench/results/p5h100/timing/timing_t27_*`.

### How the tree runs on the hybrid model (`bench/p5h100/apply_tree_patch_27b.py`, `p4tree27b/`)

The Phase 4 fast tree (factored in-graph table + best-first selection in the DSpine draft graph, FlashInfer custom mask, tree accept + KV
compaction) is reused unchanged for the 16 attention layers. For the 48 GatedDeltaNet layers it uses SGLang's own EAGLE-tree verify path,
which exists for Qwen3-Next-style models but is only enabled for `--speculative-eagle-topk > 1`:
- the GDN backend is switched to its tree metadata (`topk` forced to 2 when `RDF2_TREE_SPEC` is set), and every verify carries the tree as
  first-child / next-sibling links written into the captured bs=1 buffers (`glue_fast.py: tree_links`); the fused causal-conv kernel turns
  them into parent indices, and the gated-delta recurrence starts each node from its parent's fp32 snapshot;
- this needs the snapshotting verify (`intermediate_ssm`, no `--enable-linear-replayssm-spec`): SGLang's ReplaySSM fold path asserts a chain;
- after acceptance the Mamba commit takes the snapshot (SSM state and conv window) of the **last accepted node** `path[commit_len-1]`
  instead of verify step `commit_len-1`;
- conv windows must be **dense**. SGLang deduplicates the per-step conv windows into one sliding buffer (window of step s = tokens
  s..s+2), which is only valid for a chain; SGLang itself turns it off for EAGLE trees. With it on, the tree run was coherent but wrong
  (teacher-forced mismatch 9-12%, margins up to 29 logits, `tf_check_27b_v1.json`); `test_gdn_tree.py` (kernels vs per-path chain,
  bit-exact) and a forced-chain run (tree pipeline with a chain-shaped tree: lossless) localized it;
- DSpine's 27B plugin pins the served block to its trained 16; the pin is relaxed so the b16 drafter drafts N rows (target attention:
  flashinfer, draft attention: fa3; target AR/rssm baselines keep fa3).

### Results (c=1, T=0, mean of 5 datasets x 8 prompts)

x AR uses the AR cells measured on this box (`t27_ar`, 49.3 tok/s; the Phase 1 27B box measured 51.5, so against that AR every row
below is 4.3% lower: rssm 4.49x, trees 4.74x).

| config | tau | tok/s | x AR | gsm8k / math500 / humaneval / mbpp / mt-bench tok/s | draft ms | verify ms | round ms |
|---|---|---|---|---|---|---|---|
| AR (fa3) | 1 | 49.3 | 1.00x | 48.9 / 49.4 / 49.5 / 49.4 / 49.4 | | | |
| DSpine b16 chain, published config (fa3 + ReplaySSM, `t27_rssm16`) | 5.68 | 231.0 | 4.68x | 258.5 / 280.3 / 204.8 / 230.9 / 180.6 | 2.00 | 20.82 | 23.33 |
| tree 24 / d23 | 7.11 | 244.1 | 4.95x | 273.6 / 280.0 / 238.2 / 228.1 / 200.8 | 3.01 | 23.86 | 27.63 |
| tree 32 / d31 | 7.43 | 244.1 | 4.95x | 275.7 / 283.5 / 235.2 / 224.1 / 202.3 | 3.03 | 25.09 | 28.87 |
| tree 48 / d31 | 8.03 | 244.1 | 4.95x | 272.1 / 285.0 / 247.9 / 222.8 / 192.9 | 3.07 | 27.49 | 31.31 |

- **The tree is +5.7% over the best chain (4.95x vs 4.68x), short of the 5.2x bar.** It raises tau a lot (+25% at 24 nodes, +41% at 48),
  but each round costs 4.3-8 ms more, and the gain is the same 244 tok/s at 24, 32 and 48 nodes.
- Where the round time goes (median ms, GPU timeline):
  - verify grows by **~0.15 ms per tree node** (23.86 / 25.09 / 27.49 at 24 / 32 / 48), ten times the 8B H100 slope (0.012-0.016). The
    cause is the snapshotting GDN verify: every node writes its fp32 recurrent state, 48 layers x 48 heads x 128 x 128 x 4 B = 151 MB per
    node, and the recurrence runs node by node. ReplaySSM, which the chain baseline uses, avoids both, but SGLang's ReplaySSM is chain-only.
  - a fixed ~1.9 ms on top (verify extrapolated to 16 nodes is ~22.7 ms vs 20.8 for the ReplaySSM chain): FlashInfer custom-mask attention
    instead of fa3 and the Triton tree recurrence instead of the ReplaySSM kernel.
  - draft +1.0 ms (3.0 vs 2.0): DSpine's optimized 27B draft path only runs at exactly block 16 (`sampler.optimized`), the tree drafts
    24-48 rows on the generic path.
- Lossless: teacher-forced check against the AR server (`bench/p5h100/tf27b.py`: the AR SGLang server scores prompt + output in one
  prefill and every generated token is compared with its top-1; first 1500 output tokens of all 8 requests per dataset),
  `bench/results/p5h100/q27b/tf_check_27b.json`. Mismatch rate 0.25-0.93% for tree24/32/48 and 0.29-0.70% for the ReplaySSM chain, every
  mismatch with log-prob margin <= 0.25 (chain: <= 0.5), none > 0.5. **Pass.** The bf16-shape near-ties make T=0 outputs of different
  configs diverge after a few dozen tokens (so cells are not token-identical across configs, and per-cell tok/s carries +-5-8% noise).

### What would get it to 5.2x (tried, not landed)

A tree version of ReplaySSM (`bench/p5h100/apply_trs_patch.py`, `p4tree27b/rdf2_trs.py`, opt-in `TRS=1`, **not used for any number
above**): keep the state in registers when a node's parent is the previous node, snapshot only nodes that have a non-adjacent child, write
each node's raw (k, v, g, beta) into SGLang's existing ReplaySSM ring from the same kernel, and rebuild the last accepted node's state by
replaying the accepted path. A first (incorrectly ordered) version measured verify 23.6 ms vs 25.0 ms at 32 nodes (-1.4 ms, which would be
about +5% tok/s, i.e. ~5.2x). The corrected version (replay into the last node's snapshot before SGLang's commit) hits an out-of-bounds
gather on the first request; it was stopped there for time. Next steps if picked up: verify the `intermediate_ssm` all-layer view's layout
in the replay kernel, then fold in the larger lever, a chunked (not node-serial) tree recurrence, and DSpine's optimized draft at N > 16.

## Cost and reproduction

- One Vast H100 SXM 80GB at 2.38 USD/h, 3.1 h by the instance duration counter: **about 7.3 USD** (duration x listed price, not the invoice). Box destroyed and verified gone.
- 8B: `bash bench/setup_box.sh` (with `TORCH_CUDA_ARCH_LIST=9.0`), `bash bench/p5h100/queue8b.sh`, `bash bench/p5h100/tf8b.sh`,
  `python bench/p5h100/summarize8b.py ar chain16=... tree64_d31=...`.
- 27B: models under `/root/w/models`, `bash bench/p5h100/queue27b.sh "rssm t27_rssm16|tree t27_tree32 32:31|ar t27_ar"`,
  `bash bench/p5h100/tf27b.sh OUT.json t27_tree32 ...`, `python bench/p5h100/test_gdn_tree.py 32` (kernel unit test),
  `bash bench/p5h100/dbg27.sh TAG N:D gsm8k 2` (2-prompt serve + teacher-forced loop used for the debugging above).
- Not run: the non-ReplaySSM DSpine b16 chain (the DSpine 27B plugin fails at block 16 without ReplaySSM, `serve_t27_chain16.log`),
  c=8, T=1.
