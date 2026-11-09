# Phase 4b: lossless tree verification at temperature 1 (Qwen3-8B, 1x A100-80GB, SGLang 0.5.22, batch 1)

Question: the DSpine tree (48 nodes, depth 31) is +19.7% over the chain at T=0, but it had only a greedy accept. Can a lossless sampled
tree verifier get T=1, c=1 above 5.5x AR (chain: 4.83x)?

**Answer: not quite. The best lossless T=1 tree is 5.43x AR, +12.0% tok/s over the DSpine chain re-measured on the same box (4.83x), with tau
8.3-8.8 vs 6.89.** Two verifiers tie: deterministic tree with target draws at every node (det), and SpecInfer/RSD-style recursive rejection
sampling without replacement over children drawn from DSpine's conditional q (rrs). Both run on the tree-overhead fast path from
`docs/phase4-tree-overhead.md`, with low-overhead walk kernels written here. Output follows AR sampling by construction (proof below) and passes an
exact-law unit test. In the engine, the token distribution cannot be told apart from Tencent's stock lossless chain verifier. At 5000
samples both sit slightly off plain AR on open-ended prompts, by the same margin seen between AR at batch 1 and AR at batch 32 (bf16
verify numerics). Task accuracy is unchanged. The 5.5x target missed by 1.3%.

Setup: Qwen3-8B bf16, `tencent/Qwen3-8B-DSpine-b16` weights run at block 48 (local config copy), 1x A100-SXM4-80GB (Vast), SGLang 0.5.22,
flashinfer 0.7.1, thinking off, T=1, top_p 1, top_k off, max_new 2048, concurrency 1. Same 5 datasets x 40 prompts as Phases 1-3
(`bench/data/prompts`), **3 seeds per config** (T=1 single-seed noise is about +-4% per dataset). x AR = mean over datasets of tok/s divided
by the Phase 1 AR T=1 c=1 cells (90.1 / 89.0 / 88.9 / 88.9 / 89.0 tok/s). The chain re-measured here reproduces Phase 1 exactly for seed 0
(DSpine's sampler is seeded per request) and gives 4.83x over 3 seeds (Phase 1: 4.85x). Raw JSON: `bench/results/p4t1_<config>_<ds>_T1_c1_s<seed>.json`,
summary `bench/results/p4t1_summary.json` (`bench/p4t1_summarize.py`). AR re-measured on this box is 2-3% faster than the Phase 1 AR cells (91.5 vs 89-90 tok/s), so against same-box AR the headline is
5.29x and the chain 4.71x. The ratio to the chain (1.12) does not change.

## Results (T=1, c=1, mean of 3 seeds)
| config | tau (mean ds) | tok/s (mean ds) | x AR | vs chain | gsm8k / math500 / humaneval / mbpp / mtbench tok/s | x AR per seed |
|---|---|---|---|---|---|---|
| DSpine chain b16 (stock sampled verify) | 6.89 | 431 | 4.83x | 1.000 | 590 / 475 / 440 / 400 / 250 | 4.84 4.83 4.82 |
| det, Phase 2 glue | 8.45 | 474 | 5.30x | 1.091 | 692 / 507 / 476 / 425 / 268 | 5.33 5.30 5.28 |
| rrs, Phase 2 glue | 8.56 | 479 | 5.37x | 1.106 | 676 / 531 / 483 / 432 / 273 | 5.21 5.45 5.44 |
| det, fast path (exact table) | 8.29 | 475 | 5.32x | 1.102 | 662 / 514 / 487 / 430 / 281 | 5.33 5.18 5.45 |
| det, fast path, factored table | 8.41 | 485 | 5.43x | 1.115 | 718 / 520 / 491 / 418 / 276 | 5.43 5.43 5.42 |
| rrs, fast path, factored table, v1 kernels | 8.71 | 480 | 5.37x | 1.106 | 674 / 544 / 484 / 426 / 269 | 5.38 5.38 5.35 |
| **det, fast path, factored, v2 kernels** | 8.35 | **485** | **5.43x** | **1.123** | 689 / 532 / 486 / 434 / 284 | 5.41 5.43 5.46 |
| **rrs, fast path, factored, v2 kernels** | 8.75 | **484** | **5.43x** | **1.120** | 691 / 529 / 482 / 442 / 279 | 5.48 5.36 5.44 |

("fast path" = `bench/p4tree` in-graph table and selection, single pack kernel, FlashInfer custom-mask prefix-tile skip (`FIMASK=skip`),
plus `bench/p4tree_t1`; "factored" = `RDF2_TABLE=factored`. v2 = the lower-overhead walk kernels below.)

Tried and worse (fast path or Phase 2 glue, 1-2 seeds, raw cells in `bench/results/`):
- rrs with q sharpened or flattened (`RDF2_TREE_BUILD_SCALE`, which scales the table for tree shape, draws and verification):
  x0.7 gives 5.16x (1 seed). x2.0 lost 7-10% on gsm8k / math500 and was stopped.
- Hybrid with deterministic children where q's top probability is >= 0.6 (`RDF2_TREE_DET_THRESH=0.6`) lost 5-10% on the first three datasets and was stopped.
  The inverse (deterministic where q's top probability is < 0.3, `-0.3`) gives 5.40x (1 seed): a tie.

Reading:
- The tree raises tau at T=1 by 21-27% (8.3-8.8 vs 6.89), against +41% at T=0 (11.29 vs 7.99). A 48-node tree spends most of its
  nodes on one or two high-probability spines. At T=1 the target leaves those spines more often (det accepts a single child with
  probability p(top), rrs with sum min(p, q)), and the extra width cannot fully pay for it.
- det and rrs reach the same speed by different routes. rrs has higher tau (8.75 vs 8.35): a single-child spine accepts with sum min(p, q) instead of p(top).
  Per dataset det is ahead on gsm8k and rrs on mbpp / mtbench, both within about 1.5 seed s.e.
  rrs's extra tau was eaten by its kernels until v2 (v1 rrs cost 0.45 ms per round vs 0.19 ms for det).
- Round time is the limit. At T=1 the effective tree round (tau / tok/s) is 17.2 ms (det) to 18.1 ms (rrs), against 16.0 ms for the
  chain's sampled round (14.0 ms greedy), so +21-27% tau becomes +12% tok/s. Reaching 5.5x needs about 0.25 ms less per round, or tau +1.3% at the same round time.

## Overheads
Per-round cost of the sampled verification on the A100 (micro-benchmark, idle GPU, N=48, V=151,936):
| step | v1 | v2 |
|---|---|---|
| det: target draws per node (softmax over 48 x V + flashinfer sampling) + walk | 0.19 ms | 0.08 ms (walk with every node deterministic) |
| rrs: relabel + softmax + walk | 0.45 ms | 0.16 ms |
v2 (`tree_relabel2`, `tree_accept_rrs2` in `bench/p2tree/tree_kernels.py`, default `RDF2_TREE_RRS_V=2`) reads the logits once. Row
statistics come from one warp per vocabulary segment (online max/sum, coalesced), plus a per-row finalize. The Gumbel orders for every
(node, parent lane) are computed in parallel. The walk keeps the tree in shared memory and draws the bonus with a segment pick and a warp
scan. It computes the same algorithm. Drafting matches T=0 because DSpine's own sampled chain is skipped on tree rounds: humaneval round
period median 16.85 ms at T=0 vs 17.03 ms at T=1 with the Phase 2 glue.


## Method

The Phase 2 tree (best-first over DSpine's predecessor-conditioned table, 48 nodes, depth cap 31, CUDA build/accept/KV-compaction kernels in
`bench/p2tree/`) only had a greedy accept. Phase 4b adds two lossless sampled verifiers, selected by `RDF2_TREE_T1_MODE`. They run for every bs=1
round whose request uses plain temperature sampling (no top-k / top-p / min-p, no penalties, logit bias or grammar; other rounds keep the stock path).

**det (default): deterministic tree, target draws at every node.** The tree is built exactly as at T=0, from DSpine's exact conditional
q = softmax(table / T). After the verify forward every node i draws one token s_i ~ p_i = softmax(logits_i / T), independently
(`flashinfer.sampling.sampling_from_probs`), and the unchanged `tree_accept` kernel walks the tree with s in place of the argmax: at the current
node it moves to the child whose token equals s, and it emits s as the bonus at the first node where no child matches.

*Lossless.* Each visited node's draw is independent of the event of visiting it, because that event depends only on the draws at its ancestors.
The emitted token at depth k is therefore distributed as p(. | prefix, emitted tokens 1..k-1), which is AR sampling, given that the tree-attention
logits of a node equal the target's logits for that node's path (as at T=0, up to bf16 verify-shape numerics). This is SpecInfer's multi-round
recursive rejection sampling specialised to point-mass drafts: "accept child c with probability p_res(c), otherwise zero c and renormalise"
chains into "draw from p, follow the matching child".

*Optimal for a fixed tree.* Node v is reached with probability p(path(v)). No lossless verifier can reach a fixed token sequence with
higher probability than the target assigns it, so for a given deterministic tree det reaches the maximum expected accepted length,
sum over v of p(path(v)).

**rrs: sampled children, recursive rejection sampling without replacement** (SpecInfer / RSD-style, `tree_relabel` + `tree_accept_rrs` in
`bench/p2tree/tree_kernels.py`). The tree *shape* is the deterministic best-first shape, which depends only on the table. Each node's k_v children
then get fresh lanes drawn from that node's q without replacement (Gumbel-top-k on the table row). Verification walks the tree: at node v the
children are tried in draw order with the residual target r. Child c_i is accepted with probability min(1, r(c_i)/Z / q_i(c_i)), where q_i is
q with the already-tried lanes removed. On rejection r <- max(r/Z - q_i, 0). If every child is rejected, the bonus comes from the residual,
and at a leaf it comes from p. Because q is supported on DSpine's 16 candidates, the residual is 16 numbers plus the untouched mass outside the
candidate set. That outer part is sampled exactly by an inverse CDF over the vocabulary. *Lossless* by the standard induction (the RS lemma
at each child, with the remaining drafts distributed as q_i given the rejected ones). A node's child count is fixed by the shape and never depends
on its own children's draws. This verifier lets a single-child spine accept with probability sum min(p, q) instead of p(top token). The price
is that deterministic top-k children, which cover more p-mass when q is sharp, become random.

DSpine's own sampled chain (serial_sample_chain + dense q + rejection kernel, about 2 ms per round) is skipped on tree rounds: the glue stages
the DSpine sampler as greedy for eligible bs=1 batches (`_patch_staging`). The tree needs only the candidate table, and the draft-side work
matches T=0 (measured round period T=0 16.85 ms vs T=1 17.03 ms median, humaneval, Phase 2 glue).

**fast path.** `bench/p4tree_t1/` runs both verifiers on top of the tree-overhead work in `bench/p4tree/` (table and best-first
selection inside the draft CUDA graph, one pack kernel, FlashInfer custom-mask prefix-tile skip). It reuses the draft graph's table as q
(static graph output), derives node lanes from the in-graph tree and relabels the children before the pack kernel. The walk kernel
derives lanes from the node tokens itself. A first fast-path det run had read a stale lane buffer, giving tau 1.1 and wrong acceptance
tests on one gsm8k cell. It was discarded, and that failure mode is now impossible.

## Losslessness evidence

Three layers of evidence. The algorithm is exact by construction. The test sensitive enough to see the engine's bf16 numerics shows the
tree has the same distribution as the stock DSpine chain verifier.

**1. Exact-law unit test** (`bench/p2tree/test_t1_sampling.py`, output `bench/results/p4t1_unit_test.txt`). A random DSpine-like table
builds the real 48-node tree with the CUDA kernels. The target is a random Markov model over a 40-token vocabulary, so the AR law of
the emitted tokens is known exactly. 30,000 rounds per verifier (det, rrs, hybrid; v1 and v2 kernels), in a shallow and a deep
("aligned") scenario. Chi-square goodness of fit of o1, o2 | o1 and o3 | (o1, o2) against the exact law: all 60+ fits pass, smallest
p = 0.016 (expected among that many tests). The det walk's mean accepted length matches the exact optimum sum p(path(v)) (1.1171 vs 1.1169).
rrs accepts 1.32 vs 0.83 drafts per round for det when q is close to p, and 0.47 vs 0.20 when it is not.
Limit: V = 40 exercises the bonus draw's segment logic only with one-token segments. The full-vocabulary path is covered by test 2.

**2. Token-distribution tests in the engine** (`bench/p4_dist_sample.py`, `bench/p4_dist_compare.py`, `bench/results/p4t1_dist*.json`).
Four fixed prompts (MT-Bench #0, GSM8K #0, HumanEval #0, a one-line story prompt) get many independent 8-token continuations at T=1.
Position 1 comes from the prefill sampler; positions 2-8 come from the verifier. Each position's marginal gets a two-sample chi-square
homogeneity test (categories with pooled count >= 10, rest merged) and a plug-in KL. "Fisher" combines the 14 or 28 position p-values
(correlated, so only a rough summary).
| comparison (samples per prompt) | positions with p < 0.05 | min p | Fisher p |
|---|---|---|---|
| **negative control**: tree with node temperature 0.91 vs AR c=32 (2500) | 57% | 1e-9 | **1e-45** |
| rrs tree vs AR c=32, first 2500 | 7% | 0.007 | 0.052 |
| det tree (Phase 2 glue) vs AR c=32 (2500) | 11% | 0.001 | 0.11 |
| AR c=32 vs AR c=32 replicate (2500) | 7% | 0.019 | 0.047 |
| rrs tree vs AR c=32, pooled (5000) | 21% | 0.001 | 1e-5 |
| **stock DSpine chain (c=1) vs AR c=32** (5000, 2 prompts) | 29% | 0.0003 | **9e-6** |
| AR c=1 vs AR c=32 (5000, 2 prompts) | 21% | 0.017 | 0.023 |
| stock DSpine chain vs AR c=1 (5000, 2 prompts) | 50% | 0.004 | 5e-8 |
| rrs tree vs AR c=1 (5000, 2 prompts) | 29% | 0.004 | 3e-4 |
| **rrs tree vs stock DSpine chain** (5000, 2 prompts) | 14% | 0.008 | **0.052** |
Reading: the test is powerful. A 9% temperature error at the tree nodes is rejected at 1e-45 with 2500 samples. At 5000 samples it also
separates every speculative verifier from plain AR on the two open-ended prompts, MT-Bench #0 and the story (TV 0.02-0.05 per position,
excess KL around 0.005 nats). The stock lossless chain shows this deviation too, and more strongly than the tree. Even AR decoded at
batch 1 vs batch 32 differs at p = 0.02. The tree is indistinguishable from the stock chain (Fisher 0.05, no position below 0.007).
So the shift comes with speculative verification in this engine (bf16 logits of a 16-48-token verify pass vs a 1-token decode, the same
numerics that make T=0 speculative output non-byte-identical, Phase 1). The tree sampler adds none of its own. GSM8K #0 and HumanEval #0
(lower entropy) show no deviation for any method.

**3. Task accuracy** (`bench/p4_accuracy.py`, `bench/results/p4t1_accuracy.json`). Same grader for all methods (last `\boxed{}` vs gold,
light normalisation), T=1, the 100 frozen prompts x 3 seeds:
| method | GSM8K | MATH-500 |
|---|---|---|
| AR (c=16) | 86.3% (259/300) | 73.3% (220/300) |
| DSpine chain (c=16) | 84.0% (252/300) | 78.0% (234/300) |
| rrs tree, fast path (c=1, headline config) | 89.3% (268/300) | 74.0% (222/300) |
The binomial s.e. is about 2.0 points (GSM8K) and 2.5 points (MATH), so every difference is within 1.4 s.e. The 40-prompt speed cells
agree (3 seeds each): GSM8K chain 93.3%, det tree 93.3%, rrs tree 88.3%; MATH chain 70.8%, det 71.7%, rrs 71.7%.

## Cost
One Vast.ai 1x A100-SXM4-80GB box (label `rdf2-t1-tree`) at 1.36 USD/h for about 5.8 h: about 7.9 USD (wall clock x listed price, not
the invoice). The box was destroyed at the end.

## Reproduction (box: `bash bench/setup_box.sh`, then `source /workspace/env.sh`)
- Unit test of both walks vs an exact Markov target: `python bench/p2tree/test_t1_sampling.py 30000 det,rrs [aligned]`.
- Servers: `bench/p4_serve.sh tree 48 31` (Phase 2 glue; det, `RDF2_TREE_T1_MODE=rrs` for rrs), headline `RDF2_TABLE=factored RDF2_TREE_T1_MODE=rrs FIMASK=skip bench/p4_serve.sh fast 48 31` (or `=det`), `bench/p4_serve.sh chain`, `bench/p4_serve.sh ar`. Knobs: `RDF2_TREE_RRS_V` (1|2), `RDF2_TREE_DET_THRESH`, `RDF2_TREE_BUILD_SCALE`, `RDF2_TREE_SAMPLED=0` (greedy rounds only).
- Speed cells: `bench/p4_cells.sh TAG 1 1 40 SEED BLOCK` -> `bench/results/p4t1_TAG_<ds>_T1_c1_s<seed>.json`; summary `python bench/p4t1_summarize.py`.
- Distribution test: `python bench/p4_dist_sample.py --n 2500 --conc C [--prompts mtbench0,story] --out ...` against each server; `python bench/p4_dist_merge.py ALL.json A.json B.json`; `python bench/p4_dist_compare.py REF.json TEST.json --out cmp.json`. Negative control: serve with `RDF2_TREE_NEG_CONTROL=1.1`.
- Accuracy: `bench/p4_cells.sh TAG 1 C 100 SEED BLOCK "gsm8k math500"`, then `python bench/p4_accuracy.py files...`.
