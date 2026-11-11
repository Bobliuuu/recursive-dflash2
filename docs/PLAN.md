# Plan (rev 2026-11-08)

## Goal
Find the best speculative-decoding method (hybrid of DFlash / DFlash2 / DSpine / whatever works), then test
whether RECURSIVE / stacked drafting improves it. Run the final architecture on the best single-drafter model.
Publish a fair benchmark.

## Phases
0. Research (docs/research/*): DSpine+DSpark+ReplaySSM, prior art on recursive/stacked/hierarchical drafting, benchmark methodology.
1. Fair benchmark: AR, EAGLE-3, DFlash, DFlash2, DSpine on one engine/hardware/dataset set; acceptance length AND wall-clock tok/s;
   T=0 and T=1; concurrency 1 and 8; block sizes matched or swept. Raw JSON in repo. Pick best single drafter.
2. Improve single drafters: hybrids (DSpine depth injection + DFlash2 top-k selector), long blocks, tree verification, new ideas.
3. Recursive study (sketches below). Kill tests first, cheap and offline. Then build best stacked architecture on the phase-2 winner.
4. Write benchmark / paper-style report.

## Recursive sketches: status after prior-art survey (docs/research/recursive-prior-art.md)
Step 0 for all: from one logged run measure T_draft, T_verify(B=16), per-position acceptance, fraction of full-block accepts.
- A. Branching draft-ahead: ALREADY DONE (SSD 2603.03251 ~30% w/ drafter on own GPU; D3D, DPara; PEARL, AMUSD). Reproduction only.
  Cap 1+T_d/T_v (~1.17-1.3 in published timings). Kill if T_d/T_v < 0.2.
- B. Refine cascade: PARTLY done (D-Loop 2610.06011 tau 7.19 vs 5.77 on Qwen3-8B; XPress 2608.02438; DBloom: must train at longer length).
  OPEN GAP: pass 2 emitting a longer block (16 -> 64) from pass-1 tokens. Kill if tau_32/tau_16 < ~1.05 or full-block accepts < 10%.
- C. Stacked drafters: low value (HSD 1.03-1.17x; parallel drafter cost already flat in block length). Do not build.
- D. On-policy training: prior art (DistillSpec, HASS, Carryover Drafting, D-Loop). Support for B only, max 2 rounds.
- New, ranked first: tree verification on the best parallel drafter (DDTree 5.56x->7.52x over DFlash; nobody tried on DSpine). Baseline the
  recursive ideas must beat. Also: tree from pass-2 marginals, confidence-gated pass 2, n-gram tail extension.

## Target model (corrected)
No DFlash2 checkpoint for Qwen3-4B/8B (only 27B+). Qwen3-8B official: DFlash, DSpine, EAGLE-3 (RedHatAI). Decision pending on DFlash2
(drop / 27B / retrain). DSpine paper vs README numbers disagree (4.82 vs 7.83 tau): settle in our harness first.

## Budget
No GPU launched. Before any paid box: state hourly price + total estimate and get approval.

---
# Earlier notes (Qwen3-4B)
## Target
Qwen3-4B (bf16 ~8 GB). Cheapest model in the DFlash paper (Table 1/3). Baseline to beat, from the paper
(Transformers, temp 0, block 16): avg tau 6.54, 4.91x; SGLang B200 Math500 tau 8.01, 4.8x at concurrency 1.
DFlash2 (Inco AI, `incoai/Qwen3.8-27B-DFlash2`) adds top-k candidates per position, a path selector and
two-tap dynamic convolutions; it has no public Qwen3-4B checkpoint and no paper, so we re-implement.

## Physical limit (roofline, batch 1)
- Decode is memory-bound: ~8 GB weights / 4.8 TB/s (H200) = ~1.7 ms per target pass, ~600 tok/s autoregressive.
- Verifying B tokens costs about one pass until compute ridge (~989 TFLOPs / 4.8 TB/s ~ 200 FLOP/byte, i.e. B of order 100-250).
- So speedup ceiling ~ tau_eff = expected accepted tokens per round, bounded by B (<= ~128-256), and by
  round time = max(T_verify, T_draft) once drafting is overlapped. Today's tau ~6.5-8 is far below that ceiling.
- Deliverable 0: measure T_verify(B) for B in 1..512 on the target hardware to pin the real ceiling.

## What "recursive" means here (to validate empirically)
1. **Recursive draft-ahead**: while the target verifies block k, the drafter already drafts block k+1 assuming full
   acceptance (speculation on the speculation). Round time becomes max(T_verify, T_draft). On a rejection,
   discard and redraft.
2. **Recursive training**: the drafter is trained on its own rollouts. Round r: serve the current drafter with the target,
   log accepted/rejected trajectories and target hidden states, train round r+1 on states the drafter actually reaches
   (including positions after its own draft prefix, not just target-verified prefixes). Loss weighting on acceptance
   (DFlash positional decay gamma, extended to longer blocks).
3. **Cascade**: drafter level L is itself verified/accelerated by a smaller level L-1, so block length scales
   (16 -> 64 -> 256) while target cost stays ~1 pass.

## Earlier phases (superseded)
1. Baseline: reproduce DFlash Qwen3-4B (z-lab/dflash), measure tau and speed on 1 GPU. Roofline sweep of T_verify(B).
2. Data: ~100K target-generated samples (Nemotron Post-Training v2 + CodeAlpaca) for ablations; 800K for full run.
3. Train DFlash2-style drafter (5 layers, top-k candidates, selector, dynamic conv), block 16 then 32/64.
4. Add draft-ahead serving loop; measure vs ceiling.
5. Recursive training rounds (on-policy), 2-3 rounds; report tau, tokens/s, fraction of ceiling.

## Budget
No GPU launched. Before any paid box: state hourly price + total estimate and get approval.

## Results log
### 2026-11-04 A100-40GB, HF eager, Qwen3-4B bf16, ctx 512 (`bench/verify_sweep.py`)
Verify time is flat ~40 ms for B=1..384 (36.9 ms at B=1, 43.8 ms at B=512). Weight read alone is ~5 ms
(8 GB / 1.5 TB/s), so eager HF is launch/CPU-bound, not memory-bound. Any speedup measured under HF eager
understates the roofline ceiling and favors big blocks. Next: re-run with CUDA graphs / static cache to
get the real T_verify(B) curve, then baseline z-lab/Qwen3-4B-DFlash-b16 (5 layers, block 16, target layers 1,9,17,25,33).
Setup notes: `huggingface_hub` 2.4 + httpx2 crashes on decompress here; download files with curl.

### 2026-11-05 A100-40GB, CUDA graphs + StaticCache, Qwen3-4B bf16, ctx 512 (`bench/verify_sweep_graph.py`)
One target forward over B tokens (ms): B=1 14.06, 4 14.77, 8 14.92, 16 15.64, 32 15.92, 64 16.71, 96 18.05, 128 18.88,
192 21.50, 256 24.08, 384 32.96, 512 39.93. Eager was ~40 ms flat, so ~60% of eager time was launch overhead.
- Verify is nearly free to B~128 (1.34x of B=1); compute knee is B~200-300 on A100. At block 16, verify = 1.11x of one-token pass.
- B=1 is 14 ms vs ~5 ms weight-read floor: HF kernels are unfused, so there is a 2-3x gap to roofline on the target pass itself.
- Implication: with tau~6.5 at B=16, ceiling is ~5.8x before draft cost. Larger blocks (64-128) cost only 1.2-1.3x per round,
  so the lever is acceptance length at long blocks, not verify cost. Draft pass cost must be measured next.
- Gotcha: transformers 5.x StaticCache `cumulative_length` is a GPU tensor advanced inside the graph; reset it inside the capture or replays overflow the cache.
