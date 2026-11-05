# Plan

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

## Phases
1. Baseline: reproduce DFlash Qwen3-4B (z-lab/dflash), measure tau and speed on 1 GPU. Roofline sweep of T_verify(B).
2. Data: ~100K target-generated samples (Nemotron Post-Training v2 + CodeAlpaca) for ablations; 800K for full run.
3. Train DFlash2-style drafter (5 layers, top-k candidates, selector, dynamic conv), block 16 then 32/64.
4. Add draft-ahead serving loop; measure vs ceiling.
5. Recursive training rounds (on-policy), 2-3 rounds; report tau, tokens/s, fraction of ceiling.

## Budget
No GPU launched. Before any paid box: state hourly price + total estimate and get approval.
