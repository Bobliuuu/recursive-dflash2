# Prior art on recursive / stacked / hierarchical / draft-ahead drafting (2026-11-08)

Question: does recursive or stacked drafting help on top of parallel block drafters (DFlash, DSpine, DSpark)?

Verification level, stated honestly:
- READ IN FULL (HTML text): HSD 2510.19705, CSD 2312.11462, DDTree 2604.12989, DFlash 2602.06036,
  DSpine 2609.36173, D-Loop 2610.06011, XPress 2608.02438, SSD/Saguaro 2603.03251 (first 100k of 132k chars).
- ABSTRACT ONLY (numbers are author claims from the abstract; mechanism details not checked): everything else.
- Numbers below are as reported by the authors, on their hardware and setups. Tau values are NOT comparable
  across papers (DFlash paper Qwen3-8B T=0 avg tau 6.49; DSpine paper's DFlash reproduction on SGLang: 3.77;
  D-Loop's: 5.77; XPress's: 6.25). Compare only within one paper.
- Several papers are 2026 arXiv IDs (26xx.xxxxx); none are independently replicated.

## 1. Papers

### 1a. Stacked / hierarchical / cascaded (autoregressive drafters)

**Hierarchical Speculative Decoding (HSD), arXiv 2510.19705 (Mohri, Kaplan, Schuster, Mansour, Globerson).**
- Mechanism: chain of K drafts M_K..M_1 -> target; each model verifies the one below in one pass. Per-pair draft
  lengths T_i.
- Theory (Thm 2.2): expected latency per token = sum_i c_i * prod_{j>=i}^{K} R(alpha_{j-1,j}, j), R piecewise
  (top level (1-a)/(1-a^{T+1}); middle levels gamma(a,T_{n-1},T_n), estimated empirically, no closed form).
  Assumes iid acceptance, verify cost of a batch = cost of one token. Optimal hierarchy (which models, which T)
  found in poly time by reduction to Generalized Shortest Path (NOT DP): O(T^4 K^4 log(TK)).
- Gain (H100, batch 1, CNN-DM/XSUM): LayerSkip 7B 1.76x vs best single draft 1.62x (1.09x); 13B 1.41 vs 1.20
  (1.17x); 70B 1.77 vs 1.58 (1.12x); Gemma2-9B 1.06 vs 1.03 (CNN-DM) and 1.15 vs 1.08 (XSUM). Abstract: "up to 1.2x".
- Fails when: only helps in an intermediate band of cost and acceptance; Table 8 (A cost 1024, B cost 256,
  A-B acc 0.5): often best hierarchy is just A-B (1.20x flat), a third model adds nothing if it is expensive or
  has low acceptance into B. Needs ~1h acceptance estimation on 4 GPUs plus ~2h solve. Not compared to
  Medusa/Hydra. Batch 1, two model families.

**Cascade Speculative Drafting (CSD), arXiv 2312.11462.**
- Vertical cascade: draft model for the draft model, recursing down to a statistical Max-Gram bigram model.
  Lenient acceptance only between draft levels; target verification stays exact.
- Horizontal cascade: biggest draft model writes the earliest (most accepted) tokens, smaller ones the later tokens.
- Theory (iid acceptance, EWIF = expected walltime improvement factor): SD = (1-a^{k+1})/((1-a)(ck+1));
  horizontal T = (sum_{i=0}^k prod_{j<=i} a_j)/(1 + sum_i c_i); vertical two-draft closed form in Thm 4.3;
  Cor 4.4: adding a negligible-cost statistical model always helps.
- Gain: FLAN-T5-xxl GSM8K 3.38 (SD) -> 3.88x (base+small+mag); MMLU 3.97/4.12 -> 4.88x; "up to +44% GSM8K, +81%
  MMLU over SD". Vicuna-7B tok/s GSM8K: AR 33.34, SD 44.31, CSD 56.72, Medusa 61.87, CSD+tree 63.81.
- Fails when: hardware changes ratios; iid assumption; gain concentrated where the baseline SD is weak (FLAN-T5);
  on decoder-only the cascade LOSES to Medusa (61.87 vs 56.72) without a tree.

**Pyramid SD / "3-Model Speculative Decoding", arXiv 2510.12966 (abstract + secondary summary).** Adds a
"qualifier" model between draft and target, fuzzy (relaxed, lossy) acceptance with divergence thresholds. Up to
1.91x over SD, 124 tok/s on RTX 4090, 1B draft / 8B target. Training-free. Gain depends on fuzzy acceptance
(not lossless in the strict sense). Note: this is a different paper from any "Pyramid" tree SD; this one is the
match for "Pyramid SD".

**Staged speculative decoding, arXiv 2308.04623 (abstract).** Tree-structured speculative batch + second SD
stage on the draft. 3.16x single-batch latency on 762M GPT-2-L; small-batch local-device regime.

**TriForce, arXiv 2404.11912 (abstract).** Three levels: target w/ full KV <- same weights w/ retrieval-sparse
KV <- small model w/ StreamingLLM cache. Llama2-7B-128K: up to 2.31x on A100; offloading on 2x RTX 4090
0.108 s/token. Helps only when KV-cache loading dominates (long context, offload); at short context the
intermediate level has no cost to hide (inference from the mechanism; abstract states no failure case).

**MagicDec, arXiv 2408.11049 (abstract).** Sparse-KV self-draft; up to 2.51x for Llama3.1-8B at batch 32-256,
long sequences. Same regime caveat as TriForce.

**QuantSpec, arXiv 2502.10424 (abstract).** Self-speculative: 4-bit weights + hierarchical 4-bit KV as the draft
of the same model. >90% acceptance, up to ~2.5x, ~1.3x less memory than sparse-KV self-spec. Long-context regime.

**LongSpec, arXiv 2502.17421 (abstract).** Constant-size-KV draft + long-context position indices + tree
attention. Up to 3.26x on long-context understanding; 2.25x wall-clock cut on AIME24 with QwQ.

### 1b. Single-level drafters and trees

| Paper | Mechanism | Reported gain | When it fails |
|---|---|---|---|
| EAGLE 2401.15077 | AR draft head on 2nd-to-top-layer features + shifted token | 2.7-3.5x (LLaMA2-70B) | sequential draft cost grows with depth |
| EAGLE-2 2406.16858 | dynamic tree from draft confidence | 3.05-4.26x, 20-40% over EAGLE | needs calibrated draft |
| EAGLE-3 2503.01840 | direct token prediction, multi-layer fusion, training-time test | up to 6.5x, 1.4x over EAGLE-2; 1.38x SGLang tput @bs64 | still AR in depth; DFlash paper: avg tau 2.96-3.48 vs DFlash 6.49 |
| Medusa 2401.10774 | parallel independent heads + tree | M-1 >2.2x; M-2 2.3-3.6x | heads are sequentially independent |
| Hydra 2402.05109 | sequentially dependent heads | Hydra++ 1.31x over Medusa, 2.70x over AR | sequential head cost |
| PARD 2504.18583 | target-independent parallel draft w/ mask tokens, COD training (3x cheaper) | 3.67x LLaMA3.1-8B vLLM, 1.15x over EAGLE-3 | weaker than target-conditioned drafters |
| SpecInfer 2305.09781 | token tree from small models, tree attention verify | 1.5-2.8x dist., 2.6-3.5x offload | tree width vs verify cost |
| Sequoia 2402.12374 | DP-optimal tree shape, hardware-aware size | up to 4.04x (7B A100), 9.96x L40 offload | optimum is hardware-specific |
| SpecExec 2406.02532 | giant cache tree built from draft top mass for offload | up to 20 tok/target-iter; 4-6 tok/s 50B+ on consumer GPU | only when verify of thousands of tokens is ~free (offload) |
| Lookahead/Jacobi 2402.02057 | n-gram pool from Jacobi trajectories, no draft model | 1.8x MT-bench; up to 4x code, multi-GPU | extra FLOPs per step; low on non-repetitive text |
| Speculative Streaming 2402.11131 | target fine-tuned to predict future n-grams | 1.8-3.1x, ~10^4x fewer extra params than Medusa | needs target FT |
| DistillSpec 2310.08461 | on-policy (draft-generated) KD data | 10-45% over SD | - |
| HASS 2408.15766 | harmonized objective/context alignment (train on own-feature contexts) | 2.81-4.05x, +8-20% over EAGLE-2, no inference overhead | - |

### 1c. Parallel block drafters (the stack we want to improve)

- **DFlash 2602.06036**: 5-layer block-diffusion drafter, one pass for 16 tokens, KV-injected target features.
  Qwen3-8B T=0: avg tau 6.49, 4.86x (EAGLE-3(16) 1.76x, tau 2.96); Qwen3-4B 6.54, 4.91x. T=1: 4.03x / 4.24x.
  Draft layers: 3/5/8L tau 5.64/5.99/6.33 on Math500 but speed best at 5L. Train-16/test-8 ok (5.09 vs 6.33),
  train-8/test-16 bad (5.02). No block-32 experiments, no per-position acceptance, no ms latency split (!).
- **DSpine 2609.36173** (Tencent): adjacent causal injection at every layer + K^2 transition cache.
  Qwen3-8B T=0 mean tau 3.77 -> 4.82 (+27.8%) vs DFlash (their setup); 23.3% SGLang throughput gain.
  Verification = 72-76% of round time; injection adds <=0.15 ms/round. No long-block or tree results.
- **DSpark 2607.05147**: parallel backbone + small sequential module (semi-AR), confidence-scheduled
  verification length. Gains over DFlash grow with draft length: gamma=7 +16/15/18% tau (math/code/chat),
  gamma=15 +30/26/22%; block 4->16 adds only 0.2-1.3% round latency (secondary summary, check paper).
- **DFlare 2606.02091**: layer-wise target fusion, 2.4M training samples; 5.52x Q3-4B (+11%), 5.46x Q3-8B (+8%).
- **DDTree 2604.12989**: draft TREE from the DFlash per-position marginals, best-first heap under node budget B
  (16..1024), ancestor-mask verification. Qwen3-8B vs DFlash (T=0, speedup/tau): MATH-500 5.56->7.52x
  (7.79->10.73); GSM8K 4.78->6.75x; HumanEval 4.84->6.90x; MT-Bench 2.56->4.10x; SWE-bench Lite 2.65->4.23x.
  Optimum B ~256-512; B=1024 raises tau but lowers speed. Caveats: budget picked per dataset-model pair on the
  test set; tree-attention uses PyTorch not FA2; H200. Tree objective is drafter-marginal surrogate.
- **BASTION 2605.29727** (abstract): query-dependent trees under a roofline latency estimator; up to 6.61x vs AR,
  "+39% over block-diffusion baselines".
- **DBloom / ceiling-clipped histograms 2608.30427**: in many cycles the whole 16-block is accepted ("stranded
  speed-up"); naive inference-time widening fails (bidirectional attention shifts even early positions);
  post-training at block 24 with a short curriculum: +0.8 tokens median (up to +1.1; +1.37 with continuation FT)
  on Qwen3-4B/8B; Gemma-4-12B +0.41 median.
- **Carryover Drafting 2609.14717**: reuse rejected target hidden states as temp KV context; tau +6.5-14.7%,
  vLLM speedup +7.9-14.4% (up to 28.8% translation); training uses parallel draft-verify-draft to see
  inference-aligned states. (This is on-policy-ish training, relevant to sketch D.)
- **H-Spec 2609.24197**: no drafter-side KV cache, hybrid Mamba/attention drafter; tau +5.0-13.3%.
- **FailFast 2512.20573** (abstract): dLLM drafter with dynamic speculation length; up to 4.9x, 1.7x over EAGLE-3,
  up to 70 tokens accepted at once.

### 1d. Refinement of a block draft (directly relevant to sketch B)

- **D-Loop 2610.06011**: same backbone run in K looped passes; pass 1 proposes the block, pass 2 conditions on
  a selected PREFIX of pass 1's tokens and regenerates the suffix in parallel; prefix/suffix training objective;
  speed peaks at K=5 (Qwen3-4B GSM8K). Avg over 8 benchmarks, tau/speedup: Q3-4B T=0 DFlash 6.48/4.88x, DSpark
  6.61/4.95x, D-Loop 7.63/5.06x; Q3-4B T=1 4.46/3.36x vs 6.20/4.11x; Q3-8B T=0 5.77/4.38x vs 7.19/5.02x;
  T=1 3.87/2.94x vs 5.77/4.03x. Latency (Q3-8B GSM8K): draft 4.21 ms (DFlash) vs 9.43 ms (D-Loop), verify
  ~25.4-25.6 ms. Fails when acceptance is short (fixed schedule; Alpaca 4B T=0 loses to DSpark). Even the
  suffix-only variant gets 7.95 vs DSpark 6.03 -> loop structure itself carries most of the gain; no
  non-looped control.
- **XPress 2608.02438**: small causal refiner head, K Jacobi iterations over the whole dFlash block
  (75 us per iteration; K=4 379 us; plateau ~K=7; non-monotone). Qwen3-8B: tau +28% (T=0, 6.25->8.02), speedup
  4.9->6.2x (T=0), 3.6->5.3x (T=1); batched vLLM gain 1.16-1.47x, compresses with batch.
- Speculative Correction 2608.02625: draft-then-refine for dLLMs themselves (not SD of an AR target).

### 1e. Draft-ahead / asynchronous / pipelined

- **PEARL 2408.11850 (ICLR 2025)**: pre-verify first draft token during draft; post-verify keep drafting during
  verify assuming full acceptance; adaptive draft length. Up to 4.43x vs AR, 1.50x vs SD.
- **AMUSD 2410.17375**: draft and verify on separate devices, rollback on mismatch. Up to 1.96x vs AR, avg +29%
  vs SD (HumanEval ~1.57x, MT-Bench ~1.31x per repo README).
- **Speculative Speculative Decoding (SSD / Saguaro) 2603.03251 (Kumar, Dao, May; ICLR 2026)**: while verifying,
  the drafter (on SEPARATE hardware) precomputes speculations for many likely verification OUTCOMES
  (accepted count n, bonus token). Cache via top-F draft logits with a theorem-optimal geometric fan-out; bonus
  token predicted up to ~90% accurate; "Saguaro sampling" trades acceptance for cache hit rate; backup speculator
  when cache misses at large batch. Llama-3.1-70B (4xH100 TP) + 1B draft on another H100: ~30% over best SD,
  up to ~5x vs AR; ~20% at larger batch. Theory: SSD >= SD, speedup sandwich bounded by hidden draft latency x
  hit rate. Limits: latency not throughput (useless for RL/offline), misses grow with batch and temperature, tree
  / EAGLE-style combos "largely unexplored".
  This is, almost word for word, sketch A. Prior art.
- **D3D 2611.06425**: diffusion drafter independent of target hidden states, so successor blocks for alternative
  rejection outcomes + full accept are drafted in ONE packed pass during verify; layer pruning (5 of 36 layers
  removed keeps >99% tau). Qwen3-8B 5.00x vs DFlash 4.03x; Qwen3-32B 5.71x vs 3.67x; peak 7.48x.
  Caveat: they also swap the drafter, so the overlap share of the gain is not isolated from drafter quality.
- **DPara 2609.27396**: DSpark-style backbone precomputes draft representations for EVERY acceptance boundary
  during verification (bonus token left open); a light AR head finishes after verify. Qwen3-8B/14B 3.21x/3.52x
  avg vs AR; beats serial and parallel baselines (numbers vs. DFlash not extracted). Also sketch A, with the
  self-conditioning problem handled by "boundary-indexed representation + light head".
- **AHASD 2604.25326** (mobile async heterogeneous) - not read.

## 2. Theory: when does stacking help?

Notation: T_v(B) = target verify time for B tokens; r(B) = T_v(B)/T_ar; c_d = T_draft/T_ar (parallel drafter:
flat in block length). Speedup vs AR: S = tau / (r(B) + c_d).

### 2a. Standard SD and cascades (AR drafters)
S_SD = (1-a^{k+1}) / ((1-a)(ck+1)); drafting cost k*c grows linearly in k. This is what makes stacking pay:
replace the expensive inner AR loop by a cheaper model. HSD says gain is real but modest (1.03-1.17x over
best single draft) and confined to an intermediate band of (cost, acceptance). CSD horizontal cascade formula
T = (sum_i prod_{j<=i} a_j) / (1 + sum_i c_i) says a later position should be drafted by a model whose cost per
token is below the marginal acceptance mass it buys.

### 2b. Why it mostly dissolves for parallel block drafters
1. The drafter cost is already flat in k. The only thing an inner level could save is c_d. Perfect overlap or a
   free inner level bounds the gain at (r + c_d)/r. Measured: D-Loop/DFlash paper Qwen3-8B GSM8K draft 4.2 ms vs
   verify 25.4 ms -> c_d share ~14%, cap ~1.17x. DSpine paper (SGLang): verify 72-76% of round -> cap ~1.3-1.4x.
   Our A100/Qwen3-4B verify at B=16 is 15.6 ms (CUDA graphs); the draft pass is still UNMEASURED (PLAN.md).
2. The remaining lever is tau. Acceptance is the objective; an inner drafter that makes block-16 cheaper adds
   nothing to tau. Stacking levels of different cost only buys more tau if the outer level is a better
   (more expensive) model that sees inner tokens -- that is "refine" (D-Loop, XPress), not "stack".
3. Verify cost is nearly flat to B~128, so the break-even for longer blocks is tiny. From our A100 sweep:
   r(16)=1.11, r(64)=1.19, r(128)=1.34, r(256)=1.71 (vs B=1 14.06 ms). Extending 16->64 pays iff
   tau_64/tau_16 > (r64 + c_d)/(r16 + c_d). With c_d ~ 0.3 this is 1.057; i.e. any +6% tau wins.
   Extending 16->128 needs +10%. But the ceiling-bin evidence says tokens beyond 16 only exist in cycles where
   all 16 were accepted: gain <= p_full * E[extra | full]. DBloom's 16->24 gives only +0.8 tokens (~+11% of
   ~7) -> long blocks are worth roughly +10-25% on math/code and ~0 on chat (tau 3-4, ceiling bin empty).
4. Tree verification is the better use of the same verify slack: DDTree gets +30-40% speedup on top of DFlash
   (MATH-500 5.56->7.52x) by spending B=256-512 verify tokens width-wise; long blocks spend it depth-wise.
   Per-position acceptance decays (DFlash loss weight exp(-(k-1)/7)), so width at early positions beats depth.

### 2c. Draft-ahead bound
Hide drafter latency: round time max(T_v, T_d) instead of T_v + T_d, gain <= 1 + T_d/T_v ~ 1.14-1.33.
Costs: guess the outcome. PEARL/AMUSD bet on full acceptance (hit rate = ceiling-bin mass, ~10-35% for block
8-16 on math; ~0 on chat). SSD needs a fan-out over (n, bonus); for a parallel drafter with tau~6.5 the n
distribution is broad, and the drafter needs the target's hidden states for the NEW anchor, which do not exist
until verify ends (DFlash is target-conditioned). D3D removes the hidden-state dependence by retraining a
standalone drafter (cost: weaker drafter); DPara keeps the backbone and moves the dependence into a light head.
Expected gain = h * T_d/(T_v+T_d) where h = cache hit; and extra batched drafting costs more than one drafter
pass (packed over outcomes), so effective h*T_d is reduced by the packed-pass overhead. Needs spare compute: dies at
high concurrency (SSD, XPress and DSpine's own gains all compress with batch).

### 2d. Refine cascade: what it can add
Pass 2 turns the marginal (per-position independent) draft into a conditional draft. Evidence: XPress +28% tau
at ~0.4 ms; D-Loop +10-50% tau at +5.2 ms (T=1 gains are largest: +50% on Q3-4B). A "second drafter that
sees the first one's tokens" is therefore proven; the open part is only the LONGER output (16->64).

## 3. Assessment of our sketches

### (A) Branching draft-ahead with self-conditioned drafter -- PRIOR ART: SSD/Saguaro (2603.03251), D3D, DPara, PEARL, AMUSD
- Cap ~1.14-1.33x (T_d/T_v), only at low concurrency, minus miss cost. SSD's own ~30% over SD used a separate
  GPU and a 70B target; our target is 4-8B on one GPU, where drafter and verifier compete for the same SMs
  (verify is nearly free to B~128 on A100, so packing is plausible; SSD assumes dedicated hardware).
- The new piece in our sketch (self-conditioned hidden states) is what D3D (standalone) and DPara
  (boundary-indexed backbone + AR head) already solved two ways. Expect tau loss from losing target features
  (D3D keeps >99% tau only via pruning, but it is a different drafter).
- Verdict: do not build novel; at most reproduce DPara-style on DSpine and measure. Kill test: see section 4.

### (B) Refine cascade (long-block drafter takes short drafter's tokens) -- PARTIAL PRIOR ART: D-Loop, XPress, DBloom
- D-Loop pass 2 = "same-size drafter conditions on pass-1 prefix, regenerates the suffix" (exactly the refine,
  but length stays 16). XPress = iterative Jacobi refinement of the 16-block. DBloom = how to train a longer
  block (and warns inference-time widening breaks the front of the block, so your 64-block drafter must be
  TRAINED at 64 or with a curriculum; initializing from a 16-block drafter is not enough).
- Untested gap: pass 2 emits MORE tokens than pass 1 (16 -> 32/64) with pass-1 tokens as input. Upside
  bounded by the stranded-ceiling mass. The refine input does give pass 2 a free, strong signal for positions
  <=16, which lets it put capacity on 17-64, so it is a more favorable training problem than DBloom's.
- Verdict: the most promising of our sketches, but as an extension of D-Loop, not as a cascade.

### (C) Stacked drafters a la HSD/CSD -- low value for parallel drafters
- Bound (2b.1): gain <= (r + c_d)/r, ~14-30%, and only if the inner level is free AND lossless to tau.
  HSD with AR drafters, where inner cost was large, got 3-17%. CSD loses to Medusa without trees.
- A genuine cheap-inner variant: a statistical tail (n-gram / prompt lookup / suffix tree) extending the 16
  block "for free" past position 16 is the CSD horizontal cascade Max-Gram idea; cost ~0, r(64)=1.19. Helps on
  repetitive code/summarization/agentic text; ~0 elsewhere. Unverified here (no paper read on it).
- Verdict: kill as framed; keep the n-gram tail as a cheap experiment.

### (D) On-policy recursive training -- PRIOR ART in spirit: DistillSpec, HASS, EAGLE-3 TTT, Carryover, D-Loop's prefix objective
- DistillSpec: on-policy draft data +10-45% over SD. HASS: train on the drafter's own contexts, +8-20% over
  EAGLE-2, no inference cost. Carryover: trains on inference-aligned rejected states via parallel
  draft-verify-draft (+6.5-14.7% tau). D-Loop: prefix/suffix objective is training on one's own prefix.
- Open: multi-round DAgger-style loops for a BLOCK drafter where the "own rollout" is the pass-1 output that
  pass 2 consumes (needed for B), measured as per-round tau gain with a plateau test.
- Verdict: required plumbing for B (pass 2 must be trained on pass-1's actual errors), not a standalone project.

## 4. Ranked recommendations and kill tests

All kill tests are offline/cheap. Do #0 first; it prices everything else.

0. Measure the three numbers that decide A, C and long blocks (no new model).
   - T_draft (CUDA-graph) vs T_verify(B=16) on the A100/Qwen3-4B setup, plus per-position acceptance and the
     full-block (ceiling-bin) fraction on math, code, chat from one logged DFlash run.
   - Kill A and C if T_d/T_v < 0.2 (cap < 1.2x). Kill long blocks (B) if ceiling-bin fraction < 10% on the
     datasets you care about (expected extra tokens < 0.1 * ~10).
1. Tree verification on the best parallel drafter (DDTree/BASTION). Not recursive, but the biggest published
   lever (+30-40% speedup over DFlash) and nobody reported it on DSpine or on top of refinement. Do it before
   any recursive idea: it is the baseline recursive ideas must beat. Kill test: reproduce DDTree speedup on our
   A100 with graph-captured tree verify (their numbers use PyTorch attention on H200). If tree verify cost at
   B=256 kills the gain on A100 (r(256)=1.71) then budget the sweep at B<=128.
2. (B') Extend D-Loop's pass 2 to emit a longer block. Steps: pass 1 = frozen DFlash/DSpine 16-block; pass 2
   = same weights or a small head trained on pass-1 outputs with block 32 then 64 (DBloom curriculum).
   Kill test (offline, from logged traces): oracle experiment - given the target's own tokens as "pass-1 prefix"
   (upper bound), tau beyond position 16 of a 32/64-block drafter trained for 1 epoch; require tau_32/tau_16 >
   (r32 + c_d)/(r16 + c_d) ~ 1.04-1.06 on math/code at T=0 and T=1. If tau gain < 5% kill B.
3. Adaptive number of passes / early exit (D-Loop's own future work; DSpark's confidence-scheduled
   verification). D-Loop's second pass doubles draft time and loses on low-acceptance prompts; gate pass 2 on
   pass-1 confidence. Kill test: fraction of cycles where pass 2 changes the accepted length; if <15% gate is moot.
4. n-gram / suffix-tree tail extension of the 16-block ("free" CSD horizontal cascade). Kill test: acceptance of
   the n-gram tail on tokens 17-32 in cycles with a full 16-accept; need > 1 extra token per full-accept cycle on
   code, else drop.
5. Draft-ahead (A), only if step 0 shows T_d/T_v >= 0.25 AND we target concurrency 1. Use DPara's recipe
   (boundary-indexed representation + light AR head) on DSpine. Kill test: top-2 bonus hit rate and
   tau_self-conditioned / tau_conditioned from logs; require h * T_d/(T_v + T_d) > 8% net of packed-pass cost.
   This is a reproduction of DPara / D3D / SSD, not a research contribution.
6. Stacked HSD-style drafters (C): do not build. Kill criterion already met in theory (2b.1).
7. Multi-round on-policy training (D) as plumbing for #2 only; cap at 2 rounds; plateau test: round-2 tau gain
   < 3% -> stop.

Better ideas surfaced by the survey (not in our sketches):
- Tree built from pass-2 conditional marginals (D-Loop/XPress + DDTree). Conditionals make branches coherent
  and should raise tree efficiency per node; nobody has reported this combination (DDTree built on DFlash
  marginals; XPress did not compare to trees).
- Spend verify slack width-first (tree) then depth (long block): per-position acceptance decays, so a joint
  budget allocator (BASTION-style roofline, DSpark-style confidence schedule) should beat either.
- Carryover-style reuse of rejected target states inside pass 2 (cheap, +6-15% tau as reported).

## 5. Caveats
- Most "numbers" above for non-fully-read papers come from abstracts; mechanism rows for those are abstract-level.
- 2026 arXiv papers cited here are single-source, unreplicated, and many compare against differently trained
  DFlash reproductions; treat gains over DFlash as +/- a few points.
- Not found: any paper that does stacked (HSD-style) drafting with a parallel block drafter as an inner level;
  any paper that outputs a longer block conditioned on a shorter block's tokens (sketch B exact form).
  Absence in a handful of searches is not proof; one targeted arXiv listing search is cheap before building.
- DSpine's repo/blog describe DSpine as DFlash-family; the paper numbers are on the authors' own training setup.
