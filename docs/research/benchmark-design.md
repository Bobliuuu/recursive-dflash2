# Fair benchmark design: AR vs EAGLE-3 vs DFlash vs DFlash2 vs DSpine

Date 2026-11-08. Sources: z-lab/dflash README, DFlash paper (arXiv 2602.06036 HTML), DSpine paper
(2609.36173 HTML/abs), HF cards (z-lab, incoai, tencent, AngelSlim). Fetched via a summarizing web tool, not
raw pages. Re-verify HF ids with `huggingface-cli` / `HfApi().model_info` before launching anything.

## Headline findings

1. **No DFlash2 checkpoint exists for Qwen3-4B or Qwen3-8B** (z-lab and incoai orgs only show 27B/30B/35B/Flash-class
   repos). PLAN.md's "Qwen3-8B: DFlash, DFlash2, DSpine checkpoints all exist" is wrong for DFlash2. The DSpine
   paper retrained DFlash2 on Qwen3-8B but its release does not list that checkpoint. Options: (a) retrain DFlash2
   ourselves on the same data as the others, (b) benchmark on Qwen3.8-27B where all of DFlash2 / DSpine
   exist (DFlash and EAGLE-3 for 27B: check), (c) both. Decide before spending.
2. **DSpine has an official Qwen3-8B checkpoint and a Qwen3.8-27B one; none for Qwen3-4B** (paper has 4B, repo does not).
3. **Published numbers disagree by a lot** (DSpine paper abstract: DFlash 3.77 -> DSpine 4.82 mean tau at T=0 on 8B;
   the repo/HF card table: DFlash 5.92 -> DSpine 7.83). Different setups (probably retrained-on-ShareGPT
   vs official checkpoints, or different sets/thinking). Never quote one number without its protocol.
4. DFlash2 has no paper; only a blog post + model cards (Inco AI). Treat its claims as vendor claims.

## 1. Checkpoints (HF ids)

Legend: OK = exists and verified on a page this session; ? = likely but not verified; NONE = not found.

| Target | AR | EAGLE-3 | DFlash | DFlash2 | DSpine |
|---|---|---|---|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B` | `AngelSlim/Qwen3-4B_eagle3` OK (vLLM 0.11.2, spec tokens 2, tau 2.08 on H20) | `z-lab/Qwen3-4B-DFlash-b16` ? (paper has it; not shown on z-lab front page) | NONE | NONE (paper only) |
| Qwen3-8B | `Qwen/Qwen3-8B` | `AngelSlim/Qwen3-8B_eagle3` ? (used by DFlash paper; cited by other cards), `Tengyunw/qwen3_8b_eagle3B` ?, `RedHatAI/Qwen3-8B-speculator.eagle3` OK | `z-lab/Qwen3-8B-DFlash-b16` OK (MIT, ~1B, block 16, non-thinking only) | NONE | `tencent/Qwen3-8B-DSpine-b16` OK (Apache-2.0, ~1B BF16, 16 draft tokens, SGLang `DSPINE`) |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` (+ built-in MTP) | ? none found | ? (README lists Qwen3.6/3.5 27B, not 3.8) | `z-lab/Qwen3.8-27B-DFlash2` OK = mirror of `incoai/Qwen3.8-27B-DFlash2` (Apache-2.0); GGUF variants | `tencent/Qwen3.8-27B-DSpine-b16` OK (8K sliding window, DSpark-style dynamic verify, ReplaySSM) |
| Others | | `thoughtworks/Qwen3-Coder-Next-Eagle3` ?, `nex-agi/SGLANG-EAGLE3-Qwen3-32B-Nex-N1` ? | Qwen3.5 4B/9B/27B/35B-A3B, Qwen3.6 27B/35B-A3B, Coder-Next, Coder-30B-A3B, Llama-3.1-8B, GPT-OSS 20B/120B, Gemma 4, Kimi, MiniMax (ids in `z-lab/dflash` collection) | `z-lab/Muse-Glimmer-30B-DFlash2`, `incoai/Qwen3.6-35B-A3B-DFlash2`, `incoai/Qwen3.8-Flash-Next-DFlash2`, `incoai/GLM-5.3-Flash-DFlash2`, `incoai/Ornith-1.5-122B-DFlash2`, ... | none |

Also seen: `AngelSlim/Qwen3-8b-dflare` (unknown method, not investigated), `RedHatAI/gemma-4-31B-it-speculator.dflash`.
Recommendation: **Qwen3-8B non-thinking** is the only target with official DFlash + DSpine + (multiple) EAGLE-3
drafts; DFlash2 must be trained by us (also needed for a fair training-data match, see section 3).

## 2. Engines and version caveats

| Method | SGLang | vLLM | HF Transformers | Notes |
|---|---|---|---|---|
| AR | yes | yes | yes | |
| EAGLE-3 | yes (`--speculative-algorithm EAGLE3`) | yes (`method: eagle3`; RedHat speculators format or AngelSlim) | no first-class | AngelSlim cards benchmark on vLLM 0.11.2 only 2 spec tokens; DFlash paper used tree sizes 16 and 60 (7 steps, topk 10) -> need a tree-capable engine = SGLang. |
| DFlash | yes, `--speculative-algorithm DFLASH`; z-lab README pins PR #35371 (may be merged by now; check) | yes, `dflash` method; README pins PR #52816; vLLM-speculators docs say "under active development", `sample_from_anchor` must match model config (False for DFlash) | yes, `trust_remote_code=True`, custom `spec_generate` (paper's main results) | Qwen3 drafts: thinking mode OFF. |
| DFlash2 | yes `DFLASH` algo, 8 draft tokens on card | yes, PR build, 7 spec tokens on card | yes (Muse example) | Card says block 8 on H200; the dflash CLI hints at engine-side top-k + selector support. Engine support for the selector is the highest risk; verify path selection is actually active (look at logged acceptance not just "it ran"). |
| DSpine | **yes only**, `DSPINE` algorithm, ships SGLang 0.5.22 install, fused kernels + transition cache | no | mlx-lm / SGLang-MLX only | Training code "not yet released" per card. Needs its own SGLang fork build, so cannot share one SGLang install with others unless versions line up. |

Consequences:
- Only SGLang runs all five. Use **one pinned SGLang commit/container** that has DFLASH, DSPINE and EAGLE3
  (try DSpine's 0.5.22 pin as the base; if DFLASH is not in it, cherry-pick or run two venvs, but then record
  both and run AR in both to prove the baseline matches within noise).
- DSpine/DFlash2/DFlash differ in draft tokens per round (16 / 8 / 16 in cards). Engine arg `speculative-num-draft-tokens`
  must be set per method AND swept (see confounders).
- Add an HF Transformers path (batch 1, no engine overhead) as second axis: it gives clean acceptance length for
  methods whose engine support is flaky, but wall-clock there is not comparable to serving numbers.

## 3. Protocol

**Target:** Qwen3-8B, bf16, thinking OFF (DFlash checkpoint is non-thinking; DSpine paper also off). Optional
second target Qwen3.8-27B with its reasoning effort fixed and logged.

**Datasets** (full sets, not subsampled, as in DSpine paper; DFlash CLI default is 128 prompts):
GSM8K (1319), MATH-500, HumanEval (164), MBPP, LiveCodeBench, MT-Bench (80, multi-turn: use first turn only for
parity, or all turns and say so), Arena-Hard. Primary set = the DSpine-paper 7 so numbers can be cross-checked;
optional AIME25 + Alpaca (DFlash paper). Use the same prompt templates / chat template for every method; fixed
prompt file hashes in results.

**Sampling.**
- T=0: output must be byte-identical to AR (see losslessness check).
- T=1: top_p / top_k per Qwen card (non-thinking recommended T=0.7, top_p 0.8, top_k 20; DFlash paper and DSpine
  use plain T=1). Pick plain T=1, top_p=1, top_k=-1 (matches both papers) as primary; Qwen-recommended as secondary.
- Note DSpine paper at T=1: DSpark/DFlash2/DSpine *sample* proposals, DFlash/Domino propose greedily. Record
  `draft_sampling` ({greedy, sample}) per method and, where the engine allows, run both for DFlash.
- Seeds: 3 seeds at T=1, report mean and std; T=0 one run is enough (but run twice to measure nondeterminism).

**Lengths:** max_new_tokens 2048 (both papers). Report separately the fraction hitting the cap.

**Concurrency:** 1 (latency, comparable to papers) and 8 and 32 (throughput regime; DFlash2 card: 3.4x at c=1 drops
to 1.0-1.45x at c=32 on MT-Bench/GSM8K, so batch matters a lot). Closed-loop clients, N = concurrency.
Acceptance length for the DSpine-paper style (8 GPUs x 16 concurrent) is not needed; use engine-reported tau.

**Warmup:** 8 prompts discarded per (method, concurrency) after server start; CUDA graph capture must complete
before timing; one cold-start measurement kept separately. Drop caches: disable prefix cache (`--disable-radix-cache`)
or keep identical across methods; MT-Bench turn 2 would otherwise get free prefill.

**Metrics (per request, raw; aggregate later):**
- Acceptance length tau = emitted tokens / verify rounds (count the bonus token; document the convention. DFlash
  counts bonus token included; vLLM `sample_from_anchor` changes what "block" means, so state it).
- Per-position acceptance: P(position i accepted | i-1 accepted) and unconditional survival curve, i = 1..B-1.
  Requires per-round accepted-length histogram; log histogram per request, derive both.
- Wall-clock output tok/s: per request (n_out / (t_last_token - t_first_token)) and aggregate
  (sum n_out / makespan). TTFT and ITL percentiles separately, since spec decoding bursts tokens.
- Speedup vs AR = tok/s_method / tok/s_AR on the SAME engine build, same hardware, same concurrency, same prompts
  (ratio of means, with bootstrap CI over prompts).
- Round time breakdown: draft ms, verify ms, other (DSpine paper reports it). Needed for the recursive-drafting cap.
- GPU: mean power / peak memory (nvidia-smi sampling at 1 Hz), for the cost table.

**Losslessness checks (mandatory, gate on publishing):**
- T=0: for every prompt, compare token ids with AR run on the same engine (batch 1). Report exact-match rate;
  bf16 batch-shape nondeterminism will cause some divergences, so also run AR twice and report the AR-vs-AR
  rate as the floor. Method mismatch rate must be within AR-vs-AR noise.
- T=1: distributional check. Pick 64 prompts, 200 samples of first 1 token and first 4 tokens (hist), chi-square /
  TV distance method vs AR vs AR-vs-AR. Plus compare benchmark accuracy (GSM8K exact match, HumanEval pass@1)
  between methods: they should match within CI.

**Confounders to control (put each in the JSON):**
| Confounder | Control |
|---|---|
| Training data | DFlash official: ~800K Nemotron-V2 + CodeAlpaca regenerated by target. DSpine paper retrained: ShareGPT regenerated. DSpine release card: Open-PerfectBlend. EAGLE-3 AngelSlim: unspecified. Report **official-checkpoint** table, and an **equal-data retrained** table (same data, 5 layers, block 16, 6 epochs, LR 6e-4, max-len 3072 as DSpine paper) for DFlash, DFlash2, DSpine, EAGLE-3-style if affordable. |
| Block size / draft tokens | Official: DFlash 16, DSpine 16, DFlash2 8 (card; 7 on vLLM), EAGLE-3 chain 2-7 or tree 16/60. Run each at its own default AND at matched budget (verify width 8 and 16 tokens) because verify cost, not drafting, sets the ceiling. Also report verify width per round. |
| Verify tree vs chain | DFlash2 top-k path selector yields one chain; EAGLE-3 trees verify more tokens. Report tokens verified per round alongside tau; compare at equal verify width. |
| Engine / build | One pinned commit, flags identical (attention backend FA3 on H100, FA4/B200 only if all methods run there; CUDA graphs on; mem-fraction identical). |
| Hardware | One GPU type for the main table; repeat the c=1 row on a second type. No shared-host neighbours. Clock lock if allowed. |
| Prompt/template | Same chat template, same system prompt, thinking off flag identical. |
| Thinking mode | OFF everywhere; thinking-mode table optional, only for methods with thinking-trained drafts. |
| Quantization | bf16 target, bf16 draft. |
| Draft sampling at T=1 | greedy vs sampled proposals recorded. |
| Run order | Interleave methods per prompt batch or randomize server order; repeat AR at start and end to detect thermal drift. |

## 4. Harness design

```
bench/
  README.md                      # how to reproduce, pinned commits
  configs/
    methods/{ar,eagle3,dflash,dflash2,dspine}.yaml   # hf ids, engine args, draft tokens, draft_sampling
    datasets.yaml                # name, hf path, split, prompt template id, n
    matrix.yaml                  # methods x T x concurrency x seeds x draft-token sweep
  data/prompts/<dataset>.jsonl   # frozen prompts with sha256 recorded
  harness/
    serve.py                     # launches sglang/vllm per method, waits healthy, records versions
    run_one.py                   # one (method, dataset, T, conc, seed) cell -> raw JSONL
    client.py                    # async OpenAI-compatible streaming client, timestamps per token chunk
    collect_spec_stats.py        # pulls engine accept-length histograms (sglang /get_server_info or /metrics)
    lossless_check.py            # T=0 exact match + T=1 distribution test
    env_snapshot.py              # git sha, pip freeze, nvidia-smi -q, driver, CUDA, engine build
  results/raw/<run_id>/<cell>.jsonl      # one line per request
  results/raw/<run_id>/cell.json         # cell metadata + env
  results/agg/summary.json               # produced by analyze.py only
  analyze.py                     # raw -> tables, CIs, plots
  existing: verify_sweep*.py, a100/    # untouched
```
Rules: raw JSONL is append-only and committed; aggregate files are derived and regenerated; never edit by hand.
Run detached with logs in the repo (user rule); one cell = one resumable unit keyed by `cell_id`, skip if done.

**Per-request raw line (JSON):**
```json
{"run_id":"2026-11-10_a100_q3-8b","cell_id":"dspine|gsm8k|T1|c8|s0|d16",
 "method":"dspine","target":"Qwen/Qwen3-8B","draft":"tencent/Qwen3-8B-DSpine-b16",
 "dataset":"gsm8k","prompt_id":17,"prompt_sha":"...","temperature":1.0,"top_p":1.0,"top_k":-1,"seed":0,
 "concurrency":8,"max_new_tokens":2048,"draft_tokens":16,"draft_sampling":"sample",
 "n_in":212,"n_out":389,"hit_cap":false,
 "t_submit":0.0,"t_first":0.071,"t_last":4.82,"token_times":[...optional, downsampled...],
 "rounds":87,"accept_hist":[3,5,12,...],       // count of rounds that accepted j draft tokens, j=0..B-1
 "draft_ms":123.4,"verify_ms":987.6,           // null if engine cannot report
 "output_ids_sha":"...","output_ids_path":"ids/....npy",   // for lossless check
 "gpu_peak_mem_mb":41230,"error":null}
```
Cell metadata JSON: `{engine:{name,version,commit,flags}, gpu:{name,count,driver,clock_locked}, git_sha, hf_revision_sha
for each model, started_utc, finished_utc, price_per_hour_assumed}`.
Pin **HF revision SHAs** (not just ids) because these repos are being updated (cards mention re-measuring).

## 5. GPU needs and cost (Qwen3-8B)

Memory: target 8B bf16 ~16.4 GB + draft ~1-2 GB + KV. Fits 24 GB (A10G) only at short context/low concurrency
(c<=8 with 2K output, tight; no room for CUDA graph + mem-frac headroom at c=32). A100-40GB works comfortably
at c<=32; 80 GB is unnecessary for 8B. DFlash paper numbers were H200 (Transformers) / B200 (SGLang), so absolute
tok/s will not match; compare ratios and tau.

Matrix size: 5 methods x 7 datasets x 2 T x 3 conc x (3 seeds at T=1). At c=1 the cost is serial: roughly
(~4000 prompts across sets) x ~400 tokens at 80-250 tok/s => ~2-6 h per method/T pair at c=1 on A100. A reduced
first pass (GSM8K, HumanEval, MT-Bench, MATH-500; 128 prompts each; c in {1,8}; seeds=1) is ~6-10 GPU-hours total
for all methods. Full matrix ~40-80 GPU-hours. These are rough estimates, not measured.

**Prices are approximate recollections of list prices, NOT checked live today; they change often, spot/market prices
vary. Confirm on the provider page and get user approval before launching (docs/PLAN.md budget rule).**

| Option | GPU | ~$/h (list, unverified) | 10 GPU-h | 60 GPU-h | Notes |
|---|---|---|---|---|---|
| AWS g5.xlarge/2xlarge | A10G 24GB | ~1.0-1.2 | ~$11 | ~$66 | 24 GB too tight for c=32; fine for c<=8 and HF path |
| AWS p4d.24xlarge | 8x A100 40GB | ~32 (~4/GPU) | $320 if whole node used for 10 h; wasteful for 1 GPU | | Quota/price per memory notes; only if running 8 cells in parallel |
| AWS p5.48xlarge | 8x H100 | ~55-100 (varies by region/date) | | | Overkill; parallel cells only |
| Lambda | A100 40GB | ~1.3-1.5 | ~$14 | ~$85 | Existing `ssh lambda-final` A100 box (see memory) |
| Lambda | H100 80GB | ~2.5-3.3 | ~$30 | ~$180 | Closest to paper's FA3/H100 regime |
| Vast.ai | A100 | ~0.6-1.2 (market) | ~$9 | ~$55 | Variable hosts: reliability and noisy neighbours hurt timing fairness; lock to one host for the main table. Key at ~/.config/vastai/vast_api_key |
| Vast.ai | H100 | ~1.5-2.5 (market) | ~$20 | ~$120 | same caveat |

Suggested: single A100-40GB on Lambda (or existing box) for the main table (c=1,8,32), A10G or second Vast
card for the second-hardware row, plus a single H100 hour block for the headline cross-check. Training DFlash2 for
8B on matched data (800K samples, 6 epochs) is separate and large (multi-GPU, tens of GPU-hours at least): price
and decide separately.

## 6. Comparisons in the literature that are NOT apples-to-apples

1. **DFlash paper vs EAGLE-3 (up to 2.5x / "6.17x vs ~2.4x on Qwen3-8B").** Different drafts' training data
   (800K Nemotron+CodeAlpaca target-generated vs AngelSlim's EAGLE-3 data), draft-model size (~1B DFlash vs small
   one-layer EAGLE-3 head), and main speedups measured with Transformers on H200 at batch 1 where EAGLE-3's tree
   overhead in non-optimized HF code can be larger. The SGLang B200 serving table is more apt. Also DFlash
   verifies 16 tokens vs EAGLE-3 chain/tree budgets; compare at equal verify width.
2. **AngelSlim EAGLE-3 cards** (vLLM 0.11.2, H20, 2 spec tokens, 1024 output, tau ~2.1) are a very different
   operating point from DFlash/DSpine tau 5-8; they are not a measure of EAGLE-3's ceiling.
3. **DSpine paper vs "official" DFlash**: abstract says 3.77 -> 4.82; HF card says 5.92 -> 7.83. One of these
   retrains baselines on ShareGPT, the other uses different data. Baseline hyperparameters are not disclosed and the
   fixed 15-draft-token serving budget may not suit each baseline. DSpine/DFlash2/DSpark were retrained by the
   DSpine authors (interest in own method winning); DFlash2 is cited from a blog, not a paper.
4. **T=1 asymmetry (DSpine paper):** DSpark/DFlash2/DSpine sample proposals; DFlash/Domino propose greedily.
   Higher T=1 tau for sampled-proposal methods is partly a decoding-policy effect, not architecture.
5. **DFlash2 card (z-lab/incoai)**: block 8 on SGLang H200 with Qwen3.8's recommended sampling (T=1-ish) vs MTP and a
   community DSpark drafter, different model (27B reasoning model) from the 8B comparisons. MTP built-in is a
   different technique class. 3.4x at c=1 vs 1.0-1.45x at c=32: a method can win at c=1 and be a wash at c=32.
6. **Acceptance length vs wall-clock:** tau ignores draft cost and verify width. DSpine's SGLang +23.3% over
   DFlash throughput is much smaller than its tau gain (+27.8% in abstract) and the cards' tau gaps (+32%).
   Always show both.
7. **Different tau conventions** (bonus token counted or not; `sample_from_anchor`; MLX block <= 5 quantized) make
   cross-engine tau comparison invalid; recompute from the accept histogram with one definition.
8. **Hardware:** B200/H200 vs H20/A100/A10G change the draft/verify cost ratio; speedups do not transfer.
9. **Thinking vs non-thinking:** DFlash Qwen3 drafts are non-thinking; DFlash paper Table 2 (thinking) gives lower
   tau than Table 1. Do not mix tables.

## Open items before launch
- Confirm exact HF ids/revisions for Qwen3-4B/8B EAGLE-3 and Qwen3-4B DFlash (`HfApi.list_models(author=...)`).
- Confirm SGLang commit that has DFLASH + DSPINE + EAGLE3 together; otherwise two venvs with AR in each.
- Decide DFlash2-on-8B route (retrain vs switch to 27B) and its compute budget with user.
- Get explicit approval for hourly cost before starting any box.
