# Phase 9: deletion tuning of the DSpine drafter (kill test; Qwen3-8B, 1x A100-80GB on Modal)

**Verdict: KILL.** Neither learned deletion mask reached the preregistered bar, so no served benchmark was run.

- **gen** (open-perfectblend data): all five domains land within +-0.12% of the official drafter on proxy chain15.
- **code** (code-only data): MBPP gains **+2.6%** (95% CI +1.9% to +3.3%), while its matched-count random-deletion control gains +0.2%. The bar asked for >= +3%. HumanEval is +0.2% and not significant.

The official `tencent/Qwen3-8B-DSpine-b16` weights remain the drafter to use.

- Preregistration: [`bench/p9deletion/PREREG.md`](../bench/p9deletion/PREREG.md), committed in `62b3ec0` before any mask was trained.
- Code:
  - `train/deletion_mask.py`: mask layer, Wanda prior, `.dmask` serializer, `apply` loader.
  - `train/test_deletion_mask.py`: tests.
  - `train/train_deletion.py`: trainer and proxy.
  - `train/p9_prompts.py`: prompt sets.
  - `bench/p9deletion/{modal_app.py, box_p9.sh}`: Modal pipeline.
- Raw output: `bench/results/p9deletion/`:
  - `proxy.json`
  - `{gen,code}/{train.json, run.log, mask.dmask, wanda_init.dmask}`
  - `eval_run.log`, `logs/`

## Hypothesis
Phase 3 and Phase 6 fine-tuned all the drafter weights from a warm start, and the result was neutral or a small gain. Deletion
tuning is far more constrained:

- It only zeroes weights. Nothing else is learned.
- The paper reports that it matches full fine-tuning while forgetting less.
- The adapter is a tiny mask over the official weights.

The Qwen3-8B target is untouched, so verification stays exactly lossless; only acceptance can change. The question was whether such a
mask raises mean acceptance length, either in general or for one domain (code) without hurting the others.

## Method
Source: Lu et al., "The Surprising Effectiveness of Deleting Weights in LLM Reasoning and Adaptation", arXiv 2611.10866.
The settings below were read from the reference code (github.com/agentic-learning-ai-lab/deleting-weights:
`src/learnable_sparsity/*`, `src/wanda_calibration.py`, `src/train_sft.py`, `scripts/train_sft.sh`). Only the arXiv
abstract was read, not the full paper (UNVERIFIED against the paper text).

- **Masked weights:** the 5 drafter `layers.*.mlp.gate_proj.weight` matrices, 5 x 12288 x 4096 = 251.7M weights. All other drafter weights are frozen.
- **Logits:** one fp32 logit per weight, N(0, 0.01^2).
- **Wanda prior:**
  - Score = `|W_ij| * a_j`, where a_j sums, over 64 calibration sequences of the drafter's own training forward, the per-batch L2 norm of input column j.
  - The lowest 1/2048 of each matrix is shifted by -3 std and the rest by +3 std, giving N(-+0.03, 0.01^2).
- **Training forward:** `W * sigmoid(s * G)`, with s rising linearly from 100 to 1000.
- **Loss:** `L_DSpine - 1e-5 * (1 - t/T) * sum ||W * m||^2`. The regulariser's weight decays to 0, as in the reference code (it is not a constant).
  - L_DSpine is the SpecForge `b15ad38` DSpine objective (the Phase 3/6 trainer path) with the in-process HF Qwen3-8B target.
  - Correct-predecessor curriculum off; alignment weight at full strength from step 0.
- **Optimizer:** AdamW on the logits only, LR 5e-4 constant, weight decay 0, clip 1.0.
- **Schedule:** 300 steps x 16 sequences x 64 anchors, block 16. About 4.5 s/step, 23 min per mask.
- **Deploy:** keep `G > 0` and set the other weights to exactly 0.
- **Shipping:** `.dmask` = per matrix, the sorted deleted flat indices, delta-coded as LEB128 varints and zlib-compressed, with each base tensor's SHA-256 in the header.
  - `python train/deletion_mask.py apply --base tencent/Qwen3-8B-DSpine-b16 --mask m.dmask --out DIR` writes a draft directory SGLang serves unchanged: same `config.json`; deleted weights +0.0; every other tensor and every kept weight bit-identical.
  - Sizes: the gen mask is 0.96 MB and the code mask 1.57 MB, against a drafter checkpoint of about 2 GB.

One property of the method shows up here. With logits drawn from N(+0.03, 0.01^2), about 0.13% of the "kept" logits start
below 0 (the 3-sigma tail). So the step-0 mask deletes about 0.18% of each matrix (461.6K weights), not 1/2048 (123K):
- 123K are Wanda-chosen;
- about 340K are effectively random.

The reference implementation has the same property; we did not change it.

## Data
- **Training prompts:** `train/prep_prompts.py`, open-perfectblend with eval overlap filtered. 10,000 prompts (261 dropped for eval overlap).
- **Responses:** Qwen3-8B, thinking off, greedy, max 1536 new tokens, SGLang AR (`train/gen_responses.py`). 9,995 completed.
- **Training sets:**
  - **gen:** the first 9,739 responses (9,698 after tokenisation).
  - **code:** the 699 `evol-codealpaca` rows of that set, plus 474 MBPP train/validation/prompt-split responses = 1,173.
  - UltraInteract also contains code but was not matched by the `code` source filter.
- **Proxy eval sets:** target greedy responses, same settings. All 100 frozen ids of each `bench/data/prompts/*.jsonl` are excluded, not only the 40 that `run_cell.py` uses. Formats as in `bench/prep_prompts.py`.
  - GSM8K test: 64.
  - MATH-500: 64.
  - HumanEval: 64 (the 164 - 100 that remain).
  - MBPP test: 63 generated, 62 used.
  - **chat:** 64 held-out open-perfectblend samples with a non-math, non-code source. MT-Bench stand-in, because all 80 MT-Bench prompts are frozen.
  - **general:** first 64 of the 256 held-out samples.

## Proxy
Teacher-forced, `train/train_deletion.py`. Per domain: 64 sequences x 128 seeded anchors (about 8.2K blocks). Every
variant runs on identical blocks, with shared target features and anchors.

- **Truth:** the target's greedy argmax, cut where the data leaves the greedy path.
- **chain15:** 1 + accepted length of DSpine's refined greedy chain over 15 drafts. This is the served chain b16 and the decision metric.
- **CIs:** paired bootstrap over sequences, 95%.
- **Use:** only ratios to the official weights are used. Absolute proxy values sit above the engine's tau (greedy-path data; Phase 6 found the proxy about 3% above the engine on its data).

Variants:
- official;
- gen, code: the learned masks;
- gen_wanda, code_wanda: step-0 masks;
- gen_rand, code_rand: per matrix, the same deletion count chosen uniformly at random (seed 1234).

## Results

Learned masks:

| mask | deleted | fraction of gate_proj | step-0 deletions still deleted | final train loss (step 1) |
|---|---|---|---|---|
| gen | 654,714 | 0.260% | 256,805 of 461,584 | 2.69 (3.00) |
| code | 1,193,245 | 0.474% | 260,761 of 461,580 | 2.94 (3.30) |

Both masks grew during training, so the final masks are roughly 5x and 10x the paper's "< 0.05%". About 56% of the
step-0 deletions survived.

**Proxy chain15, ratio to official [95% CI]** (`proxy.json`):

| domain | official chain15 | gen | gen_rand | gen_wanda | code | code_rand | code_wanda |
|---|---|---|---|---|---|---|---|
| gsm8k | 11.549 | 1.0002 [0.998, 1.002] | 0.9988 | 0.9985 | 0.9981 [0.996, 1.001] | 0.9994 | 0.9987 |
| math500 | 11.325 | 1.0002 [0.998, 1.002] | 0.9994 | 0.9993 | 1.0020 [1.000, 1.004] | 0.9983 | 1.0005 |
| humaneval | 10.015 | 1.0002 [0.999, 1.002] | 0.9987 | 1.0002 | 1.0016 [0.998, 1.005] | 0.9984 | 0.9992 |
| mbpp | 9.853 | 1.0012 [0.999, 1.003] | 1.0021 | 1.0011 | **1.0258 [1.019, 1.033]** | 1.0019 | 1.0012 |
| chat | 6.328 | 1.0012 [0.999, 1.004] | 0.9991 | 0.9991 | 1.0025 [0.999, 1.006] | 1.0011 | 1.0017 |
| general (not in bar) | 7.705 | 0.9998 | 0.9984 | 0.9991 | 0.9998 | 0.9995 | 0.9983 |

**Secondary metrics, ratio to official.**

| domain | tree48 (D<=15) gen | tree48 (D<=15) code | backbone chain gen | backbone chain code | top-16 coverage gen | top-16 coverage code |
|---|---|---|---|---|---|---|
| gsm8k | 0.9996 | 0.9991 | 1.0002 | 0.9993 | 0.9997 | 0.9997 |
| math500 | 0.9999 | 1.0000 | 1.0011 | 1.0005 | 1.0000 | 1.0002 |
| humaneval | 1.0004 | 1.0020 | 1.0007 | 1.0045 | 1.0001 | 1.0018 |
| mbpp | 0.9991 | **1.0234** | 1.0016 | **1.0335** | 1.0009 | **1.0178** |
| chat | 1.0016 | 1.0016 | 1.0003 | 1.0016 | 1.0018 | 1.0025 |
| general | 1.0018 | 1.0017 | 1.0017 | 1.0013 | 1.0011 | 1.0018 |

**Decision** (PREREG bar: some domain >= 1.03 with CI lower bound > 1.00; all five domains >= 0.99; >= +0.02 over the matched random control on that domain):

| mask | best domain | condition 1 (>= 1.03, CI > 1) | condition 2 (all >= 0.99) | condition 3 (vs random) | result |
|---|---|---|---|---|---|
| gen | chat / mbpp 1.0012 | no | yes (min 1.0002) | n/a | **fail** |
| code | mbpp 1.0258 (CI 1.019-1.033) | no (1.0258 < 1.03) | yes (min 0.9981) | would pass (+0.024) | **fail** |

## Reading
- **Deletion is harmless at this scale, which fits the paper's low-forgetting claim.** Deleting 0.18-0.47% of the gate
  weights moves chain15 by at most about 0.2% on every domain, in either direction. This holds whether the deletions are
  learned, Wanda-chosen or random. The one exception is the code mask's MBPP gain. The drafter's gate_proj is very redundant at this scale.
- **The general mask learns nothing measurable.** gen, gen_wanda and gen_rand are indistinguishable everywhere (all within about 0.2%).
  This matches Phase 3: the official checkpoint was trained on essentially this distribution and objective, so there is
  nothing for a general adapter to fix. Phase 3's full warm-start fine-tune was also neutral (8.00 vs 7.99 engine tau).
- **The code mask does adapt, but narrowly.**
  - On MBPP it is +2.6% on chain15, +3.4% on the backbone chain, +2.3% on tree48 and +1.8% on candidate coverage. Its random control is flat (+0.2%), so the gain comes from which weights were deleted, not from deleting weights at all.
  - It does not transfer to HumanEval (+0.2%, not significant), the other code set.
  - Its training data contains 474 MBPP train-split prompts in the benchmark's exact "Your code should pass these tests:" format. The most likely reading is adaptation to the MBPP prompt and answer format rather than to "code".
  - Even taken at face value it is below the preregistered +3%. On the served chain (verify cost unchanged), +2.6% tau would mean about +2% tok/s on one dataset of five.
- **Why so little:** deletion can only remove capacity. A drafter already trained against this target with this objective
  gains little from losing capacity, unless the target distribution shifts (here, MBPP's format). The paper's gains come
  from changing a model's *behaviour* (task SFT, RL); a drafter's job, matching the frozen target, is what it was already trained for.

## Caveats
- **One configuration, one seed** (preregistered; no hyperparameter search). A longer schedule or a higher LR could grow the MBPP effect. With the gain below 3% and confined to the in-format dataset, I would not expect it to clear the bar on code generally.
- **Proxy only.** No served runs, as the preregistration forbids them without a pass. The proxy tracked engine ratios to within 0.1% in Phase 6, but it was not re-validated here.
- **Masks larger than the paper's < 0.05%.** Our masks (0.26% / 0.47%) are larger than that. Part of the reason is the init tail described above; part is that training grew the deletions (logits drift under Adam at LR 5e-4 with tiny gradients, grad norm about 0.01).
- **Domain stand-ins.** "chat" stands in for MT-Bench. Eval sets are 62-64 prompts per domain; the CIs above are the uncertainty.
- **Data generation stack.** Generation used SGLang AR on Modal with the same stack as `bench/setup_box.sh` (DSpine plugin at `321b0fc`, SGLang 0.5.22, torch 2.14.1+cu130, flashinfer 0.7.1), on an A100-SXM4-80GB (driver 610.57), not a Vast box.

## Compute and spend (SPEND / RUNS)
Modal, A100-80GB with 6 CPUs and 40 GB RAM. Cost is an estimate from app wall time x about 3.1 USD/h (GPU plus CPU/RAM list
price), not from the Modal invoice.

| app id | stage | wall time | result |
|---|---|---|---|
| ap-EkXuSpPhFDYgcnTGnptbzx | smoke (stack + 12 CPU tests in the image) | 3.1 min | ok |
| ap-36g7wzjeZjN4ahTM5lyKYu | data (prompts + greedy generation) | 23.6 min | ok |
| ap-u3G1yvMX1GQ6dcr3S4TiZ6 | train | 2.1 min | config error (`alignment_warmup_steps` must be an int), fixed in `f043e2b` |
| ap-35NXaIwq41glsdn5hWA7Vl | train (gen + code masks) | 48.1 min | ok |
| ap-LcgKotuOG72X3HvvjnVa7U | eval | 0.2 min | failed: CRLF line endings in the uploaded shell script after a Windows checkout |
| ap-Wz72pVZu0wLpWJPBofYUie | eval (proxy, 6 domains x 7 variants) | 8.3 min | ok |
| **total** | | **85.4 min (about 1.4 h)** | **about 4.5 USD of the 25 USD cap** |

All apps are stopped. The volume `rdf2-p9deletion` (HF cache, data, masks) was kept.

## Exact commands
```bash
# local (CPU): unit tests
python -m pytest train/test_deletion_mask.py
# Modal (each stage is resumable; state on volume rdf2-p9deletion under /vol/p9)
modal run --detach bench/p9deletion/modal_app.py --stage smoke
modal run --detach bench/p9deletion/modal_app.py --stage data     # NGEN=10000 prompts, greedy, max 1536 new tokens
modal run --detach bench/p9deletion/modal_app.py --stage train    # masks gen, code: 300 steps each
modal run --detach bench/p9deletion/modal_app.py --stage eval     # proxy: official + {gen,code}{,_rand,_wanda}
modal volume get rdf2-p9deletion p9/eval/proxy.json bench/results/p9deletion/
# deploying a mask (what a served run would use)
python train/deletion_mask.py apply --base tencent/Qwen3-8B-DSpine-b16 --mask bench/results/p9deletion/code/mask.dmask --out /workspace/dspine_b16_code_mask
```
On Windows checkouts, make sure `bench/p9deletion/box_p9.sh` has LF line endings before `modal run` (it is uploaded as-is).
