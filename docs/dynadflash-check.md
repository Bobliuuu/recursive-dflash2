# DynaDFlash check (2026-11-12)
Question: does DynaDFlash's per-token layer gate (midas-llm-research, Qwen3-8B, 20K extra params) help?
Method: their own `score_context` (teacher-forced, one anchor, 16-token block, greedy) on OUR prompts
(5 datasets x 40, Qwen3-8B greedy continuations, anchors at response offsets 1/16/64/128/256/380; 1033 contexts, A100).
Script `bench/dynadflash_eval.py`, raw `bench/results/dynadflash_tau.json`.

| drafter | all | gsm8k | math500 | humaneval | mbpp | mtbench |
|---|---|---|---|---|---|---|
| their DFlash baseline | 4.62 | 5.24 | 5.59 | 4.44 | 4.30 | 3.35 |
| DynaDFlash | 4.77 | 5.34 | 5.54 | 4.74 | 4.56 | 3.53 |
| official z-lab DFlash-b16 | **6.69** | 5.99 | 8.83 | 6.82 | 6.61 | 4.70 |

DynaDFlash - baseline = +0.158 tau (95% bootstrap CI 0.05..0.25), ~+3.4%, positive on 4/5 datasets (math500 flat/negative).
Gain is real-ish but small, and both of their drafters are far below the official DFlash (-2.0 tau); the gate was never shown on a
fully trained drafter. Not tested on DSpine (our best); would need a fine-tune with the gate added.
