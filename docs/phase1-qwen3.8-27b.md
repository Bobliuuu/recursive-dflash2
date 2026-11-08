# Phase 1 on Qwen3.8-27B: official drafters, one box, one engine

Date 2026-11-08. Hardware: 1x H100-80GB (Vast), bf16 target. Engine: SGLang 0.5.22 (torch 2.14.1+cu130, FlashInfer 0.7.1) plus Tencent's
`sglang-dspine` 0.2.0 plugin (commit 6450af7) for DSpine. Raw JSON (one file per method x dataset x temperature x concurrency, per-request
`meta_info` incl. the accepted-drafts histogram): `bench/results/qwen38_27b/`. Run logs: `logs/qwen38_27b/`. Scripts: `bench/qwen38_27b/`
(`serve.sh` launcher, `client.py`, `sweep*.sh`, `run_method*.sh`, `queue*.sh`, `make_sps_requests.py`, `summarize.py`). SPS tables + tuner measurements:
`bench/results/qwen38_27b/sps/`.

## 1. Which drafters exist for Qwen3.8-27B (HF search, 2026-11-08)

| method | repo (sha) | status | block / verify width | notes |
|---|---|---|---|---|
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2` (50307d4c, 2026-08-19); `incoai/Qwen3.8-27B-DFlash2` (015e7956, 2026-09-17) | official | block 8 (7 drafts) | same 3,848,817,896 B safetensors in both repos; incoai card says it is a mirror of z-lab |
| DSpine | `tencent/Qwen3.8-27B-DSpine-b16` (ea909c0f, 2026-11-07) | official | block 16, verify width tunable 2..16 | 4.05 GB, 8192 sliding window; plugin requires `--speculative-num-draft-tokens 16` exactly |
| MTP | built into `Qwen/Qwen3.8-27B` (37561816) | official (target's own head, `mtp_num_hidden_layers: 1`) | chain, we use 3 steps = 4 draft tokens | run via `--speculative-algorithm EAGLE`; this is what the DFlash2/DSpark cards call "MTP" |
| DSpark | `RadixArk/Qwen3.8-27B-DSpark` (b9a5dbdf, 2026-08-29) | community (RadixArk), v2 | serving block 7 (verify 8), trained 16 | native in SGLang 0.5.22 (`DSPARK`); needs flashinfer attention backend |
| EAGLE-4 | `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (f58a5985, 2026-10-19) | vendor | tree/flex | `LlamaForCausalLMEagle4`, aux layers [3,31,59], low-rank LM head; needs the separate `eagle-hq/EAGLE-4` code, not in SGLang 0.5.22; card numbers are on 4x B300 TP=4. **Not run.** |
| EAGLE-3 | none | - | - | `Qwen3.8-27B-EAGLE3` search returns nothing; only EAGLE-4 and MTP exist as EAGLE-family drafters |
| DFlash (v1) | none official | - | - | only community `kstoyanov99/Qwen3.8-27B-Dflash`, `jfan/...-heretic-dflash`; z-lab ships DFlash2 for this target (z-lab/Qwen3.5-27B-DFlash and Qwen3.6-27B-DFlash are the older targets) |
| other DFlash2/DSpark | `RedHatAI/Qwen3.8-27B-speculator.dspark`, `inference-optimization/*dflash2*`, many quants/GGUF | community | - | not run |

DFlash2 config (also added to `docs/dflash2-architecture.md`): `DFlash2DraftModel`, 5 layers, hidden 5120, `dflash_config` = {block_size 8,
selector_top_k 16, selector_rank 256, conv_kernel_size 2, conv_group_size 16, mask_token_id 248070, target_layer_ids [5,19,33,47,61]},
sliding window 2048, bidirectional in block. DSpine-b16: block 16, target_layer_ids [1,16,31,46,61], block-causal, sliding window 8192, code bridge rank 512 / size 1024.
Target: 64 layers, 48 GatedDeltaNet + 16 full attention (every 4th), hidden 5120.

## 2. Protocol (what was actually run)

- One server per method, `bench/qwen38_27b/serve.sh`; context length 8192; `--max-running-requests 8`; CUDA graphs for decode/verify (prefill graph off to save memory);
  `--mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer`; fa3 attention for AR/MTP/DFlash2/DSpine, flashinfer for DSpark; radix cache off except DSpine (plugin requires it; prompts are ~100 tokens so prefix reuse is negligible).
  Mem fraction 0.93 (AR, MTP, DFlash2, DSpark), 0.86 with 40 Mamba slots (DSpine; ReplaySSM, `--enable-linear-replayssm-spec`). All knobs are in `serve.sh`.
- Client `client.py`: OpenAI chat endpoint, prompts formatted exactly as `z-lab/dflash` benchmark.py (GSM8K/MATH-500/HumanEval/MBPP test splits, MT-Bench first turn only), shuffle seed 42, same prompts for every method.
  **Thinking mode: ON, `reasoning_effort=xhigh`** (as the model cards), `max_new_tokens=2048` (cards use 4096). T=0, or T=1.0/top_p 0.95/top_k 20 (Qwen recommended). Because of the cap, 2-43% of requests are truncated (HumanEval worst, GSM8K 2%); AR and spec runs share the cap so ratios are fine, absolute lengths are not comparable with the cards.
- Cells: 5 datasets x T in {0,1} x concurrency {1,8}. c=1: 8 prompts/cell; c=8: 24 prompts/cell (first pass) and 64 prompts/cell (second pass, T=0 for all methods, T=1 for AR/DFlash2/DSpine+SPS).
  A warmup of `concurrency` 64-token requests precedes each cell. Throughput = total completion tokens / wall time of the cell (includes prefill and tail). Acceptance = completion tokens / verify steps ("pooled") and the per-request mean of SGLang's `spec_accept_length` (the card definition).
  Per-position acceptance comes from SGLang's `spec_correct_drafts_histogram`.
- Dynamic verification: `DSpine/scripts/tune_sps.py` on this GPU (widths 4,8,12,16 x batch 1-8, 3 repeats, 512 new tokens, 20 held-out prefixes at shuffled indices 100+ of each dataset, xhigh template) for T=0; reduced sweep (widths 8,16, 1 repeat) for T=1.
  Resulting policies: T=0 `uniform_widths [16,16,16,12,16,12,16]`, `batch8_tokens 96`; T=1 `[16,16,16,16,8,8,16]`, `batch8_tokens 64` (batch8 < 128 means DSpark-style confidence-based allocation of a smaller token budget at BS 8).
- Matched block rows: (a) DSpine forced to verify width 8 via an SPS profile (`uniform_widths [8]*7`, `batch8_tokens 64`) - DSpine's drafter is block-causal so a width-8 verify equals the first 8 positions of the 16-block;
  (b) the DFlash2 checkpoint forced to block 16 (`--speculative-num-draft-tokens 16`, SGLang warns "block size mismatch" but runs; the checkpoint was trained at 8) at c=1 only, because 8 slots x 16 tokens of intermediate SSM state do not fit in 80 GB next to the weights.

## 3. Results

Reading guide: each cell is `output tok/s (speedup vs AR at the same concurrency/T/dataset) / acceptance length (pooled)`. AR: 51.5 tok/s per stream at c=1 on this H100 (bf16, 54 GB weights, close to the 3.35 TB/s roofline).
The c=1 cells (8 prompts) and the c=8 N=64 cells are the trustworthy ones. The first-pass c=8 cells (24 prompts) are dominated by a single long request: e.g. AR GSM8K c=8 took 48 s of wall time for ~8 s of work, so
AR looks 175 tok/s there and speedups are inflated (up to 5.6x). They are kept in the raw data and listed below only for completeness.


### concurrency 1, T=0: output tok/s (speedup vs AR) | acceptance length (pooled tokens/verify-step)

| method | gsm8k | math500 | humaneval | mbpp | mt-bench | mean |
|---|---|---|---|---|---|---|
| AR | 51 (1.00x) | 52 (1.00x) | 52 (1.00x) | 52 (1.00x) | 52 (1.00x) | 1.00x |
| MTP (built-in head, 4 tok) | 134 (2.61x) / 3.34 | 129 (2.50x) / 3.14 | 125 (2.42x) / 3.03 | 124 (2.40x) / 3.00 | 112 (2.17x) / 2.72 | 2.42x / 3.05 |
| DFlash2 official (blk 8) | 223 (4.35x) / 5.60 | 238 (4.62x) / 5.74 | 198 (3.84x) / 4.70 | 193 (3.74x) / 4.57 | 175 (3.38x) / 4.15 | 3.99x / 4.95 |
| DFlash2 ckpt forced blk 16 | 246 (4.80x) / 6.56 | 270 (5.24x) / 6.82 | 210 (4.07x) / 5.23 | 202 (3.92x) / 5.05 | 178 (3.44x) / 4.44 | 4.29x / 5.62 |
| DSpine-b16, full-16 verify | 277 (5.40x) / 6.97 | 261 (5.06x) / 6.16 | 216 (4.19x) / 5.04 | 228 (4.42x) / 5.35 | 194 (3.76x) / 4.57 | 4.56x / 5.62 |
| DSpine-b16 + tuned SPS (dyn. verify) | 271 (5.30x) / 6.83 | 258 (5.02x) / 6.11 | 216 (4.18x) / 5.03 | 234 (4.54x) / 5.50 | 185 (3.58x) / 4.34 | 4.52x / 5.56 |
| DSpine-b16, verify width 8 (SPS-forced) | 228 (4.46x) / 5.56 | 206 (4.01x) / 4.75 | 193 (3.74x) / 4.39 | 199 (3.85x) / 4.58 | 175 (3.40x) / 4.02 | 3.89x / 4.66 |
| DSpark community (blk 7) | 211 (4.12x) / 5.50 | 215 (4.18x) / 5.39 | 175 (3.40x) / 4.34 | 168 (3.26x) / 4.16 | 149 (2.89x) / 3.71 | 3.57x / 4.62 |

### concurrency 1, T=1: output tok/s (speedup vs AR) | acceptance length (pooled tokens/verify-step)

| method | gsm8k | math500 | humaneval | mbpp | mt-bench | mean |
|---|---|---|---|---|---|---|
| AR | 51 (1.00x) | 51 (1.00x) | 51 (1.00x) | 51 (1.00x) | 51 (1.00x) | 1.00x |
| MTP (built-in head, 4 tok) | 125 (2.45x) / 3.14 | 118 (2.30x) / 2.91 | 106 (2.06x) / 2.60 | 104 (2.02x) / 2.54 | 99 (1.93x) / 2.44 | 2.15x / 2.72 |
| DFlash2 official (blk 8) | 206 (4.04x) / 5.08 | 206 (4.02x) / 4.99 | 178 (3.47x) / 4.25 | 174 (3.40x) / 4.17 | 166 (3.23x) / 3.98 | 3.63x / 4.49 |
| DFlash2 ckpt forced blk 16 | 256 (5.02x) / 6.87 | 217 (4.23x) / 5.51 | 179 (3.48x) / 4.50 | 192 (3.75x) / 4.86 | 155 (3.03x) / 3.92 | 3.90x / 5.13 |
| DSpine-b16, full-16 verify | 255 (5.01x) / 6.50 | 187 (3.65x) / 4.46 | 190 (3.69x) / 4.49 | 197 (3.83x) / 4.68 | 164 (3.20x) / 3.90 | 3.88x / 4.81 |
| DSpine-b16 + tuned SPS (dyn. verify) | 242 (4.75x) / 6.19 | 210 (4.09x) / 5.02 | 202 (3.93x) / 4.79 | 188 (3.66x) / 4.47 | 162 (3.16x) / 3.86 | 3.92x / 4.87 |
| DSpine-b16, verify width 8 (SPS-forced) | 211 (4.14x) / 5.17 | 180 (3.51x) / 4.21 | 173 (3.36x) / 4.01 | 179 (3.48x) / 4.13 | 151 (2.95x) / 3.51 | 3.49x / 4.21 |
| DSpark community (blk 7) | 197 (3.87x) / 5.11 | 182 (3.55x) / 4.56 | 157 (3.05x) / 3.90 | 167 (3.25x) / 4.19 | 144 (2.80x) / 3.60 | 3.30x / 4.27 |

### concurrency 8, T=0: output tok/s (speedup vs AR) | acceptance length (pooled tokens/verify-step)

| method | gsm8k | math500 | humaneval | mbpp | mt-bench | mean |
|---|---|---|---|---|---|---|
| AR | 175 (1.00x) | 278 (1.00x) | 314 (1.00x) | 250 (1.00x) | 290 (1.00x) | 1.00x |
| MTP (built-in head, 4 tok) | 433 (2.47x) / 3.35 | 666 (2.40x) / 3.27 | 721 (2.30x) / 3.01 | 597 (2.38x) / 2.95 | 652 (2.25x) / 2.82 | 2.36x / 3.08 |
| DFlash2 official (blk 8) | 988 (5.65x) / 5.62 | 930 (3.34x) / 4.94 | 910 (2.90x) / 4.46 | 980 (3.91x) / 4.54 | 796 (2.74x) / 3.83 | 3.71x / 4.68 |
| DFlash2 ckpt forced blk 16 | - | - | - | - | - | - |
| DSpine-b16, full-16 verify | 749 (4.28x) / 6.53 | 936 (3.37x) / 5.42 | 887 (2.82x) / 4.90 | 899 (3.59x) / 5.11 | 820 (2.83x) / 4.01 | 3.38x / 5.19 |
| DSpine-b16 + tuned SPS (dyn. verify) | 750 (4.29x) / 6.54 | 1087 (3.91x) / 5.67 | 972 (3.09x) / 5.07 | 990 (3.95x) / 5.12 | 852 (2.94x) / 4.11 | 3.64x / 5.30 |
| DSpine-b16, verify width 8 (SPS-forced) | 682 (3.90x) / 5.45 | 964 (3.47x) / 4.97 | 920 (2.93x) / 4.77 | 935 (3.73x) / 4.65 | 820 (2.83x) / 3.76 | 3.37x / 4.72 |
| DSpark community (blk 7) | 580 (3.31x) / 5.32 | 907 (3.26x) / 5.05 | 873 (2.78x) / 4.30 | 865 (3.45x) / 4.42 | 732 (2.52x) / 3.54 | 3.07x / 4.53 |

### concurrency 8, T=1: output tok/s (speedup vs AR) | acceptance length (pooled tokens/verify-step)

| method | gsm8k | math500 | humaneval | mbpp | mt-bench | mean |
|---|---|---|---|---|---|---|
| AR | 173 (1.00x) | 294 (1.00x) | 321 (1.00x) | 271 (1.00x) | 282 (1.00x) | 1.00x |
| MTP (built-in head, 4 tok) | 544 (3.15x) / 3.16 | 541 (1.84x) / 2.81 | 573 (1.78x) / 2.54 | 521 (1.92x) / 2.55 | 521 (1.84x) / 2.44 | 2.11x / 2.70 |
| DFlash2 official (blk 8) | 682 (3.95x) / 5.23 | 870 (2.96x) / 4.57 | 830 (2.58x) / 4.04 | 818 (3.02x) / 4.33 | 716 (2.54x) / 3.57 | 3.01x / 4.35 |
| DFlash2 ckpt forced blk 16 | - | - | - | - | - | - |
| DSpine-b16, full-16 verify | 582 (3.36x) / 5.81 | 837 (2.85x) / 4.70 | 793 (2.47x) / 4.39 | 737 (2.72x) / 4.42 | 705 (2.49x) / 3.53 | 2.78x / 4.57 |
| DSpine-b16 + tuned SPS (dyn. verify) | 617 (3.57x) / 5.48 | 837 (2.85x) / 4.33 | 781 (2.43x) / 4.15 | 821 (3.03x) / 4.29 | 690 (2.44x) / 3.45 | 2.86x / 4.34 |
| DSpine-b16, verify width 8 (SPS-forced) | 595 (3.44x) / 5.04 | 839 (2.86x) / 4.40 | 773 (2.40x) / 4.11 | 789 (2.91x) / 4.26 | 757 (2.68x) / 3.45 | 2.86x / 4.25 |
| DSpark community (blk 7) | 602 (3.48x) / 5.19 | 812 (2.77x) / 4.55 | 753 (2.34x) / 3.88 | 695 (2.56x) / 3.98 | 653 (2.31x) / 3.26 | 2.69x / 4.17 |

### concurrency 8, T=0, 64 prompts per cell (much less sensitive to the long-tail request): output tok/s (speedup vs AR) | acceptance length (pooled)

| method | gsm8k | math500 | humaneval | mbpp | mt-bench | mean |
|---|---|---|---|---|---|---|
| AR | 277 (1.00x) | 361 (1.00x) | 352 (1.00x) | 341 (1.00x) | 337 (1.00x) | 1.00x |
| MTP (built-in head, 4 tok) | 626 (2.26x) / 3.23 | 795 (2.20x) / 3.15 | 766 (2.17x) / 2.99 | 748 (2.19x) / 2.99 | 727 (2.16x) / 2.85 | 2.20x / 3.04 |
| DFlash2 official (blk 8) | 857 (3.09x) / 5.03 | 1142 (3.16x) / 4.88 | 1052 (2.98x) / 4.44 | 1037 (3.04x) / 4.61 | 878 (2.60x) / 3.74 | 2.98x / 4.54 |
| DSpine-b16, full-16 verify | 874 (3.16x) / 6.06 | 1206 (3.34x) / 5.30 | 1241 (3.52x) / 5.16 | 1218 (3.57x) / 5.46 | 943 (2.80x) / 4.07 | 3.28x / 5.21 |
| DSpine-b16 + tuned SPS (dyn. verify) | 881 (3.18x) / 6.08 | 1245 (3.45x) / 5.34 | 1235 (3.50x) / 5.15 | 1188 (3.48x) / 5.29 | 970 (2.88x) / 3.93 | 3.30x / 5.16 |
| DSpark community (blk 7) | 821 (2.96x) / 4.91 | 1100 (3.04x) / 4.78 | 989 (2.80x) / 4.30 | 992 (2.91x) / 4.55 | 825 (2.45x) / 3.58 | 2.83x / 4.42 |

### concurrency 8, T=1, 64 prompts per cell (much less sensitive to the long-tail request): output tok/s (speedup vs AR) | acceptance length (pooled)

| method | gsm8k | math500 | humaneval | mbpp | mt-bench | mean |
|---|---|---|---|---|---|---|
| AR | 289 (1.00x) | 367 (1.00x) | 357 (1.00x) | 346 (1.00x) | 356 (1.00x) | 1.00x |
| DFlash2 official (blk 8) | 887 (3.07x) / 4.77 | 1048 (2.86x) / 4.60 | 934 (2.62x) / 4.11 | 964 (2.78x) / 4.36 | 803 (2.26x) / 3.59 | 2.72x / 4.29 |
| DSpine-b16 + tuned SPS (dyn. verify) | 849 (2.94x) / 4.76 | 990 (2.70x) / 4.37 | 1048 (2.94x) / 4.19 | 972 (2.81x) / 4.39 | 849 (2.39x) / 3.52 | 2.75x / 4.25 |

### per-position acceptance P(draft position i accepted), pooled over datasets, T=0, c=1+8

- MTP (built-in head, 4 tok): 0.83 0.66 0.53
- DFlash2 official (blk 8): 0.84 0.69 0.56 0.46 0.39 0.32 0.28
- DFlash2 ckpt forced blk 16: 0.84 0.67 0.54 0.43 0.36 0.30 0.25 0.21 0.17 0.14 0.12 0.10 0.08 0.06 0.05
- DSpine-b16, full-16 verify: 0.81 0.63 0.50 0.39 0.31 0.25 0.21 0.18 0.15 0.13 0.11 0.10 0.09 0.08 0.07
- DSpine-b16 + tuned SPS (dyn. verify): 0.82 0.64 0.51 0.40 0.32 0.26 0.22 0.19 0.16 0.14 0.12 0.10 0.09 0.08 0.07
- DSpine-b16, verify width 8 (SPS-forced): 0.82 0.65 0.51 0.40 0.32 0.26 0.22 0.08 0.07 0.06 0.05 0.04 0.04 0.03 0.03
- DSpark community (blk 7): 0.82 0.66 0.53 0.43 0.36 0.30 0.25

### acceptance length truncated to verify width W (1 + sum of first W-1 per-position survival), pooled over the 5 datasets, concurrency 1 only (all methods have it)

Exact for causal/greedy drafters (DSpine: block-causal, so a width-8 verify accepts exactly min(accepted, 7)+1 tokens); a lower bound on what a retrained width-W drafter would reach.

| method | T | W=4 | W=8 | W=12 | W=16 | native |
|---|---|---|---|---|---|---|
| MTP (built-in head, 4 tok) | 0 | 3.01 | - | - | - | 3.01 (W=4) |
| MTP (built-in head, 4 tok) | 1 | 2.63 | - | - | - | 2.63 (W=4) |
| DFlash2 official (blk 8) | 0 | 3.15 | 4.76 | - | - | 4.76 (W=8) |
| DFlash2 official (blk 8) | 1 | 3.07 | 4.35 | - | - | 4.35 (W=8) |
| DFlash2 ckpt forced blk 16 | 0 | 3.04 | 4.38 | 5.03 | 5.31 | 5.31 (W=16) |
| DFlash2 ckpt forced blk 16 | 1 | 2.98 | 4.10 | 4.56 | 4.73 | 4.73 (W=16) |
| DSpine-b16, full-16 verify | 0 | 3.01 | 4.27 | 4.91 | 5.31 | 5.31 (W=16) |
| DSpine-b16, full-16 verify | 1 | 2.92 | 3.92 | 4.30 | 4.48 | 4.48 (W=16) |
| DSpine-b16 + tuned SPS (dyn. verify) | 0 | 2.98 | 4.22 | 4.86 | 5.26 | 5.26 (W=16) |
| DSpine-b16 + tuned SPS (dyn. verify) | 1 | 2.92 | 3.97 | 4.40 | 4.62 | 4.62 (W=16) |
| DSpine-b16, verify width 8 (SPS-forced) | 0 | 3.08 | 4.49 | - | - | 4.49 (W=8) |
| DSpine-b16, verify width 8 (SPS-forced) | 1 | 2.95 | 4.02 | - | - | 4.02 (W=8) |
| DSpark community (blk 7) | 0 | 3.01 | 4.37 | - | - | 4.37 (W=8) |
| DSpark community (blk 7) | 1 | 2.97 | 4.10 | - | - | 4.10 (W=8) |

### per-request-mean acceptance length (the model-card definition), mean of the c=1 and c=8 cells; N=8/24 prompts per cell

| method | T | gsm8k | math500 | humaneval | mbpp | mt-bench |
|---|---|---|---|---|---|---|
| MTP (built-in head, 4 tok) | 0 | 3.40 | 3.31 | 3.07 | 3.03 | 2.98 |
| MTP (built-in head, 4 tok) | 1 | 3.23 | 3.06 | 2.70 | 2.62 | 2.67 |
| DFlash2 official (blk 8) | 0 | 5.74 | 5.66 | 4.90 | 4.87 | 4.65 |
| (model card, xhigh, 4096 tok, H100/H200) | 0 | 5.69 | 5.49 | 4.87 | 5.04 | 4.11 |
| DFlash2 official (blk 8) | 1 | 5.59 | 5.22 | 4.44 | 4.56 | 4.33 |
| (model card, xhigh, 4096 tok, H100/H200) | 1 | 5.45 | 5.27 | 4.34 | 4.79 | 3.89 |
| DFlash2 ckpt forced blk 16 | 0 | 6.90 | 7.18 | 5.80 | 5.46 | 5.15 |
| DFlash2 ckpt forced blk 16 | 1 | 6.84 | 6.30 | 4.91 | 4.98 | 4.57 |
| DSpine-b16, full-16 verify | 0 | 7.21 | 6.48 | 5.51 | 5.75 | 5.24 |
| DSpine-b16, full-16 verify | 1 | 6.54 | 5.42 | 4.89 | 5.19 | 4.42 |
| DSpine-b16 + tuned SPS (dyn. verify) | 0 | 7.19 | 6.58 | 5.81 | 5.75 | 5.18 |
| (model card, xhigh, 4096 tok, H100/H200) | 0 | 7.20 | 6.49 | 5.71 | 6.14 | 4.48 |
| DSpine-b16 + tuned SPS (dyn. verify) | 1 | 6.17 | 5.26 | 4.76 | 4.81 | 4.49 |
| (model card, xhigh, 4096 tok, H100/H200) | 1 | 6.31 | 5.66 | 4.71 | 5.41 | 4.03 |
| DSpine-b16, verify width 8 (SPS-forced) | 0 | 5.87 | 5.41 | 5.11 | 5.12 | 4.71 |
| DSpine-b16, verify width 8 (SPS-forced) | 1 | 5.44 | 4.93 | 4.43 | 4.58 | 4.17 |
| DSpark community (blk 7) | 0 | 5.79 | 5.48 | 4.67 | 4.60 | 4.32 |
| DSpark community (blk 7) | 1 | 5.46 | 5.08 | 4.16 | 4.50 | 4.09 |

## 4. Findings

1. **Model-card acceptance reproduces.** Per-request-mean acceptance at T=0 for DSpine (SPS) is 7.19/6.58/5.81/5.75/5.18 vs card 7.20/6.49/5.71/6.14/4.48 (GSM8K/MATH/HumanEval/MBPP/MT-Bench); DFlash2 5.74/5.66/4.90/4.87/4.65 vs 5.69/5.49/4.87/5.04/4.11 (see table; T=1 also within ~0.3-0.5). The engine, plugin and checkpoints work as documented.
2. **Ranking on one H100, thinking xhigh, 2048 tokens.** Speedup over AR (mean of 5 datasets): c=1 T=0: DSpine 4.56x (full-16) / 4.52x (tuned SPS), DFlash2 3.99x, DSpark(block 7) 3.57x, MTP 2.42x. c=1 T=1: DSpine+SPS 3.92x, DSpine 3.88x, DFlash2 3.63x, DSpark 3.30x, MTP 2.15x.
   c=8 (N=64) T=0: DSpine+SPS 3.30x, DSpine 3.28x, DFlash2 2.98x, DSpark 2.83x, MTP 2.20x. c=8 T=1: DSpine+SPS 2.75x vs DFlash2 2.72x (a tie; DSpine is ahead on code, behind on GSM8K/MATH).
   So DSpine's advantage over DFlash2 is +14% at c=1/T=0, +8% at c=1/T=1, +11% at c=8/T=0, ~0% at c=8/T=1. The ordering is stable but the margins are small relative to a single-seed, 8-64 prompt measurement.
3. **Matched block size changes the story.** DSpine's acceptance gain over DFlash2 is mostly a block-length effect. At verify width 8 DSpine accepts *less* than the official DFlash2-b8: 4.49 vs 4.76 (c=1, T=0, hist-based), and the width-8 DSpine run is 3.89x vs DFlash2 3.99x (c=1 T=0), 3.49x vs 3.63x (T=1).
   Conversely, the DFlash2 checkpoint, although trained at block 8, accepts 5.31 tokens when asked for 16 (hist-based, c=1, T=0), identical to DSpine's 5.31, and gets 4.29x vs DSpine's 4.56x at c=1 T=0; at T=1 it is 3.90x vs 3.88x (a tie). So at matched block 16 and c=1 the two are close in acceptance and DSpine is ahead by ~6% only at T=0 (probably because DFlash2's per-round draft cost is higher: selector + dynamic convs).
   The DFlash2-b16 row is out-of-recipe (not trained at 16, c=1 only, 2 slots) and should not be read as an official number; a fair "DFlash2 retrained at 16" is Phase 2 work.
   DSpine's real contribution is visible in the per-position survival: it decays more slowly in the tail (P(pos 15)=0.07 vs DFlash2-forced 0.05; P(pos 8) 0.18 vs 0.21 is the exception) but starts lower (P(pos 1)=0.81 vs 0.84-0.87).
4. **Dynamic verification (SPS) buys ~nothing on an H100 at 8 streams.** The tuner measurements (`sps/measurements_T0.json`) show width 16 is best or within 2% of best at every batch size 1-8 (e.g. BS1: 229 tok/s at 16 vs 198 at 8; BS8: 958 vs 909 at 8, 966 at 12), because verifying 16 tokens of a 27B model is still memory-bound on this GPU. The tuned policy (all 16 except two 12s, batch8 budget 96/128 at T=0) therefore changes throughput by -1% to +3%:
   c=1 T=0 4.52x vs 4.56x, c=8 T=0 3.30x vs 3.28x, first-pass c=8 T=1 (24 prompts) 2.86x with vs 2.78x without; the N=64 c=8 T=1 run exists only for the tuned profile. The model card's gains from dynamic verification likely need a larger batch or a more compute-bound GPU; we did not test BS > 8.
   Width 8 without the confidence allocation is clearly worse (3.89x vs 4.56x at c=1 T=0), so the DFlash2-style short block is the weaker operating point for a 16-block drafter.
5. **Community DSpark (block 7, verify 8)** lands between MTP and DFlash2: 3.57x at c=1 T=0, 2.83x at c=8 T=0. Its acceptance (4.62 pooled c=1 T=0) is below the official DFlash2-b8 (4.95), consistent with the DFlash2 card.
6. **MTP** (built-in head, 4 draft tokens) gives a robust 2.2-2.4x at T=0 and 2.1-2.2x at T=1 but accepts only ~3.0 tokens.
7. **Absolute speed.** DFlash2 reaches ~223-238 tok/s on GSM8K/MATH at c=1 on H100 (card, H200: 236/231); DSpine up to 277 tok/s (GSM8K). At c=8 the aggregate is 0.85-1.25k tok/s.

## 5. Caveats

- Single seed, T=0 is not bit-deterministic across methods (different kernels), so outputs, lengths and prompts-in-flight differ slightly; speedups are per cell ratios of aggregate throughput, not of identical token sequences. N is small (8 prompts at c=1, 24/64 at c=8); expect +-5-8% noise per cell. Cell means over 5 datasets are less noisy.
- Thinking mode xhigh + 2048-token cap truncates 2-43% of generations; the cards use 4096. Acceptance on truncated reasoning traces differs from complete outputs. Runs with thinking off (e.g. the Qwen3-8B work) are not comparable with these.
- H100 vs the cards' H200 (DFlash2) and unknown hardware (DSpine); our AR is 51.5 tok/s at c=1 vs 68.9 on H200.
- DSpine ran with radix cache on and 40 Mamba slots at mem-fraction 0.86; the others with radix cache off and 0.93. The first DSpine GSM8K c=8 cell (24 prompts) was low in both of its runs (750 tok/s) because of one long request; ignore the 24-prompt c=8 numbers.
- SPS tuning used only 20 prefixes, T=1 tuning was reduced (widths 8,16, one repeat), and tuning ran at a different time than evaluation. Width 16 was nearly always optimal so this is unlikely to matter.
- "DSpine width 8" uses `batch8_tokens 64`, which at BS 8 is a confidence-allocated budget rather than a strict uniform 8.
- DFlash2 block-16 rows are c=1 only, and the draft was never trained at that length.
- EAGLE-4 (vendor checkpoint) and the community DSpark/DFlash2 variants other than RadixArk's DSpark were not run. No EAGLE-3 checkpoint exists for this target.
- Draft/verify timing split (T_draft vs T_verify) was not logged here; `bench/patches/apply_timing_patch.py` from the Qwen3-8B work could be applied to this setup for that.

## 6. Setup notes (for reproducing on a plain pytorch/cuda12.4 image)

- `uv venv` + `uv pip install -e DSpine` pulls sglang 0.5.22 / torch 2.14.1+cu130 / flashinfer 0.7.1 (the driver on the box supports CUDA 13.4).
- JIT kernels (deep_gemm, sgl-kernel, flashinfer top-k) need nvcc >= 12.9 and matching headers: the image's nvcc is 12.4, so use the pip `nvidia-cuda-nvcc==13.0.88` + `nvidia-cuda-crt==13.0.88` + `nvidia-nvvm==13.0.88` (must match the 13.0.96 runtime headers; mixing 13.5 nvcc with 13.0 headers fails),
  `CUDA_HOME=<venv>/site-packages/nvidia/cu13` and symlinks `lib64 -> lib` and `lib/libcudart.so -> libcudart.so.13`. First launch of each mode spends 5-9 minutes compiling the flashinfer top-k module (cached afterwards).
- Memory: with a 54 GB target the intermediate-SSM cache of non-ReplaySSM spec modes (10 GB at 8 slots x 8 tokens) leaves almost no KV at `--mem-fraction-static 0.85` (2k tokens, silently serialising concurrency); 0.93 fixes it. DSpine's ReplaySSM avoids that cache but its plugin OOMs at 0.90 with 40 slots (0.86 works).
- The tuner (`tune_sps.py`) needs the full `sglang serve` command with `--speculative-algorithm DSPINE` and without host/port flags; `serve.sh` has a `DRYRUN=1` mode that prints it.
