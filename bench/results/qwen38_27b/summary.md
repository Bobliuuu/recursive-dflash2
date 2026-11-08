
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