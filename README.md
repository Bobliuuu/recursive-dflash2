# recursive-dflash2 ⚡

**How fast can speculative decoding go, and does stacking drafters help?**
A head-to-head benchmark of DFlash, DFlash2, DSpine and EAGLE-3, plus a tree-verification
upgrade that makes the best drafter **~20% faster** with no retraining.

```
Qwen3-8B · A100 · batch 1 · greedy · speedup over plain autoregressive decoding

autoregressive  ████ 1.0x
EAGLE-3 (chain) ████████ 2.0x
DFlash          ██████████████████ 4.5x
DSpine          ███████████████████████ 5.8x
DSpine + tree   ████████████████████████████ 6.9x   <- this repo
```

## TL;DR

- **DSpine is the best single drafter.** It beats DFlash on Qwen3-8B at every temperature and concurrency we ran, and the official DFlash2 on Qwen3.8-27B (no DFlash checkpoint exists for that model).
- **Tree verification on DSpine gives +19.7% tok/s** (6.92x vs 5.79x of its own chain) at batch 1. DSpine already scores 16 candidates per position conditioned on the previous position, so the tree is built from exact conditional logits each round. Verify costs 11.05 ms at 16 tokens and 11.91 ms at 48, so extra branches are nearly free on an A100.
- **Published acceptance numbers reproduce, with a catch.** Our runs match the DSpine README (7.97 vs 6.08 mean acceptance length against DFlash), not the paper's smaller retrained-baseline table (4.82 vs 3.77). Always read the protocol next to the number.
- **Training a longer-block drafter and a recursive second pass did not beat the tree** (see [Phase 3](docs/phase3-findings.md)).

## Benchmark results

Everything below is from one engine build (SGLang 0.5.22), a frozen prompt set and raw per-request JSON in `bench/results/`.

### Qwen3-8B (1x A100-80GB, thinking off, mean over GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench)

| T | concurrency | EAGLE-3 (chain) | DFlash | DSpine |
|---|---|---|---|---|
| 0 | 1 | 2.03x | 4.50x | **5.75x** |
| 0 | 8 | 1.82x | 3.40x | **4.11x** |
| 1 | 1 | 1.95x | 3.57x | **4.85x** |
| 1 | 8 | 1.70x | 2.76x | **3.42x** |

Mean acceptance length at T=0: DSpine 7.99, DFlash 6.03. Full table and protocol: [docs/phase1-qwen3-8b.md](docs/phase1-qwen3-8b.md).

### Qwen3.8-27B (1x H100-80GB, thinking on, 64 prompts per concurrency-8 cell)

| Method | c=1 T=0 | c=1 T=1 | c=8 T=0 | c=8 T=1 |
|---|---|---|---|---|
| DSpine + tuned SPS | 4.52x | 3.92x | 3.30x | 2.75x |
| DFlash2 (official, block 8) | 3.99x | 3.63x | 2.98x | 2.72x |
| DSpark (community, block 7) | 3.57x | 3.30x | 2.83x | n/a |
| MTP (built-in head) | 2.42x | 2.15x | 2.20x | n/a |

Most of DSpine's edge over DFlash2 is block length: at matched verify width 8 it is slightly behind. Details and caveats: [docs/phase1-qwen3.8-27b.md](docs/phase1-qwen3.8-27b.md).

### Tree verification (Qwen3-8B, A100, batch 1, T=0)

| Config | Acceptance length | tok/s | x AR |
|---|---|---|---|
| DSpine b16 chain | 7.99 | 523 | 5.78x |
| DSpine b32 chain (same weights) | 9.03 | 557 | 6.15x |
| **DSpine tree, 48 nodes** | **11.29** | **627** | **6.9x** |

The tree path, CUDA kernels and SGLang patches live in `bench/p2tree/`. Write-up: [docs/phase2-findings.md](docs/phase2-findings.md).

## How to read these numbers

- Speedups are same-cell ratios to autoregressive decoding, with run-to-run noise of about 1-2% (more at T=1, which is single-seed).
- The tree gain exists at **batch 1, greedy** only. At concurrency 8 verify is no longer free, so wider trees and longer blocks lose. The tree needs multi-draft rejection sampling for T=1, which is not implemented.
- Our EAGLE-3 baseline is a 3-step chain. The DFlash paper's EAGLE-3 uses a tree, so ours is the weaker one.
- Speculative output is not byte-identical to autoregressive at T=0 (bf16 numerics at a 16-token verify shape). DFlash and DSpine produce identical text to each other, and GSM8K accuracy is equal across methods.
- 8B runs use 40 prompts per dataset at concurrency 1 (80 at 8). LiveCodeBench and Arena-Hard were not run.

## Repo map

| Path | What |
|---|---|
| `docs/research/` | DSpine internals, prior art on recursive and stacked drafters, benchmark design |
| `docs/step0-findings.md` | draft vs verify timing, per-position acceptance, full-block accept rate |
| `docs/phase1-*.md` | the benchmarks above |
| `docs/phase2-findings.md` | tree verification, longer blocks, confidence gating, n-gram tail |
| `docs/phase3-findings.md` | training experiments and the recursive second pass |
| `docs/dflash2-architecture.md` | DFlash2 architecture and its released config |
| `bench/` | serving and benchmark scripts, SGLang patches, raw results (`bench/results/`) |
| `bench/p2tree/` | tree build, tree-attention mask, accept and KV-compaction kernels |
| `train/` | DSpine trainer (SpecForge model and loss, in-process target) and the pass-2 experiment |
| [PLAN.md](PLAN.md) | goals and plan |

## What we tried

We also tested longer-block training and a recursive second drafting pass on top of DSpine; neither beat tree verification, details in [docs/phase3-findings.md](docs/phase3-findings.md).

## Credits

DFlash ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036), z-lab), DSpine ([Tencent/DSpine](https://github.com/Tencent/DSpine), arXiv 2609.36173), DFlash2 (Inco AI), SGLang and SpecForge.
