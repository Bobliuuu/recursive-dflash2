| Path | What |
|---|---|
| `docs/research/` | DSpine internals, prior art on recursive and stacked drafters, benchmark design |
| `docs/step0-findings.md` | draft vs verify timing, per-position acceptance, full-block accept rate |
| `docs/phase1-*.md` | the benchmarks above |
| `docs/phase2-findings.md` | tree verification, longer blocks, confidence gating, n-gram tail |
| `docs/phase3-findings.md` | training experiments and the recursive second pass |
| `docs/phase4-gb10-spark.md` | DGX Spark (GB10): DSpine tree 32 nodes 7.80x AR on bf16 (A100 best 7.20x); with an FP8 / NVFP4 target 126 / 157 tok/s (11.7x / 14.6x bf16 AR) |
| `docs/dflash2-architecture.md` | DFlash2 architecture and its released config |
| `bench/` | serving and benchmark scripts, SGLang patches, raw results (`bench/results/`) |
| `bench/p2tree/` | tree build, tree-attention mask, accept and KV-compaction kernels |
| `train/` | DSpine trainer (SpecForge model and loss, in-process target) and the pass-2 experiment |
| [docs/PLAN.md](docs/PLAN.md) | goals and plan |
