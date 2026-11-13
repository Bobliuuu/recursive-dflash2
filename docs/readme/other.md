### Qwen3.8-27B (1x H100-80GB, thinking on, 64 prompts per concurrency-8 cell)

| Method | c=1 T=0 | c=1 T=1 | c=8 T=0 | c=8 T=1 |
|---|---|---|---|---|
| DSpine + tuned SPS | 4.52x | 3.92x | 3.30x | 2.75x |
| DFlash2 (official, block 8) | 3.99x | 3.63x | 2.98x | 2.72x |
| DSpark (community, block 7) | 3.57x | 3.30x | 2.83x | n/a |
| MTP (built-in head) | 2.42x | 2.15x | 2.20x | n/a |

Most of DSpine's edge over DFlash2 is block length: at matched verify width 8 it is slightly behind. Details and caveats: [docs/phase1-qwen3.8-27b.md](docs/phase1-qwen3.8-27b.md).

![Tree verification vs the same drafter's chain across hardware and sampling](docs/img/tree_gain.png)

Other hardware and sampling, same tree:

| Setup | Result | Doc |
|---|---|---|
| Qwen3-8B, H100, 64-node tree | 6.49x AR, +27% over the stock chain | [phase4-h100-27b.md](docs/phase4-h100-27b.md) |
| Qwen3-8B, DGX Spark (GB10), 32-node tree | 7.80x AR (bf16); 157 tok/s with an NVFP4 target | [phase4-gb10-spark.md](docs/phase4-gb10-spark.md) |
| Qwen3-8B, A100, **T=1** lossless sampled tree | 5.43x AR, +12% over the chain | [phase4-t1-tree.md](docs/phase4-t1-tree.md) |
| Qwen3.8-27B hybrid, H100 | 4.95x AR, +5.7% over the chain | [phase4-h100-27b.md](docs/phase4-h100-27b.md) |
