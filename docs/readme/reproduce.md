Needs one NVIDIA GPU (A100-80GB tested; H100 and GB10 also run), CUDA 13 toolchain (installed by the script) and about 60 GB of disk. The stack is pinned: SGLang 0.5.22, torch 2.14.1+cu130, flashinfer 0.7.1, [Tencent/DSpine](https://github.com/Tencent/DSpine) for the drafter weights.

```bash
# 1. fresh box: clone to /workspace/recursive-dflash2, then install the stack, download Qwen3-8B + DSpine-b16,
#    and build the block-16/32/48 draft dirs (same weights, longer block)
git clone https://github.com/Bobliuuu/recursive-dflash2 /workspace/recursive-dflash2
bash /workspace/recursive-dflash2/bench/setup_box.sh

# 2. baseline: stock DSpine chain, then the benchmark cell (GSM8K, greedy, batch 1)
bash bench/p4_serve.sh chain
python bench/run_cell.py --method dspine_chain --dataset gsm8k --temp 0 --conc 1 --n 40 --block 16 --out /tmp/chain_gsm8k.json

# 3. the upgrade: same weights, 48-node tree verification (patches SGLang in place; --revert undoes it)
RDF2_TABLE=factored FIMASK=skip bash bench/p4_serve.sh fast 48 31
python bench/run_cell.py --method dspine_tree --dataset gsm8k --temp 0 --conc 1 --n 40 --block 48 --out /tmp/tree_gsm8k.json
```

Each cell prints acceptance length and tok/s. Compare with the committed raw results in `bench/results/` (`p1_*` for baselines, `p4_*` for the tree) and `python bench/p4_summarize.py <prefix>` for the table. Add `RDF2_TREE_T1_MODE=rrs` and `--temp 1` for the lossless sampled tree. Scripts take no hostnames: run them on the box itself.
