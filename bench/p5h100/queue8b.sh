#!/bin/bash
# H100 Qwen3-8B sweep queue (box): tree48 sanity, chains, tree node-count sweep, AR. Log: /workspace/p5/queue8b.log
R=/workspace/recursive-dflash2/bench/p5h100/run8b.sh
bash $R tree tree48_d31 48:31
bash $R chain chain16 16
bash $R chain chain48 48
bash $R tree tree32_d31 32:31
bash $R tree tree64_d31 64:31
bash $R tree tree96_d31 96:31
bash $R chain chain32 32
bash $R ar ar
echo QUEUE DONE
