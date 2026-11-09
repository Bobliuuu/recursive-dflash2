#!/bin/bash
# H100 Qwen3.8-27B c=1 T=0 queue (box). Log: /workspace/p5/queue27b.log
R=/workspace/recursive-dflash2/bench/p5h100/run27b.sh
IFS='|' read -ra JOBS <<< "${1:-tree t27_tree32 32:31|chain t27_chain16 16|rssm t27_rssm16|tree t27_tree48 48:31|ar t27_ar}"
for j in "${JOBS[@]}"; do echo "== $j"; bash $R $j; done
echo QUEUE DONE
