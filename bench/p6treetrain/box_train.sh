#!/bin/bash
# Phase 6: official-weights proxy, then V1 (mix1) and V2 (mix4). Outputs /workspace/p6/<tag>/{proxy.json,train.log}
source /workspace/env.sh
cd /workspace/recursive-dflash2/train
D=/workspace/p6/data.jsonl; O=/workspace/p6
[ -f $O/official/proxy.json ] || python train_treeaware.py --data $D --out $O/official --eval-only > $O/official.log 2>&1
for spec in ${VARIANTS:-mix1:1.0 mix4:4.0}; do
  tag=${spec%%:*}; lam=${spec##*:}
  [ -f $O/$tag/final/step.json ] && continue
  python train_treeaware.py --data $D --out $O/$tag --tree-lambda $lam > $O/$tag.log 2>&1
done
echo TRAIN_ALL_DONE > $O/TRAIN_DONE
