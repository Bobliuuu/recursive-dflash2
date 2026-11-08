#!/bin/bash
# Tree verify sweep (c=1, T=0, 5 datasets x 40): "N:D" specs
for spec in "$@"; do
  N=${spec%%:*}; D=${spec##*:}
  bash /workspace/recursive-dflash2/bench/p2_tree_run.sh $N $D "gsm8k math500 humaneval mbpp mtbench" 40 tree${N}_d${D}
done
echo done > /workspace/p2/tree_sweep_done
