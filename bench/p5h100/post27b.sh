#!/bin/bash
# After the 27B queue (box): Tree-ReplaySSM runs (TRS=1) at N=32/48, then the teacher-forced check of all tree configs.
B=/workspace/recursive-dflash2/bench; source /workspace/env.sh
TRS=1 bash $B/p5h100/run27b.sh tree t27_trs32 32:31
TRS=1 bash $B/p5h100/run27b.sh tree t27_trs48 48:31
bash $B/p5h100/tf27b.sh $B/results/p5h100/q27b/tf_check_27b.json t27_rssm16 t27_tree24 t27_tree32 t27_tree48 t27_trs32 t27_trs48
echo POST DONE
