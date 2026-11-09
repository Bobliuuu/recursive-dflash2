#!/bin/bash
# Queue B2 on spark2 (replaces the unstarted holds of queue_b.sh after the GEMM sweep showed the 32-row cliff):
# small trees and depth at 32 nodes, then teacher-forced checks for the queue A/B cells. One GPU-lock hold.
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_b.log
ALL="gsm8k math500 humaneval mbpp mtbench"
hold() { echo "=== hold: $1 $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
hold "bash $S treefast f_tree16_d15 '$ALL' 20 16 15; bash $S treefast f_tree32_d15 '$ALL' 20 32 15; bash $S treefast f_tree32_d23 '$ALL' 20 32 23; python $B/p2_tf_check.py $B/results/gb10/tf_check_a.json gb10/tree48_d31:gsm8k gb10/tree48_d31:math500 gb10/tree48_d31:humaneval gb10/tree48_d31:mbpp gb10/tree48_d31:mtbench gb10/f_tree48_d31:gsm8k gb10/f_tree48_d31:mtbench gb10/f_tree64_d31:gsm8k gb10/f_tree64_d31:mtbench gb10/dspine_b32:gsm8k gb10/dflash_b16:gsm8k"
echo "QUEUE_B DONE $(date)" >> $Q
