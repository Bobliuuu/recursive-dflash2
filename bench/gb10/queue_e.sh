#!/bin/bash
# Queue E on spark2: the GB10 sweet spot (32-node tree: every target GEMM stays in cuBLAS's fast <=32-row regime) at 40 prompts,
# 24 nodes, FP8 drafter at 32 nodes, and the teacher-forced check of the 32-node outputs.
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_e.log
ALL="gsm8k math500 humaneval mbpp mtbench"
hold() { echo "=== hold: $1 $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
hold "bash $S treefast f40_tree32_d31 '$ALL' 40 32 31; bash $S treefast f_tree24_d23 '$ALL' 20 24 23"
hold "EXTRA='--speculative-draft-model-quantization fp8' bash $S treefast f_tree32_d31_fp8draft '$ALL' 20 32 31; python $B/p2_tf_check.py $B/results/gb10/tf_check_e.json gb10/f40_tree32_d31:gsm8k gb10/f40_tree32_d31:math500 gb10/f40_tree32_d31:humaneval gb10/f40_tree32_d31:mbpp gb10/f40_tree32_d31:mtbench gb10/dspine_b16:gsm8k gb10/dspine_b16:mtbench"
echo "QUEUE_E DONE $(date)" >> $Q
