#!/bin/bash
# Phase B queue on spark2: tree width x depth sweep (20 prompts/dataset = the first half of the 40-prompt set), GEMM micro-sweep,
# teacher-forced losslessness checks. Several suites share one GPU-lock hold (<= ~50 min per hold).
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_b.log
ALL="gsm8k math500 humaneval mbpp mtbench"
hold() { echo "=== hold: $1 $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
hold "python $B/gb10/linear_sweep.py $B/results/gb10/linear_sweep.json; bash $S tree tree64_d31 '$ALL' 20 64 31; bash $S tree tree32_d31 '$ALL' 20 32 31; bash $S tree tree96_d31 '$ALL' 20 96 31"
hold "bash $S tree tree96_d47 '$ALL' 20 96 47; bash $S tree tree16_d15 '$ALL' 20 16 15; bash $S tree tree128_d47 '$ALL' 20 128 47"
hold "python $B/p2_tf_check.py $B/results/gb10/tf_check_a.json gb10/dspine_b16:gsm8k gb10/dspine_b16:mtbench gb10/tree48_d31:gsm8k gb10/tree48_d31:math500 gb10/tree48_d31:humaneval gb10/tree48_d31:mbpp gb10/tree48_d31:mtbench gb10/tree64_d31:gsm8k gb10/tree64_d31:mtbench gb10/tree96_d31:gsm8k gb10/tree96_d31:mtbench"
echo "QUEUE_B DONE $(date)" >> $Q
