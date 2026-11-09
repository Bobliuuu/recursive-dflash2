#!/bin/bash
# Phase B queue on spark2: tree width x depth sweep, GEMM micro-sweep, teacher-forced losslessness checks. One lock per job.
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_b.log
ALL="gsm8k math500 humaneval mbpp mtbench"
job() { echo "=== $* $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash $S "$@" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
lk() { echo "=== $* $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock "$@" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
lk python $B/gb10/linear_sweep.py $B/results/gb10/linear_sweep.json
job tree tree64_d31 "$ALL" 40 64 31
job tree tree96_d31 "$ALL" 40 96 31
job tree tree32_d31 "$ALL" 40 32 31
job tree tree96_d47 "$ALL" 40 96 47
job tree tree16_d15 "$ALL" 40 16 15
lk python $B/p2_tf_check.py $B/results/gb10/tf_check_a.json gb10/dspine_b16:gsm8k gb10/dspine_b16:mtbench gb10/tree48_d31:gsm8k gb10/tree48_d31:math500 gb10/tree48_d31:humaneval gb10/tree48_d31:mbpp gb10/tree48_d31:mtbench
echo "QUEUE_B DONE $(date)" >> $Q
