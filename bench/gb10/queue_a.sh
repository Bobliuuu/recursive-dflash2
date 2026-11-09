#!/bin/bash
# Phase A queue on spark2: each job takes the shared GPU lock separately (<= ~40 min each) so other jobs can interleave.
R=${R:-$HOME/rdf2}; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; Q=$R/logs/queue_a.log
job() { echo "=== $* $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash $S "$@" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
WARMUP=1 job ar ar "gsm8k math500 humaneval mbpp mtbench" 8
job dspine dspine_b16 "gsm8k math500 humaneval mbpp mtbench" 40
job tree tree48_d31 "gsm8k math500 humaneval mbpp mtbench" 40 48 31
job dflash dflash_b16 "gsm8k math500 humaneval mbpp mtbench" 40
job dspine_blk dspine_b32 "gsm8k math500 humaneval mbpp mtbench" 40 32
echo "QUEUE_A DONE $(date)" >> $Q
