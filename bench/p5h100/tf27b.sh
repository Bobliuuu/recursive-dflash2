#!/bin/bash
# Start the 27B AR server (no speculation) and teacher-force the given result files against it. Usage: tf27b.sh OUT.json TAG...
OUT=$1; shift
B=/workspace/recursive-dflash2/bench; R=$B/results/p5h100/q27b
DSS=serve bash $B/p5h100/run27b.sh ar t27_ar_tfsrv || exit 1
source /workspace/env.sh; A=""
for t in "$@"; do for d in gsm8k math500 humaneval mbpp mt-bench; do [ -f $R/${t}__${d}__T0__c1.json ] && A="$A $R/${t}__${d}__T0__c1.json:8"; done; done
python $B/p5h100/tf27b.py $OUT $A
pkill -f "[s]glang serve"
