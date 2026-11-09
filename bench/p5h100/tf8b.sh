#!/bin/bash
# Teacher-forced lossless check (bench/p2_tf_check.py, HF Qwen3-8B argmax) for the H100 8B configs, 8 requests per dataset. Server must be stopped.
source /workspace/env.sh; pkill -f "[s]glang serve"; sleep 5
B=/workspace/recursive-dflash2/bench; A=""
for t in ${TAGS:-chain16 tree48_d31 tree64_d31 tree96_d31}; do for d in gsm8k math500 humaneval mbpp mtbench; do A="$A p5h100/$t:$d:8"; done; done
python $B/p2_tf_check.py $B/results/p5h100/tf_check_8b.json $A
