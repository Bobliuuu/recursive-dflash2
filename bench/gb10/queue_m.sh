#!/bin/bash
# Master queue on spark2 (replaces queue_b2/c/e, whose 2-hour flock waits timed out on the shared box). One GPU-lock hold per line;
# a lock timeout (flock exit 75) is retried, so nothing is skipped.
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_m.log
ALL="gsm8k math500 humaneval mbpp mtbench"
hold() { echo "=== hold: $1 $(date)" >> $Q
  while true; do flock -E 75 -w 3600 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; rc=$?; [ $rc = 75 ] || break; echo "lock timeout, retry $(date)" >> $Q; done
  echo "=== exit $rc $(date)" >> $Q; }
F8="export TARGET=Qwen/Qwen3-8B-FP8"
F4="export TARGET=nvidia/Qwen3-8B-NVFP4 EXTRA='--quantization modelopt_fp4'"
FD="--speculative-draft-model-quantization fp8"
TF="python $B/p2_tf_check.py"
hold "EXTRA='$FD' bash $S treefast f_tree32_d31_fp8draft '$ALL' 20 32 31; $TF $B/results/gb10/tf_check_e.json gb10/f40_tree32_d31:gsm8k gb10/f40_tree32_d31:math500 gb10/f40_tree32_d31:humaneval gb10/f40_tree32_d31:mbpp gb10/f40_tree32_d31:mtbench gb10/dspine_b16:gsm8k gb10/dspine_b16:mtbench gb10/f_tree32_d31_fp8draft:gsm8k gb10/f_tree32_d31_fp8draft:mtbench"
hold "$F8; WARMUP=1 bash $S ar fp8_ar '$ALL' 8; bash $S treefast fp8_f_tree32_d31 '$ALL' 20 32 31"
hold "$F8; bash $S dspine fp8_dspine_b16 '$ALL' 20; bash $S treefast fp8_f_tree48_d31 '$ALL' 20 48 31; EXTRA='$FD' bash $S treefast fp8_f_tree32_d31_fp8draft '$ALL' 20 32 31"
hold "$F4; WARMUP=1 bash $S ar nvfp4_ar '$ALL' 8; bash $S treefast nvfp4_f_tree32_d31 '$ALL' 20 32 31"
hold "$F4; bash $S dspine nvfp4_dspine_b16 '$ALL' 20; bash $S treefast nvfp4_f_tree48_d31 '$ALL' 20 48 31; EXTRA=\"\$EXTRA $FD\" bash $S treefast nvfp4_f_tree32_d31_fp8draft '$ALL' 20 32 31"
hold "bash $S treefast f_tree16_d15 '$ALL' 20 16 15; bash $S treefast f_tree32_d15 '$ALL' 20 32 15; bash $S treefast f_tree32_d23 '$ALL' 20 32 23; $TF $B/results/gb10/tf_check_a.json gb10/tree48_d31:gsm8k gb10/tree48_d31:math500 gb10/tree48_d31:humaneval gb10/tree48_d31:mbpp gb10/tree48_d31:mtbench gb10/f_tree48_d31:gsm8k gb10/f_tree48_d31:mtbench gb10/p_tree64_d31:gsm8k gb10/p_tree64_d31:mtbench gb10/dspine_b32:gsm8k gb10/dflash_b16:gsm8k"
echo "QUEUE_M DONE $(date)" >> $Q
