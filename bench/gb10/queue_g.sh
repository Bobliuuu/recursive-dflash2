#!/bin/bash
# Queue G on spark2 (after queue_f): SGLang-prefill teacher-forced checks for bf16 (calibration against the HF check), FP8 and NVFP4 cells.
R=${R:-$HOME/rdf2}; source $R/env.sh; B=$R/recursive-dflash2/bench; T=$B/gb10/sgl_tf_run.sh; O=$B/results/gb10; Q=$R/logs/queue_g.log
hold() { echo "=== hold: $1 $(date)" >> $Q
  while true; do flock -E 75 -w 3600 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; rc=$?; [ $rc = 75 ] || break; echo "lock timeout, retry $(date)" >> $Q; done
  echo "=== exit $rc $(date)" >> $Q; }
sp() { for d in gsm8k math500 humaneval mbpp mtbench; do echo -n "gb10/$1:$d "; done; }
hold "bash $T $O/sgltf_bf16.json $(sp f40_tree32_d31) gb10/dspine_b16:gsm8k gb10/dspine_b16:mtbench gb10/ar:gsm8k gb10/ar:mtbench; \
TARGET=Qwen/Qwen3-8B-FP8 bash $T $O/sgltf_fp8.json $(sp fp8_f_tree32_d31) gb10/fp8_f_tree48_d31:gsm8k gb10/fp8_dspine_b16:gsm8k gb10/fp8_ar:gsm8k gb10/fp8_ar:mtbench; \
TARGET=nvidia/Qwen3-8B-NVFP4 EXTRA='--quantization modelopt_fp4' bash $T $O/sgltf_nvfp4.json $(sp nvfp4_f_tree32_d31) gb10/nvfp4_f_tree48_d31:gsm8k gb10/nvfp4_dspine_b16:gsm8k gb10/nvfp4_ar:gsm8k gb10/nvfp4_ar:mtbench"
hold "python $B/p2_tf_check.py $O/tf_check_sweep.json $(for t in f_tree16_d15 f_tree24_d23 f_tree32_d15 f_tree32_d23 f_tree32_d31 f_tree64_d31 p_tree48_d31 p_tree96_d31; do echo -n "gb10/$t:gsm8k gb10/$t:mtbench "; done)"
echo "QUEUE_G DONE $(date)" >> $Q
