#!/bin/bash
# Queue F on spark2 (after queue_m): teacher-forced checks of the quantized-target cells against the HF quantized target.
# The SGLang FP8 AR cell is checked the same way, as the noise floor (HF and SGLang FP8 GEMMs differ numerically).
R=${R:-$HOME/rdf2}; source $R/env.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_f.log
hold() { echo "=== hold: $1 $(date)" >> $Q
  while true; do flock -E 75 -w 3600 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; rc=$?; [ $rc = 75 ] || break; echo "lock timeout, retry $(date)" >> $Q; done
  echo "=== exit $rc $(date)" >> $Q; }
S=$R/recursive-dflash2/bench/gb10/run_suite.sh; ALL="gsm8k math500 humaneval mbpp mtbench"
python $B/gb10/apply_dspine_draftquant_patch.py >> $Q 2>&1
QD="export RDF2_DSPINE_DRAFT_QUANT=1; EXTRA='--speculative-draft-model-quantization fp8'"
hold "$QD; bash $S treefast q_tree32_d31_fp8draft '$ALL' 20 32 31; export TARGET=Qwen/Qwen3-8B-FP8; bash $S treefast fp8_q_tree32_d31_fp8draft '$ALL' 20 32 31"
hold "TF_MODEL=Qwen/Qwen3-8B-FP8 python $B/p2_tf_check.py $B/results/gb10/tf_check_fp8.json gb10/fp8_ar:gsm8k gb10/fp8_ar:mtbench gb10/fp8_f_tree32_d31:gsm8k gb10/fp8_f_tree32_d31:math500 gb10/fp8_f_tree32_d31:humaneval gb10/fp8_f_tree32_d31:mbpp gb10/fp8_f_tree32_d31:mtbench"
hold "python $B/p2_tf_check.py $B/results/gb10/tf_check_q.json gb10/q_tree32_d31_fp8draft:gsm8k gb10/q_tree32_d31_fp8draft:math500 gb10/q_tree32_d31_fp8draft:humaneval gb10/q_tree32_d31_fp8draft:mbpp gb10/q_tree32_d31_fp8draft:mtbench"
echo "QUEUE_F DONE $(date)" >> $Q
