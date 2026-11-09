#!/bin/bash
# Phase C queue on spark2: quantized targets (FP8 block-128 = Qwen/Qwen3-8B-FP8; NVFP4 = nvidia/Qwen3-8B-NVFP4, ModelOpt, FP8 KV) with the
# unchanged bf16 DSpine b16 drafter. AR on 8 prompts/dataset, spec on 20 (first half of the frozen set).
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; Q=$R/logs/queue_c.log
ALL="gsm8k math500 humaneval mbpp mtbench"
hold() { echo "=== hold: $1 $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
hold "export TARGET=Qwen/Qwen3-8B-FP8; WARMUP=1 bash $S ar fp8_ar '$ALL' 8; bash $S dspine fp8_dspine_b16 '$ALL' 20; bash $S treefast fp8_f_tree48_d31 '$ALL' 20 48 31; EXTRA='--speculative-draft-model-quantization fp8' bash $S treefast fp8_f_tree48_d31_fp8draft '$ALL' 20 48 31"
hold "export TARGET=nvidia/Qwen3-8B-NVFP4 EXTRA='--quantization modelopt_fp4'; WARMUP=1 bash $S ar nvfp4_ar '$ALL' 8; bash $S dspine nvfp4_dspine_b16 '$ALL' 20; bash $S treefast nvfp4_f_tree48_d31 '$ALL' 20 48 31"
echo "QUEUE_C DONE $(date)" >> $Q
