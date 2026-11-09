#!/bin/bash
# Queue D on spark2: cuBLAS row-padding workaround (apply_gemm_pad_patch.py, RDF2_GEMM_PAD=128: bf16 GEMMs with 32 < M < 128 rows run at 128 rows).
R=${R:-$HOME/rdf2}; source $R/env.sh; S=$R/recursive-dflash2/bench/gb10/run_suite.sh; B=$R/recursive-dflash2/bench; Q=$R/logs/queue_d.log
ALL="gsm8k math500 humaneval mbpp mtbench"
hold() { echo "=== hold: $1 $(date)" >> $Q; flock -w 7200 $HOME/.gpu.lock bash -c "$1" >> $Q 2>&1; echo "=== exit $? $(date)" >> $Q; }
python $B/gb10/apply_gemm_pad_patch.py >> $Q 2>&1
hold "export RDF2_GEMM_PAD=128; bash $S treefast p_tree48_d31 '$ALL' 20 48 31; bash $S treefast p_tree64_d31 '$ALL' 20 64 31; bash $S treefast p_tree96_d31 '$ALL' 20 96 31"
echo "QUEUE_D DONE $(date)" >> $Q
