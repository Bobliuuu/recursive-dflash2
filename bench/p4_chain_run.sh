#!/bin/bash
# Phase 4 chain reference (box): DSpine b16 weights at block N, no tree, causal verify, Phase 4 round timer.
# Usage: p4_chain_run.sh TAG N ["datasets"] [nprompts]
TAG=$1; N=$2; DSS=${3:-"gsm8k math500 humaneval mbpp mtbench"}; NP=${4:-40}
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench
mkdir -p /workspace/p4
python $B/patches/apply_timing_patch.py; python $B/patches/apply_timing_patch_p4.py
pkill -f "[s]glang serve"; sleep 5
unset RDF2_TREE_SPEC; export TORCH_CUDA_ARCH_LIST=8.0
export SPEC_TIMING_FILE=/workspace/p4/timing_$TAG.jsonl; rm -f $SPEC_TIMING_FILE
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$N --speculative-num-draft-tokens $N --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p4/serve_$TAG.log 2>&1 &
for i in $(seq 1 180); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; done
[ "$DSS" = serve ] && exit 0   # leave the server running (profiling)
for ds in $DSS; do
  python $B/run_cell.py --method dspine_$TAG --dataset $ds --temp 0 --conc 1 --n $NP --block $N --out $B/results/p4_${TAG}_${ds}_T0_c1.json > /workspace/p4/run_${TAG}_$ds.log 2>&1
done
pkill -f "[s]glang serve"
echo done > /workspace/p4/done_$TAG
