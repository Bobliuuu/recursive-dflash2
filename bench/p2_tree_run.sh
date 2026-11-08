#!/bin/bash
# In-engine tree verification run. Usage: p2_tree_run.sh N D "datasets" nprompts tag   (box). Draft = b16 checkpoint at block N.
N=$1; D=$2; DSS=${3:-"gsm8k math500 humaneval mbpp mtbench"}; NP=${4:-40}; TAG=${5:-tree${1}_d$2}
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench
python $B/p2tree/apply_tree_patch.py
pkill -f "[s]glang serve"; sleep 5
export RDF2_TREE_SPEC=$N:$D RDF2_TREE_STATS=/workspace/p2/treestats_$TAG.txt TORCH_CUDA_ARCH_LIST=8.0
rm -f $RDF2_TREE_STATS
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$N --speculative-num-draft-tokens $N --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p2/serve_$TAG.log 2>&1 &
for i in $(seq 1 150); do sleep 5; curl -s localhost:30000/health_generate >/dev/null && break; done
for ds in $DSS; do
  python $B/run_cell.py --method dspine_$TAG --dataset $ds --temp 0 --conc 1 --n $NP --block $N --out $B/results/p2_${TAG}_${ds}_T0_c1.json > /workspace/p2/run_${TAG}_$ds.log 2>&1
done
pkill -f "[s]glang serve"
echo done > /workspace/p2/tree_done_$TAG
