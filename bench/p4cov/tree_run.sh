#!/bin/bash
# Phase 4 coverage run (box): bench/p4_tree_run.sh with the K-generic glue (bench/p4cov/p4tree_k) and draft dir /workspace/dspine_b<N>_k<K>.
# Usage: RDF2_TREE_K=K [RDF2_TREE_E=E] RDF2_TABLE=factored tree_run.sh TAG FIMASK [N:D] ["datasets"] [nprompts]
#   FIMASK = stock | skip (FlashInfer custom-mask prefill skips mask lookups on full-prefix KV tiles, bench/p4tree/apply_fi_mask_patch.py)
# Results: bench/results/p4cov_<TAG>_<ds>_T0_c1.json, per-round timer /workspace/p4/timing_<TAG>.jsonl (bench/patches/apply_timing_patch_p4.py).
TAG=$1; GLUE=p4cov/p4tree_k; FIMASK=$2; SPEC=${3:-48:31}; DSS=${4:-"gsm8k math500 humaneval mbpp mtbench"}; NP=${5:-40}
K=${RDF2_TREE_K:-16}; export RDF2_TREE_K=$K
N=${SPEC%%:*}
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench
mkdir -p /workspace/p4
python $B/patches/apply_timing_patch.py
python $B/patches/apply_timing_patch_p4.py
python $B/p4cov/apply_cov_dump_hook.py --revert
python $B/p4tree/apply_tree_patch_fast.py
[ -d /workspace/dspine_b${N}_k$K ] || bash $B/p4cov/mkdraft.sh $N $K
if [ "$FIMASK" = skip ]; then python $B/p4tree/apply_fi_mask_patch.py; else python $B/p4tree/apply_fi_mask_patch.py --revert; fi
pkill -f "[s]glang serve"; sleep 5
export RDF2_TREE_SPEC=$SPEC RDF2_TREE_DIR=$B/$GLUE TORCH_CUDA_ARCH_LIST=8.0
export SPEC_TIMING_FILE=/workspace/p4/timing_$TAG.jsonl; rm -f $SPEC_TIMING_FILE
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b${N}_k$K --speculative-num-draft-tokens $N --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p4/serve_$TAG.log 2>&1 &
for i in $(seq 1 180); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; done
[ "$DSS" = serve ] && exit 0   # leave the server running (profiling)
for ds in $DSS; do
  python $B/run_cell.py --method dspine_$TAG --dataset $ds --temp 0 --conc 1 --n $NP --block $N --out $B/results/p4cov_${TAG}_${ds}_T0_c1.json > /workspace/p4/run_${TAG}_$ds.log 2>&1
done
pkill -f "[s]glang serve"
echo done > /workspace/p4/done_$TAG
