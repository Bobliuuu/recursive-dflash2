#!/bin/bash
# Phase 4 server launcher (box). Usage: p4_serve.sh <ar|chain|tree> [N D] ; logs to /workspace/p4/serve_<mode>.log, waits for health.
# tree = DSpine b16 weights at block N with the in-engine tree (greedy + lossless sampled verification); chain = stock DSpine b16.
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench
M=$1; N=${2:-48}; D=${3:-31}
pkill -f "[s]glang serve"; sleep 5
COMMON="--model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache"
case $M in
  ar) python $B/p4tree/apply_tree_patch_fast.py --revert >/dev/null; python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; unset RDF2_TREE_SPEC; A="--cuda-graph-bs-decode 1 2 4 8 16 32 64" ;;
  chain) python $B/p4tree/apply_tree_patch_fast.py --revert >/dev/null; python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; unset RDF2_TREE_SPEC; A="--cuda-graph-bs-decode 1 2 4 8 16 32 64 --speculative-algorithm DSPINE --speculative-draft-model-path tencent/Qwen3-8B-DSpine-b16 --speculative-num-draft-tokens 16" ;;
  tree) python $B/p4tree/apply_tree_patch_fast.py --revert >/dev/null; python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; python $B/p2tree/apply_tree_patch.py >/dev/null; export RDF2_TREE_SPEC=$N:$D RDF2_TREE_T1_MODE=${RDF2_TREE_T1_MODE:-det}; A="--cuda-graph-bs-decode 1 --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$N --speculative-num-draft-tokens $N" ;;
  fast) python $B/p4tree/apply_tree_patch_fast.py >/dev/null; if [ "$FIMASK" = skip ]; then python $B/p4tree/apply_fi_mask_patch.py >/dev/null; else python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; fi
        export RDF2_TREE_SPEC=$N:$D RDF2_TREE_DIR=$B/p4tree_t1; A="--cuda-graph-bs-decode 1 --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$N --speculative-num-draft-tokens $N" ;;
esac
TAG=${P4_TAG:-$M}
[ -n "$P4_TIMING" ] && export SPEC_TIMING_FILE=/workspace/p4/timing_$TAG.jsonl && rm -f $SPEC_TIMING_FILE
nohup sglang serve $COMMON $A > /workspace/p4/serve_$TAG.log 2>&1 &
for i in $(seq 1 200); do sleep 5; curl -s localhost:30000/health_generate >/dev/null && echo up && exit 0; done
echo "server failed"; tail -30 /workspace/p4/serve_$TAG.log; exit 1
