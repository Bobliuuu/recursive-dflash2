#!/bin/bash
# Phase 3 eval of a trained/official DSpine drafter in the Phase 1/2 harness (c=1, T=0, 5 datasets x 40 prompts, A100).
# Usage: p3_eval.sh TAG WEIGHTS_DIR(or HF id) BLOCK [TREE=N:D]
#   chain:  p3_eval.sh ours_b16 /workspace/ckpt/b16/final 16
#   tree:   p3_eval.sh ours_b16_tree48 /workspace/ckpt/b16/final 48 48:31   (draft block set to BLOCK, tree nodes/depth N:D)
# Results: bench/results/p3_<TAG>_<ds>_T0_c1.json ; logs /workspace/p3/
TAG=$1; W=$2; BLK=$3; TREE=${4:-}
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench
mkdir -p /workspace/p3
if [ -d "$W" ]; then
  D=/workspace/p3/draft_$TAG; mkdir -p $D
  ln -sf $W/model.safetensors $D/model.safetensors
  python - <<PY
import json
c = json.load(open("$W/config.json")); c["block_size"] = $BLK
json.dump(c, open("$D/config.json", "w"), indent=1)
PY
  # tokenizer-free: sglang reads only config + weights for the draft
else
  D=$W   # HF id; block size can only be the checkpoint's own
fi
if [ -n "$TREE" ]; then python $B/p2tree/apply_tree_patch.py; export RDF2_TREE_SPEC=$TREE; else python $B/p2tree/apply_tree_patch.py --revert; unset RDF2_TREE_SPEC; fi
pkill -f "[s]glang serve"; sleep 6
export TORCH_CUDA_ARCH_LIST=8.0
export SPEC_TIMING_FILE=/workspace/p3/timing_$TAG.jsonl; rm -f $SPEC_TIMING_FILE
GB="1 2 4 8 16 32 64"; [ -n "$TREE" ] && GB="1"
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path $D --speculative-num-draft-tokens $BLK --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode $GB --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p3/serve_$TAG.log 2>&1 &
for i in $(seq 1 180); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; done
for ds in gsm8k math500 humaneval mbpp mtbench; do
  python $B/run_cell.py --method dspine_$TAG --dataset $ds --temp 0 --conc 1 --n 40 --block $BLK --out $B/results/p3_${TAG}_${ds}_T0_c1.json > /workspace/p3/run_${TAG}_$ds.log 2>&1
done
pkill -f "[s]glang serve"
echo done > /workspace/p3/done_$TAG
