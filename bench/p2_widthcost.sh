#!/bin/bash
# Real-engine verify cost vs width: DSpine drafter with block_size patched to N (garbage drafts, real verify forward), SPEC_TIMING_FILE round timer.
# Usage: p2_widthcost.sh N   (runs on the box)
N=$1
source /workspace/env.sh
mkdir -p /workspace/p2/width; rm -f /workspace/p2/width/t$N.jsonl
export SPEC_TIMING_FILE=/workspace/p2/width/t$N.jsonl
D=/workspace/dspine_b$N
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path $D --speculative-num-draft-tokens $N --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 2 4 8 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p2/width/serve$N.log 2>&1 &
for i in $(seq 1 90); do sleep 5; curl -s localhost:30000/health_generate >/dev/null && break; done
python /workspace/recursive-dflash2/bench/p2_collect.py --dump /nonexistent --out /workspace/p2/width/meta$N.json --n 12 --datasets gsm8k,mtbench > /workspace/p2/width/collect$N.log 2>&1
tail -2 /workspace/p2/width/serve$N.log
pkill -f "[s]glang serve"; sleep 6
