#!/bin/bash
# Collect DSpine rounds dump at block N (draft b16 weights, block_size patched). Usage: p2_collect_blk.sh N
N=$1; source /workspace/env.sh
pkill -f "[s]glang serve"; sleep 5
export RDF2_DUMP=/workspace/p2/dump$N.bin; rm -f $RDF2_DUMP
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$N --speculative-num-draft-tokens $N --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 2 4 8 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p2/collect_serve$N.log 2>&1 &
for i in $(seq 1 120); do sleep 5; curl -s localhost:30000/health_generate >/dev/null && break; done
export P2_T1=$((N-1))
python /workspace/recursive-dflash2/bench/p2_collect.py --dump $RDF2_DUMP --out /workspace/p2/meta$N.json --n 40 > /workspace/p2/collect$N.log 2>&1
pkill -f "[s]glang serve"
echo done > /workspace/p2/collect_done_$N
