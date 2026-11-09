#!/bin/bash
# Coverage dump on the stock block-48 DSpine chain with head_topk 64 (box). Output /workspace/p4cov/dump_k64.bin + meta_k64.json.
# Usage: run_dump.sh [nprompts]
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench; NP=${1:-40}; O=/workspace/p4cov
mkdir -p $O
bash $B/p4cov/mkdraft.sh 48 64
python $B/p4tree/apply_tree_patch_fast.py --revert; python $B/p4tree/apply_fi_mask_patch.py --revert
python $B/p4cov/apply_cov_dump_hook.py
pkill -f "[s]glang serve"; sleep 5
rm -f $O/dump_k64.bin
export RDF2_DUMP=$O/dump_k64.bin RDF2_DUMP_D=31
nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b48_k64 --speculative-num-draft-tokens 48 --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > $O/serve_dump.log 2>&1 &
for i in $(seq 1 180); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; done
rm -f $O/dump_k64.bin   # health probes
P2_T1=47 P4COV_K=64 P4COV_D=31 python $B/p4cov/collect.py --dump $O/dump_k64.bin --out $O/meta_k64.json --n $NP > $O/collect.log 2>&1
pkill -f "[s]glang serve"; sleep 3
python $B/p4cov/apply_cov_dump_hook.py --revert
echo done > $O/done_dump
