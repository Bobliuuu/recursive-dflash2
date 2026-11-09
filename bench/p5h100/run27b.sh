#!/bin/bash
# Phase 5 (H100) Qwen3.8-27B c=1 T=0 cells (box). Target flags follow bench/qwen38_27b/serve.sh (ctx 8192, mamba ssm fp32, thinking xhigh,
# 2048 new tokens, 8 prompts per dataset); client bench/p5h100/client27b.py (native /generate, saves output ids).
# Usage: run27b.sh ar TAG | rssm TAG (published DSpine: fa3 + ReplaySSM + radix) | chain TAG N (fa3, no ReplaySSM: per-step SSM snapshots like the tree; flashinfer at block 16 fails, DSpine's ragged verify graph)
#        | tree TAG N:D [table] (TRS=1: tree ReplaySSM, apply_trs_patch.py)   ; env DSS="..." NP=8 MAXNEW=2048 EXTRA="..."
MODE=$1; TAG=$2; ARG=$3; TABLE=${4:-factored}
DSS=${DSS:-"gsm8k math500 humaneval mbpp mt-bench"}; NP=${NP:-8}; MAXNEW=${MAXNEW:-2048}
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench; R=$B/results/p5h100/q27b; L=/workspace/p5/q27b; mkdir -p $R $L
M=/root/w/models; T=$M/Qwen/Qwen3.8-27B; DSP=$M/tencent/Qwen3.8-27B-DSpine-b16
mkd() { local D=/root/w/dspine27_b$1; [ -f $D/config.json ] && return; mkdir -p $D
  for f in $DSP/*; do [ "$(basename $f)" = config.json ] || ln -sf $f $D/; done
  python -c "import json; c=json.load(open('$DSP/config.json')); c['block_size']=$1; json.dump(c, open('$D/config.json','w'), indent=1)"; }
python $B/patches/apply_timing_patch.py >/dev/null; python $B/patches/apply_timing_patch_p4.py >/dev/null
pkill -f "[s]glang serve"; sleep 8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--model-path $T --tp-size 1 --dtype bfloat16 --trust-remote-code --context-length 8192 --chunked-prefill-size 2048 --cuda-graph-backend-prefill disabled --mamba-ssm-dtype float32 --linear-attn-backend triton --host 127.0.0.1 --port 30000"
unset RDF2_TREE_SPEC RDF2_TREE_DIR RDF2_TABLE RDF2_ALLOW_BLOCK
P=$B/p5h100/apply_tree_patch_27b.py
python $B/p5h100/apply_trs_patch.py --revert >/dev/null   # always first: TRS patches sit on top of the tree patches
case $MODE in
  ar) python $P --revert >/dev/null; A="--attention-backend fa3 --mem-fraction-static 0.93 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --mamba-radix-cache-strategy extra_buffer --disable-radix-cache" ;;
  rssm) python $P --revert >/dev/null; A="--attention-backend fa3 --speculative-draft-attention-backend fa3 --enable-linear-replayssm-spec --max-mamba-cache-size 40 --min-free-slots-delay 1 --sampling-backend flashinfer --mem-fraction-static 0.86 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --mamba-radix-cache-strategy extra_buffer --speculative-algorithm DSPINE --speculative-draft-model-path $DSP --speculative-num-draft-tokens 16" ;;
  chain) mkd $ARG; python $P --revert >/dev/null; python $P >/dev/null; python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; export RDF2_ALLOW_BLOCK=1
     A="--attention-backend fa3 --speculative-draft-attention-backend fa3 --page-size 1 --mem-fraction-static ${MEM:-0.92} --max-running-requests 1 --cuda-graph-bs-decode 1 --disable-radix-cache --speculative-algorithm DSPINE --speculative-draft-model-path /root/w/dspine27_b$ARG --speculative-num-draft-tokens $ARG" ;;
  tree) N=${ARG%%:*}; mkd $N; python $P >/dev/null; if [ "${FIMASK:-skip}" = skip ]; then python $B/p4tree/apply_fi_mask_patch.py >/dev/null; else python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; fi
     export RDF2_TREE_SPEC=$ARG RDF2_TREE_DIR=$B/p5h100/p4tree27b RDF2_TABLE=$TABLE
     if [ -n "$TRS" ]; then python $B/p5h100/apply_trs_patch.py >/dev/null; export RDF2_TRS=1; fi
     A="--attention-backend flashinfer --speculative-draft-attention-backend fa3 --page-size 1 --mem-fraction-static ${MEM:-0.92} --max-running-requests 1 --cuda-graph-bs-decode 1 --disable-radix-cache --speculative-algorithm DSPINE --speculative-draft-model-path /root/w/dspine27_b$N --speculative-num-draft-tokens $N" ;;
esac
export SPEC_TIMING_FILE=$L/timing_$TAG.jsonl; rm -f $SPEC_TIMING_FILE
nohup sglang serve $COMMON $A $EXTRA > $L/serve_$TAG.log 2>&1 &
for i in $(seq 1 360); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; pgrep -f "[s]glang serve" >/dev/null || break; done
curl -sf localhost:30000/health_generate >/dev/null || { echo "SERVER FAILED $TAG"; grep -E "Error|error|assert" $L/serve_$TAG.log | tail -15; exit 1; }
[ "$DSS" = serve ] && exit 0
for ds in $DSS; do python $B/p5h100/client27b.py --method $TAG --dataset $ds --num-prompts $NP --max-new-tokens $MAXNEW --out $R 2>&1 | grep -v Warning | tail -2; done
pkill -f "[s]glang serve"; sleep 5
