#!/bin/bash
# Usage: serve.sh <ar|eagle3|dflash|dspine> [extra sglang args]   (run on the box; env in /workspace/env.sh)
# Attention backend: flashinfer (A100/sm80 has no FA3; fa3 is Hopper-only).
source /workspace/env.sh
M=$1; shift
COMMON="--model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 2 4 8 16 32 64 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache"
case $M in
  ar) A="" ;;
  eagle3) A="--speculative-algorithm EAGLE3 --speculative-draft-model-path ${EAGLE3_PATH:-RedHatAI/Qwen3-8B-speculator.eagle3} --speculative-num-steps ${E_STEPS:-3} --speculative-eagle-topk ${E_TOPK:-1} --speculative-num-draft-tokens ${E_DRAFT:-4}" ;;
  dflash) A="--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --speculative-num-draft-tokens 16" ;;
  dspine) A="--speculative-algorithm DSPINE --speculative-draft-model-path tencent/Qwen3-8B-DSpine-b16 --speculative-num-draft-tokens 16" ;;
esac
exec sglang serve $COMMON $A "$@"
