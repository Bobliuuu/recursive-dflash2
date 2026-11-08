#!/usr/bin/env bash
# Launch SGLang 0.5.22 server for Qwen3.8-27B in one of several speculative modes.
# usage: serve.sh MODE [PORT] [extra sglang args...]
# MODE: ar | mtp | dflash2 | dspine | dspark
# env: W=/root/w (work dir)  MEM=0.93  CTX=8192  VERIFY_N (dspine/dspark width override)  SPS (path to sps.json for DSpine dynamic verification)
set -euo pipefail
MODE=$1; PORT=${2:-30000}; shift || true; shift || true
W=${W:-/root/w}; MEM=${MEM:-$([ "$MODE" = dspine ] && echo 0.86 || echo 0.93)}; CTX=${CTX:-8192}
source $W/venv/bin/activate
T=$W/models/Qwen/Qwen3.8-27B
COMMON=(--model-path $T --tp-size 1 --dtype bfloat16 --trust-remote-code
  --context-length $CTX --mem-fraction-static $MEM --chunked-prefill-size 2048 --cuda-graph-backend-prefill disabled --max-running-requests 8
  --cuda-graph-max-bs-decode 8 --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer
  --reasoning-parser qwen3 --host 127.0.0.1 --port $PORT)
# system nvcc is 12.4; deep_gemm/flashinfer JIT need >=12.9: use pip nvcc 13.x from the venv
export CUDA_HOME=$W/venv/lib/python3.11/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
# radix cache off except DSPINE, whose plugin requires it (prompts are ~100 tokens, so prefix reuse is negligible)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $MODE in
 ar)
  ARGS=(--attention-backend fa3 --linear-attn-backend triton) ;;
 mtp)
  # built-in MTP head (mtp_num_hidden_layers=1), chain of 3 steps -> 4 draft tokens (as in RadixArk's EAGLE/MTP row)
  ARGS=(--attention-backend fa3 --linear-attn-backend triton --speculative-algorithm EAGLE
        --speculative-num-steps ${MTP_STEPS:-3} --speculative-eagle-topk 1 --speculative-num-draft-tokens $((${MTP_STEPS:-3}+1))) ;;
 dflash2)
  ARGS=(--attention-backend fa3 --speculative-draft-attention-backend fa3 --linear-attn-backend triton
        --speculative-algorithm DFLASH --speculative-draft-model-path $W/models/z-lab/Qwen3.8-27B-DFlash2
        --speculative-num-draft-tokens ${BLOCK:-8}) ;;
 dspine)
  ARGS=(--attention-backend fa3 --speculative-draft-attention-backend fa3 --linear-attn-backend triton
        --enable-linear-replayssm-spec --max-mamba-cache-size ${MAMBA:-40} --min-free-slots-delay 1 --sampling-backend flashinfer
        --speculative-algorithm DSPINE --speculative-draft-model-path $W/models/tencent/Qwen3.8-27B-DSpine-b16
        --speculative-num-draft-tokens ${BLOCK:-16})
  [ -n "${SPS:-}" ] && ARGS+=(--speculative-dspark-sps-table-path $SPS) ;;
 dspark)
  ARGS=(--attention-backend flashinfer --speculative-draft-attention-backend flashinfer --linear-attn-backend triton
        --speculative-algorithm DSPARK --speculative-draft-model-path $W/models/RadixArk/Qwen3.8-27B-DSpark
        --speculative-draft-model-quantization unquant --speculative-dspark-block-size 7
        --speculative-num-steps 1 --speculative-eagle-topk 1)
  export SGLANG_RAGGED_VERIFY_MODE=static ;;
 *) echo bad mode; exit 2 ;;
esac
[ "$MODE" != dspine ] && ARGS+=(--disable-radix-cache)
exec sglang serve "${COMMON[@]}" "${ARGS[@]}" "$@"
