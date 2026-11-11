#!/bin/bash
# Phase 7 tier A on one A100 box: setup -> data (68K prompts, T=0.7) -> official proxy (ref) -> from-scratch train -> proxy.
# Run on the box: nohup bash bench/p7scratch/box_p7a.sh > /workspace/p7a_all.log 2>&1 &   Marker: /workspace/p7/A_DONE
set -uo pipefail
R=/workspace/recursive-dflash2; W=/workspace/p7; mkdir -p $W
cd $R
[ -f /workspace/env.sh ] || bash bench/setup_box.sh > /workspace/setup.log 2>&1
source /workspace/env.sh; export PATH=$HOME/.local/bin:$PATH
if ! python -c "import specforge" 2>/dev/null; then
  [ -d /workspace/SpecForge ] || git clone -q https://github.com/sgl-project/SpecForge /workspace/SpecForge
  (cd /workspace/SpecForge && git checkout -q b15ad38)
  uv pip install -q --no-deps -e /workspace/SpecForge
  uv pip install -q --no-deps "liger-kernel>=0.8.3,<0.9" yunchang
  uv pip install -q datasets accelerate click openai-harmony pydantic psutil pyyaml tensorboard aiohttp
fi
if ! grep -q DATA_OK $W/data.log 2>/dev/null; then
  [ -f $W/prompts.jsonl ] || python train/prep_prompts.py $W/prompts.jsonl ${N:-68000} > $W/prep.log 2>&1
  pkill -f "[s]glang serve" || true; sleep 3
  nohup sglang serve --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --mem-fraction-static 0.85 \
    --context-length 4096 --max-running-requests 384 --trust-remote-code --port 30000 --host 127.0.0.1 > $W/serve_ar.log 2>&1 &
  for i in $(seq 1 240); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; done
  python train/gen_responses.py $W/prompts.jsonl $W/data.jsonl --conc 384 --max-new 1536 --temp 0.7 > $W/gen.log 2>&1
  pkill -f "[s]glang serve" || true; wc -l $W/data.jsonl > $W/data.log; echo DATA_OK >> $W/data.log
fi
cd train
[ -f $W/official/proxy.json ] || python train_treeaware.py --data $W/data.jsonl --out $W/official --eval-only > $W/official.log 2>&1
[ -f $W/A/final/sf_state.pt ] || python train_dspine.py --data $W/data.jsonl --out $W/A --block 16 --epochs 1 --anchors 128 --lr 6e-4 \
    --replace-prob 0.5 --micro-bs 2 --accum 16 --save-every 500 --log $W/A/train.log > $W/A.log 2>&1
[ -f $W/A_proxy/proxy.json ] || python train_treeaware.py --data $W/data.jsonl --out $W/A_proxy --init $W/A/final --eval-only > $W/A_proxy.log 2>&1
echo A_DONE > $W/A_DONE
