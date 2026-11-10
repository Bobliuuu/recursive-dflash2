#!/bin/bash
# Phase 6 box steps (A100-80GB, after bench/setup_box.sh). Usage: box_data.sh  -> /workspace/p6/data.jsonl
# SpecForge pinned at the commit read for this phase; installed --no-deps into the DSpine venv (keeps torch 2.14.1+cu130).
set -euo pipefail
source /workspace/env.sh
export PATH=$HOME/.local/bin:$PATH
R=/workspace/recursive-dflash2; W=/workspace/p6; mkdir -p $W
if ! python -c "import specforge" 2>/dev/null; then
  [ -d /workspace/SpecForge ] || git clone -q https://github.com/sgl-project/SpecForge /workspace/SpecForge
  (cd /workspace/SpecForge && git checkout -q b15ad38)
  uv pip install -q --no-deps -e /workspace/SpecForge
  uv pip install -q --no-deps "liger-kernel>=0.8.3,<0.9" yunchang
  uv pip install -q datasets accelerate click openai-harmony pydantic psutil pyyaml tensorboard aiohttp
fi
N=${N:-16000}
[ -f $W/prompts.jsonl ] || python $R/train/prep_prompts.py $W/prompts.jsonl $N > $W/prep.log 2>&1
pkill -f "[s]glang serve" || true; sleep 3
nohup sglang serve --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --mem-fraction-static 0.85 \
  --context-length 4096 --max-running-requests 384 --trust-remote-code --port 30000 --host 127.0.0.1 > $W/serve_ar.log 2>&1 &
for i in $(seq 1 240); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; done
python $R/train/gen_responses.py $W/prompts.jsonl $W/data.jsonl --conc 384 --max-new 1536 --temp 0 > $W/gen.log 2>&1
pkill -f "[s]glang serve" || true
wc -l $W/data.jsonl
echo DATA_OK
