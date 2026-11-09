#!/bin/bash
# Torch-profiler trace of ~12 decode rounds on a running server (box). Usage: p4_profile.sh OUTDIR   (server started with SGLANG_TORCH_PROFILER_DIR=OUTDIR)
OUT=$1; source /workspace/env.sh
python - <<PY &
import json, urllib.request
p = open("/workspace/recursive-dflash2/bench/data/prompts/gsm8k.jsonl").readline()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
t = tok.apply_chat_template([{"role": "user", "content": json.loads(p)["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
r = urllib.request.Request("http://127.0.0.1:30000/generate", data=json.dumps({"text": t, "sampling_params": {"temperature": 0, "max_new_tokens": 1500, "top_k": 1}}).encode(), headers={"Content-Type": "application/json"})
urllib.request.urlopen(r).read()
PY
sleep 4
curl -s -X POST localhost:30000/start_profile -H "Content-Type: application/json" -d "{\"output_dir\": \"$OUT\", \"num_steps\": 12, \"activities\": [\"GPU\"]}"
wait
ls -la $OUT
