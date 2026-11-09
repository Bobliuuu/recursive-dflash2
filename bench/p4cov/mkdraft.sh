#!/bin/bash
# Draft dir /workspace/dspine_b<N>_k<K>: the official b16 weights with block_size N and head_topk K (config only). Usage: mkdraft.sh N K
source /workspace/env.sh
N=$1; K=$2; D=/workspace/dspine_b${N}_k$K
DSP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('tencent/Qwen3-8B-DSpine-b16'))")
mkdir -p $D
for f in $DSP/*; do [ "$(basename $f)" = config.json ] || ln -sf $f $D/; done
python - <<PY
import json
c = json.load(open("$DSP/config.json")); c["block_size"] = $N; n = 0
def walk(o):
    global n
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "head_topk": o[k] = $K; n += 1
            else: walk(v)
walk(c); assert n >= 1, "no head_topk in config"
json.dump(c, open("$D/config.json", "w"), indent=1); print("$D", "head_topk entries set:", n)
PY
