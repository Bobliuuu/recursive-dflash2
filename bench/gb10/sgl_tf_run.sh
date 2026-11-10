#!/bin/bash
# Start an AR SGLang server for TARGET (same flags as run_suite.sh), run sgl_tf_check.py on the given specs, stop the server.
# Usage: TARGET=... EXTRA=... sgl_tf_run.sh OUT.json spec ...   (inside the GPU lock)
R=${R:-$HOME/rdf2}; source $R/env.sh; B=$R/recursive-dflash2/bench; PORT=${RDF2_PORT:-30000}; OUT=$1; shift
python $B/p4tree/apply_tree_patch_fast.py --revert >/dev/null
setsid sglang serve --model-path ${TARGET:-Qwen/Qwen3-8B} --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.25 \
  --context-length 8192 --max-running-requests 4 --cuda-graph-bs-decode 1 --trust-remote-code --port $PORT --host 127.0.0.1 --disable-radix-cache \
  --disable-prefill-cuda-graph ${EXTRA:-} > $R/logs/serve_sgltf_$(basename $OUT .json).log 2>&1 &
SPID=$!; trap 'kill -TERM -- -$SPID 2>/dev/null; sleep 5; kill -KILL -- -$SPID 2>/dev/null' EXIT
for i in $(seq 1 150); do sleep 5; curl -s 127.0.0.1:$PORT/health_generate >/dev/null && break; done
python $B/gb10/sgl_tf_check.py --url http://127.0.0.1:$PORT $OUT "$@"
