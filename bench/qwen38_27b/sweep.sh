#!/usr/bin/env bash
# Run the full cell grid against an ALREADY RUNNING server.  usage: sweep.sh METHOD_NAME [PORT]
# env: N1 (prompts at c=1, default 8) N8 (prompts at c=8, default 24) OUT (results dir) THINK (xhigh|off) MAXNEW (2048)
M=$1; PORT=${2:-30000}; W=${W:-/root/w}; OUT=${OUT:-$W/results}; N1=${N1:-8}; N8=${N8:-24}
THINK=${THINK:-xhigh}; MAXNEW=${MAXNEW:-2048}; TAG=${TAG:-}
source $W/venv/bin/activate; mkdir -p $OUT
for C in 8 1; do for T in 0 1; do for D in gsm8k math500 humaneval mbpp mt-bench; do
  N=$N8; [ $C = 1 ] && N=$N1
  python $W/client.py --base-url http://127.0.0.1:$PORT --method $M --tag "$TAG" --dataset $D --temperature $T --concurrency $C \
     --num-prompts $N --max-new-tokens $MAXNEW --thinking $THINK --out $OUT 2>&1 | grep -E "^$M|Error|error" | cut -c1-300
done; done; done
