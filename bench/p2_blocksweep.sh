#!/bin/bash
# E2b: DSpine-b16 checkpoint run at longer blocks (config block_size patched; drafter weights untouched). Real tok/s, c=1, T=0, 5 datasets x 40 prompts.
# Usage: p2_blocksweep.sh "16 24 32 48 64" [conc] [temp]   (on the box)
source /workspace/env.sh
CONC=${2:-1}; TEMP=${3:-0}; NP=40; [ "$CONC" -gt 1 ] && NP=80
B=/workspace/recursive-dflash2/bench
for N in $1; do
  if [ "$N" = 16 ]; then D=tencent/Qwen3-8B-DSpine-b16; else D=/workspace/dspine_b$N; fi
  pkill -f "[s]glang serve"; sleep 6
  nohup sglang serve --model-path Qwen/Qwen3-8B --speculative-algorithm DSPINE --speculative-draft-model-path $D --speculative-num-draft-tokens $N --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --cuda-graph-bs-decode 1 2 4 8 16 32 64 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache > /workspace/p2/blk_serve$N.log 2>&1 &
  for i in $(seq 1 120); do sleep 5; curl -s localhost:30000/health_generate >/dev/null && break; done
  for ds in gsm8k math500 humaneval mbpp mtbench; do
    python $B/run_cell.py --method dspine_blk$N --dataset $ds --temp $TEMP --conc $CONC --n $NP --block $N --out $B/results/p2_blk${N}_${ds}_T${TEMP}_c${CONC}.json > /workspace/p2/blk_${N}_${ds}.log 2>&1
  done
done
pkill -f "[s]glang serve"
echo done > /workspace/p2/blocksweep_done_$CONC
