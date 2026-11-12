#!/bin/bash
# Phase 9 pipeline (one A100-80GB; run by bench/p9deletion/modal_app.py, or on any box after bench/setup_box.sh + SpecForge).
# Usage: box_p9.sh <data|train|eval|serve>   State and outputs under $P9 (default /vol/p9); every step is resumable.
set -uo pipefail
R=${R:-/workspace/recursive-dflash2}; P9=${P9:-/vol/p9}; mkdir -p $P9/data $P9/logs
[ -f /workspace/env.sh ] && source /workspace/env.sh
cd $R
D=$P9/data
serve_ar() {
  pkill -f "[s]glang serve" || true; sleep 3
  nohup sglang serve --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --mem-fraction-static 0.85 \
    --context-length 4096 --max-running-requests 384 --trust-remote-code --port 30000 --host 127.0.0.1 > $P9/logs/serve_ar.log 2>&1 &
  for i in $(seq 1 360); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && return 0; done
  echo "AR server failed"; tail -40 $P9/logs/serve_ar.log; return 1
}
case $1 in
data)
  grep -q DATA_OK $D/DATA_OK 2>/dev/null && { echo data already done; exit 0; }
  [ -f $D/general.prompts.jsonl ] || python train/prep_prompts.py $D/general.prompts.jsonl ${NGEN:-10000} > $P9/logs/prep.log 2>&1
  [ -f $D/mbpp_train.prompts.jsonl ] || python train/p9_prompts.py domains $D --frozen bench/data/prompts > $P9/logs/domains.log 2>&1
  serve_ar || exit 1
  # greedy (T=0) responses from the target itself, thinking off, max 1536 new tokens (the Phase 6 recipe)
  for f in eval_gsm8k eval_math500 eval_humaneval eval_mbpp mbpp_train general; do
    python train/gen_responses.py $D/$f.prompts.jsonl $D/$f.jsonl --conc 384 --max-new 1536 --temp 0 > $P9/logs/gen_$f.log 2>&1
    echo "$f $(wc -l < $D/$f.jsonl)"
  done
  pkill -f "[s]glang serve" || true
  python train/p9_prompts.py split $D/general.jsonl $D/mbpp_train.jsonl $D > $P9/logs/split.log 2>&1
  cat $P9/logs/split.log
  echo DATA_OK > $D/DATA_OK ;;
train)
  cd train
  for m in ${MASKS:-gen code}; do
    [ -f $P9/$m/mask.dmask ] && continue
    case $m in gen) TD=$D/train_general.jsonl ;; code) TD=$D/train_code.jsonl ;; esac
    python train_deletion.py --train-data $TD --out $P9/$m --steps ${STEPS:-300} > $P9/logs/train_$m.log 2>&1 || { tail -40 $P9/logs/train_$m.log; exit 1; }
  done ;;
eval)
  cd train
  EV="gsm8k=$D/eval_gsm8k.jsonl math500=$D/eval_math500.jsonl humaneval=$D/eval_humaneval.jsonl mbpp=$D/eval_mbpp.jsonl chat=$D/eval_chat.jsonl general=$D/eval_general.jsonl"
  MS=""
  for m in ${MASKS:-gen code}; do
    MS="$MS $m=$P9/$m/mask.dmask ${m}_rand=random:$P9/$m/mask.dmask ${m}_wanda=$P9/$m/wanda_init.dmask"
  done
  python train_deletion.py --out $P9/eval --eval $EV --masks $MS > $P9/logs/eval.log 2>&1 || { tail -40 $P9/logs/eval.log; exit 1; } ;;
*) echo "unknown stage $1"; exit 2 ;;
esac
echo "STAGE $1 OK"
