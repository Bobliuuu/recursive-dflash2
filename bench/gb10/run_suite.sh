#!/bin/bash
# GB10 (DGX Spark) benchmark suite runner: start one SGLang server (own process group), run the c=1 T=0 cells, stop the server.
# Usage: run_suite.sh MODE TAG "datasets" NPROMPTS [N] [D]
#   MODE: ar | dflash | dspine (b16 chain) | dspine_blk (chain at block N) | tree (DSpine tree, N nodes, depth D)
# Env (from $R/env.sh): R (work dir, default ~/rdf2), RDF2_PORT, MEMF (mem-fraction-static, default 0.25), EXTRA (extra sglang args)
# Wrap in `flock -w 7200 ~/.gpu.lock` for timing runs.
set -u
R=${R:-$HOME/rdf2}; source $R/env.sh
MODE=$1; TAG=$2; DSS=${3:-"gsm8k math500 humaneval mbpp mtbench"}; NP=${4:-40}; N=${5:-16}; D=${6:-31}
PORT=${RDF2_PORT:-30000}; MEMF=${MEMF:-0.25}
B=$R/recursive-dflash2/bench; OUT=$B/results/gb10; LOG=$R/logs; mkdir -p $OUT $LOG
COMMON="--model-path ${TARGET:-Qwen/Qwen3-8B} --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static $MEMF --context-length 8192 --max-running-requests 4 --cuda-graph-bs-decode 1 --trust-remote-code --port $PORT --host 127.0.0.1 --disable-radix-cache --disable-prefill-cuda-graph ${EXTRA:-}"
unset RDF2_TREE_SPEC
AVAIL=$(free -g | awk '/Mem:/{print $7}')
[ "$AVAIL" -ge 50 ] || { echo "only ${AVAIL} GB available, backing off"; exit 2; }
echo "start $TAG $(date) avail=${AVAIL}GB"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
case $MODE in
  ar) A=""; BLK=1 ;;
  dflash) A="--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --speculative-num-draft-tokens 16"; BLK=16 ;;
  dspine) A="--speculative-algorithm DSPINE --speculative-draft-model-path tencent/Qwen3-8B-DSpine-b16 --speculative-num-draft-tokens 16"; BLK=16 ;;
  dspine_blk) A="--speculative-algorithm DSPINE --speculative-draft-model-path $R/dspine_b$N --speculative-num-draft-tokens $N"; BLK=$N ;;
  tree) export RDF2_TREE_SPEC=$N:$D RDF2_TREE_DIR=$B/p2tree
        A="--speculative-algorithm DSPINE --speculative-draft-model-path $R/dspine_b$N --speculative-num-draft-tokens $N"; BLK=$N ;;
esac
if [ "$MODE" = tree ]; then python $B/p2tree/apply_tree_patch.py; else python $B/p2tree/apply_tree_patch.py --revert; fi
export SPEC_TIMING_FILE=$LOG/timing_$TAG.jsonl; rm -f $SPEC_TIMING_FILE
setsid sglang serve $COMMON $A > $LOG/serve_$TAG.log 2>&1 &
SPID=$!
trap 'kill -TERM -- -$SPID 2>/dev/null; sleep 5; kill -KILL -- -$SPID 2>/dev/null' EXIT
ok=0
for i in $(seq 1 240); do sleep 5; curl -s 127.0.0.1:$PORT/health_generate >/dev/null && { ok=1; break; }; kill -0 $SPID 2>/dev/null || break; done
[ $ok = 1 ] || { echo "server failed"; tail -30 $LOG/serve_$TAG.log; exit 1; }
for ds in $DSS; do
  python $B/run_cell.py --url http://127.0.0.1:$PORT --prompts-dir $B/data/prompts --method $TAG --dataset $ds --temp 0 --conc 1 --n $NP --warmup ${WARMUP:-2} --block $BLK \
    --out $OUT/${TAG}_${ds}_T0_c1.json > $LOG/run_${TAG}_$ds.log 2>&1
  python -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; print(sys.argv[2], round(s['agg_tok_s'],1), 'tok/s tau', round(s.get('accept_len_pooled') or 1,2))" $OUT/${TAG}_${ds}_T0_c1.json $ds
done
echo "suite $TAG done"
