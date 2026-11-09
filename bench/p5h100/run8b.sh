#!/bin/bash
# Phase 5 (H100) Qwen3-8B cell runner (box). Same server flags / harness as bench/p4_tree_run.sh and p4_chain_run.sh, but arch-neutral
# (TORCH_CUDA_ARCH_LIST from /workspace/env.sh, 9.0 on H100) and with an AR mode. Results: bench/results/p5h100/<TAG>_<ds>_T0_c1.json
# Usage: run8b.sh ar TAG | chain TAG N | tree TAG N:D [table=factored] [fimask=skip]   ; env DSS="..." NP=40
MODE=$1; TAG=$2; ARG=$3; TABLE=${4:-factored}; FIMASK=${5:-skip}
DSS=${DSS:-"gsm8k math500 humaneval mbpp mtbench"}; NP=${NP:-40}
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench; R=$B/results/p5h100; L=/workspace/p5; mkdir -p $R $L
mkd() { local N=$1 D=/workspace/dspine_b$1; [ -f $D/config.json ] && return
  local DSP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('tencent/Qwen3-8B-DSpine-b16'))"); mkdir -p $D
  for f in $DSP/*; do [ "$(basename $f)" = config.json ] || ln -sf $f $D/; done
  python -c "import json; c=json.load(open('$DSP/config.json')); c['block_size']=$N; json.dump(c, open('$D/config.json','w'), indent=1)"; }
python $B/patches/apply_timing_patch.py >/dev/null; python $B/patches/apply_timing_patch_p4.py >/dev/null
pkill -f "[s]glang serve"; sleep 5
COMMON="--model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 64 --trust-remote-code --port 30000 --host 127.0.0.1 --disable-radix-cache --cuda-graph-bs-decode 1"
unset RDF2_TREE_SPEC RDF2_TREE_DIR RDF2_TABLE
case $MODE in
  ar) python $B/p4tree/apply_tree_patch_fast.py --revert >/dev/null; python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; A=""; BLK=1 ;;
  chain) python $B/p4tree/apply_tree_patch_fast.py --revert >/dev/null; python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; mkd $ARG
     A="--speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$ARG --speculative-num-draft-tokens $ARG"; BLK=$ARG ;;
  tree) N=${ARG%%:*}; mkd $N; python $B/p4tree/apply_tree_patch_fast.py >/dev/null
     if [ "$FIMASK" = skip ]; then python $B/p4tree/apply_fi_mask_patch.py >/dev/null; else python $B/p4tree/apply_fi_mask_patch.py --revert >/dev/null; fi
     export RDF2_TREE_SPEC=$ARG RDF2_TREE_DIR=$B/p4tree RDF2_TABLE=$TABLE
     A="--speculative-algorithm DSPINE --speculative-draft-model-path /workspace/dspine_b$N --speculative-num-draft-tokens $N"; BLK=$N ;;
esac
export SPEC_TIMING_FILE=$L/timing_$TAG.jsonl; rm -f $SPEC_TIMING_FILE
nohup sglang serve $COMMON $A > $L/serve_$TAG.log 2>&1 &
for i in $(seq 1 240); do sleep 5; curl -sf localhost:30000/health_generate >/dev/null && break; pgrep -f "[s]glang serve" >/dev/null || break; done
curl -sf localhost:30000/health_generate >/dev/null || { echo "SERVER FAILED $TAG"; tail -40 $L/serve_$TAG.log; exit 1; }
[ "$DSS" = serve ] && exit 0
for ds in $DSS; do
  python $B/run_cell.py --method ${MODE}_$TAG --dataset $ds --temp 0 --conc 1 --n $NP --block $BLK --out $R/${TAG}_${ds}_T0_c1.json > $L/run_${TAG}_$ds.log 2>&1
  python -c "import json; s=json.load(open('$R/${TAG}_${ds}_T0_c1.json'))['summary']; print('$TAG', '$ds', round(s['agg_tok_s'],1), round(s.get('accept_len_pooled') or 1,3))"
done
pkill -f "[s]glang serve"; sleep 3
