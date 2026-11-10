#!/bin/bash
# Phase 6 box driver (1x A100-80GB, Vast). Repo at /workspace/recursive-dflash2, after bench/setup_box.sh.
# Stages: specforge | prompts | gen | timing | train   (run all by default). Everything under /workspace/p6.
set -euo pipefail
source /workspace/env.sh
R=/workspace/recursive-dflash2; P=/workspace/p6; mkdir -p $P
stage=${1:-all}
if [[ $stage == all || $stage == specforge ]]; then
  [ -d /workspace/SpecForge ] || git clone -q https://github.com/sgl-project/SpecForge /workspace/SpecForge
  (cd /workspace/SpecForge && git checkout -q b15ad386)
  uv pip install -q --no-deps -e /workspace/SpecForge
  uv pip install -q "liger-kernel>=0.8.3,<0.9" datasets aiohttp
  python -c "from specforge.modeling.draft.dspine import DSpineDraftModel; from specforge.algorithms.dspine.model import OnlineDSpineModel; import transformers; print('specforge ok', transformers.__version__)"
fi
if [[ $stage == all || $stage == prompts ]]; then
  [ -s $P/prompts.jsonl ] || python $R/train/prep_prompts.py $P/prompts.jsonl 10600
  head -n 10344 $P/prompts.jsonl > $P/prompts_train.jsonl; tail -n 256 $P/prompts.jsonl > $P/prompts_eval.jsonl
fi
if [[ $stage == all || $stage == gen ]]; then
  python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --mem-fraction-static 0.85 --max-running-requests 256 \
      --disable-radix-cache > $P/sglang_gen.log 2>&1 &
  SP=$!
  for i in $(seq 1 120); do curl -s localhost:30000/health >/dev/null 2>&1 && break; sleep 5; done
  python $R/train/gen_responses.py $P/prompts_eval.jsonl $P/eval_greedy.jsonl --temp 0 --max-new 1536 --conc 256
  python $R/train/gen_responses.py $P/prompts_train.jsonl $P/train_t07.jsonl --temp 0.7 --max-new 1536 --conc 256
  kill $SP; sleep 10
fi
if [[ $stage == all || $stage == timing ]]; then
  python $R/bench/p6refine/timing.py $P/timing.json
fi
if [[ $stage == all || $stage == train ]]; then
  python $R/train/p6_pass2_tree.py --train $P/train_t07.jsonl --eval $P/eval_greedy.jsonl --out $P/run ${P6_ARGS:-} 2>&1 | tee -a $P/train_stdout.log
fi
