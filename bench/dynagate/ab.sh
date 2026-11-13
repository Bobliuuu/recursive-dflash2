#!/bin/bash
# A/B: warm-start DSpine-b16 fine-tune with vs without DynaDFlash-style layer gate. Same data/seed/steps. Box: /workspace.
source /workspace/env.sh; cd /workspace/recursive-dflash2/train; W=/workspace/dg; mkdir -p $W
D=/workspace/p7/data.jsonl
C="--data $D --limit 30000 --init-official tencent/Qwen3-8B-DSpine-b16 --block 16 --anchors 64 --lr 2e-4 --replace-prob 0 --epochs 1 --plan-steps 400 --micro-bs 2 --accum 16 --save-every 200"
python train_dspine.py $C --gate 1 --eval-only 64 --out $W/gate_e0 > $W/gate_e0.log 2>&1
python train_dspine.py $C --gate 0 --eval-only 64 --out $W/ctrl_e0 > $W/ctrl_e0.log 2>&1
python train_dspine.py $C --gate 0 --out $W/ctrl > $W/ctrl.log 2>&1
python train_dspine.py $C --gate 1 --out $W/gate > $W/gate.log 2>&1
echo done > $W/AB_DONE
