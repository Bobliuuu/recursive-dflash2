#!/bin/bash
# Phase 6 full box pipeline: setup -> greedy data -> official proxy -> mix1 -> mix4. Marker /workspace/p6/TRAIN_DONE.
cd /workspace/recursive-dflash2
[ -f /workspace/env.sh ] || bash bench/setup_box.sh > /workspace/setup.log 2>&1
grep -q DATA_OK /workspace/data.log 2>/dev/null || bash bench/p6treetrain/box_data.sh > /workspace/data.log 2>&1
bash bench/p6treetrain/box_train.sh > /workspace/train_all.log 2>&1
