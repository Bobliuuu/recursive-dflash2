#!/bin/bash
# Usage: run_cells.sh <tag> <method> "<temps>" "<concs>" "<datasets>" [n]   (server for <method> must be up)
source /workspace/env.sh
cd /workspace/recursive-dflash2
TAG=$1; M=$2; TEMPS=$3; CONCS=$4; DS=${5:-"gsm8k math500 humaneval mbpp mtbench"}; N=${6:-100}
for T in $TEMPS; do for C in $CONCS; do for D in $DS; do
  OUT=bench/results/${TAG}_${M}_${D}_T${T}_c${C}.json
  [ -s $OUT ] && continue
  # larger concurrency => more prompts are not needed; same prompts for every method
  python bench/run_cell.py --method $M --dataset $D --temp $T --conc $C --n $N --out $OUT > logs/${TAG}_${M}_${D}_T${T}_c${C}.log 2>&1 || echo "FAILED $OUT"
done; done; done
echo ALLDONE $TAG $M
