#!/bin/bash
# Run benchmark cells against the running server. Usage: p4_cells.sh TAG TEMP CONC NPROMPTS SEED BLOCK "datasets"
source /workspace/env.sh
B=/workspace/recursive-dflash2/bench
TAG=$1; T=$2; C=$3; NP=$4; SEED=$5; BLK=$6; DSS=${7:-"gsm8k math500 humaneval mbpp mtbench"}
for ds in $DSS; do
  python $B/run_cell.py --method $TAG --dataset $ds --temp $T --conc $C --n $NP --seed $SEED --block $BLK \
    --out $B/results/p4t1_${TAG}_${ds}_T${T%.*}_c${C}_s${SEED}.json > /workspace/p4/run_${TAG}_${ds}_s$SEED.log 2>&1
  python -c "import json;s=json.load(open('$B/results/p4t1_${TAG}_${ds}_T${T%.*}_c${C}_s${SEED}.json'))['summary'];print('$TAG',s['dataset'],round(s['agg_tok_s'],1),round(s.get('accept_len_pooled',1),2))"
done
