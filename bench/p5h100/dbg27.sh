#!/bin/bash
# Debug loop (box): serve TAG with the given env, run N prompts of DS, then teacher-force them on the AR server.  usage: dbg27.sh TAG SPEC DS N
TAG=$1; SPEC=$2; DS=${3:-gsm8k}; NP=${4:-1}
B=/workspace/recursive-dflash2/bench; O=/workspace/p5/dbg_$TAG; mkdir -p $O
DSS=serve bash $B/p5h100/run27b.sh tree $TAG $SPEC > $O/serve.out 2>&1 || { tail $O/serve.out; exit 1; }
source /workspace/env.sh
python $B/p5h100/client27b.py --method $TAG --dataset $DS --num-prompts $NP --max-new-tokens 300 --warmup 0 --out $O 2>&1 | grep tok/s
DSS=serve bash $B/p5h100/run27b.sh ar t27_ar_tfsrv > /dev/null 2>&1
TF_VERBOSE=1 python $B/p5h100/tf27b.py $O/tf.json $O/${TAG}__${DS}__T0__c1.json:$NP 2>&1 | grep -v Warn | tail -4
