#!/usr/bin/env bash
# c=8 with 64 prompts per cell (the N=24 cells are dominated by one long tail request: gsm8k AR 48 s wall for ~8 s of real work).
W=${W:-/root/w}; cd $W
while pgrep -f "[q]ueue6.sh" >/dev/null; do sleep 10; done
run() { name=$1; shift; ./run_method_t.sh $name "$@" CLIST=8 N8=64 TAG=n64 > logs/run_${name}_n64_T$TL.log 2>&1; echo "=== $(date) done $name n64 T$TL"; }
TL=0
run ar ar TLIST=0
run dflash2 dflash2 TLIST=0
run dspine16sps dspine SPS=$W/sps_T0.json TLIST=0
run dspine16 dspine TLIST=0
run dspark dspark TLIST=0
run mtp mtp TLIST=0
TL=1
run ar ar TLIST=1
run dflash2 dflash2 TLIST=1
[ -f sps_T1.json ] && run dspine16sps dspine SPS=$W/sps_T1.json TLIST=1
echo "=== $(date) queue7 done"
