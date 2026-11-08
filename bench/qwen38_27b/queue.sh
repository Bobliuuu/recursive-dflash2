#!/usr/bin/env bash
# Sequential method queue on the box (waits for any running run_method.sh first).
W=${W:-/root/w}; cd $W
while pgrep -f "[r]un_method.sh" >/dev/null; do sleep 10; done
for spec in "dspine16 dspine" "ar ar" "mtp mtp" "dspine8 dspine BLOCK=8" "dspark dspark" "dflash2b16 dflash2 BLOCK=16 TAG=b16"; do
  set -- $spec
  echo "=== $(date) start $1"; ./run_method.sh "$@" > logs/run_$1.log 2>&1; echo "=== $(date) done $1: $(tail -1 logs/run_$1.log)"
done
