#!/usr/bin/env bash
# After queue3: DFlash2 drafter forced to block 16 (trained for 8) at c=1 only (intermediate SSM cache for 8 slots x 16 tokens does not fit).
W=${W:-/root/w}; cd $W
while pgrep -f "[q]ueue3.sh" >/dev/null; do sleep 10; done
./run_method_t.sh dflash2b16 dflash2 BLOCK=16 MAXREQ=2 CLIST=1 TAG=b16 > logs/run_dflash2b16.log 2>&1
echo "=== $(date) queue4 done"
