#!/bin/bash
# Usage: phase1.sh <method>...   Starts the server per method, runs the Phase 1 matrix, stops the server.
# Matrix: T in {0,1} x conc in {1 (n=40), 8 (n=80)} x 5 datasets. Raw JSON -> bench/results/p1_<method>_*.json
cd /workspace/recursive-dflash2
for M in "$@"; do
  pkill -f "[s]glang serve"; sleep 8
  setsid nohup bench/serve.sh $M > logs/p1_server_$M.log 2>&1 < /dev/null &
  sleep 20
  for i in $(seq 1 400); do grep -q "fired up" logs/p1_server_$M.log && break; pgrep -f "[s]glang serve" >/dev/null || { echo "SERVER DEAD $M"; break; }; sleep 3; done
  grep -q "fired up" logs/p1_server_$M.log || { echo "SKIP $M (no server)"; continue; }
  BLOCK=$([ $M = ar ] && echo 1 || ([ $M = eagle3 ] && echo 1 || echo 16)) bench/run_cells.sh p1 $M "0 1" "1" "gsm8k math500 humaneval mbpp mtbench" 40
  BLOCK=$([ $M = ar ] && echo 1 || ([ $M = eagle3 ] && echo 1 || echo 16)) bench/run_cells.sh p1 $M "0 1" "8" "gsm8k math500 humaneval mbpp mtbench" 80
  pkill -f "[s]glang serve"; sleep 5
done
echo PHASE1DONE "$@"
