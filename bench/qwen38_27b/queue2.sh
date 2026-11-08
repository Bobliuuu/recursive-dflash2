#!/usr/bin/env bash
# After queue.sh: tune DSpine SPS tables per temperature (DSpark-style dynamic verification), then sweep DSpine with them.
W=${W:-/root/w}; cd $W; source venv/bin/activate
while pgrep -f "[q]ueue.sh" >/dev/null; do sleep 10; done
export CUDA_HOME=$W/venv/lib/python3.11/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
for T in 0 1; do
  echo "=== $(date) tune T=$T"
  pkill -f "[s]glang serve"; sleep 5
  python DSpine/scripts/tune_sps.py --requests sps_req_T$T.jsonl --output sps_T$T.json --port 30010 \
     -- ./serve.sh dspine 30010 > logs/tune_T$T.log 2>&1
  echo "=== $(date) tune T=$T done: $(tail -1 logs/tune_T$T.log | cut -c1-200)"
  [ -f sps_T$T.json ] && ./run_method_t.sh dspine16sps_T$T dspine SPS=$W/sps_T$T.json TLIST=$T TAG= > logs/run_dspine16sps_T$T.log 2>&1
done
echo "=== $(date) queue2 done"
