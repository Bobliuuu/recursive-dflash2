#!/usr/bin/env bash
# Replacement for queue2/queue3 once tune_sps T=0 is running: sweeps with tuned SPS tables, reduced tuning for T=1, forced width-8 profile.
W=${W:-/root/w}; cd $W; source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export CUDA_HOME=$W/venv/lib/python3.11/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
while pgrep -f "[t]une_sps.py" >/dev/null; do sleep 10; done
echo "=== $(date) T=0 tuning finished"; cat sps_T0.json | python -c "import json,sys;print(json.load(sys.stdin)['policy'])"
mkw8() { python - <<PY
import json
d = json.load(open("sps_T$1.json")); d["policy"] = {"uniform_widths": [8]*7, "batch8_tokens": 64}
json.dump(d, open("sps_w8_T$1.json", "w"))
PY
}
[ -f sps_T0.json ] && ./run_method_t.sh dspine16sps dspine SPS=$W/sps_T0.json TLIST=0 > logs/run_dspine16sps_T0.log 2>&1
echo "=== $(date) sweep sps T=0 done"
pkill -f "[s]glang serve"; sleep 5
python DSpine/scripts/tune_sps.py --requests sps_req_T1.jsonl --output sps_T1.json --port 30010 --widths 8 12 16 --repeats 1 \
   -- $(DRYRUN=1 ./serve.sh dspine 30010 | sed "s/--host 127.0.0.1 --port 30010//") > logs/tune_T1.log 2>&1
echo "=== $(date) tune T=1 done"; python -c "import json;print(json.load(open('sps_T1.json'))['policy'])"
[ -f sps_T1.json ] && ./run_method_t.sh dspine16sps dspine SPS=$W/sps_T1.json TLIST=1 > logs/run_dspine16sps_T1.log 2>&1
for T in 0 1; do
  [ -f sps_T$T.json ] || continue
  mkw8 $T
  ./run_method_t.sh dspine_w8 dspine SPS=$W/sps_w8_T$T.json TLIST=$T > logs/run_dspine_w8_T$T.log 2>&1
done
echo "=== $(date) queue5 done"
