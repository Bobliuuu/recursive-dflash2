#!/usr/bin/env bash
# After queue2.sh: DSpine verify width forced to 8 (matches DFlash2 block 8) via a crafted SPS profile derived from the tuned one.
W=${W:-/root/w}; cd $W; source venv/bin/activate
while pgrep -f "[q]ueue2.sh" >/dev/null; do sleep 10; done
for T in 0 1; do
  [ -f sps_T$T.json ] || { echo "no sps_T$T.json"; continue; }
  python - <<PY
import json
d = json.load(open("sps_T$T.json")); d["policy"] = {"uniform_widths": [8]*7, "batch8_tokens": 64}
json.dump(d, open("sps_w8_T$T.json", "w"))
PY
  ./run_method_t.sh dspine_w8 dspine SPS=$W/sps_w8_T$T.json TLIST=$T > logs/run_dspine_w8_T$T.log 2>&1
done
echo "=== $(date) queue3 done"
