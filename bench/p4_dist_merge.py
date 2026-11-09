"""Merge distribution-test sample files (same prompts): python p4_dist_merge.py OUT.json IN1.json IN2.json ..."""
import json, sys
out = None
for f in sys.argv[2:]:
    d = json.load(open(f))
    if out is None: out = d; continue
    for k in d["samples"]:
        out["samples"][k] += d["samples"][k]; out["verify_ct"][k] += d["verify_ct"][k]
    out["n"] += d["n"]
json.dump(out, open(sys.argv[1], "w"))
print({k: len(v) for k, v in out["samples"].items()})
