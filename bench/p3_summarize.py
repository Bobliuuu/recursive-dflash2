"""Summarize Phase 3 cells: bench/results/p3_<tag>_<ds>_T0_c1.json -> bench/results/p3_summary.json + printed table.
x AR = per-dataset tok/s ratio to the Phase 1 AR cell (p1_ar_*_T0_c1), averaged over the 5 datasets (same convention as Phase 1/2).
Usage: python p3_summarize.py [tag ...]   (default: every tag found)"""
import json, glob, os, sys, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
dss = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
ar = {d: json.load(open(f"{R}/p1_ar_{d}_T0_c1.json"))["summary"]["agg_tok_s"] for d in dss}
tags = sys.argv[1:] or sorted({re.match(r"p3_(.+)_gsm8k_T0_c1\.json", os.path.basename(f)).group(1) for f in glob.glob(f"{R}/p3_*_gsm8k_T0_c1.json")})
out = {}
for t in tags:
    rows = {}
    for d in dss:
        f = f"{R}/p3_{t}_{d}_T0_c1.json"
        if os.path.exists(f):
            s = json.load(open(f))["summary"]
            rows[d] = {"tau": s["accept_len_pooled"], "tok_s": s["agg_tok_s"], "x_ar": s["agg_tok_s"] / ar[d]}
    if len(rows) == 5:
        out[t] = {"by_ds": rows, "tau": sum(r["tau"] for r in rows.values()) / 5, "tok_s": sum(r["tok_s"] for r in rows.values()) / 5,
                  "x_ar": sum(r["x_ar"] for r in rows.values()) / 5}
base = out.get("off_b16", {}).get("tok_s")
for t, v in out.items():
    v["vs_off_b16"] = v["tok_s"] / base if base else None
json.dump(out, open(f"{R}/p3_summary.json", "w"), indent=1)
for t, v in out.items():
    print(f"{t:26s} tau {v['tau']:.2f}  tok/s {v['tok_s']:.0f}  x AR {v['x_ar']:.2f}  vs off_b16 {v['vs_off_b16'] if v['vs_off_b16'] is None else round(v['vs_off_b16'],3)}  ",
          {d: (round(r['tau'], 1), round(r['tok_s'])) for d, r in v['by_ds'].items()})
