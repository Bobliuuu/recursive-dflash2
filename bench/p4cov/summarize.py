"""Phase 4 coverage summary (copy of bench/p4_summarize.py writing results/p4cov/summary.json): tau / tok/s / x AR per config (5 datasets, c=1, T=0) plus the per-round timer breakdown (median ms).
Usage: python bench/p4cov/summarize.py PREFIX[=TIMING.jsonl] ...  (PREFIX as in bench/results/PREFIX_<ds>_T0_c1.json)."""
import json, os, sys, gzip, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
dss = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
ar = {d: json.load(open(f"{R}/p1_ar_{d}_T0_c1.json"))["summary"]["agg_tok_s"] for d in dss}
out_f = f"{R}/p4cov/summary.json"
allres = {}
for spec in sys.argv[1:]:
    pre, _, tf = spec.partition("=")
    c = {}
    for d in dss:
        f = f"{R}/{pre}_{d}_T0_c1.json"
        if os.path.exists(f):
            s = json.load(open(f))["summary"]; c[d] = {"tau": s.get("accept_len_pooled"), "tok_s": s["agg_tok_s"], "x_ar": s["agg_tok_s"] / ar[d]}
    row = {"by_ds": c}
    if len(c) == 5:
        row.update(mean_tau=sum(v["tau"] for v in c.values()) / 5, mean_tok_s=sum(v["tok_s"] for v in c.values()) / 5,
                   mean_x_ar=sum(v["x_ar"] for v in c.values()) / 5)
    if tf and os.path.exists(tf):
        rows = [json.loads(l) for l in (gzip.open(tf, "rt") if tf.endswith(".gz") else open(tf))][50:]
        if rows and "round_ms" in rows[0]:
            rows = [r for r in rows if r["round_ms"] < 60]
        keys = [k for k in rows[0] if k != "bs"] if rows else []
        row["timing_median_ms"] = {k: round(st.median(r[k] for r in rows if k in r), 3) for k in keys}
        row["timing_mean_ms"] = {k: round(st.mean(r[k] for r in rows if k in r), 3) for k in keys}
        row["timing_rounds"] = len(rows)
    allres[pre] = row
    print(pre, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k.startswith("mean")},
          {d: round(v["tok_s"]) for d, v in c.items()}, row.get("timing_median_ms"))
json.dump(allres, open(out_f, "w"), indent=1)
