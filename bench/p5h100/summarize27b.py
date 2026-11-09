"""Phase 5 H100 Qwen3.8-27B summary (c=1, T=0, thinking xhigh, 8 prompts x 5 datasets): tok/s, tau (pooled tokens / verify step),
x AR against the AR cells measured on the same box (t27_ar), per-round timer medians. Usage: summarize27b.py TAG[=TIMING.jsonl] ...
-> results/p5h100/q27b/summary27b.json"""
import json, os, sys, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "p5h100", "q27b")
dss = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
ar = {d: json.load(open(f"{R}/t27_ar__{d}__T0__c1.json"))["tok_per_s"] for d in dss if os.path.exists(f"{R}/t27_ar__{d}__T0__c1.json")}
out_f = f"{R}/summary27b.json"; allres = json.load(open(out_f)) if os.path.exists(out_f) else {}
for spec in sys.argv[1:]:
    tag, _, tf = spec.partition("=")
    c = {}
    for d in dss:
        f = f"{R}/{tag}__{d}__T0__c1.json"
        if os.path.exists(f):
            j = json.load(open(f)); c[d] = {"tok_s": j["tok_per_s"], "tau": j["pooled_accept_length"] or 1.0, "trunc": j["truncated_frac"],
                                             "x_ar": j["tok_per_s"] / ar[d] if d in ar else None}
    row = {"by_ds": c}
    if len(c) == 5:
        row.update(mean_tok_s=sum(v["tok_s"] for v in c.values()) / 5, mean_tau=sum(v["tau"] for v in c.values()) / 5)
        if len(ar) == 5: row["mean_x_ar"] = sum(v["x_ar"] for v in c.values()) / 5
    if tf and os.path.exists(tf):
        rows = [json.loads(l) for l in open(tf)][50:]
        rows = [r for r in rows if r.get("round_ms", 0) < 200]
        row["timing_median_ms"] = {k: round(st.median(r[k] for r in rows), 3) for k in rows[0] if not k.startswith("cpu") and k != "bs"} if rows else {}
    allres[tag] = row
    print(tag, {k: round(v, 3) for k, v in row.items() if k.startswith("mean")}, {d: round(v["tok_s"], 1) for d, v in c.items()},
          {k: row.get("timing_median_ms", {}).get(k) for k in ("draft_fwd", "verify", "round_ms")})
json.dump(allres, open(out_f, "w"), indent=1)
