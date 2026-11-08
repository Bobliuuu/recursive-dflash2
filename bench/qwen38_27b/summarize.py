#!/usr/bin/env python3
"""Aggregate bench/results/qwen38_27b/*.json into markdown tables (+ summary.json).
usage: summarize.py RESULTS_DIR [OUT.md]"""
import json, glob, sys, collections, os
d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(d + "/*.json")):
    if f.endswith("summary.json"): continue
    r = json.load(open(f))
    key = (r["method"] + ("_" + r["tag"] if r.get("tag") else ""), r["dataset"], r["temperature"], r["concurrency"])
    hist = None
    for q in r["requests"]:
        h = q["meta"].get("spec_correct_drafts_histogram")
        if h: hist = h[:] if hist is None else [a + b for a, b in zip(hist, h)]
    rounds = sum(hist) if hist else 0
    surv = [sum(hist[i:]) / rounds for i in range(1, len(hist))] if rounds else None  # P(draft position i accepted) = P(>= i correct)
    rows[key] = {"tok_s": r["tok_per_s"], "acc": r["pooled_accept_length"], "acc_req": r["mean_req_accept_length"], "surv": surv, "hist": hist,
                 "trunc": r["truncated_frac"], "n": r["num_prompts"], "tokens": r["output_tokens"]}
methods = sorted({k[0] for k in rows}, key=lambda m: (m != "ar", m))
dsets = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
out = []
def cell(m, ds, T, c): return rows.get((m, ds, T, c))
for c in (1, 8):
    for T in (0.0, 1.0):
        out.append(f"\n### concurrency {c}, T={T:g}: output tok/s (speedup vs AR) | acceptance length (pooled tokens/verify-step)\n")
        out.append("| method | " + " | ".join(dsets) + " | mean |"); out.append("|---|" + "---|" * (len(dsets) + 1))
        for m in methods:
            cs, sp, ac = [], [], []
            for ds in dsets:
                x = cell(m, ds, T, c); a = cell("ar", ds, T, c)
                if not x: cs.append("-"); continue
                s = x["tok_s"] / a["tok_s"] if a else None
                cs.append(f"{x['tok_s']:.0f}" + (f" ({s:.2f}x)" if s else "") + (f" / {x['acc']:.2f}" if x["acc"] else ""))
                if s: sp.append(s)
                if x["acc"]: ac.append(x["acc"])
            mean = (f"{sum(sp)/len(sp):.2f}x" if sp else "-") + (f" / {sum(ac)/len(ac):.2f}" if ac else "")
            out.append(f"| {m} | " + " | ".join(cs) + f" | {mean} |")
out.append("\n### per-position acceptance P(draft position i accepted), pooled over datasets, T=0, c=1+8\n")
for m in methods:
    agg = None
    for (mm, ds, T, c), x in rows.items():
        if mm == m and T == 0.0 and x["hist"]:
            agg = x["hist"][:] if agg is None else [a + b for a, b in zip(agg, x["hist"])]
    if agg:
        n = sum(agg); out.append(f"- {m}: " + " ".join(f"{sum(agg[i:])/n:.2f}" for i in range(1, len(agg))))
txt = "\n".join(out); print(txt)
if len(sys.argv) > 2: open(sys.argv[2], "w").write(txt)
json.dump({"|".join(map(str, k)): v for k, v in rows.items()}, open(d + "/summary.json", "w"))
