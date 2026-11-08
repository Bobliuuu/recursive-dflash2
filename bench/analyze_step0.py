"""Aggregate step0 cell JSONs + timing jsonl into bench/results/step0_summary.json and print a markdown table.
Usage: python bench/analyze_step0.py"""
import json, glob, statistics as st, os
R = "bench/results"
out = {}
for m in ["dflash", "dspine"]:
    cells = {}
    for f in sorted(glob.glob(f"{R}/step0_{m}_*_T0_c1.json")):
        j = json.load(open(f)); s = j["summary"]; cells[s["dataset"]] = s
        hist = [0] * 16  # re-pool: per-request histograms are trimmed of trailing zeros
        for r in j["requests"]:
            for k, c in enumerate(r["spec_hist"] or []): hist[k] += c
        rounds = sum(hist); surv = [sum(hist[i:]) / rounds for i in range(16)]
        s.update({"block_size": 16, "rounds": rounds, "hist_pooled": hist, "survival": surv,
                  "accept_len_pooled": sum((k + 1) * c for k, c in enumerate(hist)) / rounds,
                  "cond_accept": [surv[i] / surv[i - 1] if surv[i - 1] > 0 else None for i in range(1, 16)],
                  "frac_full_block": hist[-1] / rounds})
    t = [json.loads(l) for l in open(f"{R}/timing/{m}_step0.jsonl")] if os.path.exists(f"{R}/timing/{m}_step0.jsonl") else []
    t = [x for x in t if x["bs"] == 1]
    tm = None
    if t:
        d = [x["draft_ms"] for x in t]; v = [x["verify_ms"] for x in t]
        tm = {"rounds": len(t), "T_draft_ms_median": st.median(d), "T_verify_ms_median": st.median(v),
              "T_draft_ms_mean": st.mean(d), "T_verify_ms_mean": st.mean(v),
              "ratio_Td_over_Tv_median": st.median(d) / st.median(v), "ratio_Td_over_Tv_mean": st.mean(d) / st.mean(v)}
    out[m] = {"cells": cells, "timing": tm}
json.dump(out, open(f"{R}/step0_summary.json", "w"), indent=1)
ds = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
print("| method | dataset | n | tau | tok/s | full-block frac | cap-hit |")
print("|---|---|---|---|---|---|---|")
for m in out:
    for d in ds:
        c = out[m]["cells"].get(d)
        if c: print(f"| {m} | {d} | {c['n']} | {c['accept_len_pooled']:.2f} | {c['agg_tok_s']:.0f} | {c['frac_full_block']:.3f} | {c['frac_hit_cap']:.2f} |")
    cs = list(out[m]["cells"].values())
    if cs: print(f"| {m} | mean over datasets | | {sum(c['accept_len_pooled'] for c in cs)/len(cs):.2f} | {sum(c['agg_tok_s'] for c in cs)/len(cs):.0f} | {sum(c['frac_full_block'] for c in cs)/len(cs):.3f} | |")
for m in out:
    print(m, out[m]["timing"])

# derived: draft-ahead gain bound and per-position acceptance
for m in out:
    tm = out[m]["timing"]
    if not tm: continue
    td, tv = tm["T_draft_ms_median"], tm["T_verify_ms_median"]
    print(f"\n{m}: naive cap 1+Td/Tv = {1+td/tv:.3f}; round = {td+tv:.2f} ms")
    for d, c in out[m]["cells"].items():
        pf = c["frac_full_block"]
        # next draft can only be hidden behind verify if the whole block is accepted (bonus guess also must match: ignored, upper bound)
        gain = 1 / (1 - pf * td / (td + tv))
        c["draft_ahead_gain_upper_bound"] = gain
        print(f"  {d}: P(full)={pf:.3f} -> draft-ahead speedup upper bound {gain:.3f}")
        print("   cond accept per position 1..15:", " ".join("na" if x is None else f"{x:.2f}" for x in c["cond_accept"]))
json.dump(out, open(f"{R}/step0_summary.json", "w"), indent=1)
