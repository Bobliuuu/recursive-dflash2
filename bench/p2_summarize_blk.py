"""Summarize block-length sweep (p2_blk{N}_*) vs Phase-1 b16 and AR. Usage: python p2_summarize_blk.py [conc] [temp] -> bench/results/p2_e2b_blocksweep_T{T}_c{c}.json"""
import json, sys, glob, os
conc = sys.argv[1] if len(sys.argv) > 1 else "1"; T = sys.argv[2] if len(sys.argv) > 2 else "0"
R = os.path.join(os.path.dirname(__file__), "results")
dss = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
def ld(p):
    return json.load(open(p))["summary"] if os.path.exists(p) else None
ar = {d: ld(f"{R}/p1_ar_{d}_T{T}_c{conc}.json") for d in dss}
out = {}
for N in (16, 24, 32, 48, 64):
    rows = {}
    for d in dss:
        s = ld(f"{R}/p2_blk{N}_{d}_T{T}_c{conc}.json")
        if s is None: continue
        rows[d] = {"tau": s.get("accept_len_pooled"), "tok_s": s["agg_tok_s"], "speedup_vs_ar": s["agg_tok_s"] / ar[d]["agg_tok_s"], "full_block_frac": (s["survival"][-1] if s.get("survival") else None)}
    if rows:
        out[f"b{N}"] = {"by_ds": rows, "mean_tau": sum(r["tau"] for r in rows.values()) / len(rows), "mean_tok_s": sum(r["tok_s"] for r in rows.values()) / len(rows),
                        "mean_speedup_vs_ar": sum(r["speedup_vs_ar"] for r in rows.values()) / len(rows)}
p1 = {d: ld(f"{R}/p1_dspine_{d}_T{T}_c{conc}.json") for d in dss}
out["phase1_b16"] = {"by_ds": {d: {"tau": p1[d]["accept_len_pooled"], "tok_s": p1[d]["agg_tok_s"], "speedup_vs_ar": p1[d]["agg_tok_s"] / ar[d]["agg_tok_s"]} for d in dss if p1[d]}}
json.dump(out, open(f"{R}/p2_e2b_blocksweep_T{T}_c{conc}.json", "w"), indent=1)
for k, v in out.items():
    if "mean_tau" in v: print(k, "tau %.2f tok/s %.0f speedup %.2fx" % (v["mean_tau"], v["mean_tok_s"], v["mean_speedup_vs_ar"]), {d: (round(r["tau"], 2), round(r["tok_s"])) for d, r in v["by_ds"].items()})
    else: print(k, {d: (round(r["tau"], 2), round(r["tok_s"])) for d, r in v["by_ds"].items()})
