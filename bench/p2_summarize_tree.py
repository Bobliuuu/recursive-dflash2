"""Summarize in-engine tree-verify cells (c=1, T=0) vs chain b16 control / chain b32. Writes bench/results/p2_e1_tree_engine_summary.json."""
import json, os, glob, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
dss = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
def cell(prefix):
    out = {}
    for d in dss:
        f = f"{R}/{prefix}_{d}_T0_c1.json"
        if os.path.exists(f):
            s = json.load(open(f))["summary"]; out[d] = {"tau": s["accept_len_pooled"], "tok_s": s["agg_tok_s"]}
    return out
ar = {d: json.load(open(f"{R}/p1_ar_{d}_T0_c1.json"))["summary"]["agg_tok_s"] for d in dss}
rows = {}
for name, pre in [("chain_b16", "p2_blk16"), ("chain_b32", "p2_blk32"), ("chain_b48", "p2_blk48")] + [(os.path.basename(f).replace("_gsm8k_T0_c1.json", "").replace("p2_", ""), os.path.basename(f).replace("_gsm8k_T0_c1.json", "")) for f in sorted(glob.glob(f"{R}/p2_tree*_gsm8k_T0_c1.json"))]:
    c = cell(pre)
    if len(c) == 5:
        rows[name] = {"by_ds": c, "mean_tau": sum(v["tau"] for v in c.values()) / 5, "mean_tok_s": sum(v["tok_s"] for v in c.values()) / 5,
                      "mean_speedup_vs_ar": sum(c[d]["tok_s"] / ar[d] for d in dss) / 5}
base = rows["chain_b16"]["mean_tok_s"]
for k, v in rows.items(): v["tok_s_vs_chain_b16"] = v["mean_tok_s"] / base
json.dump(rows, open(f"{R}/p2_e1_tree_engine_summary.json", "w"), indent=1)
for k, v in rows.items():
    print(f"{k:22s} tau {v['mean_tau']:.2f}  tok/s {v['mean_tok_s']:.0f}  speedup vs AR {v['mean_speedup_vs_ar']:.2f}x  vs chain16 {v['tok_s_vs_chain_b16']:.3f}  ", {d: round(x['tok_s']) for d, x in v['by_ds'].items()})
