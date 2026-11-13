"""Derive README / video numbers from the raw per-request result files in bench/results/.
Writes bench/results/derived/readme_data.json. Every value is a mean of same-cell ratios to autoregressive decoding."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "results")
DS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]

def S(f):
    return json.load(open(os.path.join(R, f)))["summary"]

def mean_x(pre, ar_pre, suffix, ar_suffix=None, d=""):
    xs, taus, tks = [], [], []
    for ds in DS:
        s = S(f"{d}{pre}_{ds}_{suffix}.json"); a = S(f"{d}{ar_pre}_{ds}_{ar_suffix or suffix}.json")
        xs.append(s["agg_tok_s"] / a["agg_tok_s"]); tks.append(s["agg_tok_s"]); taus.append(s.get("accept_len_pooled") or 0)
    n = len(DS)
    return {"x": sum(xs) / n, "tok_s": sum(tks) / n, "tau": sum(taus) / n}

out = {}
# phase 1: Qwen3-8B, A100, mean over 5 datasets
p1 = {}
for m, label in [("eagle3", "EAGLE-3 (chain)"), ("dflash", "DFlash"), ("dspine", "DSpine")]:
    for T in (0, 1):
        for c in (1, 8):
            r = mean_x(f"p1_{m}", "p1_ar", f"T{T}_c{c}")
            p1[f"{m}_T{T}_c{c}"] = r
out["p1"] = p1
out["p1_rows"] = [{"T": T, "conc": c, "eagle3": p1[f"eagle3_T{T}_c{c}"]["x"], "dflash": p1[f"dflash_T{T}_c{c}"]["x"],
                   "dspine": p1[f"dspine_T{T}_c{c}"]["x"]} for T in (0, 1) for c in (1, 8)]
out["p1_bars"] = [{"cfg": f"T={T}, c={c}", "method": lab, "x": p1[f"{m}_T{T}_c{c}"]["x"]}
                  for T, c in [(0, 1), (0, 8), (1, 1), (1, 8)]
                  for m, lab in [("eagle3", "EAGLE-3"), ("dflash", "DFlash"), ("dspine", "DSpine")]]
# tree verification, batch 1, T=0
ar = "p1_ar"
chain16 = p1["dspine_T0_c1"]; chain32 = mean_x("p2_blk32", ar, "T0_c1"); tree48 = mean_x("p4_tree48_factored", ar, "T0_c1")
out["tree"] = {"chain16": chain16, "chain32": chain32, "tree48": tree48,
               "gain_pct": (tree48["x"] / chain16["x"] - 1) * 100}
out["tree_bars"] = [{"label": "Autoregressive", "x": 1.0}, {"label": "EAGLE-3 (chain)", "x": p1["eagle3_T0_c1"]["x"]},
                    {"label": "DFlash", "x": p1["dflash_T0_c1"]["x"]}, {"label": "DSpine (chain, block 16)", "x": chain16["x"]},
                    {"label": "DSpine chain, block 32", "x": chain32["x"]}, {"label": "DSpine + tree (48 nodes)", "x": tree48["x"]}]
out["tree_table"] = [{"cfg": "DSpine b16 chain", **chain16}, {"cfg": "DSpine b32 chain (same weights)", **chain32}, {"cfg": "DSpine tree, 48 nodes", **tree48}]
# block sweep (chain, same weights)
sweep = []
for blk, c in [(8, 8), (12, 8), (16, 1), (16, 8), (24, 1), (32, 1), (32, 8), (48, 1), (64, 1)]:
    r = mean_x(f"p2_blk{blk}", ar, f"T0_c{c}", f"T0_c{c}")
    sweep.append({"block": blk, "conc": f"concurrency {c}", "x": r["x"], "tau": r["tau"]})
out["block_sweep"] = sweep
# GB10 (DGX Spark), bf16
gb = lambda pre: mean_x(pre, "ar", "T0_c1", d="gb10/")
g_ar_tok = sum(S(f"gb10/ar_{d}_T0_c1.json")["agg_tok_s"] for d in DS) / 5
out["gb10"] = {"dspine_chain": gb("dspine_b16"), "dspine_b32": gb("dspine_b32"), "tree32": gb("f_tree32_d31"), "tree48": gb("f_tree48_d31"), "ar_tok_s": g_ar_tok}
out["gb10_bars"] = [{"label": "Autoregressive", "x": 1.0}, {"label": "DFlash (chain)", "x": gb("dflash_b16")["x"]},
                    {"label": "DSpine (chain)", "x": out["gb10"]["dspine_chain"]["x"]},
                    {"label": "DSpine + tree (32 nodes)", "x": out["gb10"]["tree32"]["x"]}]
json.dump(out, open(os.path.join(R, "derived", "readme_data.json"), "w") if os.makedirs(os.path.join(R, "derived"), exist_ok=True) is None else None, indent=1)
print(json.dumps({k: out[k] for k in ("tree", "gb10")}, indent=1)[:1500])
