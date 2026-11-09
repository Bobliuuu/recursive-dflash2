"""Phase 4 (T=1 tree) summary: per config, mean over seeds of per-dataset tok/s and tau; x AR = mean over datasets of tok/s / AR tok/s
(AR = Phase 1 cells p1_ar_<ds>_T1_c1.json; x AR against this box's p4t1_ar_* cells is printed too), vs chain = same ratio
against this box's chain16 cells. Usage: python p4t1_summarize.py [results_dir] [out.json]"""
import json, glob, os, sys, re
from collections import defaultdict
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
cells = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{R}/p4t1_*_T1_c1_s*.json"):
    m = re.match(r"p4t1_(.+)_(gsm8k|math500|humaneval|mbpp|mtbench)_T1_c1_s(\d+)\.json", os.path.basename(f))
    s = json.load(open(f))["summary"]
    cells[m.group(1)][m.group(2)].append((int(m.group(3)), s["agg_tok_s"], s.get("accept_len_pooled", 1.0)))
ar = {d: json.load(open(f"{R}/p1_ar_{d}_T1_c1.json"))["summary"]["agg_tok_s"] for d in DS}          # Phase 1 AR (as in Phases 1-3)
arbox = {d: sum(x[1] for x in cells["ar"][d]) / len(cells["ar"][d]) for d in DS} if all(d in cells.get("ar", {}) for d in DS) else None
def mean(x): return sum(x) / len(x)
out = {"ar_tok_s": ar, "ar_tok_s_same_box": arbox, "configs": {}}
chain = {d: mean([x[1] for x in cells["chain16"][d]]) for d in DS} if "chain16" in cells else None
print(f"AR T=1 c=1 tok/s (Phase 1): " + " ".join(f"{d} {ar[d]:.1f}" for d in DS))
if arbox: print(f"AR T=1 c=1 tok/s (same box): " + " ".join(f"{d} {arbox[d]:.1f}" for d in DS))
print(f"{'config':14s} seeds  tau(mean ds)  tok/s(mean ds)  xAR    vs chain  xAR(same-box AR) | per-dataset tok/s")
for c in sorted(cells):
    if not all(d in cells[c] for d in DS): continue
    tok = {d: mean([x[1] for x in cells[c][d]]) for d in DS}; tau = {d: mean([x[2] for x in cells[c][d]]) for d in DS}
    xar = mean([tok[d] / ar[d] for d in DS]); vc = mean([tok[d] / chain[d] for d in DS]) if chain else None
    seeds = sorted({x[0] for d in DS for x in cells[c][d]})
    # per-seed x AR (spread)
    per_seed = [mean([next(x[1] for x in cells[c][d] if x[0] == s) / ar[d] for d in DS]) for s in seeds if all(any(x[0] == s for x in cells[c][d]) for d in DS)]
    xbox = mean([tok[d] / arbox[d] for d in DS]) if arbox else None
    out["configs"][c] = dict(seeds=seeds, tok_s=tok, tau=tau, x_ar=xar, x_ar_same_box_ar=xbox, vs_chain=vc, x_ar_per_seed=per_seed)
    print(f"{c:14s} {len(seeds):5d}  {mean(list(tau.values())):11.2f}  {mean(list(tok.values())):14.1f}  {xar:5.2f}x  {vc if vc else float('nan'):.3f}   {xbox if xbox else float('nan'):.2f}x | "
          + " ".join(f"{tok[d]:.0f}" for d in DS) + "   per-seed xAR " + " ".join(f"{v:.2f}" for v in per_seed))
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], "w"), indent=1)
