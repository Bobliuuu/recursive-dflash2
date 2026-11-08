"""Losslessness vs AR (T=0, c=1): byte-identical fraction and mean common-prefix fraction for p2 cells vs p1 AR. Usage: python p2_lossless.py PREFIX [PREFIX...]"""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
dss = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
out = {}
for pre in sys.argv[1:]:
    ident = 0; tot = 0; fr = []
    for d in dss:
        f = f"{R}/{pre}_{d}_T0_c1.json"
        if not os.path.exists(f): continue
        ar = {r["id"]: r for r in json.load(open(f"{R}/p1_ar_{d}_T0_c1.json"))["requests"]}
        for r in json.load(open(f))["requests"]:
            a = ar[r["id"]]["text"]; b = r["text"]; tot += 1; ident += a == b
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            fr.append(n / max(len(a), len(b), 1))
    out[pre] = {"identical": ident, "total": tot, "mean_common_prefix_frac": sum(fr) / max(len(fr), 1)}
    print(pre, out[pre])
json.dump(out, open(f"{R}/p2_lossless_check.json", "w"), indent=1)
