"""Quantized-target cells vs bf16 / quantized AR on the first NF prompts (default 20). Usage: python summarize_quant.py [tags...]"""
import json, os, sys, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results", "gb10"); L = os.path.join(H, "..", "..", "logs", "gb10")
DS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]; NF = int(os.environ.get("NFIRST", "20"))
def cells(tag):
    out = {}
    for d in DS:
        p = f"{R}/{tag}_{d}_T0_c1.json"
        if not os.path.exists(p): continue
        rq = json.load(open(p))["requests"][:NF]
        no = sum(r["n_out"] for r in rq); lat = sum(r["latency"] for r in rq); vc = sum(r["spec_verify_ct"] or 0 for r in rq)
        out[d] = (no / lat, no / vc if vc else 1.0)
    return out
ar = cells("ar")
tags = sys.argv[1:] or ["fp8_ar", "fp8_dspine_b16", "fp8_f_tree32_d31", "fp8_f_tree48_d31", "nvfp4_ar", "nvfp4_dspine_b16", "nvfp4_f_tree32_d31", "nvfp4_f_tree48_d31"]
res = {}
print("| tag | tau | tok/s | x bf16 AR | x own AR | gsm8k / math500 / humaneval / mbpp / mtbench tok/s | draft ms | verify ms | GPU round ms |")
print("|---|---|---|---|---|---|---|---|---|")
for t in tags:
    c = cells(t)
    if len(c) < 5: continue
    own = cells(t.split("_")[0] + "_ar") if t.split("_")[0] in ("fp8", "nvfp4") else ar
    tf = f"{L}/timing_{t}.jsonl"; seg = {}
    if os.path.exists(tf):
        rows = [json.loads(l) for l in open(tf) if l.strip()][50:]
        if rows: seg = {k: st.median(x[k] for x in rows) for k in ("draft_ms", "verify_ms", "round_ms") if k in rows[0]}
    r = {"tau": st.mean(v[1] for v in c.values()), "tok_s": st.mean(v[0] for v in c.values()),
         "x_bf16_ar": st.mean(c[d][0] / ar[d][0] for d in c), "x_own_ar": st.mean(c[d][0] / own[d][0] for d in c),
         "by_ds_tok_s": {d: v[0] for d, v in c.items()}, **seg}
    res[t] = r
    f = lambda k: f"{seg[k]:.1f}" if k in seg else "-"
    print(f"| {t} | {r['tau']:.2f} | {r['tok_s']:.1f} | {r['x_bf16_ar']:.2f}x | {r['x_own_ar']:.2f}x | " + " / ".join(f"{c[d][0]:.0f}" for d in DS)
          + f" | {f('draft_ms')} | {f('verify_ms')} | {f('round_ms')} |")
json.dump(res, open(f"{R}/summary_quant_first{NF}.json", "w"), indent=1)
