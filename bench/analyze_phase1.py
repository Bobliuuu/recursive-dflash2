"""Aggregate p1_* cell JSONs -> bench/results/phase1_summary.json and markdown tables (stdout).
Speedup = method tok/s / AR tok/s for the same (dataset, T, concurrency), same engine build."""
import json, glob, os, collections
R = "bench/results"
DS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
METHODS = ["ar", "eagle3", "dflash", "dspine"]
cells = {}
raw = {}
for f in glob.glob(f"{R}/p1_*_T*_c*.json"):
    j = json.load(open(f)); s = j["summary"]
    key = (s["method"], s["dataset"], s["temperature"], s["concurrency"])
    hist = None
    L = max([len(r["spec_hist"] or []) for r in j["requests"]] + [1])
    hist = [0] * L
    for r in j["requests"]:
        for k, c in enumerate(r["spec_hist"] or []): hist[k] += c
    rounds = sum(hist)
    s["tau"] = sum((k + 1) * c for k, c in enumerate(hist)) / rounds if rounds else 1.0
    s["frac_full"] = hist[-1] / rounds if rounds and L > 1 else None
    s["hist"] = hist
    cells[key] = s
    raw[key] = {r["id"]: r for r in j["requests"]}
out = {"cells": {"|".join(map(str, k)): v for k, v in cells.items()}}
def mean(x): x = [v for v in x if v is not None]; return sum(x) / len(x) if x else None
rows = []
print("| T | conc | method | tau (mean of datasets) | tok/s (mean) | speedup vs AR (mean of per-dataset ratios) | datasets |")
print("|---|---|---|---|---|---|---|")
for T in [0.0, 1.0]:
    for C in [1, 8]:
        for m in METHODS:
            ds = [d for d in DS if (m, d, T, C) in cells]
            if not ds: continue
            tau = mean([cells[(m, d, T, C)]["tau"] for d in ds])
            tps = mean([cells[(m, d, T, C)]["agg_tok_s"] for d in ds])
            sp = mean([cells[(m, d, T, C)]["agg_tok_s"] / cells[("ar", d, T, C)]["agg_tok_s"] for d in ds if ("ar", d, T, C) in cells])
            print(f"| {T:g} | {C} | {m} | {tau:.2f} | {tps:.0f} | {'' if sp is None else f'{sp:.2f}x'} | {len(ds)} |")
            rows.append(dict(T=T, conc=C, method=m, tau=tau, tok_s=tps, speedup=sp, n_datasets=len(ds)))
out["table"] = rows
print("\nPer-dataset tau / tok/s / speedup")
print("| T | conc | method | " + " | ".join(DS) + " |")
print("|---|---|---|" + "---|" * len(DS))
for T in [0.0, 1.0]:
    for C in [1, 8]:
        for m in METHODS:
            vals = []
            for d in DS:
                c = cells.get((m, d, T, C)); a = cells.get(("ar", d, T, C))
                vals.append("" if not c else f"{c['tau']:.2f} / {c['agg_tok_s']:.0f}" + (f" / {c['agg_tok_s']/a['agg_tok_s']:.2f}x" if a and m != 'ar' else ""))
            if any(vals): print(f"| {T:g} | {C} | {m} | " + " | ".join(vals) + " |")
# losslessness: T=0, c=1 and c=8, token text identical to AR on the same prompt ids
print("\nLosslessness vs AR (T=0, exact text match on identical prompts; bf16 batch-shape noise expected)")
print("| conc | method | dataset | matched / compared |")
print("|---|---|---|---|")
ll = []
for C in [1, 8]:
    for m in METHODS[1:]:
        for d in DS:
            if (m, d, 0.0, C) in raw and ("ar", d, 0.0, C) in raw:
                A = raw[("ar", d, 0.0, C)]; M = raw[(m, d, 0.0, C)]
                ids = [i for i in M if i in A]
                eq = sum(M[i]["text_sha"] == A[i]["text_sha"] for i in ids)
                import os.path
                pre = []
                for i in ids:
                    a, b = A[i]["text"], M[i]["text"]
                    n = len(os.path.commonprefix([a, b])); pre.append(n / max(1, max(len(a), len(b))))
                pre.sort()
                print(f"| {C} | {m} | {d} | {eq} / {len(ids)} | {sum(pre)/len(pre):.2f} | {pre[len(pre)//2]:.2f} |")
                ll.append(dict(conc=C, method=m, dataset=d, matched=eq, compared=len(ids), mean_prefix_frac=sum(pre)/len(pre), median_prefix_frac=pre[len(pre)//2]))
out["lossless"] = ll
# drafter-independence at T=0: do the two block-16 drafters emit identical text?
print("\nDFlash vs DSpine text identity (T=0)")
for C in [1, 8]:
    for d in DS:
        if ("dflash", d, 0.0, C) in raw and ("dspine", d, 0.0, C) in raw:
            A = raw[("dflash", d, 0.0, C)]; B = raw[("dspine", d, 0.0, C)]; ids = [i for i in A if i in B]
            print(f"  c={C} {d}: {sum(A[i]['text_sha']==B[i]['text_sha'] for i in ids)} / {len(ids)} identical")
# AR-vs-AR floor if a repeat exists
for f in glob.glob(f"{R}/p1b_ar_*_T0_c1.json"):
    j2 = {r["id"]: r for r in json.load(open(f))["requests"]}; d = os.path.basename(f).split("_")[2]
    A = raw.get(("ar", d, 0.0, 1))
    if A:
        ids = [i for i in j2 if i in A]; eq = sum(j2[i]["text_sha"] == A[i]["text_sha"] for i in ids)
        print(f"AR-vs-AR repeat floor ({d}, T=0, c=1): {eq} / {len(ids)}")
        out.setdefault("ar_vs_ar", []).append(dict(dataset=d, matched=eq, compared=len(ids)))
json.dump(out, open(f"{R}/phase1_summary.json", "w"), indent=1)

# functional equivalence: GSM8K accuracy from \\boxed{} answers (gold from the dataset parquet if available)
import re, glob as g
def boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; d = 1; k = j
    while k < len(t) and d:
        d += (t[k] == "{") - (t[k] == "}"); k += 1
    return t[j:k - 1].replace(",", "").replace("$", "").replace("\\!", "").strip()
try:
    import pandas as pd
    df = pd.read_parquet(g.glob(os.environ.get("HF_HOME", "/workspace/hf") + "/hub/datasets--openai--gsm8k/snapshots/*/main/test-00000-of-00001.parquet")[0])
    gold = {i: a.split("####")[-1].strip().replace(",", "") for i, a in enumerate(df.answer)}
    print("\nGSM8K accuracy (boxed answer == gold), n per cell in parentheses")
    print("| T | conc | method | accuracy |\n|---|---|---|---|")
    acc = []
    for T in [0.0, 1.0]:
        for C in [1, 8]:
            for m in METHODS:
                if (m, "gsm8k", T, C) in raw:
                    R_ = raw[(m, "gsm8k", T, C)]; ok = sum((boxed(r["text"]) == gold[i]) or (boxed(r["text"]) is not None and gold[i] in boxed(r["text"]).split("=")[-1].split()) for i, r in R_.items())
                    print(f"| {T:g} | {C} | {m} | {ok/len(R_):.3f} ({len(R_)}) |"); acc.append(dict(T=T, conc=C, method=m, acc=ok / len(R_), n=len(R_)))
    out["gsm8k_acc"] = acc
except Exception as e:
    print("gsm8k accuracy skipped:", e)
json.dump(out, open(f"{R}/phase1_summary.json", "w"), indent=1)
