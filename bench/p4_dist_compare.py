"""Phase 4 distribution test, analysis: two-sample chi-square tests and plug-in KL between token samples of two servers
(per prompt: marginal of each generated position, and the joint of the first 2 / 3 tokens). Positions >= 2 are produced by the
speculative verifier (position 1 comes from the prefill sampler). Also reports the same statistics for a split-half of the
reference (null calibration). Usage: python p4_dist_compare.py REF.json TEST.json [--out x.json]"""
import json, sys, math, random
import numpy as np
from scipy.stats import chi2_contingency, chi2, combine_pvalues
ref = json.load(open(sys.argv[1])); tst = json.load(open(sys.argv[2]))
outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
MINC = 10
def cats(seqs, f): return [f(s) for s in seqs]
def test(a, b):
    """chi-square homogeneity over categories with pooled count >= MINC (rest merged); plug-in KL(a||b) with +0.5 smoothing."""
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    keys = [k for k in set(ca) | set(cb) if ca[k] + cb[k] >= MINC]
    other_a = len(a) - sum(ca[k] for k in keys); other_b = len(b) - sum(cb[k] for k in keys)
    ta = [ca[k] for k in keys] + ([other_a] if other_a + other_b > 0 else [])
    tb = [cb[k] for k in keys] + ([other_b] if other_a + other_b > 0 else [])
    if len(ta) < 2: return dict(cats=len(ta), p=1.0, kl=0.0, tv=0.0)
    stat, p, dof, _ = chi2_contingency(np.array([ta, tb]))
    pa = (np.array(ta) + .5) / (len(a) + .5 * len(ta)); pb = (np.array(tb) + .5) / (len(b) + .5 * len(tb))
    return dict(cats=len(ta), chi2=float(stat), dof=int(dof), p=float(p), kl=float((pa * np.log(pa / pb)).sum()),
                tv=float(0.5 * np.abs(np.array(ta) / len(a) - np.array(tb) / len(b)).sum()))
res = {}; allp = []; nullp = []
for k in ref["samples"]:
    R = ref["samples"][k]; T = tst["samples"][k]
    L = min(min(len(s) for s in R), min(len(s) for s in T))
    rng = random.Random(0); Rs = R[:]; rng.shuffle(Rs); h = len(Rs) // 2
    rows = {}
    feats = {f"pos{i+1}": (lambda i: lambda s: s[i])(i) for i in range(L)}
    feats["joint12"] = lambda s: tuple(s[:2]); feats["joint123"] = lambda s: tuple(s[:3])
    for name, f in feats.items():
        t = test(cats(R, f), cats(T, f)); n0 = test(cats(Rs[:h], f), cats(Rs[h:2 * h], f))
        rows[name] = dict(test=t, null_split_half=n0)
        if name != "pos1": allp.append(t["p"]); nullp.append(n0["p"])
    vc = [v for v in (tst.get("verify_ct", {}).get(k) or []) if v is not None]
    res[k] = dict(n_ref=len(R), n_test=len(T), rows=rows, mean_verify_rounds_test=(sum(vc) / len(vc) if vc else None))
    print(f"\n{k}: n_ref={len(R)} n_test={len(T)} mean verify rounds (test) {res[k]['mean_verify_rounds_test']}")
    print("feature    cats   p(test vs ref)  KL(ref||test)  TV     | split-half ref: p    KL")
    for name, r in rows.items():
        t, n0 = r["test"], r["null_split_half"]
        print(f"{name:10s} {t['cats']:5d}  {t['p']:.3f}           {t['kl']:.4f}        {t['tv']:.3f}  |  {n0['p']:.3f}  {n0['kl']:.4f}")
summ = dict(n_tests=len(allp), min_p=min(allp), frac_p_below_005=sum(p < .05 for p in allp) / len(allp),
            fisher_p=float(combine_pvalues(allp, method="fisher")[1]),
            null_min_p=min(nullp), null_fisher_p=float(combine_pvalues(nullp, method="fisher")[1]))
print("\nsummary (positions >= 2 and joints):", json.dumps(summ))
if outp: json.dump(dict(per_prompt=res, summary=summ), open(outp, "w"), indent=1)
