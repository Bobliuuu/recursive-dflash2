"""E3: does draft confidence (conditional max-prob / entropy / cumulative survival from DSpine's own table) predict acceptance?
Truncating the verify width (SPS-like) and chain-vs-tree gating. Usage: python p2_e3_conf.py DUMP META OUT.json"""
import sys, json, numpy as np
from p2_common import *
from p2_e1_tree import build_tree, replay, softmax

def auroc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    n1 = y.sum(); n0 = (~y).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); 
    # average ranks for ties
    ss = s[order]; i = 0; r = np.empty(len(s))
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        r[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    ranks[order] = r
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def chain_stats(rd):
    P = softmax(rd["tab"]); prev = 0; p = []; ent = []
    for i in range(T1):
        row = P[i, prev]; lane = int(np.argmax(row)); p.append(float(row[lane]))
        ent.append(float(-(row * np.log(row + 1e-12)).sum())); prev = lane
    return np.array(p), np.array(ent)

if __name__ == "__main__":
    reqs = load(sys.argv[1], sys.argv[2])
    allr = [(r["ds"], rd) for r in reqs for rd in r["rounds"]]
    res = {"n_rounds": len(allr)}
    # per-position (conditional on all previous drafts accepted)
    S = []; Y = []; E = []; pos = []
    for ds, rd in allr:
        p, e = chain_stats(rd); rd["p"] = p; rd["e"] = e; rd["S"] = np.cumprod(p)
        for i in range(min(T1, rd["L"] + 1)):
            if i >= len(rd["truth"]): break
            S.append(p[i]); E.append(-e[i]); Y.append(i < rd["L"]); pos.append(i)
    S = np.array(S); E = np.array(E); Y = np.array(Y); pos = np.array(pos)
    res["pos_cond_accept_rate"] = {str(i): float(Y[pos == i].mean()) for i in range(T1) if (pos == i).any()}
    res["auroc_cond_maxprob_all"] = auroc(S, Y); res["auroc_cond_negentropy_all"] = auroc(E, Y)
    res["auroc_cond_maxprob_by_pos"] = {str(i): auroc(S[pos == i], Y[pos == i]) for i in (0, 1, 3, 7, 11, 14)}
    # cumulative survival vs P(L>=i)
    cs = np.array([rd["S"] for _, rd in allr]); Ls = np.array([rd["L"] for _, rd in allr])
    res["auroc_cumsurv_predicts_L_ge"] = {str(k): auroc(cs[:, k - 1], Ls >= k) for k in (2, 4, 8, 12, 15)}
    res["auroc_cumsurv_full_block"] = auroc(cs[:, -1], Ls >= T1)
    res["mean_cumsurv_vs_actual"] = {str(k): [float(cs[:, k - 1].mean()), float((Ls >= k).mean())] for k in (1, 2, 4, 8, 12, 15)}
    # truncation policy: width w_r = 1 + #{i : S_i >= theta}, min 2 -> accepted L' = min(L, w-1).
    # Round-time vs verify width (ms): c=1 from the real-engine timer (T_verify 16/24/32 = 11.05/11.57/11.70) plus the HF graph increments below 16;
    # c=8 from the measured fixed-block runs p2_blk{8,12,16}_*_c8.json (round = tau*8/tok_s). Piecewise-linear interpolation.
    import os, glob
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    c8 = {}
    for w in (8, 12, 16):
        vals = []
        for f in glob.glob(f"{R}/p2_blk{w}_*_T0_c8.json"):
            sm = json.load(open(f))["summary"]; vals.append(sm["accept_len_pooled"] * 8 / sm["agg_tok_s"] * 1e3)
        c8[w] = float(np.mean(vals)) if vals else None
    c1 = {1: 14.0 - 0.80, 8: 14.0 - 0.57, 12: 14.0 - 0.30, 16: 14.0}
    curves = {"c1": c1, "c8": c8 if all(v for v in c8.values()) else None}
    res["round_ms_by_width"] = curves
    def rt(model, w):
        cv = curves[model]; xs = sorted(cv); return float(np.interp(w, xs, [cv[x] for x in xs]))
    base = {k: np.mean(Ls + 1) / rt(k, 16) for k in curves if curves[k]}
    trunc = []
    for th in [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        w = np.maximum(2, 1 + (cs >= th).sum(1)); w = np.minimum(w, 16)
        emit = np.minimum(Ls, w - 1) + 1
        row = {"theta": th, "mean_width": float(w.mean()), "emit": float(emit.mean())}
        for k in base:
            # per-round adaptive width: expected round time = mean over rounds of rt(w_r) (c=8 rounds share a batch; this is the optimistic per-round accounting)
            xs = sorted(curves[k]); row["speedup_" + k] = float(emit.mean() / np.mean(np.interp(w, xs, [curves[k][x] for x in xs])) / base[k])
        trunc.append(row)
    res["truncation"] = trunc
    fixed = []
    for wf in (8, 12, 16):
        emit = np.minimum(Ls, wf - 1) + 1
        row = {"width": wf, "emit": float(emit.mean())}
        for k in base: row["speedup_" + k] = float(emit.mean() / rt(k, wf) / base[k])
        fixed.append(row)
    res["fixed_width_model"] = fixed
    # tree-vs-chain gating (N=32 dynamic tree): does confidence predict where the tree helps?
    sub = allr[::3]
    Lt = np.array([replay(rd, build_tree(rd, 32)) for _, rd in sub]); Lc = np.array([rd["L"] for _, rd in sub])
    gain = Lt - Lc; sc = np.array([rd["S"][-1] for _, rd in sub]); s4 = np.array([rd["S"][3] for _, rd in sub])
    res["tree32_gain_mean"] = float(gain.mean()); res["tree_helps_frac"] = float((gain > 0).mean())
    res["auroc_tree_helps_vs_cumsurv15"] = auroc(-sc, gain > 0); res["auroc_tree_helps_vs_cumsurv4"] = auroc(-s4, gain > 0)
    # oracle gating: use tree only on rounds whose cumsurv15 < thr
    gate = []
    for thr in (0.01, 0.05, 0.1, 0.2, 0.5, 1.01):
        use = sc < thr
        emit = np.where(use, Lt, Lc).mean() + 1
        gate.append({"thr": thr, "tree_frac": float(use.mean()), "emit": float(emit)})
    res["tree_gating"] = gate
    json.dump(res, open(sys.argv[3], "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("truncation", "tree_gating")}, indent=1))
    for t in trunc: print(t)
    for g in gate: print(g)
