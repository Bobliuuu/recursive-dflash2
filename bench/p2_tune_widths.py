"""Greedy search of the per-depth beam widths (static tree shape) maximizing replayed emitted tokens/round. Vectorized over rounds.
Train on even requests, report held-out on odd (and vice versa). Usage: P2_T1=31 python p2_tune_widths.py DUMP META OUT.json N1,N2,..."""
import sys, json, numpy as np
from p2_common import *

def prep(reqs):
    P = []; C = []; TR = []; L = []; par = []
    for ri, r in enumerate(reqs):
        for rd in r["rounds"]:
            t = rd["tab"] - rd["tab"].max(-1, keepdims=True); e = np.exp(t); P.append((e / e.sum(-1, keepdims=True)).astype(np.float32))
            C.append(rd["cand"]); tr = -np.ones(T1, np.int64); tr[:len(rd["truth"])] = rd["truth"][:T1]; TR.append(tr); L.append(rd["L"]); par.append(ri % 2)
    return np.stack(P), np.stack(C), np.stack(TR), np.array(L), np.array(par)

def evalw(P, C, TR, widths):
    """mean emitted tokens (accepted + 1) for a beam tree with given widths."""
    R = P.shape[0]; ar = np.arange(R)
    bp = np.ones((R, 1), np.float32); blane = np.zeros((R, 1), np.int64); onp = np.ones((R, 1), bool)
    acc = np.zeros(R, np.int64)
    for d, w in enumerate(widths):
        if w == 0: break
        n = bp.shape[1]
        pl = blane if d > 0 else np.zeros_like(blane)
        rows = P[ar[:, None], d, pl]                       # [R, n, 16]
        sc = (bp[:, :, None] * rows).reshape(R, n * 16)
        idx = np.argsort(-sc, axis=1, kind="stable")[:, :w]
        m = idx // 16; c = idx % 16
        bp = np.take_along_axis(sc, idx, 1); blane = c
        tok = C[ar[:, None], d, c]
        onp = np.take_along_axis(onp, m, 1) & (tok == TR[:, d:d + 1])
        acc = np.where(onp.any(1), d + 1, acc)
        if w < 1: break
    return float(acc.mean() + 1)

def search(P, C, TR, budget, D0=1):
    widths = [1] * D0
    best = evalw(P, C, TR, widths)
    while sum(widths) < budget:
        cand = []
        for d in range(min(len(widths) + 1, T1)):
            w2 = list(widths) + ([0] if d == len(widths) else [])
            w2[d] += 1
            if w2[d] > 8 or (d > 0 and w2[d] > 16 * w2[d - 1]): continue
            if d == len(widths): w2[d] = 1
            cand.append((evalw(P, C, TR, w2), w2))
        v, widths = max(cand, key=lambda x: x[0])
    return widths, evalw(P, C, TR, widths)

if __name__ == "__main__":
    reqs = load(sys.argv[1], sys.argv[2]); P, C, TR, L, par = prep(reqs)
    budgets = [int(x) for x in sys.argv[4].split(",")]
    out = {"chain_emit": float(L.mean() + 1)}
    for N in budgets:
        res = {}
        for tr in (0, 1):
            m_tr = par == tr; m_te = par != tr
            w, v = search(P[m_tr], C[m_tr], TR[m_tr], N - 1)
            res[f"fold{tr}"] = {"widths": w, "train_emit": v, "heldout_emit": evalw(P[m_te], C[m_te], TR[m_te], w)}
        w, v = search(P, C, TR, N - 1)
        res["all"] = {"widths": w, "emit": v}
        out[str(N)] = res; print(N, res, flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)
