"""E1: offline tree verification replay. For each logged round (at chain anchors) build a best-first tree over DSpine's exact
predecessor-conditioned candidate table and replay against the target's true greedy continuation.
Usage: python p2_e1_tree.py DUMP META OUT.json [tau]"""
import sys, json, heapq, numpy as np
from p2_common import *

def softmax(x, tau=1.0):
    x = x / tau; x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)

def build_tree(rd, N, tau=1.0, expand=8):
    """Return nodes list: (depth, lane, parent_idx) with up to N-1 draft nodes (root=anchor excluded).
    probs: row0 = softmax(tab[0,0]); child of (depth d, lane j) = softmax(tab[d, j])."""
    P = softmax(rd["tab"], tau)  # [15, pred, cand]
    nodes = []  # (depth, lane, parent)
    heap = []; cnt = 0
    top0 = np.argsort(-P[0, 0])[:expand]
    for c in top0:
        heapq.heappush(heap, (-P[0, 0, c], cnt, 1, int(c), -1)); cnt += 1
    while heap and len(nodes) < N - 1:
        negp, _, d, c, par = heapq.heappop(heap)
        idx = len(nodes); nodes.append((d, c, par, -negp))
        if d < T1:
            row = P[d, c]
            for c2 in np.argsort(-row)[:expand]:
                heapq.heappush(heap, (negp * row[c2], cnt, d + 1, int(c2), idx)); cnt += 1
    return nodes

def replay(rd, nodes):
    cand = rd["cand"]; truth = rd["truth"]
    children = {}
    for i, (d, c, par, _) in enumerate(nodes):
        children.setdefault(par, {})[int(cand[d - 1, c])] = i
    cur = -1; L = 0
    while L < len(truth) and L < T1:
        nxt = children.get(cur, {}).get(int(truth[L]))
        if nxt is None: break
        cur = nxt; L += 1
    return L

if __name__ == "__main__":
    reqs = load(sys.argv[1], sys.argv[2])
    taus = [float(sys.argv[4])] if len(sys.argv) > 4 else [1.0]
    import os
    budgets = [int(x) for x in os.environ.get('P2_BUDGETS', '8,16,24,32,48,64').split(',')]
    res = {"n_requests": len(reqs), "bad_alignment": sum(r["bad"] for r in reqs), "by_tau": {}}
    allr = [(r["ds"], rd) for r in reqs for rd in r["rounds"]]
    res["n_rounds"] = len(allr)
    res["chain"] = {"emit_mean": float(np.mean([rd["L"] + 1 for _, rd in allr]))}
    # coverage: truth token present in candidate lanes at each depth along true path
    cov = []
    for _, rd in allr:
        ok = 0
        for d in range(min(T1, len(rd["truth"]))):
            if int(rd["truth"][d]) in set(rd["cand"][d].tolist()): ok += 1
            else: break
        cov.append(ok)
    res["cand_coverage_prefix_mean"] = float(np.mean(cov))
    dss = sorted(set(d for d, _ in allr))
    for tau in taus:
        out = {}
        for N in budgets:
            Ls = np.array([replay(rd, build_tree(rd, N, tau)) for _, rd in allr])
            out[N] = {"emit_mean": float(Ls.mean() + 1), "full_accept_frac": float((Ls == T1).mean()),
                      "by_ds": {d: float(np.mean([L + 1 for L, (dd, _) in zip(Ls, allr) if dd == d])) for d in dss}}
            print("tau", tau, "N", N, out[N]["emit_mean"], flush=True)
        res["by_tau"][str(tau)] = out
    res["chain"]["by_ds"] = {d: float(np.mean([rd["L"] + 1 for dd, rd in allr if dd == d])) for d in dss}
    json.dump(res, open(sys.argv[3], "w"), indent=1)
