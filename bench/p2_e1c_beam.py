"""E1c: GPU-friendly dynamic tree = per-depth beam (keep top-W joint-probability partial paths at every depth; fixed shape 1+15W nodes,
no heap). Replay vs true greedy continuation. Usage: python p2_e1c_beam.py DUMP META OUT.json"""
import sys, json, numpy as np
from p2_common import *
from p2_e1_tree import softmax

def beam_tree(rd, W, depth=T1, tau=1.0):
    P = softmax(rd["tab"], tau)
    nodes = []  # (depth, lane, parent_idx)
    beam = [(1.0, -1, 0)]  # (pathprob, node_idx, lane) root: lane irrelevant (row 0 uses pred 0)
    for d in range(depth):
        cands = []
        for (pp, nid, lane) in beam:
            row = P[d, lane if d > 0 else 0]
            top = np.argsort(-row)[:W]
            for c in top: cands.append((pp * row[c], nid, int(c)))
        cands.sort(key=lambda x: -x[0]); new = []
        for (pp, par, c) in cands[:W]:
            nodes.append((d + 1, c, par)); new.append((pp, len(nodes) - 1, c))
        beam = new
    return nodes

def replay(rd, nodes):
    cand = rd["cand"]; truth = rd["truth"]; children = {}
    for i, (d, c, par) in enumerate(nodes): children.setdefault(par, {})[int(cand[d - 1, c])] = i
    cur = -1; L = 0
    while L < len(truth) and L < T1:
        nxt = children.get(cur, {}).get(int(truth[L]))
        if nxt is None: break
        cur = nxt; L += 1
    return L

if __name__ == "__main__":
    reqs = load(sys.argv[1], sys.argv[2]); allr = [(r["ds"], rd) for r in reqs for rd in r["rounds"]]
    out = {"chain_emit": float(np.mean([rd["L"] + 1 for _, rd in allr]))}
    for W in (1, 2, 3, 4, 5):
        Ls = np.array([replay(rd, beam_tree(rd, W)) for _, rd in allr])
        out[f"W{W}"] = {"nodes_incl_anchor": 1 + 15 * W, "emit_mean": float(Ls.mean() + 1)}
        print(W, out[f"W{W}"], flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)
