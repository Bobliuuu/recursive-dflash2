"""E1b: STATIC tree topology (rank-path trie, no per-round tree build on GPU). Topology chosen on a train half of the requests
(by empirical frequency of the true rank path), evaluated on the held-out half (and swapped). Usage: python p2_e1b_static.py DUMP META OUT.json"""
import sys, json, heapq, numpy as np
from p2_common import *
MAXR = 8

def truth_ranks(rd):
    """rank sequence of the true continuation along the table (rank 0 = argmax of conditional row), stops when truth leaves the 16 candidates."""
    ranks = []; prev = 0
    for d in range(min(T1, len(rd["truth"]))):
        lane = np.where(rd["cand"][d] == rd["truth"][d])[0]
        if len(lane) == 0: break
        lane = int(lane[0])
        row = rd["tab"][d, prev]
        r = int((row > row[lane]).sum())
        if r >= MAXR: break
        ranks.append(r); prev = lane
    return ranks

def build_static(paths, N):
    """paths: list of rank tuples (truth paths). Greedy best-first over prefix frequencies; returns set of rank-tuples (nodes, excl. root)."""
    freq = {}
    for p in paths:
        for d in range(1, len(p) + 1):
            freq[p[:d]] = freq.get(p[:d], 0) + 1
    kids = {}
    for k in freq: kids.setdefault(k[:-1], []).append(k)
    heap = [(-freq[k], k) for k in kids.get((), [])]; heapq.heapify(heap)
    tree = set()
    while heap and len(tree) < N - 1:
        f, k = heapq.heappop(heap); tree.add(k)
        for c in kids.get(k, []): heapq.heappush(heap, (-freq[c], c))
    return tree

def acc_len(p, tree):
    L = 0
    while L < len(p) and p[:L + 1] in tree: L += 1
    return L

if __name__ == "__main__":
    reqs = load(sys.argv[1], sys.argv[2])
    for r in reqs:
        for rd in r["rounds"]: rd["rk"] = tuple(truth_ranks(rd))
    budgets = [8, 16, 24, 32, 48, 64]
    out = {}
    folds = [(0, 1), (1, 0)]
    idx = np.arange(len(reqs)) % 2
    tot = {N: [] for N in budgets}; chain = []
    for tr, te in folds:
        train = [rd["rk"] for r, i in zip(reqs, idx) if i == tr for rd in r["rounds"]]
        test = [(r["ds"], rd) for r, i in zip(reqs, idx) if i == te for rd in r["rounds"]]
        for N in budgets:
            tree = build_static(train, N)
            tot[N] += [(ds, acc_len(rd["rk"], tree) + 1) for ds, rd in test]
        chain += [(ds, rd["L"] + 1) for ds, rd in test]
    out["chain_emit"] = float(np.mean([x for _, x in chain]))
    dss = sorted(set(d for d, _ in chain))
    for N in budgets:
        out[N] = {"emit_mean": float(np.mean([x for _, x in tot[N]])), "by_ds": {d: float(np.mean([x for dd, x in tot[N] if dd == d])) for d in dss}}
        print(N, out[N]["emit_mean"], flush=True)
    # print the full-data tree depth/branch profile for N=32
    allp = [rd["rk"] for r in reqs for rd in r["rounds"]]
    t32 = build_static(allp, 32)
    out["tree32_profile_depth_counts"] = {str(d): sum(1 for k in t32 if len(k) == d) for d in range(1, 16)}
    out["tree32_nodes"] = sorted(["".join(map(str, k)) for k in t32], key=lambda s: (len(s), s))
    json.dump(out, open(sys.argv[3], "w"), indent=1)
