"""Unit test: CUDA best-first tree + accept vs python reference (p2_e1_tree.build_tree/replay) on dumped rounds.
Usage: P2_T1=31 python test_kernels.py DUMP META N D"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_common import *
from p2_e1_tree import build_tree, replay
from tree_kernels import TreeVerifier
import p2_common
if __name__ == "__main__":
    N = int(sys.argv[3]); D = int(sys.argv[4])
    reqs = load(sys.argv[1], sys.argv[2]); rounds = [rd for r in reqs for rd in r["rounds"]][:400]
    tv = TreeVerifier(N, D)
    diff = 0; kl = []; rl = []
    for rd in rounds:
        tab = torch.tensor(rd["tab"][:D], device="cuda"); cand = torch.tensor(rd["cand"][:D], device="cuda")
        tv.build(tab, cand, torch.tensor([int(rd["truth"][0])], device="cuda"), torch.tensor([100], device="cuda"))
        tk = tv.tokens.cpu().numpy(); pk = tv.parent.cpu().numpy(); dep = tv.depth.cpu().numpy()
        truth = rd["truth"]
        on = np.zeros(N, bool); on[0] = True
        for n in range(1, N):
            on[n] = on[pk[n]] and dep[n] <= len(truth) and tk[n] == truth[dep[n] - 1]
        am = np.full(N, -5, np.int64)
        for n in range(N):
            if on[n]: am[n] = truth[dep[n]] if dep[n] < len(truth) else 7
        tv.accept(torch.tensor(am, device="cuda"), torch.arange(1000, 1000 + N, device="cuda"), torch.tensor([100], device="cuda"))
        L = int(tv.accept_len.item())
        nodes = build_tree(rd, N, 1.0)
        # reference replay needs cand/trunc depth cap of D; e1 uses T1 cap, so only compare when D == T1
        Lref = replay(rd, nodes) if D == T1 else -1
        kl.append(L); rl.append(Lref)
        # check path validity
        path = tv.path.cpu().numpy()[:L + 1]
        assert all(pk[path[i]] == path[i - 1] for i in range(1, L + 1))
        assert (np.array([dep[p] for p in path]) == np.arange(L + 1)).all()
    kl = np.array(kl); rl = np.array(rl)
    print("rounds", len(rounds), "kernel mean accept", kl.mean(), "ref", rl.mean() if D == T1 else None, "mismatch", int((kl != rl).sum()) if D == T1 else None, "chain", np.mean([rd["L"] for rd in rounds]))
