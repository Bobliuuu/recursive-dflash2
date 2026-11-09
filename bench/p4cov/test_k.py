"""Check FastTreeK against the Phase 4 kernel (K=16, E=8) and against the offline best-first reference (bench/p4cov/sim.py) for K=32/64.
Usage (box): python bench/p4cov/test_k.py"""
import os, sys, torch, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(os.path.dirname(here), "p4tree"))
from tree_kernels_k import FastTreeK
from tree_kernels_fast import FastTree
import sim

N, D = 48, 31
g = torch.Generator(device="cuda").manual_seed(0)
bad16 = 0; badref = {}
for it in range(300):
    ft = FastTree(N, D) if it == 0 else ft
    tab = (torch.randn(D, 16, 16, device="cuda", generator=g) * 3).contiguous()
    cand = torch.randint(0, 150000, (D, 16), device="cuda", generator=g)
    anc = torch.tensor([7], device="cuda")
    if it == 0: fk = FastTreeK(N, D, 16, 8)
    ft.select(tab, cand, anc); fk.select(tab, cand, anc)
    bad16 += int(not (torch.equal(ft.tokens, fk.tokens) and torch.equal(ft.parent, fk.parent) and torch.equal(ft.anc, fk.anc)))
print("K16 E8 vs Phase 4 kernel: mismatching trees", bad16, "/ 300")
sim.D = D
for K, E in ((32, 8), (64, 8), (64, 16)):
    fk = FastTreeK(N, D, K, E); bad = 0
    for it in range(200):
        tab = (torch.randn(D, K, K, device="cuda", generator=g) * 3).contiguous()
        cand = torch.stack([torch.randperm(150000, device="cuda", generator=g)[:K] for _ in range(D)])
        fk.select(tab, cand, torch.tensor([7], device="cuda"))
        P = torch.softmax(tab.double(), -1).cpu().numpy()
        nodes = sim.best_first(P, N, E)
        c = cand.cpu().numpy()
        ref = [7] + [int(c[d - 1, l]) for (d, l, p) in nodes]
        refpar = [-1] + [p + 1 for (d, l, p) in nodes]
        bad += int(ref != fk.tokens.tolist() or refpar != fk.parent.tolist())
    badref[(K, E)] = bad
    print(f"K{K} E{E} vs offline best-first: mismatching trees {bad} / 200")
