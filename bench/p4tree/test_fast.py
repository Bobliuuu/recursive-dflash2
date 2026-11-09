"""Equivalence + timing: Phase 4 tree_select/tree_pack vs Phase 2 tree_build on random DSpine-like tables (tokens, parents, depths, positions and
the packed FlashInfer mask must be bit-identical). Usage: python test_fast.py [N D rounds]"""
import sys, os, torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.append(os.path.join(os.path.dirname(here), "p2tree"))
from tree_kernels import TreeVerifier
from tree_kernels_fast import FastTree
N, D, R = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (48, 31, 300)))
torch.manual_seed(0)
old = TreeVerifier(N, D); old.set_mask_buffer(torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"))
new = FastTree(N, D); fm = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
pos = torch.zeros(N, dtype=torch.int64, device="cuda"); tok = torch.zeros(N, dtype=torch.int64, device="cuda")
bad = 0
for r in range(R):
    sharp = [0.5, 2.0, 6.0][r % 3]
    tab = (torch.randn(D, 16, 16, device="cuda") * sharp).float()
    if r % 7 == 0: tab[:, :, 3] = tab[:, :, 5]        # ties
    cand = torch.randint(0, 150000, (D, 16), device="cuda")
    anchor = torch.randint(0, 150000, (1,), device="cuda")
    P = int(torch.randint(1, 3000, (1,)).item())
    pl = torch.tensor([P], device="cuda", dtype=torch.int64 if r % 2 else torch.int32)
    old.build(tab, cand, anchor, pl.to(torch.int64))
    new.select(tab, cand, anchor); new.pack(pl, fm, pos, tok)
    nb = (N * (P + N) + 7) // 8
    ok = (torch.equal(old.tokens, new.tokens) and torch.equal(old.parent, new.parent) and torch.equal(old.depth, new.depth)
          and torch.equal(old.positions, pos) and torch.equal(new.tokens, tok) and torch.equal(old.fmask[:nb], fm[:nb])
          and torch.equal(new.commit_pos, P + torch.arange(N, device="cuda")))
    bad += not ok
print(f"N={N} D={D} rounds={R} mismatches={bad}")
# timing (GPU, back-to-back launches)
tab = torch.randn(D, 16, 16, device="cuda") * 2; cand = torch.randint(0, 150000, (D, 16), device="cuda"); anchor = cand[0, :1].clone()
pl = torch.tensor([1500], device="cuda")
def t(fn, it=200):
    for _ in range(10): fn()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return 1000 * s.elapsed_time(e) / it
print("old tree_build us", round(t(lambda: old.build(tab, cand, anchor, pl)), 1))
print("new select us", round(t(lambda: new.select(tab, cand, anchor)), 1), " pack us", round(t(lambda: new.pack(pl, fm, pos, tok)), 1))
