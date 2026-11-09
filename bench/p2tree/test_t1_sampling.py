"""Unit test of the sampled tree verifiers (Phase 4). Target = a random Markov model over a small vocab, p(. | path) = M[depth][last token],
so the exact law of the emitted tokens is known. A random DSpine-like [D,16,16] table -> CUDA best-first tree (shape + deterministic labels).
Per trial:
  det: every node draws s_i ~ p_i and tree_accept walks the deterministic tree (TreeGlue sampled mode 'det');
  rrs: children are re-drawn without replacement from q (tree_relabel, fresh Gumbels) and tree_accept_rrs walks it (mode 'rrs').
Checks the emitted o1, o2 | o1, o3 | (o1, o2) against the exact Markov law (chi-square goodness of fit) and prints mean accepted drafts.
Usage: python test_t1_sampling.py [trials] [det,rrs] [aligned]"""
import sys, os, numpy as np, torch
from scipy.stats import chisquare as _cs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_kernels import TreeVerifier
def chisquare(f, e):
    f = np.asarray(f, float); e = np.asarray(e, float)
    keep = e > 5; f = np.append(f[keep], f[~keep].sum()); e = np.append(e[keep], e[~keep].sum())
    if e[-1] < 1e-9: f, e = f[:-1], e[:-1]
    return _cs(f, e * (f.sum() / e.sum())).pvalue
torch.manual_seed(0); np.random.seed(0)
N, D, V = 48, 31, 40
M = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["det", "rrs", "hyb", "det2", "rrs2", "hyb2"]   # *2 = v2 kernels
try:
    from flashinfer.sampling import sampling_from_probs as sfp
except Exception:
    sfp = None
dev = "cuda"
cand = torch.stack([torch.randperm(V, device=dev)[:16] for _ in range(D)])
table = torch.randn(D, 16, 16, device=dev) * 1.5
anchor = 3
aligned = len(sys.argv) > 3 and sys.argv[3] == "aligned"   # q also favours lanes 0..3 -> deep accepted paths
if aligned:
    table[:, :, :4] += 3.0
# Markov target: logits noise + boost on the q-favoured candidate of the next depth (so paths go deep)
ML = torch.randn(D + 1, V, V, device=dev) * 1.5
for d in range(D):
    for l in range(16):
        ML[d, :, cand[d, l]] += float(np.random.uniform(0, 2.5) + 2.0 * aligned) * (l < 4)
Mp = torch.softmax(ML, -1)                        # Mp[d, last_token] = p(next | depth d, last token)
loc = torch.arange(1000, 1000 + N, device=dev); pl = torch.tensor([100], device=dev)
one = torch.ones(1, dtype=torch.float32, device=dev)
ok = True
for mode in modes:
    tv = TreeVerifier(N, D); tv.set_mask_buffer(torch.zeros(1 << 20, dtype=torch.uint8, device=dev))
    tv.build(table, cand, torch.tensor([anchor], device=dev), pl)
    base_tokens = tv.tokens.clone(); base_lane = tv.lane.clone()
    O = np.full((M, 3), -1, np.int64); L = np.zeros(M, np.int64)
    for t in range(M):
        th = {"hyb": 0.35, "hyb2": 0.35, "det2": 0.0}.get(mode, 2.0)    # hyb: confident nodes keep deterministic children; det2 = all det
        if mode in ("rrs", "hyb"):
            tv.tokens.copy_(base_tokens); tv.lane.copy_(base_lane)
            tv.relabel(table, cand, torch.rand(N * 16, device=dev), th)
        if mode in ("rrs2", "hyb2"):
            tv.tokens.copy_(base_tokens); tv.lane.copy_(base_lane)
            tv.relabel2(table, cand, torch.rand(N * 16, device=dev), th)
        probs = Mp[tv.depth.long(), tv.tokens].contiguous()
        if mode.endswith("2"):
            tv.accept_rrs2(torch.log(probs), one, table, cand, torch.rand(N * 17 + 2, device=dev), loc, pl, th)
        elif mode == "det":
            s = sfp(probs).to(torch.int64) if sfp is not None else torch.multinomial(probs, 1).view(-1)
            tv.accept(s, loc, pl)
        else:
            tv.accept_rrs(probs, table, cand, torch.rand(N * 17 + 2, device=dev), loc, pl, th)
        l = int(tv.accept_len.item()); out = tv.out_tokens[:3].cpu().numpy()
        L[t] = l; O[t, : min(l + 1, 3)] = out[: min(l + 1, 3)]
    P = Mp.double().cpu().numpy()
    ps = [chisquare(np.bincount(O[:, 0], minlength=V), P[0, anchor] * M)]
    print(f"[{mode}] o1 ~ p_root: p = {ps[-1]:.3f}")
    top1 = np.argsort(-np.bincount(O[:, 0][O[:, 1] >= 0], minlength=V))[:3]
    for x in top1:
        sel = (O[:, 0] == x) & (O[:, 1] >= 0); n = sel.sum()
        if n < 200: continue
        # o2 is observed only if the round continued past o1; under the AR law o2 | o1 ~ M[1][o1] regardless, but the round-continuation
        # event may depend on o2 through nothing (o2 is drawn after the continuation decision) -> unbiased
        ps.append(chisquare(np.bincount(O[sel, 1], minlength=V), P[1, x] * n)); print(f"[{mode}] o2 | o1={x} (n={n}): p = {ps[-1]:.3f}")
    sel2 = O[:, 2] >= 0
    if sel2.sum() > 500:
        pairs = O[sel2, :2]; u, c = np.unique(pairs, axis=0, return_counts=True); x1, x2 = u[np.argmax(c)]
        sel = sel2 & (O[:, 0] == x1) & (O[:, 1] == x2); n = sel.sum()
        ps.append(chisquare(np.bincount(O[sel, 2], minlength=V), P[2, x2] * n)); print(f"[{mode}] o3 | o1={x1}, o2={x2} (n={n}): p = {ps[-1]:.3f}")
    print(f"[{mode}] mean accepted drafts {L.mean():.4f} (min chi2 p {min(ps):.3f})")
    ok &= min(ps) > 1e-3
print("OK" if ok else "FAIL")
