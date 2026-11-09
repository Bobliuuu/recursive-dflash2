"""Microbenchmark: FlashInfer fa2 paged prefill for a 48-token verify window (Qwen3-8B attention shape: 32 q heads, 8 kv heads, d=128,
page_size 1) with causal mask vs custom (tree) mask vs split (prefix no-mask + 48x48 custom + merge_state). Per-layer time x 36 = per-verify cost.
Usage: python p4_mask_bench.py"""
import torch, flashinfer, json, sys
torch.manual_seed(0)
HQ, HK, D, N, L = 32, 8, 128, 48, 36
dev = "cuda"; bf = torch.bfloat16
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
NP = 20000
kc = torch.randn(NP, 1, HK, D, dtype=bf, device=dev); vc = torch.randn(NP, 1, HK, D, dtype=bf, device=dev)
q = torch.randn(N, HQ, D, dtype=bf, device=dev)

def tree_mask(P):
    par = [-1] + [torch.randint(0, i, (1,)).item() for i in range(1, N)]
    m = torch.zeros(N, N, dtype=torch.bool)
    for i in range(N):
        j = i
        while j >= 0: m[i, j] = True; j = par[j]
    full = torch.cat([torch.ones(N, P, dtype=torch.bool), m], 1)
    return full.reshape(-1).to(dev), m.reshape(-1).to(dev)

def timeit(fn, it=200):
    for _ in range(10): fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(L): fn()
    g.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it // 10): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (it // 10)  # ms per 36 layers

res = {}
for P in [256, 512, 1024, 2048, 3072]:
    kv_idx = torch.randperm(NP, device=dev)[: P + N].int()
    qo = torch.tensor([0, N], dtype=torch.int32, device=dev)
    kvp = torch.tensor([0, P + N], dtype=torch.int32, device=dev)
    lpl = torch.ones(1, dtype=torch.int32, device=dev)
    full, small = tree_mask(P)
    r = {}
    for name, kw in [("causal", dict(causal=True)), ("custom", dict(custom_mask=full)), ("nomask", dict(causal=False))]:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(qo, kvp, kv_idx, lpl, HQ, HK, D, 1, q_data_type=bf, kv_data_type=bf, **kw)
        r[name] = timeit(lambda: w.run(q, (kc, vc)))
    # split: prefix (no mask) + tree part (custom 48x48) + merge
    wa = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    ws2 = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    wb = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws2, "NHD", backend="fa2")
    wa.plan(qo, torch.tensor([0, P], dtype=torch.int32, device=dev), kv_idx[:P].contiguous(), lpl, HQ, HK, D, 1, q_data_type=bf, kv_data_type=bf, causal=False)
    wb.plan(qo, torch.tensor([0, N], dtype=torch.int32, device=dev), kv_idx[P:].contiguous(), lpl, HQ, HK, D, 1, q_data_type=bf, kv_data_type=bf, custom_mask=small)
    def split():
        oa, la = wa.run(q, (kc, vc), return_lse=True); ob, lb = wb.run(q, (kc, vc), return_lse=True)
        return flashinfer.merge_state(oa, la, ob, lb)
    r["split"] = timeit(split)
    o_ref = None
    res[P] = r
    print(P, {k: round(v, 3) for k, v in r.items()}, flush=True)
# correctness of split vs custom at last P
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
w.plan(qo, kvp, kv_idx, lpl, HQ, HK, D, 1, q_data_type=bf, kv_data_type=bf, custom_mask=full)
o1 = w.run(q, (kc, vc)); o2, _ = split()
print("split vs custom max abs diff", (o1.float() - o2.float()).abs().max().item())
if len(sys.argv) > 1: json.dump(res, open(sys.argv[1], "w"), indent=1)
