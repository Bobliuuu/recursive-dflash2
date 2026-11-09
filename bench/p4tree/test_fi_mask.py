"""Correctness of the patched FlashInfer custom-mask prefill (apply_fi_mask_patch.py): tree-masked paged prefill vs an fp32 torch reference,
for several prefix lengths (incl. non-multiples of the KV tile) and with/without forced split-KV. Usage: python test_fi_mask.py"""
import torch, flashinfer
torch.manual_seed(0)
HQ, HK, D, N = 32, 8, 128, 48
dev, bf = "cuda", torch.bfloat16
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
NP = 12000
kc = torch.randn(NP, 1, HK, D, dtype=bf, device=dev); vc = torch.randn(NP, 1, HK, D, dtype=bf, device=dev)
worst = 0.0
for P in [1, 37, 63, 64, 65, 127, 200, 511, 1000, 2047, 2048, 3001, 7000]:
    for split in [None, 64, 256]:
        q = torch.randn(N, HQ, D, dtype=bf, device=dev)
        par = [-1] + [int(torch.randint(0, i, (1,))) for i in range(1, N)]
        m = torch.zeros(N, N, dtype=torch.bool)
        for i in range(N):
            j = i
            while j >= 0: m[i, j] = True; j = par[j]
        full = torch.cat([torch.ones(N, P, dtype=torch.bool), m], 1).to(dev)
        idx = torch.randperm(NP, device=dev)[: P + N].int()
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
        kw = {} if split is None else {"fixed_split_size": split}
        w.plan(torch.tensor([0, N], dtype=torch.int32, device=dev), torch.tensor([0, P + N], dtype=torch.int32, device=dev), idx,
               torch.ones(1, dtype=torch.int32, device=dev), HQ, HK, D, 1, q_data_type=bf, kv_data_type=bf, custom_mask=full.reshape(-1), **kw)
        o = w.run(q, (kc, vc)).float()
        K = kc[idx.long(), 0].float(); V = vc[idx.long(), 0].float()           # [P+N, HK, D]
        K = K.repeat_interleave(HQ // HK, 1); V = V.repeat_interleave(HQ // HK, 1)
        s = torch.einsum("qhd,khd->hqk", q.float(), K) / D ** 0.5
        s = s.masked_fill(~full[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), V)
        err = (o - ref).abs().max().item(); worst = max(worst, err)
        print(P, split, round(err, 4), flush=True)
print("patched custom-mask prefill vs fp32 reference: max abs err", worst, "OK")
