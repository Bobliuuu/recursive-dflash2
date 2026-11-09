"""Why does a 48-token verify cost more than a 16-token one on GB10? cuBLAS heuristics for bf16 F.linear at 32 < M < 128.
Times the Qwen3-8B per-layer GEMMs (x36) + lm_head for: default cuBLAS, cuBLASLt (torch.backends.cuda.preferred_blas_library), and
padding the token dim up to 128 rows. Usage: python gemm_variants.py OUT.json"""
import torch, json, sys
import torch.nn.functional as F
def t(fn, it=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it
H, I, V = 4096, 12288, 151936
shapes = {"qkv": (6144, H), "o": (H, 4096), "gate_up": (2 * I, H), "down": (H, I)}
Ws = {k: torch.randn(o, i, device="cuda", dtype=torch.bfloat16) * 0.02 for k, (o, i) in shapes.items()}
Wl = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.02
def stack(M, pad_to=None):
    tot = 0.0
    for k, w in list(Ws.items()) + [("lm_head", Wl)]:
        x = torch.randn(M, w.shape[1], device="cuda", dtype=torch.bfloat16)
        if pad_to and pad_to > M:
            xp = torch.zeros(pad_to, w.shape[1], device="cuda", dtype=torch.bfloat16)
            f = lambda: F.linear(torch.cat([x, xp[M:]], 0), w)[:M]
        else:
            f = lambda: F.linear(x, w)
        ms = t(f, 10 if k == "lm_head" else 30)
        tot += ms if k == "lm_head" else 36 * ms
    return tot
res = {}
for lib in ["cublas", "cublaslt"]:
    torch.backends.cuda.preferred_blas_library(lib)
    for M in [16, 32, 40, 48, 56, 64, 80, 96, 112, 128]:
        res[f"{lib}_M{M}"] = stack(M)
        if lib == "cublas" and 32 < M < 128: res[f"{lib}_M{M}_pad128"] = stack(M, 128)
        print(lib, M, round(res[f"{lib}_M{M}"], 2), round(res.get(f"{lib}_M{M}_pad128", 0), 2), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
