"""GB10 roofline micro-benchmark: achievable DRAM bandwidth (device copy / read-reduce) and dense bf16 / fp8 GEMM throughput.
Also times a bf16 GEMV-like (M = 1..256 tokens x 4096 x 12288) shape to show where verify width stops being free.
Usage: python roofline_micro.py OUT.json"""
import torch, json, sys, time
torch.backends.cuda.matmul.allow_tf32 = True
def t(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it
res = {"device": torch.cuda.get_device_name(), "cap": torch.cuda.get_device_capability()}
n = 1 << 30  # 2 GiB bf16
a = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_(); b = torch.empty_like(a)
ms = t(lambda: b.copy_(a)); res["copy_GBps"] = 2 * a.numel() * 2 / ms / 1e6
ms = t(lambda: a.sum()); res["read_sum_GBps"] = a.numel() * 2 / ms / 1e6
del a, b
for dt in [torch.bfloat16]:
    x = torch.randn(8192, 8192, device="cuda", dtype=dt); ms = t(lambda: x @ x, 10); res["gemm_bf16_TFLOPs"] = 2 * 8192**3 / ms / 1e9
try:
    x8 = torch.randn(8192, 8192, device="cuda").to(torch.float8_e4m3fn); w8 = torch.randn(8192, 8192, device="cuda").to(torch.float8_e4m3fn).t()
    one = torch.tensor(1.0, device="cuda")
    ms = t(lambda: torch._scaled_mm(x8, w8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), 10); res["gemm_fp8_TFLOPs"] = 2 * 8192**3 / ms / 1e9
except Exception as ex: res["gemm_fp8_err"] = str(ex)[:200]
# weight-streaming shape: Qwen3-8B MLP gate_up (4096 -> 2*12288), x M tokens
W = torch.randn(4096, 24576, device="cuda", dtype=torch.bfloat16)
res["gemv_sweep"] = {}
for M in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
    x = torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16)
    ms = t(lambda: x @ W, 50); res["gemv_sweep"][M] = {"ms": ms, "GBps": W.numel() * 2 / ms / 1e6}
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], "w"), indent=1)
