"""Per-layer GEMM cost vs token count M for Qwen3-8B shapes with the real F.linear weight layout ([out, in], bf16 and FP8 per-tensor),
plus one full 36-layer weight stream estimate. Shows where verify width stops being free on this GPU.
Usage: python linear_sweep.py OUT.json"""
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
Ms = [1, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
res = {"device": torch.cuda.get_device_name(), "bf16": {}, "fp8": {}}
Ws = {k: torch.randn(o, i, device="cuda", dtype=torch.bfloat16) * 0.02 for k, (o, i) in shapes.items()}
Wl = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) * 0.02
W8 = {k: w.to(torch.float8_e4m3fn) for k, w in Ws.items()}
one = torch.tensor(1.0, device="cuda")
for M in Ms:
    per = {}
    for k, w in Ws.items():
        x = torch.randn(M, w.shape[1], device="cuda", dtype=torch.bfloat16)
        per[k] = t(lambda: F.linear(x, w))
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16); per["lm_head"] = t(lambda: F.linear(x, Wl), 10)
    per["layer_stack_ms"] = 36 * sum(per[k] for k in shapes) + per["lm_head"]
    res["bf16"][M] = per
    per8 = {}
    Mp = max(M, 16)  # _scaled_mm needs M % 16 == 0 on some arches; pad like serving kernels do
    for k, w in W8.items():
        x8 = torch.randn(Mp, w.shape[1], device="cuda").to(torch.float8_e4m3fn)
        try: per8[k] = t(lambda: torch._scaled_mm(x8, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
        except Exception as ex: per8[k] = None; per8["err"] = str(ex)[:120]
    if all(per8.get(k) for k in shapes): per8["layer_stack_ms_no_lmhead"] = 36 * sum(per8[k] for k in shapes)
    res["fp8"][M] = per8
    print(M, round(per["layer_stack_ms"], 2), per8.get("layer_stack_ms_no_lmhead"), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
