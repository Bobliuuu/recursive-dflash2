"""Aggregate GPU kernel time per kernel name from a torch-profiler chrome trace (per decode round). Usage: python p4_trace_kernels.py TRACE.json.gz ROUNDS"""
import json, gzip, sys, collections
f = sys.argv[1]; R = float(sys.argv[2]) if len(sys.argv) > 2 else 12
tr = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
ev = [e for e in tr["traceEvents"] if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
agg = collections.defaultdict(lambda: [0.0, 0])
for e in ev:
    n = e["name"]
    for key in ["BatchPrefillWithPagedKVCache", "gemm", "gemv", "Gemm", "nvjet", "cutlass", "triton", "rmsnorm", "RMSNorm", "rotary", "silu", "act_and_mul", "flashinfer"]:
        pass
    agg[n[:110]][0] += e["dur"]; agg[n[:110]][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"total GPU kernel time per round {tot / R / 1000:.3f} ms, kernels/round {sum(v[1] for v in agg.values()) / R:.0f}")
for n, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:30]:
    print(f"{d / R / 1000:8.3f} ms  {c / R:6.1f}/round  {n}")
