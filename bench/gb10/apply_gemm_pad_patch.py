"""GB10 (sm_121) workaround for cuBLAS bf16 heuristics: F.linear with 32 < M < 128 rows picks kernels that are up to 25% slower than M=128
(bench/gb10/gemm_variants.py). Patch SGLang's UnquantizedLinearMethod.apply so that, when env RDF2_GEMM_PAD=<P> is set, a 2-D input with
lo < M < P rows is zero-padded to P rows and the result sliced back. Rows are independent in a GEMM, so outputs for the real rows are the
same math (cuBLAS may still pick a different kernel, i.e. bf16 rounding can differ in the last bit, like any shape change).
Env: RDF2_GEMM_PAD=128 (target rows), RDF2_GEMM_PAD_LO=32. Idempotent. Usage: python apply_gemm_pad_patch.py [--revert]"""
import sys, os, sglang
f = os.path.dirname(sglang.__file__) + "/srt/layers/quantization/unquant.py"
MARK = "# RDF2_GEMM_PAD"
src = open(f).read()
if "--revert" in sys.argv:
    if os.path.exists(f + ".prepad"): os.replace(f + ".prepad", f); print("reverted", f)
    sys.exit()
if MARK in src: print("already patched"); sys.exit()
old = """            return _bf16_gemm_dispatch_impl(x, layer.weight, bias)

        return F.linear(x, layer.weight, bias)
"""
new = """            return _bf16_gemm_dispatch_impl(x, layer.weight, bias)

        if _RDF2_PAD and x.ndim == 2 and _RDF2_PAD_LO < x.shape[0] < _RDF2_PAD:  """ + MARK + """
            m = x.shape[0]
            xp = x.new_zeros((_RDF2_PAD, x.shape[1]))
            xp[:m].copy_(x)
            return F.linear(xp, layer.weight, bias)[:m]
        return F.linear(x, layer.weight, bias)
"""
assert src.count(old) == 1, "anchor not found"
open(f + ".prepad", "w").write(src)
src = src.replace(old, new)
src = src.replace("class UnquantizedLinearMethod(LinearMethodBase):",
                  "import os as _rdf2_os  " + MARK + "\n_RDF2_PAD = int(_rdf2_os.environ.get('RDF2_GEMM_PAD', '0'))\n_RDF2_PAD_LO = int(_rdf2_os.environ.get('RDF2_GEMM_PAD_LO', '32'))\n\n\nclass UnquantizedLinearMethod(LinearMethodBase):", 1)
open(f, "w").write(src); print("patched", f)
