"""Phase 5: in SGLang's GDN tree-verify kernel (fused_sigmoid_gating_recurrent.py, EAGLE-tree branch) skip reloading the parent's
intermediate state when the parent is the previous step: the state is already in registers (same fp32 values that were just stored),
so results are bit-identical and only the redundant reload goes away. Idempotent. Usage: python apply_gdn_skipload.py [--revert]"""
import sys, os, sglang
f = os.path.dirname(sglang.__file__) + "/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py"
MARK = "# RDF2_SKIPLOAD"; src = open(f).read()
if "--revert" in sys.argv:
    if os.path.exists(f + ".orig"): os.replace(f + ".orig", f); os.utime(f, None); print("reverted", f)
    sys.exit()
if MARK in src: print("already patched"); sys.exit()
old = """                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)
"""
assert src.count(old) == 1
new = old.replace("                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)\n",
                  "                if parent_step_idx != step_idx - 1:  " + MARK + "\n"
                  "                    b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)\n")
open(f + ".orig", "w").write(src); open(f, "w").write(src.replace(old, new)); print("patched", f)
