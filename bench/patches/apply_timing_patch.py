"""Patch the installed SGLang 0.5.22 dflash_worker_v2.py (used by DFLASH and, via subclass, DSPINE) to log
per-round GPU time split into DRAFT (draft-model forward + draft sampling + verify-input prep) and VERIFY
(target forward over the block) with CUDA events. Active only when env SPEC_TIMING_FILE is set.
Idempotent. Usage: python apply_timing_patch.py [--revert]"""
import sys, os, sglang
d = os.path.dirname(sglang.__file__) + "/srt/speculative"
f = d + "/dflash_worker_v2.py"
src = open(f).read()
MARK = "# SPEC_TIMER_PATCH"
if "--revert" in sys.argv:
    if os.path.exists(f + ".orig"):
        os.replace(f + ".orig", f); print("reverted")
    sys.exit()
if MARK in src:
    print("already patched"); sys.exit()
open(f + ".orig", "w").write(src)
timer = '''import os, json, torch
class SpecTimer:
    """CUDA-event round timer. mark(name, bs); flush at the start of the next round."""
    def __init__(self):
        p = os.environ.get("SPEC_TIMING_FILE")
        self.f = open(p, "a", buffering=1) if p else None
        self.cur = {}
    def mark(self, name, bs=None):
        if self.f is None:
            return
        if name == "d0" and "d0" in self.cur:
            self.flush()
        e = torch.cuda.Event(enable_timing=True); e.record(torch.cuda.current_stream())
        self.cur[name] = e
        if bs is not None:
            self.cur["bs"] = bs
    def flush(self):
        c = self.cur; self.cur = {}
        if all(k in c for k in ("d0", "v0", "v1")):
            c["v1"].synchronize()
            self.f.write(json.dumps({"bs": c["bs"], "draft_ms": c["d0"].elapsed_time(c["v0"]),
                                     "verify_ms": c["v0"].elapsed_time(c["v1"])}) + "\\n")
SPEC_TIMER = SpecTimer()
'''
open(d + "/spec_timer.py", "w").write(timer)
def rep(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
rep("        with (\n            torch.inference_mode(),\n            draft_tp_context(self.draft_owns_attention),\n        ):\n            draft_out = self.draft_model_runner.forward(forward_batch)",
    "        from sglang.srt.speculative.spec_timer import SPEC_TIMER  " + MARK + "\n        SPEC_TIMER.mark('d0', bs)\n        with (\n            torch.inference_mode(),\n            draft_tp_context(self.draft_owns_attention),\n        ):\n            draft_out = self.draft_model_runner.forward(forward_batch)")
rep("        target_out = self.target_worker.forward_batch_generation(\n            batch=None,\n            forward_batch=verify_forward_batch,",
    "        SPEC_TIMER.mark('v0')\n        target_out = self.target_worker.forward_batch_generation(\n            batch=None,\n            forward_batch=verify_forward_batch,")
rep("        logits_output = target_out.logits_output\n        can_run_cuda_graph = target_out.can_run_cuda_graph\n",
    "        SPEC_TIMER.mark('v1')\n        logits_output = target_out.logits_output\n        can_run_cuda_graph = target_out.can_run_cuda_graph\n")
open(f, "w").write(src)
print("patched", f)
