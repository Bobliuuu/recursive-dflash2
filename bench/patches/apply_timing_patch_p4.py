"""Phase 4 fine-grained round timer, on top of bench/patches/apply_timing_patch.py (apply that first). Adds CUDA-event marks inside the
DFlash/DSpine v2 decode round and writes, per round, the GPU-timeline ms of each segment (CPU launch gaps included):
  draft_fwd  d0->d1  draft graph replay (DSpine table/tree select runs in here in the Phase 4 fast path)
  tree_prep  d1->d2  draft-token copy + Phase 2 tree build / Phase 4 pack
  vprep      d2->v0  verify input prep, FlashInfer plan, graph input copies
  verify     v0->v1  target verify forward
  accept     v1->a1  logits adjust + chain/tree accept (+ KV compaction)
  append     a1->a2  target-hidden -> draft KV append
  tail       a2->next d0  next-draft-input, scheduler, output processing until the next round's draft starts
cpu_<segment> = host wall time between the same marks (perf_counter at the mark call). Keeps the old draft_ms (d0->v0) / verify_ms (v0->v1) fields. Idempotent. Usage: python apply_timing_patch_p4.py [--revert]"""
import sys, os, sglang
d = os.path.dirname(sglang.__file__) + "/srt/speculative"
f = d + "/dflash_worker_v2.py"
MARK = "# SPEC_TIMER_P4"
src = open(f).read()
if "--revert" in sys.argv:
    if os.path.exists(f + ".p4orig"): os.replace(f + ".p4orig", f); print("reverted")
    sys.exit()
assert "# SPEC_TIMER_PATCH" in src, "apply bench/patches/apply_timing_patch.py first"
timer = '''import os, json, time, torch
class SpecTimer:
    """CUDA-event round timer (Phase 4). mark(name, bs); a round is written once the next round's d0 event has completed (no host sync)."""
    SEG = [("draft_fwd", "d0", "d1"), ("tree_prep", "d1", "d2"), ("vprep", "d2", "v0"), ("verify", "v0", "v1"),
           ("accept", "v1", "a1"), ("append", "a1", "a2"), ("tail", "a2", "next")]
    def __init__(self):
        p = os.environ.get("SPEC_TIMING_FILE")
        self.f = open(p, "a", buffering=1) if p else None
        self.cur = {}
        self.pending = []
    def mark(self, name, bs=None):
        if self.f is None:
            return
        e = torch.cuda.Event(enable_timing=True); e.record(torch.cuda.current_stream())
        tc = time.perf_counter()
        if name == "d0" and "d0" in self.cur:
            self.cur["next"] = e; self.cur["cpu_next"] = tc
            self.pending.append(self.cur); self.cur = {}
            self.flush()
        self.cur[name] = e; self.cur["cpu_" + name] = tc
        if bs is not None:
            self.cur["bs"] = bs
    def flush(self):
        # non-blocking: only rounds whose closing event has completed are written (no host sync in the decode loop)
        while self.pending and self.pending[0]["next"].query():
            self.write(self.pending.pop(0))
        if len(self.pending) > 64:
            self.pending = self.pending[-64:]
    def write(self, c):
        if all(k in c for k in ("d0", "v0", "v1")):
            r = {"bs": c["bs"], "draft_ms": c["d0"].elapsed_time(c["v0"]), "verify_ms": c["v0"].elapsed_time(c["v1"])}
            for n, a, b in self.SEG:
                if a in c and b in c:
                    r[n] = c[a].elapsed_time(c[b]); r["cpu_" + n] = 1000 * (c["cpu_" + b] - c["cpu_" + a])
            r["round_ms"] = c["d0"].elapsed_time(c["next"])
            self.f.write(json.dumps(r) + "\\n")
SPEC_TIMER = SpecTimer()
'''
if MARK in src:
    print("already patched"); sys.exit()
open(f + ".p4orig", "w").write(src)
open(d + "/spec_timer.py", "w").write(timer)
def rep(old, new):
    global src
    assert src.count(old) == 1, old
    src = src.replace(old, new)
rep("        draft_logits_output = draft_out.logits_output\n",
    "        draft_logits_output = draft_out.logits_output\n        SPEC_TIMER.mark('d1')  " + MARK + "\n")
rep("        # Must stay ahead of the target verify launch below.\n",
    "        SPEC_TIMER.mark('d2')  " + MARK + "\n        # Must stay ahead of the target verify launch below.\n")
rep("        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(",
    "        SPEC_TIMER.mark('a1')  " + MARK + "\n        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(")
rep("        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.\n        logits_output.hidden_states = None\n\n        next_draft_input",
    "        SPEC_TIMER.mark('a2')  " + MARK + "\n        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.\n        logits_output.hidden_states = None\n\n        next_draft_input")
open(f, "w").write(src)
print("patched", f)
