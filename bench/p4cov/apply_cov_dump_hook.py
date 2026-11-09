"""Phase 4 coverage experiment: patch DSpine's SGLang worker to dump, per bs=1 draft round, the anchor, the greedy chain (T-1),
and for the first D draft positions the K candidate ids, their marginal (pre-refinement) draft logits and the exact
predecessor-conditioned refinement table [D,K,K] (serial_rs_table), all float32. Run with head_topk = 64 so the K=16/32 candidate
sets can be recovered offline as the top-16/32 of the marginal logits. Enabled by env RDF2_DUMP=<file>, RDF2_DUMP_D (default 31).
Record = 1 + (T-1) + D*K + D*K + D*K*K float32.
Usage: python apply_cov_dump_hook.py [--revert]   (patches the installed sglang_dspine/runtime/worker.py, keeps <file>.precov)"""
import os, sys
import sglang_dspine.runtime.worker as w
p = w.__file__
if "--revert" in sys.argv:
    if os.path.exists(p + ".precov"): os.replace(p + ".precov", p); print("reverted", p)
    sys.exit(0)
s = open(p).read()
if "_p4cov_dump" in s:
    print("already patched", p); sys.exit(0)
open(p + ".precov", "w").write(s)
# static buffer for the marginal logits of the candidates (allocated with the other sampler buffers, before graph capture)
a = "        self.q_out = torch.zeros((self.max_bs, gamma, k), dtype=f32, device=device)\n"
assert a in s
s = s.replace(a, a + "        self._p4cov_m = torch.zeros((n, k), dtype=f32, device=device)\n")
b = """            cand = dspine_kernels.topk_ids(logits, k)
            del logits"""
assert b in s
s = s.replace(b, """            cand = dspine_kernels.topk_ids(logits, k)
            self._p4cov_m[:n].copy_(torch.gather(logits, 1, cand))
            del logits""")
hook = '''
    def _p4cov_dump(self, bs: int) -> None:
        import os
        path = os.environ.get("RDF2_DUMP")
        if not path or bs != 1:
            return
        import numpy as np
        dm = self.draft_model
        bridge = dm.selected_prefix_bridge
        (t, k) = (self.block_size, self.k)
        (m, n) = (bs * t, bs * (t - 1))
        D = min(int(os.environ.get("RDF2_DUMP_D", "31")), n)
        cand = self.candidate_out[:bs].view(n, k)
        table = dspine_kernels.serial_rs_table(
            self.raw[:m], self.rms[:m], self.query[:m], self.anchors[:bs], cand, bridge.codebook,
            self.pred_gate_f32, self.prev_f32, self.gate_b_f32, self.write_f32, self.norm_f32,
            self.lm_head_weight, self.norm_eps, t, res_emb=self.res_emb, res_proj=self.res_proj_f32)
        rec = torch.cat([self.anchors[:1].float(), self.greedy_out[:n].float(), cand[:D].float().reshape(-1),
                         self._p4cov_m[:D].reshape(-1), table[:D].reshape(-1)]).cpu().numpy().astype(np.float32)
        with open(path, "ab") as f:
            f.write(rec.tobytes())
'''
marker = "    def after_graph_replay(self, forward_batch) -> None:"
assert marker in s
s = s.replace(marker, hook + "\n" + marker)
c = """        if not self._round_samples:
            return
        if int(forward_batch.batch_size) != self._round_bs:"""
assert c in s
s = s.replace(c, "        self._p4cov_dump(int(forward_batch.batch_size))\n" + c)
d = """        sampler.propose(hidden_states, self._draft_block_ids_buf[:bs].reshape(-1))
        if sampler.round_samples:"""
assert d in s
s = s.replace(d, d.replace("        if sampler.round_samples:", "        sampler._p4cov_dump(bs)\n        if sampler.round_samples:"))
open(p, "w").write(s)
print("patched", p)
