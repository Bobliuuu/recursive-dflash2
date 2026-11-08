"""Phase 2: patch DSpine's SGLang worker to dump, per draft round (bs=1), the anchor, greedy chain,
the K=16 candidate ids per position and the exact predecessor-conditioned logit table [15,16,16]
(serial_rs_table). Enabled by env RDF2_DUMP=<file>; records are 1+15+15*16+15*256 float64 values.
Usage: python apply_p2_dump_hook.py /workspace/DSpine/sglang_dspine/runtime/worker.py"""
import sys

p = sys.argv[1]
s = open(p).read()
hook = '''
    def _p2_dump(self, bs: int) -> None:
        import os
        path = os.environ.get("RDF2_DUMP")
        if not path or bs != 1:
            return
        import numpy as np
        dm = self.draft_model
        bridge = dm.selected_prefix_bridge
        (t, k) = (self.block_size, self.k)
        (m, n) = (bs * t, bs * (t - 1))
        cand = self.candidate_out[:bs].view(n, k)
        table = dspine_kernels.serial_rs_table(
            self.raw[:m], self.rms[:m], self.query[:m], self.anchors[:bs], cand, bridge.codebook,
            self.pred_gate_f32, self.prev_f32, self.gate_b_f32, self.write_f32, self.norm_f32,
            self.lm_head_weight, self.norm_eps, t, res_emb=self.res_emb, res_proj=self.res_proj_f32)
        rec = np.concatenate([
            self.anchors[:1].cpu().numpy().astype(np.float64),
            self.greedy_out[:n].cpu().numpy().astype(np.float64),
            cand.cpu().numpy().astype(np.float64).ravel(),
            table.cpu().numpy().astype(np.float64).ravel()])
        with open(path, "ab") as f:
            f.write(rec.tobytes())
'''
marker = "    def after_graph_replay(self, forward_batch) -> None:"
assert marker in s and "_p2_dump" not in s
s = s.replace(marker, hook + "\n" + marker)
a = """        if not self._round_samples:
            return
        if int(forward_batch.batch_size) != self._round_bs:"""
assert a in s
s = s.replace(a, "        self._p2_dump(int(forward_batch.batch_size))\n" + a)
b = """        sampler.propose(hidden_states, self._draft_block_ids_buf[:bs].reshape(-1))
        if sampler.round_samples:"""
assert b in s
s = s.replace(b, b.replace("        if sampler.round_samples:", "        sampler._p2_dump(bs)\n        if sampler.round_samples:"))
open(p, "w").write(s)
print("patched", p)
