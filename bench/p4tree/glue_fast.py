"""Phase 4 glue (bs=1, greedy): same tree verify as bench/p2tree/glue.py, with the predecessor-conditioned table and the best-first selection
folded into the DSpine draft CUDA graph (hook on DSpineDraftSampler.propose), so the per-round host work before verify is one pack kernel.
Env: RDF2_TREE_SPEC="N:D". Installed by bench/p4tree/apply_tree_patch_fast.py (module name `glue` is kept for the worker patch)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p2tree"))   # tree_kernels (accept / kv_move)
from tree_kernels_fast import FastTree

_INST = None

def instance():
    global _INST
    if _INST is None:
        assert not torch.cuda.is_current_stream_capturing(), "tree glue must be built before graph capture"
        _INST = TreeGlue()
    return _INST


def table(sampler, D):
    """[D,16,16] predecessor-conditioned logits for draft rows 1..D (same math as sglang_dspine serial_rs_table; Phase 2 glue.table)."""
    bridge = sampler.draft_model.selected_prefix_bridge
    k = sampler.k
    f32 = torch.float32
    r = sampler.pred_gate_f32.shape[0]
    cand = sampler.candidate_out[0, :D]
    x = sampler.raw[1:D + 1].to(f32); scale = sampler.rms[1:D + 1].to(f32); q = sampler.query[1:D + 1, :r].to(f32)
    sid = torch.cat((sampler.anchors[:1].view(1, 1).expand(1, k), cand[:-1]), dim=0)
    sqrt_c = float(bridge.codebook.shape[1]) ** 0.5
    exact = sqrt_c * bridge.codebook[sid].to(f32)
    gate = torch.sigmoid(q[:, None, :] + exact @ sampler.pred_gate_f32.T + sampler.gate_b_f32)
    value = exact @ sampler.prev_f32.T
    if sampler.res_proj_f32 is not None:
        value = value + sampler.res_emb[sid].to(f32) @ sampler.res_proj_f32.T
    refined = x[:, None, :] + scale[:, None, None] * ((gate * value) @ sampler.write_f32.T)
    normalized = refined / torch.sqrt(refined.square().mean(-1, keepdim=True) + sampler.norm_eps) * sampler.norm_f32
    head = sampler.lm_head_weight[cand].to(f32)
    return torch.einsum("njh,nch->njc", normalized, head).contiguous(), cand


def install_propose_hook(S):
    """Wrap DSpineDraftSampler.propose (class S) so every bs=1 draft (graph capture and replay) also builds the tree."""
    if getattr(S.propose, "_rdf2", False): return
    orig = S.propose
    def propose(self, rows, input_ids):
        orig(self, rows, input_ids)
        if rows.shape[0] == self.block_size - 1:      # bs == 1
            g = instance()
            if self.block_size == g.N:
                if not os.environ.get("RDF2_SKIP_SELECT"):   # SKIP_SELECT: timing ablation only (stale tree, still lossless)
                    tab, cand = table(self, g.D)
                    g.ft.select(tab, cand, self.anchors[:1])
                g.selected = True
    propose._rdf2 = True
    S.propose = propose


class TreeGlue:
    def __init__(self):
        from tree_kernels import TreeVerifier
        n, d = os.environ["RDF2_TREE_SPEC"].split(":")
        self.N, self.D = int(n), int(d)
        self.ft = FastTree(self.N, self.D)
        self.tv = TreeVerifier(self.N, self.D)          # Phase 2 accept + KV-compaction kernels, reading the fast tree's buffers
        self.tv.tokens = self.ft.tokens; self.tv.parent = self.ft.parent
        self.commit_pos_buf = self.ft.commit_pos
        self.kv_ready = False; self.fmask = None; self.selected = False

    def prepare(self, worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d):
        assert bs == 1 and int(worker.block_size) == self.N and self.selected, (bs, worker.block_size, self.N, self.selected)
        if self.fmask is None:
            be = worker.target_worker.model_runner.attn_backend
            for _ in range(4):
                if hasattr(be, "cuda_graph_custom_mask"): break
                be = getattr(be, "full_attn_backend")
            self.fmask = be.cuda_graph_custom_mask
            assert self.fmask.dtype == torch.uint8
            self.dummy_mask = torch.ones(self.N * (worker.model_runner.model_config.context_len + 2 * self.N) + self.N * self.N, dtype=torch.bool, device="cuda")
        p0, t0 = positions_2d[0], draft_tokens[0]
        assert p0.dtype == torch.int64 and t0.dtype == torch.int64 and p0.is_contiguous() and t0.is_contiguous()
        self.ft.pack(prefix_lens[:1], self.fmask, p0, t0)
        return self

    def mask(self):
        return self.dummy_mask

    def accept(self, worker, logits_output, kv_loc_plan, prefix_lens):
        tv = self.tv
        if not self.kv_ready:
            pool = worker.target_worker.model_runner.token_to_kv_pool
            tv.set_kv(pool.k_buffer, pool.v_buffer); self.kv_ready = True
        am = torch.argmax(logits_output.next_token_logits, dim=-1)
        loc = kv_loc_plan.write_ids(worker.target_worker.model_runner.kv_index_translator).reshape(-1)
        if loc.dtype != torch.int64: loc = loc.to(torch.int64)
        pl = prefix_lens[:1]
        if pl.dtype != torch.int64: pl = pl.to(torch.int64)
        tv.accept(am, loc, pl)
        hs = logits_output.hidden_states
        logits_output.hidden_states = hs.view(self.N, -1).index_select(0, tv.path)
        return tv.accept_len, tv.commit_lens, tv.bonus, tv.out_tokens.view(1, self.N), tv.new_seq_lens


def chain_mask(N, prefix_lens_host, device="cuda"):
    parts = []
    for P in prefix_lens_host:
        P = int(P)
        parts.append(torch.cat([torch.ones(N, P, dtype=torch.bool, device=device), torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))], 1).reshape(-1))
    return torch.cat(parts)
