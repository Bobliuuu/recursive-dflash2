"""Phase 5 glue for the hybrid 27B target: the Phase 4 fast tree (bench/p4tree/glue_fast.py, table + selection in the DSpine draft graph)
plus (a) tree links in the EAGLE layout the GDN tree-verify kernels read (retrieve_next_token = first child, retrieve_next_sibling =
next sibling, -1 = none; nodes are in pop order so parent < child), written straight into the linear backend's captured bs=1 buffers,
and (b) KV compaction on the full-attention sub-pool of the hybrid KV pool. Installed by bench/p5h100/apply_tree_patch_27b.py. Named glue_fast.py
because the DSpine worker hook imports `glue_fast` from RDF2_TREE_DIR; glue.py (imported by the SGLang worker patch) re-exports it."""
import os, sys, importlib.util, torch
# load the Phase 4 module under another name (this file shadows it as `glue_fast`, which the DSpine worker hook imports)
_P4 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "p4tree", "glue_fast.py")
if "glue_fast_p4" not in sys.modules:
    _s = importlib.util.spec_from_file_location("glue_fast_p4", _P4); _m = importlib.util.module_from_spec(_s)
    sys.modules["glue_fast_p4"] = _m; _s.loader.exec_module(_m)
glue_fast = sys.modules["glue_fast_p4"]
chain_mask = glue_fast.chain_mask
install_propose_hook = glue_fast.install_propose_hook

_INST = None


class TreeGlue27(glue_fast.TreeGlue):
    def __init__(self):
        super().__init__()
        N = self.N; dev = "cuda"
        self.idx = torch.arange(N, dtype=torch.int32, device=dev)
        self.big = torch.full((N, N), N, dtype=torch.int32, device=dev)
        self.chain = {}
        self.lin = None

    def _linear_backend(self, worker):
        if self.lin is None:
            be = worker.target_worker.model_runner.attn_backend
            self.lin = getattr(be, "linear_attn_backend", None)
            assert self.lin is not None and hasattr(self.lin, "retrieve_next_token_list"), type(be)
        return self.lin

    def tree_links(self):
        p = self.ft.parent                                     # [N] int32, p[0] = root marker
        N = self.N; idx = self.idx
        child = (p[None, :] == idx[:, None]) & (idx[None, :] > 0)          # [i, j]: j is a child of i
        nxt = torch.where(child, idx[None, :].expand(N, N), self.big).amin(1)
        sib = (p[None, :] == p[:, None]) & (idx[None, :] > idx[:, None]) & (idx[:, None] > 0)
        nsb = torch.where(sib, idx[None, :].expand(N, N), self.big).amin(1)
        nxt = torch.where(nxt >= N, -1, nxt); nsb = torch.where(nsb >= N, -1, nsb)
        return nxt.view(1, N).to(torch.int32), nsb.view(1, N).to(torch.int32)

    def links(self, worker, bs, is_tree):
        lin = self._linear_backend(worker)
        if is_tree:
            nt, ns = lin.retrieve_next_token_list[0], lin.retrieve_next_sibling_list[0]
            a, b = self.tree_links(); nt.copy_(a); ns.copy_(b)
            return nt, ns
        if bs not in self.chain:
            N = int(worker.block_size)
            a = torch.arange(1, N + 1, dtype=torch.int32, device="cuda"); a[-1] = -1
            self.chain[bs] = (a.view(1, N).repeat(bs, 1).contiguous(), torch.full((bs, N), -1, dtype=torch.int32, device="cuda"))
        return self.chain[bs]

    def force_chain(self, worker, draft_tokens):
        """Debug only (RDF2_FORCE_CHAIN): replace the selected tree by the draft's top-1 chain (same N), keeping the whole tree pipeline."""
        N = self.N; ft = self.ft; dev = ft.parent.device
        i = torch.arange(N, device=dev)
        ft.parent.copy_((i - 1).to(torch.int32)); ft.depth.copy_(i.to(torch.int32))
        w = torch.arange(2, device=dev)
        bits = (i[:, None, None] >= (w[None, :, None] * 64 + torch.arange(64, device=dev)[None, None, :]))       # [N, 2, 64]
        vals = (bits.long() << torch.arange(64, device=dev)).sum(-1)                                            # int64 wraps like uint64
        ft.anc.copy_(vals.reshape(-1))
        ft.tokens[1:].copy_(draft_tokens[0, 1:N])          # the draft's own chain, before tree_pack overwrites it

    def prepare(self, worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d):
        if os.environ.get("RDF2_FORCE_CHAIN"):
            self.force_chain(worker, draft_tokens)
        r = super().prepare(worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d)
        if os.environ.get("RDF2_DEBUG"):
            P = int(prefix_lens[0]); k = (self.N * (P + self.N) + 7) // 8
            self.dbg_mask = (k, self.fmask[:k].clone(), positions_2d[0].clone())
        return r

    def accept(self, worker, logits_output, kv_loc_plan, prefix_lens):
        if os.environ.get("RDF2_DEBUG") and getattr(self, "dbg_n", 0) % 50 == 0:
            k, m0, p0 = self.dbg_mask
            be = worker.target_worker.model_runner.attn_backend; fb = getattr(be, "full_attn_backend", be)
            print("RDF2DBG mask_same", bool(torch.equal(m0, self.fmask[:k])), "nbytes", k, "prefill_backend", getattr(fb, "prefill_backend", None),
                  "dispatch", getattr(fb, "dispatch_reason", None), "pos", p0.tolist()[:20], flush=True)
        if not self.kv_ready:
            pool = worker.target_worker.model_runner.token_to_kv_pool
            pool = getattr(pool, "full_kv_pool", pool)
            self.tv.set_kv(pool.k_buffer, pool.v_buffer); self.kv_ready = True
        if os.environ.get("RDF2_DEBUG"):
            loc0 = kv_loc_plan.write_ids(worker.target_worker.model_runner.kv_index_translator).reshape(-1).long()
            kb = self.tv_kbuf if hasattr(self, "tv_kbuf") else None
            pool = getattr(worker.target_worker.model_runner.token_to_kv_pool, "full_kv_pool", None)
            kb = pool.k_buffer if pool is not None else None
            snap = [kb[j][loc0].clone() for j in (0, len(kb) - 1)] if kb is not None else None
        out = super().accept(worker, logits_output, kv_loc_plan, prefix_lens)
        if os.environ.get("RDF2_DEBUG") and snap is not None:
            L = int(self.tv.accept_len.view(-1)[0]); pth = self.tv.path[: L + 1]
            ok = all(torch.equal(kb[j][loc0[: L + 1]], snap[t][pth]) for t, j in enumerate((0, len(kb) - 1)))
            if not ok or pth.tolist() != list(range(L + 1)):
                print("RDF2DBG kvmove ok", ok, "L", L, "path", pth.tolist(), "nbuf", len(kb), tuple(kb[0].shape), "loc", loc0[:6].tolist(), flush=True)
        if os.environ.get("RDF2_DEBUG"):
            self.dbg_n = getattr(self, "dbg_n", 0) + 1
            if self.dbg_n % 50 == 1:
                lin = self._linear_backend(worker); fm = lin.forward_metadata
                print("RDF2DBG parent", self.ft.parent.tolist()[:20], "\nRDF2DBG kparent", lin.retrieve_parent_token_list[0][0].tolist()[:20],
                      "\nRDF2DBG next", lin.retrieve_next_token_list[0][0].tolist()[:20], "fm_parent_none", fm.retrieve_parent_token is None,
                      "topk", lin.topk, "path", self.tv.path.tolist()[:12], "acc", int(self.tv.accept_len.view(-1)[0]), flush=True)
        return out


def instance():
    global _INST
    if _INST is None:
        assert not torch.cuda.is_current_stream_capturing()
        _INST = TreeGlue27()
    return _INST


glue_fast.instance = instance    # the draft-graph propose hook (glue_fast_p4.install_propose_hook) resolves instance() at call time


if os.environ.get("RDF2_DEBUG"):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch as _FB
    _orig_csm = _FB.compute_spec_mrope_positions
    def _dbg_csm(self, model_runner, batch, seq_positions=None):
        _orig_csm(self, model_runner, batch, seq_positions)
        if self.forward_mode.is_target_verify():
            _dbg_csm.n = getattr(_dbg_csm, "n", 0) + 1
            if _dbg_csm.n % 20 == 1:
                print("RDF2DBG mrope", self.mrope_positions[0, :20].tolist(), "pos", None if self.positions is None else self.positions[:20].tolist(), flush=True)
    _FB.compute_spec_mrope_positions = _dbg_csm
