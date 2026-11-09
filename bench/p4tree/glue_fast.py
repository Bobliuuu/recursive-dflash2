"""Phase 4 glue (bs=1, greedy): same tree verify as bench/p2tree/glue.py, with the predecessor-conditioned table and the best-first selection
folded into the DSpine draft CUDA graph (hook on DSpineDraftSampler.propose), so the per-round host work before verify is one pack kernel.
Env: RDF2_TREE_SPEC="N:D". Installed by bench/p4tree/apply_tree_patch_fast.py (module name `glue` is kept for the worker patch)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p2tree"))   # tree_kernels (accept / kv_move)
from tree_kernels_fast import FastTree

_INST = None
# exact: fp32 refinement table (Phase 2 math, torch ops inside the draft graph)
# approx: DSpine's own greedy-chain table (dspine_table.py) = the logit numerator refined.(head*norm_w) WITHOUT the final RMSNorm factor
# rescaled: approx * rho_j, rho_j = rsqrt(mean(refined_j^2)+eps) with ||refined_j||^2 expanded as ||x||^2 + 2s (W^T x).m_j + s^2 m_j^T (W^T W) m_j
#           (no [rows*16, H] fp32 intermediate; m_j from DSpine's bf16 TSV factor table)
# factored: the exact fp32 table re-associated so that no [rows*16, H] fp32 intermediate or large fp32 GEMM is needed per round:
#           logit[j,c] = rho_j * ( (x*norm_w).head_c + s * m_j . TGf[c] ),  TGf[v] = W^T (head_v*norm_w), m_j from TSVf[v] = [G_p; A] sqrt(C) code_v,
#           TGf / TSVf precomputed once in fp32 (~0.9 GB), rho_j as in "rescaled". Same math as "exact" up to fp32 summation order.
TABLE_MODE = os.environ.get("RDF2_TABLE", "exact")   # Phase 4 headline uses RDF2_TABLE=factored (default kept at exact: bench/p4tree_t1 loads this module)
_GRAM = {}
_FT = {}


def factored_tables(sampler):
    key = id(sampler)
    if key not in _FT:
        assert not torch.cuda.is_current_stream_capturing(), "factored tables must be built before graph capture"
        assert sampler.res_proj_f32 is None, "terminal residual not supported by the factored table"
        f32 = torch.float32
        bridge = sampler.draft_model.selected_prefix_bridge
        cb = bridge.codebook; sqrt_c = float(cb.shape[1]) ** 0.5
        PV = torch.cat([sampler.pred_gate_f32, sampler.prev_f32], 0)       # [2R, C]
        W = sampler.write_f32                                                # [H, R]
        V = cb.shape[0]
        tsv = torch.empty(V, PV.shape[0], dtype=f32, device=cb.device)
        tg = torch.empty(V, W.shape[1], dtype=f32, device=cb.device)
        for s in range(0, V, 8192):
            e = min(V, s + 8192)
            tsv[s:e] = (sqrt_c * cb[s:e].to(f32)) @ PV.T
            tg[s:e] = (sampler.lm_head_weight[s:e].to(f32) * sampler.norm_f32) @ W
        _FT[key] = (tsv, tg, (W.T @ W).contiguous())
    return _FT[key]


def factored_table(sampler, D):
    tsv, tg, G = factored_tables(sampler)
    f32 = torch.float32
    k = sampler.k; R = G.shape[0]; W = sampler.write_f32; H = W.shape[0]
    cand = sampler.candidate_out[0, :D]
    sid = torch.cat((sampler.anchors[:1].view(1, 1).expand(1, k), cand[:-1]), dim=0)    # [D, k]
    t = tsv[sid]                                                             # [D, k(j), 2R]
    q = sampler.query[1:D + 1, :R].to(f32)
    m = torch.sigmoid(q[:, None, :] + t[..., :R] + sampler.gate_b_f32) * t[..., R:]       # [D, j, R]
    x = sampler.raw[1:D + 1].to(f32); sc = sampler.rms[1:D + 1].to(f32)
    base = torch.einsum("nh,nch->nc", x * sampler.norm_f32, sampler.lm_head_weight[cand].to(f32))   # [D, c]
    num = base[:, None, :] + sc[:, None, None] * torch.bmm(m, tg[cand].transpose(1, 2))          # [D, j, c]
    quad = ((m @ G) * m).sum(-1)
    lin = (m * (x @ W)[:, None, :]).sum(-1)
    nsq = x.square().sum(-1, keepdim=True) + 2 * sc[:, None] * lin + sc[:, None].square() * quad
    rho = torch.rsqrt(nsq / H + sampler.norm_eps)
    return (num * rho[:, :, None]).contiguous(), cand


def rescale_approx(sampler, s_tab, D):
    """s_tab [D,16,16] (dspine_table numerator) -> logits [D,16,16]."""
    f32 = torch.float32
    W = sampler.write_f32                                                    # [H, R]
    G = _GRAM.get(W.data_ptr())
    if G is None:
        assert not torch.cuda.is_current_stream_capturing()
        G = _GRAM[W.data_ptr()] = (W.T @ W).contiguous()
    k = sampler.k; R = W.shape[1]; H = W.shape[0]
    cand = sampler.candidate_out[0, :D]
    sid = torch.cat((sampler.anchors[:1].view(1, 1).expand(1, k), cand[:-1]), dim=0)    # [D, k]
    tsv = sampler.tsv[sid].to(f32)                                           # [D, k, 2R]
    q = sampler.query[1:D + 1, :R].to(f32)
    m = torch.sigmoid(q[:, None, :] + tsv[..., :R] + sampler.gate_b_f32) * tsv[..., R:]   # [D, k, R]
    x = sampler.raw[1:D + 1].to(f32); sc = sampler.rms[1:D + 1].to(f32)    # [D, H], [D]
    xw = x @ W                                                               # [D, R]
    quad = ((m @ G) * m).sum(-1)                                             # [D, k]
    lin = (m * xw[:, None, :]).sum(-1)
    nsq = x.square().sum(-1, keepdim=True) + 2 * sc[:, None] * lin + sc[:, None].square() * quad
    rho = torch.rsqrt(nsq / H + sampler.norm_eps)                            # [D, k]
    return (s_tab * rho[:, :, None]).contiguous()

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
    if TABLE_MODE in ("approx", "rescaled"):
        import dspine_table
        dspine_table.install()
    orig = S.propose
    def propose(self, rows, input_ids):
        orig(self, rows, input_ids)
        if rows.shape[0] == self.block_size - 1:      # bs == 1
            g = instance()
            if self.block_size == g.N:
                if not os.environ.get("RDF2_SKIP_SELECT"):   # SKIP_SELECT: timing ablation only (stale tree, still lossless)
                    if TABLE_MODE in ("approx", "rescaled"):
                        import dspine_table
                        tab, cand = dspine_table.TABLES[self.block_size - 1][: g.D], self.candidate_out[0, : g.D]
                        if TABLE_MODE == "rescaled":
                            tab = rescale_approx(self, tab, g.D)
                    elif TABLE_MODE == "factored":
                        tab, cand = factored_table(self, g.D)
                    else:
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
        if os.environ.get("RDF2_TABLE_DEBUG"):
            self.debug_compare(worker)
        return self

    def debug_compare(self, worker):
        """Eager per-round check (debug runs only): exact vs approx table and the trees they give."""
        import json
        smp = worker._draft_sampler
        ex, cand = table(smp, self.D)
        if TABLE_MODE == "factored":
            ap = factored_table(smp, self.D)[0]
        else:
            import dspine_table
            ap = dspine_table.TABLES[smp.block_size - 1][: self.D]
            if TABLE_MODE == "rescaled":
                ap = rescale_approx(smp, ap, self.D)
        if not hasattr(self, "dbg"):
            self.dbg = {"rounds": 0, "tree_same": 0, "node_overlap": 0.0, "argmax_same": 0.0, "max_abs_logit_diff": 0.0, "logprob_mae": 0.0}
            self.ft2 = FastTree(self.N, self.D)
        self.ft2.select(ex, cand, smp.anchors[:1])
        a = set(zip(self.ft.depth.tolist(), self.ft.tokens.tolist())); b = set(zip(self.ft2.depth.tolist(), self.ft2.tokens.tolist()))
        d = self.dbg; d["rounds"] += 1
        d["tree_same"] += int(torch.equal(self.ft.tokens, self.ft2.tokens) and torch.equal(self.ft.parent, self.ft2.parent))
        d["node_overlap"] += len(a & b) / self.N
        d["argmax_same"] += (ex.argmax(-1) == ap.argmax(-1)).float().mean().item()
        d["max_abs_logit_diff"] = max(d["max_abs_logit_diff"], (ex - ap).abs().max().item())
        d["logprob_mae"] += (ex.log_softmax(-1) - ap.log_softmax(-1)).abs().mean().item()
        if d["rounds"] % 100 == 0:
            r = d["rounds"]
            out = {k: (v / r if k in ("node_overlap", "argmax_same", "logprob_mae", "tree_same") else v) for k, v in d.items()}
            open(os.environ["RDF2_TABLE_DEBUG"], "w").write(json.dumps(out) + "\n")

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
