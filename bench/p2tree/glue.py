"""Glue between SGLang's DFlash/DSpine v2 worker and the tree kernels (bs=1). Env: RDF2_TREE_SPEC="N:D" (N must equal the verify block size). Installed by apply_tree_patch.py.

Greedy rounds: walk the tree against the target's per-node argmax (Phase 2).
Sampled rounds (Phase 4, T>0 without top-k/top-p/min-p/penalties/grammar): every tree node i draws one token s_i ~ p_i = softmax(logits_i / T)
independently, and the same walk (tree_accept) follows the child whose token equals s at each visited node and emits s at the first node where
no child matches. This is SpecInfer-style multi-round recursive rejection sampling specialised to a deterministic draft tree (point-mass drafts:
"accept child c with prob p_res(c), else zero it and renormalise" chains to exactly "draw from p, follow the matching child"), so the committed
tokens are distributed exactly as AR sampling, and each node v is reached with probability p(path(v)), the maximum any lossless verifier can
reach for a fixed tree. The draft tree itself is built best-first from DSpine's exact predecessor-conditioned q (table / T).
"""
import os, torch

class TreeGlue:
    def __init__(self):
        from tree_kernels import TreeVerifier
        n, d = os.environ["RDF2_TREE_SPEC"].split(":")   # "N:D" = verify width N (= block size), max tree depth D
        self.tv = TreeVerifier(int(n), int(d))
        self.N = self.tv.N; self.D = self.tv.D
        self.kv_ready = False
        self.stats = [0, 0]
        self.commit_pos_buf = torch.zeros(self.N, dtype=torch.int64, device="cuda")
        self.ar = torch.arange(self.N, dtype=torch.int64, device="cuda")
        self.dump = os.environ.get("RDF2_TREE_STATS")
        self.sampled = False          # mode of the open round
        self.inv_t = torch.ones(1, 1, dtype=torch.float32, device="cuda")
        self.staging_patched = False
        self.n_sampled = 0
        # negative control for the distribution test only: deliberately wrong node temperature (inv_t multiplier), e.g. 1.1
        nc = os.environ.get("RDF2_TREE_NEG_CONTROL")
        self.neg_control = float(nc) if nc else None
        # sampled rounds: extra inverse temperature on the tree-building q (shape only; cannot affect losslessness)
        self.build_scale = float(os.environ.get("RDF2_TREE_BUILD_SCALE", "1.0"))
        # sampled-round verifier: "det" = deterministic best-first tree, every node draws from p, walk (optimal for a fixed tree);
        # "rrs" = same shape, children re-drawn from q without replacement, recursive rejection sampling without replacement.
        self.t1_mode = os.environ.get("RDF2_TREE_T1_MODE", "det")
        assert self.t1_mode in ("det", "rrs"), self.t1_mode
        # rrs hybrid: nodes whose q top-probability >= RDF2_TREE_DET_THRESH keep deterministic children (point-mass drafts); > 1 = pure rrs;
        # a negative value -t keeps deterministic children where the top probability is < t instead
        self.det_thresh = float(os.environ.get("RDF2_TREE_DET_THRESH", "2.0"))
        self.rrs_v = int(os.environ.get("RDF2_TREE_RRS_V", "2"))     # 2: low-overhead kernels (row stats + walk; det = walk with det_thresh 0)
        self.gum_u = torch.empty(self.N * 16, dtype=torch.float32, device="cuda")
        self.unif = torch.empty(self.N * 17 + 2, dtype=torch.float32, device="cuda")
        try:
            from flashinfer.sampling import sampling_from_probs as _sfp
            self._sfp = _sfp
        except Exception:
            self._sfp = None

    # ---- which rounds the tree may take
    @staticmethod
    def eligible(si):
        """greedy, or plain temperature sampling (no top-k/top-p/min-p filter, no penalties/logit bias/grammar)."""
        if si is None or si.is_all_greedy:
            return True
        if getattr(si, "need_top_k_sampling", True) or getattr(si, "need_top_p_sampling", True) or getattr(si, "need_min_p_sampling", True):
            return False
        if getattr(si, "vocab_mask", None) is not None or getattr(si, "logit_bias", None) is not None:
            return False
        if getattr(si, "has_custom_logit_processor", False):
            return False
        po = getattr(si, "penalizer_orchestrator", None)
        if po is not None and getattr(po, "is_required", False):
            return False
        return True

    def _patch_staging(self, sampler):
        """Draft tree rounds with DSpine's greedy staging: the tree is built from the predecessor-conditioned table, so the plugin's own
        sampled chain (serial_sample_chain + q) would be dead work. Non-eligible batches stage unchanged."""
        if self.staging_patched:
            return
        orig = sampler.stage_sampling_params
        glue = self
        def staged(*, bs, sampling_info):
            if int(bs) == 1 and glue.eligible(sampling_info):
                return orig(bs=bs, sampling_info=None)
            return orig(bs=bs, sampling_info=sampling_info)
        sampler.stage_sampling_params = staged
        self.staging_patched = True

    # ---- table from the DSpine sampler (static buffers after the draft graph replay)
    def table(self, sampler):
        """Predecessor-conditioned logit table for the first D draft rows only (same math as sglang_dspine serial_rs_table, restricted to D rows)."""
        bridge = sampler.draft_model.selected_prefix_bridge
        t, k, D = sampler.block_size, sampler.k, self.D
        f32 = torch.float32
        r = sampler.pred_gate_f32.shape[0]
        cand = sampler.candidate_out[0, :D]                                   # [D, k]
        x = sampler.raw[1:D + 1].to(f32); scale = sampler.rms[1:D + 1].to(f32); q = sampler.query[1:D + 1, :r].to(f32)
        sid = torch.cat((sampler.anchors[:1].view(1, 1).expand(1, k), cand[:-1]), dim=0)      # [D, k]
        sqrt_c = float(bridge.codebook.shape[1]) ** 0.5
        exact = sqrt_c * bridge.codebook[sid].to(f32)                         # [D, k, C]
        gate = torch.sigmoid(q[:, None, :] + exact @ sampler.pred_gate_f32.T + sampler.gate_b_f32)
        value = exact @ sampler.prev_f32.T
        if sampler.res_proj_f32 is not None:
            value = value + sampler.res_emb[sid].to(f32) @ sampler.res_proj_f32.T
        refined = x[:, None, :] + scale[:, None, None] * ((gate * value) @ sampler.write_f32.T)
        normalized = refined / torch.sqrt(refined.square().mean(-1, keepdim=True) + sampler.norm_eps) * sampler.norm_f32
        head = sampler.lm_head_weight[cand].to(f32)                           # [D, k, H]
        return torch.einsum("njh,nch->njc", normalized, head), cand

    def prepare(self, worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d, sampling_info=None):
        assert bs == 1 and int(worker.block_size) == self.N, (bs, worker.block_size, self.N)
        if self.tv.fmask is None:
            be = worker.target_worker.model_runner.attn_backend
            for _ in range(4):
                if hasattr(be, "cuda_graph_custom_mask"): break
                be = getattr(be, "full_attn_backend")
            self.tv.set_mask_buffer(be.cuda_graph_custom_mask)
            self.dummy_mask = torch.ones(self.N * (worker.model_runner.model_config.context_len + 2 * self.N) + self.N * self.N, dtype=torch.bool, device="cuda")
        self._patch_staging(worker._draft_sampler)
        self.sampled = not (sampling_info is None or sampling_info.is_all_greedy)
        if self.sampled and self.n_sampled in (0, 1, 100):
            import sys
            print(f"[RDF2_TREE] sampled tree round #{self.n_sampled}: DSpine sampled-chain staged={worker._draft_sampler.round_samples}", file=sys.stderr, flush=True)
        tab, cand = self.table(worker._draft_sampler)
        if self.sampled:
            # q at the request's temperature (T=1: unchanged)
            torch.reciprocal(sampling_info.temperatures.view(-1)[:1].to(torch.float32).clamp_min(1e-5), out=self.inv_t.view(-1))
            tab = tab * (self.inv_t.view(1, 1, 1) * self.build_scale)
        pl = prefix_lens[:1].to(torch.int64)
        self.tv.build(tab[: self.D], cand[: self.D], block_ids[0, 0:1].to(torch.int64), pl)
        if self.sampled and self.t1_mode == "rrs":
            self.tab_s = tab[: self.D].contiguous(); self.cand_s = cand[: self.D].contiguous()
            self.gum_u.uniform_()
            (self.tv.relabel2 if self.rrs_v == 2 else self.tv.relabel)(self.tab_s, self.cand_s, self.gum_u, self.det_thresh)
        elif self.sampled and self.rrs_v == 2:
            self.tab_s = tab[: self.D].contiguous(); self.cand_s = cand[: self.D].contiguous()
        draft_tokens[0].copy_(self.tv.tokens)
        positions_2d[0].copy_(self.tv.positions)
        torch.add(pl, self.ar, out=self.commit_pos_buf)
        return self

    def mask(self):
        # dummy non-None allow-mask (the real packed mask is written in place into the attention backend's custom-mask buffer by tree_build);
        # must be >= bs*(seq+N)*N + N^2 elements so SGLang does not re-pad it every round.
        return self.dummy_mask

    def node_tokens(self, logits):
        """[N] int64: greedy argmax, or one independent draw per node from softmax(logits / T)."""
        if not self.sampled:
            return torch.argmax(logits, dim=-1)
        inv_t = self.inv_t if self.neg_control is None else self.inv_t * self.neg_control
        probs = torch.softmax(logits.to(torch.float32) * inv_t, dim=-1)
        self.n_sampled += 1
        if self._sfp is not None:
            return self._sfp(probs).to(torch.int64)
        return torch.multinomial(probs, 1).view(-1)

    def accept(self, worker, logits_output, kv_loc_plan, prefix_lens, sampling_info=None):
        tv = self.tv
        if not self.kv_ready:
            pool = worker.target_worker.model_runner.token_to_kv_pool
            tv.set_kv(pool.k_buffer, pool.v_buffer); self.kv_ready = True
        loc = kv_loc_plan.write_ids(worker.target_worker.model_runner.kv_index_translator).reshape(-1).to(torch.int64)
        if self.sampled and self.rrs_v == 2:
            inv_t = self.inv_t if self.neg_control is None else self.inv_t * self.neg_control
            lg = logits_output.next_token_logits
            if lg.dtype != torch.float32 or not lg.is_contiguous(): lg = lg.float().contiguous()
            self.unif.uniform_(); self.n_sampled += 1
            tv.accept_rrs2(lg, inv_t.view(-1), self.tab_s, self.cand_s, self.unif, loc, prefix_lens[:1].to(torch.int64),
                           self.det_thresh if self.t1_mode == "rrs" else 0.0)
        elif self.sampled and self.t1_mode == "rrs":
            inv_t = self.inv_t if self.neg_control is None else self.inv_t * self.neg_control
            probs = torch.softmax(logits_output.next_token_logits.to(torch.float32) * inv_t, dim=-1).contiguous()
            self.unif.uniform_(); self.n_sampled += 1
            tv.accept_rrs(probs, self.tab_s, self.cand_s, self.unif, loc, prefix_lens[:1].to(torch.int64), self.det_thresh)
        else:
            am = self.node_tokens(logits_output.next_token_logits)
            tv.accept(am, loc, prefix_lens[:1].to(torch.int64))
        hs = logits_output.hidden_states
        logits_output.hidden_states = hs.view(self.N, -1).index_select(0, tv.path)
        return tv.accept_len, tv.commit_lens, tv.bonus, tv.out_tokens.view(1, self.N), tv.new_seq_lens

def chain_mask(N, prefix_lens_host, device="cuda"):
    """causal verify mask for the non-tree fallback: per request [N rows x (P+N)] allowed matrix, flattened."""
    parts = []
    for P in prefix_lens_host:
        P = int(P)
        parts.append(torch.cat([torch.ones(N, P, dtype=torch.bool, device=device), torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))], 1).reshape(-1))
    return torch.cat(parts)
