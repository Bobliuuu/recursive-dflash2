"""Phase 4 T=1 on the fast tree path: loads bench/p4tree/glue_fast.py (table + best-first selection inside the DSpine draft CUDA graph) and
adds the lossless sampled verification of bench/p2tree/glue.py (mode "det": every tree node draws s_i ~ softmax(logits_i / T), tree_accept
walks the deterministic tree). Use with RDF2_TREE_DIR=<repo>/bench/p4tree_t1 and bench/p4tree/apply_tree_patch_fast.py (the 'p4tree'
substring in the dir name enables its draft hook, which imports this module as `glue_fast`)."""
import importlib.util, os, sys, torch
_here = os.path.dirname(os.path.abspath(__file__))
_bench = os.path.dirname(_here)
for p in (os.path.join(_bench, "p4tree"), os.path.join(_bench, "p2tree")):
    if p not in sys.path: sys.path.append(p)
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
_fast = _load("rdf2_glue_fast_p4", os.path.join(_bench, "p4tree", "glue_fast.py"))
_p2 = _load("rdf2_glue_p2", os.path.join(_bench, "p2tree", "glue.py"))
install_propose_hook = _fast.install_propose_hook
instance = _fast.instance
chain_mask = _fast.chain_mask
G = _fast.TreeGlue
P2 = _p2.TreeGlue

_orig_init, _orig_prepare = G.__init__, G.prepare
def _init(self):
    _orig_init(self)
    self.sampled = False; self.staging_patched = False; self.n_sampled = 0; self.neg_control = None
    self.inv_t = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    try:
        from flashinfer.sampling import sampling_from_probs as _sfp
        self._sfp = _sfp
    except Exception:
        self._sfp = None
def _prepare(self, worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d, sampling_info=None):
    P2._patch_staging(self, worker._draft_sampler)
    self.sampled = not (sampling_info is None or sampling_info.is_all_greedy)
    if self.sampled:
        torch.reciprocal(sampling_info.temperatures.view(-1)[:1].to(torch.float32).clamp_min(1e-5), out=self.inv_t.view(-1))
        if self.n_sampled == 0:
            print(f"[RDF2_TREE fast] sampled tree round: DSpine sampled-chain staged={worker._draft_sampler.round_samples}", file=sys.stderr, flush=True)
    return _orig_prepare(self, worker, bs, block_ids, prefix_lens, draft_tokens, positions_2d)
def _accept(self, worker, logits_output, kv_loc_plan, prefix_lens, sampling_info=None):
    tv = self.tv
    if not self.kv_ready:
        pool = worker.target_worker.model_runner.token_to_kv_pool
        tv.set_kv(pool.k_buffer, pool.v_buffer); self.kv_ready = True
    am = P2.node_tokens(self, logits_output.next_token_logits)
    loc = kv_loc_plan.write_ids(worker.target_worker.model_runner.kv_index_translator).reshape(-1)
    if loc.dtype != torch.int64: loc = loc.to(torch.int64)
    pl = prefix_lens[:1]
    if pl.dtype != torch.int64: pl = pl.to(torch.int64)
    tv.accept(am, loc, pl)
    hs = logits_output.hidden_states
    logits_output.hidden_states = hs.view(self.N, -1).index_select(0, tv.path)
    return tv.accept_len, tv.commit_lens, tv.bonus, tv.out_tokens.view(1, self.N), tv.new_seq_lens
G.__init__ = _init; G.prepare = _prepare; G.accept = _accept; G.eligible = staticmethod(P2.eligible)
G._patch_staging = P2._patch_staging
