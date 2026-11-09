"""Phase 5: Tree ReplaySSM (bench/p5h100/p4tree27b/rdf2_trs.py) on top of apply_tree_patch_27b.py. Patches
  - fused_sigmoid_gating_recurrent.py: optional `rdf2_store_flag` (RDF2_SPARSE): in the EAGLE-tree verify branch, skip the parent-state
    reload when the parent is the previous node and store a node snapshot only where flag[node] != 0;
  - gdn_triton.py: TritonGDNKernel.target_verify forwards the ring / flag kwargs;
  - gdn_backend.py: tree verify passes the TRS ring views of this layer (module rdf2_trs, when its buffers exist);
  - hybrid_linear_attn_backend.py: allocate the TRS buffers with the CUDA-graph state (before capture) when RDF2_TRS=1;
  - dflash_worker_v2.py: before the normal Mamba commit of a tree round, rebuild the last accepted node's snapshot by replay.
Idempotent; --revert restores all five files."""
import sys, os, sglang
base = os.path.dirname(sglang.__file__) + "/"
MARK = "# RDF2_TRS"
here = os.path.dirname(os.path.abspath(__file__))

def patch(rel, edits):
    f = base + rel; src = open(f).read()
    if "--revert" in sys.argv:
        if os.path.exists(f + ".pretrs"): os.replace(f + ".pretrs", f); os.utime(f, None); print("reverted", rel)
        return
    if MARK in src: print("already patched", rel); return
    open(f + ".pretrs", "w").write(src)
    for old, new in edits:
        assert src.count(old) == 1, (rel, old[:80])
        src = src.replace(old, new)
    open(f, "w").write(src); print("patched", rel)

K = "kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py"
patch(K, [
 ("    ROUND_STATE_TO_BF16: tl.constexpr = False,\n):",
  "    ROUND_STATE_TO_BF16: tl.constexpr = False,\n    rdf2_store_flag=None,  " + MARK + "\n    RDF2_SPARSE: tl.constexpr = False,\n):"),
 ("""                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)
""", """                    + o_k[:, None]
                )
                if RDF2_SPARSE:  """ + MARK + """
                    if parent_step_idx != step_idx - 1:
                        b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)
                else:
                    b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)
"""),
 ("""        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V""",
  """        if CACHE_INTERMEDIATE_STATES:
            rdf2_do = cache_idx >= 0  """ + MARK + """
            if RDF2_SPARSE:
                rdf2_do = rdf2_do & (tl.load(rdf2_store_flag + step_idx) != 0)
            if rdf2_do:
                step_offset = step_idx * HV * K * V"""),
 ("    match_cutedsl_decode: bool = False,\n):", "    match_cutedsl_decode: bool = False,\n    rdf2_store_flag: Optional[torch.Tensor] = None,  " + MARK + "\n):"),
 ("        h0_source=initial_state_source,\n", "        h0_source=initial_state_source,\n        rdf2_store_flag=rdf2_store_flag, RDF2_SPARSE=rdf2_store_flag is not None,  " + MARK + "\n"),
])
patch("srt/layers/attention/linear/kernels/gdn_triton.py", [
 ("            retrieve_parent_token=retrieve_parent_token,\n",
  "            retrieve_parent_token=retrieve_parent_token,\n"
  "            **{k_: kwargs[k_] for k_ in ('cache_ring', 'replayssm_rawv', 'replayssm_rawk', 'replayssm_g', 'replayssm_beta', 'rdf2_store_flag') if k_ in kwargs},  " + MARK + "\n"),
])
patch("srt/layers/attention/linear/gdn_backend.py", [
 ("                    stable_rows=pp_spec_stable_rows_enabled(),\n",
  "                    stable_rows=pp_spec_stable_rows_enabled(),\n"
  "                    **(__import__('sys').modules['rdf2_trs'].kwargs_for(self.req_to_token_pool.mamba2_layer_index(layer.layer_id)) if retrieve_parent_token is not None and 'rdf2_trs' in __import__('sys').modules else {}),  " + MARK + "\n"),
])
patch("srt/layers/attention/hybrid_linear_attn_backend.py", [
 ("            device=self.device,\n        )\n\n    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):\n",
  "            device=self.device,\n        )\n"
  "        import os as _o, sys as _s  " + MARK + "  (end of init_cuda_graph_state)\n"
  "        if _o.environ.get('RDF2_TRS') and _o.environ.get('RDF2_TREE_SPEC') and self.topk > 1:\n"
  "            _s.path.insert(0, _o.environ['RDF2_TREE_DIR']); import rdf2_trs as _t\n"
  "            _t.alloc(self.req_to_token_pool.get_speculative_mamba2_params_all_layers().temporal, self._model_runner.model_config.hf_text_config.linear_num_key_heads, draft_token_num, self.device)\n"
  "\n    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):\n"),
])
patch("srt/speculative/dflash_worker_v2.py", [
 ("        model_runner = self.target_worker.model_runner\n        if hasattr(attn_backend, \"update_mamba_state_after_mtp_verify\"):\n",
  "        if _rdf2_t is not None and 'rdf2_trs' in __import__('sys').modules and __import__('sys').modules['rdf2_trs'].BUF is not None:  " + MARK + "\n"
  "            _tm = __import__('sys').modules['rdf2_trs']; _lb = attn_backend.linear_attn_backend\n"
  "            _ms = _lb.req_to_token_pool.get_speculative_mamba2_params_all_layers()\n"
  "            _tm.replay(_ms.temporal, _ms.intermediate_ssm, _lb.forward_metadata.mamba_cache_indices[:1], _lb.verify_intermediate_state_indices[:1], _rdf2_t.tv.path, _rdf2_t.tv.commit_lens)\n"
  "        model_runner = self.target_worker.model_runner\n        if hasattr(attn_backend, \"update_mamba_state_after_mtp_verify\"):\n"),
])
