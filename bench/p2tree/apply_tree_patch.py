"""Patch SGLang 0.5.22 dflash_worker_v2.py + dflash_utils.py for in-engine tree verification (env RDF2_TREE_SPEC; greedy and, unless RDF2_TREE_SAMPLED=0, plain-temperature sampled rounds). Idempotent. Usage: python apply_tree_patch.py [--revert]"""
import sys, os, sglang
d = os.path.dirname(sglang.__file__) + "/srt/speculative"
MARK = "# RDF2_TREE"
def patch(fname, edits):
    f = d + "/" + fname; src = open(f).read()
    if "--revert" in sys.argv:
        if os.path.exists(f + ".pretree"): os.replace(f + ".pretree", f); print("reverted", fname)
        return
    if MARK in src: print("already patched", fname); return
    open(f + ".pretree", "w").write(src)
    for old, new in edits:
        assert src.count(old) == 1, (fname, old)
        src = src.replace(old, new)
    open(f, "w").write(src); print("patched", fname)

patch("dflash_utils.py", [(
"    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)",
"    import os as _os  " + MARK + "\n    return backend_name, (bool(_os.environ.get('RDF2_TREE_SPEC')) or backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)")])

patch("dflash_worker_v2.py", [
("def _commit_accept(candidates, accept_len, bonus_tokens):",
 "import os as _rdf2_os  " + MARK + "\n_RDF2_TREE = None\n\n\ndef _commit_accept(candidates, accept_len, bonus_tokens):"),
# tree build right after the draft tokens are written
("        draft_tokens[:, 1:].copy_(draft_next)\n",
 "        draft_tokens[:, 1:].copy_(draft_next)\n"
 "        _rdf2_tree = None  " + MARK + "\n"
 "        if _rdf2_os.environ.get('RDF2_TREE_SPEC'):\n"
 "            global _RDF2_TREE\n"
 "            if _RDF2_TREE is None:\n"
 "                import sys as _sys\n"
 "                _sys.path.insert(0, _rdf2_os.environ.get('RDF2_TREE_DIR', '/workspace/recursive-dflash2/bench/p2tree'))\n"
 "                from glue import TreeGlue\n"
 "                _RDF2_TREE = TreeGlue()\n"
 "            if bs == 1 and (_is_all_greedy(batch.sampling_info) or (_rdf2_os.environ.get('RDF2_TREE_SAMPLED', '1') == '1' and _RDF2_TREE.eligible(batch.sampling_info))) and self._draft_sampler is not None and draft_out.can_run_graph:\n"
 "                _rdf2_tree = _RDF2_TREE.prepare(self, bs, block_ids, prefix_lens, draft_tokens, positions_2d, sampling_info=batch.sampling_info)\n"),
# mask
("        custom_mask = None\n\n        verify_input_ids = draft_tokens.reshape(-1)",
 "        custom_mask = None\n"
 "        if _rdf2_os.environ.get('RDF2_TREE_SPEC'):  " + MARK + "\n"
 "            if _rdf2_tree is not None:\n"
 "                custom_mask = _rdf2_tree.mask()\n"
 "            else:\n"
 "                from glue import chain_mask\n"
 "                _hl = batch.seq_lens_cpu if batch.seq_lens_cpu is not None else draft_input.nxt_kv_lens_cpu\n"
 "                custom_mask = chain_mask(block_size, [int(x) - (0 if batch.seq_lens_cpu is not None else block_size) for x in _hl[:bs].tolist()])\n"
 "\n        verify_input_ids = draft_tokens.reshape(-1)"),
# accept
("        candidates = draft_tokens\n        (\n            accept_len,\n            commit_lens,\n            bonus,\n            out_tokens,\n            new_seq_lens,\n        ) = self._accept_block(\n            candidates=candidates,\n            next_token_logits=logits_output.next_token_logits,\n            sampling_info=sampling_info,\n            draft_input=draft_input,\n            prefix_lens=prefix_lens,\n            bs=bs,\n        )\n",
 "        candidates = draft_tokens\n"
 "        if _rdf2_tree is not None:  " + MARK + "\n"
 "            (accept_len, commit_lens, bonus, out_tokens, new_seq_lens) = _rdf2_tree.accept(self, logits_output, kv_loc_plan, prefix_lens, sampling_info=sampling_info)\n"
 "        else:\n"
 "          (\n            accept_len,\n            commit_lens,\n            bonus,\n            out_tokens,\n            new_seq_lens,\n        ) = self._accept_block(\n            candidates=candidates,\n            next_token_logits=logits_output.next_token_logits,\n            sampling_info=sampling_info,\n            draft_input=draft_input,\n            prefix_lens=prefix_lens,\n            bs=bs,\n        )\n"),
# draft KV append positions
("            cache_loc_2d=cache_loc.view(verify_out_cache_loc_2d.shape),\n            positions=positions,\n            commit_lens=commit_lens,\n        )",
 "            cache_loc_2d=cache_loc.view(verify_out_cache_loc_2d.shape),\n            positions=(_rdf2_tree.commit_pos_buf if _rdf2_tree is not None else positions),  " + MARK + "\n            commit_lens=commit_lens,\n        )"),
])

# --- flashinfer backend: sync-free fast plan for the custom-mask verify graph (mask is written in place by tree_build)
import sglang.srt.layers.attention.flashinfer_backend as _fb
f = os.path.dirname(_fb.__file__) + "/flashinfer_backend.py"; src = open(f).read()
if "--revert" in sys.argv:
    if os.path.exists(f + ".pretree"): os.replace(f + ".pretree", f); print("reverted flashinfer_backend.py")
elif MARK in src: print("already patched flashinfer_backend.py")
else:
    open(f + ".pretree", "w").write(src)
    a = '            and getattr(spec_info, "custom_mask", None) is None\n            and self.prefill_backend == "fa2"'
    b = '            and (getattr(spec_info, "custom_mask", None) is None or bool(os.environ.get("RDF2_TREE_SPEC")))  ' + MARK + '\n            and self.prefill_backend == "fa2"'
    assert src.count(a) == 1; src = src.replace(a, b)
    a = '            assert use_custom_mask is None, ('
    b = '            assert use_custom_mask is None or os.environ.get("RDF2_TREE_SPEC"), ('
    assert src.count(a) == 1; src = src.replace(a, b)
    if "\nimport os\n" not in src: src = "import os\n" + src
    open(f, "w").write(src); print("patched flashinfer_backend.py")
