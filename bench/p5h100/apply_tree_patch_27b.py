"""Phase 5: tree verify on the hybrid Qwen3.8-27B (48 GatedDeltaNet + 16 attention layers). Applies the Phase 4 fast tree patches
(bench/p4tree/apply_tree_patch_fast.py) and, on top:
  0. memory_pool.py: dense (not deduplicated) per-step conv windows, as SGLang does for EAGLE trees.
  1. hybrid_linear_attn_backend.py: the linear-attention (GDN) backend runs its EAGLE-tree verify path (topk>1 buffers, parent-linked
     causal conv + recurrent kernel with per-node intermediate states) when RDF2_TREE_SPEC is set; spec_info link fields read via getattr.
  2. dflash_worker_v2.py: the verify input carries the tree links (first child / next sibling, chain links on non-tree rounds), and the
     post-verify Mamba commit takes the state of the last ACCEPTED NODE (path[commit_len-1]) instead of verify step commit_len-1.
Needs the non-ReplaySSM verify (no --enable-linear-replayssm-spec): ReplaySSM's fold path asserts a chain. Idempotent; --revert undoes all."""
import sys, os, subprocess, sglang
here = os.path.dirname(os.path.abspath(__file__))
FAST = [sys.executable, os.path.join(os.path.dirname(here), "p4tree", "apply_tree_patch_fast.py")] + sys.argv[1:]
if "--revert" not in sys.argv: subprocess.check_call(FAST)      # apply: Phase 4 patches first, these on top; revert: reverse order
MARK = "# RDF2_TREE27"
base = os.path.dirname(sglang.__file__) + "/srt/"

def patch(rel, edits):
    f = base + rel; src = open(f).read()
    if "--revert" in sys.argv:
        if os.path.exists(f + ".pretree27"): os.replace(f + ".pretree27", f); print("reverted", rel)
        return
    if MARK in src: print("already patched", rel); return
    open(f + ".pretree27", "w").write(src)
    for old, new in edits:
        assert src.count(old) == 1, (rel, old)
        src = src.replace(old, new)
    open(f, "w").write(src); print("patched", rel)

patch("layers/attention/hybrid_linear_attn_backend.py", [
 ("        self.is_draft_worker = model_runner.is_draft_worker\n",
  "        self.is_draft_worker = model_runner.is_draft_worker\n"
  "        import os as _o  " + MARK + "\n"
  "        if _o.environ.get('RDF2_TREE_SPEC') and not self.is_draft_worker and self.topk <= 1:\n"
  "            self.topk = 2\n"),
 ("retrieve_next_token = forward_batch.spec_info.retrieve_next_token\n",
  "retrieve_next_token = getattr(forward_batch.spec_info, 'retrieve_next_token', None)  " + MARK + "\n"),
 ("forward_batch.spec_info.retrieve_next_sibling\n",
  "getattr(forward_batch.spec_info, 'retrieve_next_sibling', None)\n"),
])

patch("mem_cache/memory_pool.py", [
 # the deduplicated conv-window layout (window of step s = tokens s..s+K-2 of one shared buffer) assumes a chain; tree nodes need dense windows
 ("        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)\n    )\n",
  "        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)\n        and not __import__('os').environ.get('RDF2_TREE_SPEC')  " + MARK + "\n    )\n"),
])

patch("speculative/dflash_worker_v2.py", [
 ("            kv_loc_plan=kv_loc_plan,\n        )\n",
  "            kv_loc_plan=kv_loc_plan,\n        )\n"
  "        self._rdf2_cur_tree = _rdf2_tree  " + MARK + "\n"
  "        if _rdf2_os.environ.get('RDF2_TREE_SPEC') and _RDF2_TREE is not None and hasattr(_RDF2_TREE, 'links'):\n"
  "            verify_input.retrieve_next_token, verify_input.retrieve_next_sibling = _RDF2_TREE.links(self, bs, _rdf2_tree is not None)\n"),
 ("        last_correct_step_indices = commit_lens.to(torch.int64) - 1\n",
  "        last_correct_step_indices = commit_lens.to(torch.int64) - 1\n"
  "        _rdf2_t = getattr(self, '_rdf2_cur_tree', None)  " + MARK + "\n"
  "        if _rdf2_t is not None:\n"
  "            last_correct_step_indices = _rdf2_t.tv.path.index_select(0, last_correct_step_indices.clamp(min=0))\n"),
 ("        model_runner = self.target_worker.model_runner\n        if hasattr(attn_backend, \"update_mamba_state_after_mtp_verify\"):\n",
  "        if _rdf2_t is not None and mamba_steps_to_track is not None:  " + MARK + "\n"
  "            mamba_steps_to_track = torch.where(mamba_steps_to_track >= 0, _rdf2_t.tv.path.index_select(0, mamba_steps_to_track.clamp(min=0)), mamba_steps_to_track)\n"
  "        model_runner = self.target_worker.model_runner\n        if hasattr(attn_backend, \"update_mamba_state_after_mtp_verify\"):\n"),
])

import sglang_dspine.layers.attention as _da    # DSpine 27B plugin pins the served block to its trained 16; the tree drafts N > 16 rows
f = _da.__file__; src = open(f).read()
if "--revert" in sys.argv:
    if os.path.exists(f + ".pretree27"): os.replace(f + ".pretree27", f); print("reverted", f)
elif MARK not in src:
    open(f + ".pretree27", "w").write(src)
    for old in ['        or (_get(config, "block_size") != BLOCK_SIZE)\n', '    if served_block_size != BLOCK_SIZE:\n']:
        assert src.count(old) == 1, old
        src = src.replace(old, old.replace("!= BLOCK_SIZE", "!= BLOCK_SIZE and not (__import__('os').environ.get('RDF2_TREE_SPEC') or __import__('os').environ.get('RDF2_ALLOW_BLOCK'))").rstrip("\n") + "  " + MARK + "\n")
    open(f, "w").write(src); print("patched", f)

if "--revert" in sys.argv: subprocess.check_call(FAST)
