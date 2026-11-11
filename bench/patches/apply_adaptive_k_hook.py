"""Phase 8: shadow-mode adaptive-K hook for SGLang 0.5.22 dflash_worker_v2.py (DFLASH and, via subclass, DSPINE).

After every verify round's accept step (chain `_accept_block`, or the tree accept when bench/p2tree is applied) and only
at batch size 1, passes (prefix length, block size, accepted drafts) to bench/adaptive_k.py:ShadowLogger, which appends
{seq, k, accepted, next_k} to a JSONL file. It does NOT change the served block size: switching K per round needs verify
CUDA graphs captured at several widths, which this repo does not implement (docs/phase8-adaptive-block.md).
The log gives the per-round acceptance order that the committed results lack, plus the policy's would-be choices.

Active only when env RDF2_ADAPTIVE_K_LOG=<file.jsonl> is set (default off: one cached env check per round).
Optional: RDF2_ADAPTIVE_K_POLICY (default ema:3.0; also hazard[:decay], fixed:K), RDF2_ADAPTIVE_K_MENU (default "8 16 32"),
RDF2_ADAPTIVE_K_DIR (directory holding adaptive_k.py, default /workspace/recursive-dflash2/bench).
Assumes `accept_len[0]` counts accepted draft tokens excluding the bonus (as SGLang's spec_correct_drafts_histogram);
check the first run by comparing the logged histogram with the cell's spec_hist. Apply after the tree patch if both are wanted.
Idempotent. Usage: python apply_adaptive_k_hook.py [--revert]"""
import sys, os, sglang

f = os.path.dirname(sglang.__file__) + "/srt/speculative/dflash_worker_v2.py"
MARK = "# RDF2_ADAPTIVE_K"
if "--revert" in sys.argv:
    if os.path.exists(f + ".preak"):
        os.replace(f + ".preak", f); print("reverted", f)
    sys.exit()
src = open(f).read()
if MARK in src:
    print("already patched", f); sys.exit()

helper = '''
import os as _rdf2_ak_os  ''' + MARK + '''
_RDF2_AK_PATH = _rdf2_ak_os.environ.get("RDF2_ADAPTIVE_K_LOG")
_RDF2_AK = None


def _rdf2_ak_observe(accept_len, prefix_lens, bs, block_size):
    global _RDF2_AK
    if not _RDF2_AK_PATH or bs != 1 or block_size is None:
        return
    if _RDF2_AK is None:
        import sys as _s
        _s.path.insert(0, _rdf2_ak_os.environ.get("RDF2_ADAPTIVE_K_DIR", "/workspace/recursive-dflash2/bench"))
        import adaptive_k as _ak
        _RDF2_AK = _ak.ShadowLogger(_RDF2_AK_PATH, _rdf2_ak_os.environ.get("RDF2_ADAPTIVE_K_POLICY", "ema:3.0"),
                                    _ak.parse_menu(_rdf2_ak_os.environ.get("RDF2_ADAPTIVE_K_MENU", "8 16 32")))
    _RDF2_AK.step(int(prefix_lens[0]), int(block_size), int(accept_len[0]))

'''
anchor_def = "def _commit_accept(candidates, accept_len, bonus_tokens):"
anchor_call = ("        ) = self._accept_block(\n            candidates=candidates,\n            next_token_logits=logits_output.next_token_logits,\n"
               "            sampling_info=sampling_info,\n            draft_input=draft_input,\n            prefix_lens=prefix_lens,\n"
               "            bs=bs,\n        )\n")
assert src.count(anchor_def) == 1, anchor_def
assert src.count(anchor_call) == 1, "accept call not found (unexpected SGLang version?)"
open(f + ".preak", "w").write(src)
src = src.replace(anchor_def, helper + "\n" + anchor_def)
# 8-space indent: after the if/else of the tree patch when present, otherwise in the function body
src = src.replace(anchor_call, anchor_call + "        if _RDF2_AK_PATH:  " + MARK + "\n"
                  "            _rdf2_ak_observe(accept_len, prefix_lens, bs, locals().get('block_size'))\n")
open(f, "w").write(src)
print("patched", f)
