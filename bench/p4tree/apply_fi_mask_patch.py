"""Patch the installed FlashInfer prefill kernels so a CUSTOM-mask prefill skips the per-element mask lookup on KV tiles that lie entirely in the
prefix (kv_idx < kv_len - qo_len). Stock FlashInfer calls logits_mask() on every KV tile in kCustom mode (one global byte load + bit test per
score element), while kCausal skips it on all but the diagonal tiles; for a 48-token tree verify over a ~1-2K token prefix that is the
custom-mask verify overhead.

ASSUMPTION (holds for tree / EAGLE-style verify, where every query sees the whole committed prefix): a custom mask is all-ones over the prefix
columns. A custom mask that hides prefix tokens would be silently ignored on full prefix tiles, so only use this on a box that serves tree verify.
Idempotent; clears the FlashInfer JIT cache for prefill modules so they are rebuilt. Usage: python apply_fi_mask_patch.py [--revert]"""
import sys, os, glob, shutil, flashinfer
changed = False
inc = os.path.join(os.path.dirname(flashinfer.__file__), "data/include/flashinfer/attention/prefill.cuh")
MARK = "/*RDF2_CM_SKIP*/"
src = open(inc).read()
if "--revert" in sys.argv:
    if os.path.exists(inc + ".orig"):
        os.replace(inc + ".orig", inc); print("reverted"); changed = True
else:
    if MARK in src:
        print("already patched")
    else:
        open(inc + ".orig", "w").write(src)
        SKIP = "(iter >= min(chunk_size, sub_if_greater_or_zero(kv_len, qo_len + chunk_start)) / CTA_TILE_KV) " + MARK   # tiles fully inside prefix AND chunk
        # single / ragged kernels: `if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {`
        old = "if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {"
        new = "if ((MASK_MODE == MaskMode::kCustom && " + SKIP + ") || (MASK_MODE != MaskMode::kCustom && (iter >= mask_iteration || iter < window_iteration))) {"
        n1 = src.count(old); src = src.replace(old, new)
        # paged kernel: `if (MASK_MODE == MaskMode::kCustom) {\n          logits_mask<KTraits>(` -> guard the call
        old2 = "        if (MASK_MODE == MaskMode::kCustom) {\n          logits_mask<KTraits>("
        new2 = "        if (MASK_MODE == MaskMode::kCustom) {\n          if " + SKIP + "\n          logits_mask<KTraits>("
        n2 = src.count(old2); src = src.replace(old2, new2)
        assert n1 >= 1 and n2 >= 1, (n1, n2)
        changed = True
        open(inc, "w").write(src); print(f"patched {inc}: {n1} single/ragged sites, {n2} paged sites")
# drop cached JIT prefill modules (they are keyed by URI, not by header contents)
roots = [os.path.expanduser("~/.cache/flashinfer"), os.environ.get("FLASHINFER_WORKSPACE_BASE", "")]
n = 0
for r in (roots if changed else []):
    if not r or not os.path.isdir(r): continue
    for d in glob.glob(os.path.join(r, "**", "*prefill*"), recursive=True):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True); n += 1
print("removed", n, "cached prefill JIT dirs")
