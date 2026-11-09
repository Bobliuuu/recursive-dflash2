"""Patch the installed FlashInfer prefill kernels so a CUSTOM-mask prefill does not pay the per-element mask lookup on the prefix.
Stock FlashInfer calls logits_mask() on every KV tile in kCustom mode (one global byte load + bit test per score element), while kCausal
skips it on all but the diagonal tiles; for a 48-token tree verify over a ~1-2K token prefix that is most of the custom-mask verify overhead.
Edit: prefill.cuh: in kCustom mode skip logits_mask() on KV tiles that lie fully inside the prefix (kv_idx < kv_len - qo_len) and inside the
KV chunk (partial tiles at split-KV chunk ends keep the mask, they need its chunk-end bound check).
Tried and dropped (see docs/phase4-tree-overhead.md): skipping the bit lookup for prefix columns inside LogitsMask (no gain), and running the
tree verify on the causal kernel with the tree bits applied on the diagonal tiles (correct, but the extra mask code slows the causal kernel
itself by about as much as it saves; the custom-mask kernel's remaining cost is register pressure, both are at the 255-register cap).

ASSUMPTION (holds for tree / EAGLE-style verify, where every query sees the whole committed prefix): a custom mask is all-ones over the prefix
columns. A custom mask that hides prefix tokens would be silently ignored there, so only use this on a box that serves tree verify.
Idempotent; clears the FlashInfer JIT cache for prefill modules when the headers change. Usage: python apply_fi_mask_patch.py [--revert]"""
import sys, os, glob, shutil, flashinfer
changed = False
base = os.path.join(os.path.dirname(flashinfer.__file__), "data/include/flashinfer/attention")
MARK = "/*RDF2_CM_SKIP*/"
SKIP = "(iter >= min(chunk_size, sub_if_greater_or_zero(kv_len, qo_len + chunk_start)) / CTA_TILE_KV) " + MARK   # tiles fully inside prefix AND chunk
EDITS = {
    "prefill.cuh": [
        # single / ragged kernels
        ("if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {",
         "if ((MASK_MODE == MaskMode::kCustom && " + SKIP + ") || (MASK_MODE != MaskMode::kCustom && (iter >= mask_iteration || iter < window_iteration))) {"),
        # paged kernel
        ("        if (MASK_MODE == MaskMode::kCustom) {\n          logits_mask<KTraits>(",
         "        if (MASK_MODE == MaskMode::kCustom) {\n          if " + SKIP + "\n          logits_mask<KTraits>("),
    ],

}
for key, edits in EDITS.items():
    fname = key.split("#")[0]
    inc = os.path.normpath(os.path.join(base, fname))
    src = open(inc).read()
    if "--revert" in sys.argv:
        if os.path.exists(inc + ".orig"):
            os.replace(inc + ".orig", inc); os.utime(inc); print("reverted", fname); changed = True   # utime: .orig is older than the built .so, ninja must see a change
        continue
    if all(new.split(MARK)[0][-60:] in src for _, new in edits):
        print("already patched", key); continue
    if not os.path.exists(inc + ".orig"): open(inc + ".orig", "w").write(src)
    for old, new in edits:
        c = src.count(old); assert c >= 1, (fname, old)
        src = src.replace(old, new); print(f"patched {fname}: {c} site(s)")
    open(inc, "w").write(src); changed = True
# drop cached JIT prefill modules (they are keyed by URI, not by header contents)
# (SGLang's server keeps its own FlashInfer JIT cache under ~/.cache/sglang/.cache/flashinfer and rebuilds via ninja's header-mtime scan.)
roots = [os.path.expanduser("~/.cache/flashinfer"), os.path.expanduser("~/.cache/sglang/.cache/flashinfer"), os.environ.get("FLASHINFER_WORKSPACE_BASE", "")]
n = 0
for r in (roots if changed else []):
    if not r or not os.path.isdir(r): continue
    for d in glob.glob(os.path.join(r, "**", "*prefill*"), recursive=True):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True); n += 1
print("removed", n, "cached prefill JIT dirs")
