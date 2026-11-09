"""Experimental: let the DSpine SGLang plugin load its draft with online quantization (--speculative-draft-model-quantization fp8).
The plugin refuses any quant_config for the draft (DSpineDraftModel.__init__); with env RDF2_DSPINE_DRAFT_QUANT=1 the check is skipped and the
draft's linear layers go through SGLang's normal online-FP8 path. The DSpine refinement tail and the tree glue must still find bf16 tensors
they need; if they do not, the server fails at load or capture. Idempotent. Usage: python apply_dspine_draftquant_patch.py [--revert]"""
import sys, os
import sglang_dspine.models.dspine as m
f = m.__file__; MARK = "# RDF2_DRAFT_QUANT"
src = open(f).read()
if "--revert" in sys.argv:
    if os.path.exists(f + ".prequant"): os.replace(f + ".prequant", f); print("reverted", f)
    sys.exit()
if MARK in src: print("already patched"); sys.exit()
old = "        if quant_config is not None:\n            raise ValueError(\"The DSPINE draft does not support quantized weights\")"
new = "        if quant_config is not None and not __import__('os').environ.get('RDF2_DSPINE_DRAFT_QUANT'):  " + MARK + "\n            raise ValueError(\"The DSPINE draft does not support quantized weights\")"
assert src.count(old) == 1
open(f + ".prequant", "w").write(src); open(f, "w").write(src.replace(old, new)); print("patched", f)
