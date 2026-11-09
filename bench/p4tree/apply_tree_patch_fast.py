"""Phase 4 fast tree path: applies the Phase 2 SGLang patches (bench/p2tree/apply_tree_patch.py) plus a hook at the end of the DSpine plugin's
worker module that folds the tree table + best-first selection into the draft CUDA graph (bench/p4tree/glue_fast.py). Active only when
RDF2_TREE_SPEC is set and RDF2_TREE_DIR points at bench/p4tree. Idempotent. Usage: python apply_tree_patch_fast.py [--revert]"""
import sys, os, subprocess
here = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(here), "p2tree", "apply_tree_patch.py")] + sys.argv[1:])
import sglang_dspine.runtime.worker as w
f = w.__file__; src = open(f).read(); MARK = "# RDF2_TREE_FAST"
if "--revert" in sys.argv:
    if os.path.exists(f + ".pretree"): os.replace(f + ".pretree", f); print("reverted", f)
elif MARK in src: print("already patched", f)
else:
    open(f + ".pretree", "w").write(src)
    src += ("\n\nimport os as _rdf2_o  " + MARK + "\n"
            "if _rdf2_o.environ.get('RDF2_TREE_SPEC') and 'p4tree' in _rdf2_o.environ.get('RDF2_TREE_DIR', ''):\n"
            "    import sys as _rdf2_s\n"
            "    _rdf2_s.path.insert(0, _rdf2_o.environ['RDF2_TREE_DIR'])\n"
            "    import glue_fast as _rdf2_gf\n"
            "    _rdf2_gf.install_propose_hook(DSpineDraftSampler)\n")
    open(f, "w").write(src); print("patched", f)
