"""Phase 4 coverage experiment: the Phase 4 fast glue (bench/p4tree/glue_fast.py, unchanged) with its tree selection swapped for the
K-generic kernel (bench/p4cov/tree_kernels_k.py). The draft dir must set head_topk = RDF2_TREE_K (DSpine computes K candidates and the
factored table then is [D,K,K]). Env: RDF2_TREE_DIR=<this dir>, RDF2_TREE_SPEC=N:D, RDF2_TREE_K (default 16), RDF2_TREE_E (children per
node, default 8), RDF2_TABLE=factored. The directory name contains 'p4tree' so bench/p4tree/apply_tree_patch_fast.py's hook loads it."""
import os, sys, importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
_p4 = os.path.join(os.path.dirname(os.path.dirname(_here)), "p4tree")
sys.path.insert(1, os.path.dirname(_here))   # tree_kernels_k
_spec = importlib.util.spec_from_file_location("glue_fast_p4", os.path.join(_p4, "glue_fast.py"))
_g = importlib.util.module_from_spec(_spec); sys.modules["glue_fast_p4"] = _g; _spec.loader.exec_module(_g)
from tree_kernels_k import FastTreeK

K = int(os.environ.get("RDF2_TREE_K", "16")); E = int(os.environ.get("RDF2_TREE_E", "8"))
_orig_init = _g.TreeGlue.__init__


def _init(self):
    _orig_init(self)
    self.ft = FastTreeK(self.N, self.D, K, E)
    self.tv.tokens = self.ft.tokens; self.tv.parent = self.ft.parent
    self.commit_pos_buf = self.ft.commit_pos
    print(f"[RDF2_TREE] K-generic tree select: N={self.N} D={self.D} K={K} E={E} table={_g.TABLE_MODE}", file=sys.stderr, flush=True)


_g.TreeGlue.__init__ = _init
_orig_propose_hook = _g.install_propose_hook


def install_propose_hook(S):
    _orig_propose_hook(S)
    orig = S.propose
    def propose(self, rows, input_ids):
        assert self.k == K, f"draft head_topk {self.k} != RDF2_TREE_K {K}"
        return orig(self, rows, input_ids)
    propose._rdf2 = True
    S.propose = propose


instance = _g.instance; chain_mask = _g.chain_mask; factored_table = _g.factored_table; table = _g.table
