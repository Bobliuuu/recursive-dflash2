"""Shim so the Phase 2 worker patch (`from glue import TreeGlue`, RDF2_TREE_DIR=<this dir>) picks up the Phase 4 fast glue."""
from glue_fast import instance as TreeGlue, chain_mask  # noqa: F401
