"""Shim for the Phase 2 worker patch (`from glue import TreeGlue, chain_mask`): Phase 5 27B glue."""
from glue_fast import instance as TreeGlue, chain_mask  # noqa: F401
