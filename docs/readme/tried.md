### What we tried

We also tested longer-block training and recursive second drafting passes (on pass-1 tokens, and on pass-1 features scored as a tree) on top of DSpine; none beat tree verification (these were short warm-start fine-tunes of the official weights, not from-scratch training; a from-scratch run is in progress), details in [docs/phase3-findings.md](docs/phase3-findings.md) and [docs/phase6-recursive-tree-pass2.md](docs/phase6-recursive-tree-pass2.md).
