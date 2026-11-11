"""Regenerates docs/img/*.png from the numbers in README.md / docs/phase*.md. Usage: python docs/img/make_charts.py"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, BASE, ACC = "#1f2328", "#6e7781", "#9aa4ae", "#0b6bcb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})

def bars(path, title, labels, vals, hi, xmax, note):
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    colors = [ACC if i == hi else BASE for i in range(len(vals))]
    y = range(len(vals))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + xmax * 0.012, yy, f"{v:.2f}x", va="center", fontsize=10, fontweight="bold" if vals.index(v) == hi else "normal")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, xmax); ax.set_xlabel("speedup over plain autoregressive decoding")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path); plt.close(fig)

bars(f"{out}/speedup.png", "Qwen3-8B, 1x A100, batch 1, greedy",
     ["Autoregressive", "EAGLE-3 (chain)", "DFlash", "DSpine", "DSpine + tree (48 nodes)"],
     [1.0, 2.03, 4.50, 5.78, 7.20], 4, 8.2, "Mean of GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench. Source: docs/phase1-qwen3-8b.md, docs/phase4-tree-overhead.md")

# tree vs same-weights chain across setups
fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
names = ["A100\ngreedy", "H100\ngreedy", "A100\nT=1 sampled", "H100\nQwen3.8-27B"]
chain = [5.78, 5.09, 4.83, 4.68]; tree = [7.20, 6.49, 5.43, 4.95]; pct = [25, 27, 12, 6]  # as stated in the docs
x = range(len(names)); w = 0.36
ax.bar([i - w/2 for i in x], chain, w, color=BASE, label="DSpine chain (stock)")
ax.bar([i + w/2 for i in x], tree, w, color=ACC, label="DSpine + tree")
for i, (c, t) in enumerate(zip(chain, tree)):
    p_ = pct[i]
    ax.text(i - w/2, c + 0.08, f"{c:.2f}x", ha="center", fontsize=9, color=MUTED)
    ax.text(i + w/2, t + 0.08, f"{t:.2f}x", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, t + 0.62, f"+{p_}%", ha="center", fontsize=10, color=ACC, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9.5); ax.set_ylim(0, 9.0)
ax.set_ylabel("speedup over autoregressive")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=2, bbox_to_anchor=(1, 1.0))
ax.set_title("Tree verification vs the same drafter's chain", loc="left", fontsize=11.5, fontweight="bold")
fig.text(0.01, 0.01, "Qwen3-8B unless noted, batch 1. Source: docs/phase4-*.md", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{out}/tree_gain.png"); plt.close(fig)
