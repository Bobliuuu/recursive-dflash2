"""Build the Phase 3 training prompt set (box). Source: mlabonne/open-perfectblend (the DSpine README's training mix).
Takes the FIRST human turn of each conversation, drops anything that overlaps the eval sets (GSM8K/MATH-500/HumanEval/MBPP/MT-Bench:
exact normalised match or 8-gram Jaccard-ish overlap), dedups, length-filters, samples N.
For math-flavoured prompts (source contains 'math'), ~50% get the same 'reason step by step ... \\boxed{}' suffix the eval harness uses
(format matching only; no eval data is used).
Usage: python prep_prompts.py out.jsonl N
"""
import json, sys, re, random, glob, os
from datasets import load_dataset
out, N = sys.argv[1], int(sys.argv[2])
rng = random.Random(1234)
MATHSUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."

def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
def grams(s, n=8):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

# eval texts: full test sets (not just the 40-prompt subsets)
ev = []
ev += [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")]
ev += [r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
ev += [r["prompt"] + " " + r["canonical_solution"] for r in load_dataset("openai/openai_humaneval", split="test")]
ev += [r["text"] + " " + " ".join(r["test_list"]) for r in load_dataset("google-research-datasets/mbpp", "full", split="test")]
ev += [x for r in load_dataset("HuggingFaceH4/mt_bench_prompts", split="train") for x in r["prompt"]]
evn = {norm(x) for x in ev}
evg = set()
for x in ev: evg |= grams(x)
print("eval texts", len(ev), "grams", len(evg), flush=True)

ds = load_dataset("mlabonne/open-perfectblend", split="train")
print(ds, flush=True)
idx = list(range(len(ds))); rng.shuffle(idx)
rows, seen, drop_ev, drop_len = [], set(), 0, 0
for i in idx:
    ex = ds[i]
    conv = ex["conversations"]
    first = next((c for c in conv if c["from"] in ("human", "user")), None)
    if first is None: continue
    p = first["value"].strip()
    if not (20 <= len(p) <= 6000): drop_len += 1; continue
    k = norm(p)
    if k in seen: continue
    if k in evn or (len(grams(p) & evg) > 0): drop_ev += 1; continue
    seen.add(k)
    src = str(ex.get("source", ""))
    if "math" in src.lower() and rng.random() < 0.5: p = p + MATHSUF
    rows.append({"id": int(i), "source": src, "prompt": p})
    if len(rows) >= N: break
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
import collections
print("wrote", len(rows), "dropped_eval_overlap", drop_ev, "dropped_len", drop_len)
print(collections.Counter(r["source"] for r in rows).most_common(15))
