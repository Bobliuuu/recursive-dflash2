"""Freeze benchmark prompts to bench/data/prompts/<name>.jsonl (run on the box; reads the HF cache).
Prompt templates (recorded in the file): plain user turn, thinking OFF, no system prompt.
N is a deterministic first-N subsample (seeded shuffle) so all methods see identical prompts."""
import json, glob, hashlib, random, sys, os
import pandas as pd
HF = os.environ.get("HF_HOME", "/workspace/hf") + "/hub"
OUT = sys.argv[1] if len(sys.argv) > 1 else "bench/data/prompts"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
os.makedirs(OUT, exist_ok=True)
def snap(p): return glob.glob(f"{HF}/{p}/snapshots/*")[0]
MATHSUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
sets = {}
df = pd.read_parquet(snap("datasets--openai--gsm8k") + "/main/test-00000-of-00001.parquet")
sets["gsm8k"] = [r.question + MATHSUF for r in df.itertuples()]
sets["math500"] = [json.loads(l)["problem"] + MATHSUF for l in open(snap("datasets--HuggingFaceH4--MATH-500") + "/test.jsonl")]
df = pd.read_parquet(snap("datasets--openai--openai_humaneval") + "/openai_humaneval/test-00000-of-00001.parquet")
sets["humaneval"] = ["Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + r.prompt + "\n```" for r in df.itertuples()]
df = pd.read_parquet(snap("datasets--google-research-datasets--mbpp") + "/full/test-00000-of-00001.parquet")
sets["mbpp"] = [r.text + "\nYour code should pass these tests:\n" + "\n".join(r.test_list) for r in df.itertuples()]
q = [json.loads(l) for l in open(snap("datasets--HuggingFaceH4--mt_bench_prompts") + "/raw/question.jsonl")]
sets["mtbench"] = [x["turns"][0] for x in q]  # first turn only
for k, v in sets.items():
    idx = list(range(len(v)))
    random.Random(0).shuffle(idx)
    idx = sorted(idx[:N])
    rows = [{"id": i, "prompt": v[i]} for i in idx]
    with open(f"{OUT}/{k}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    h = hashlib.sha256(open(f"{OUT}/{k}.jsonl", "rb").read()).hexdigest()[:12]
    print(k, len(v), "->", len(rows), h)
