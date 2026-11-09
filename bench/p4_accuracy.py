"""Phase 4: task accuracy of run_cell.py outputs (GSM8K, MATH-500: last \\boxed{} vs gold; same grader for every method).
Usage: python p4_accuracy.py results/a.json [b.json ...]  (gold from the HF cache, ids = dataset row index as in prep_prompts.py)"""
import json, sys, glob, os, re
HF = os.environ.get("HF_HOME", "/workspace/hf") + "/hub"
def snap(p): return glob.glob(f"{HF}/{p}/snapshots/*")[0]
def boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; d = 1; k = j
    while k < len(t) and d:
        d += (t[k] == "{") - (t[k] == "}"); k += 1
    return t[j:k - 1]
def norm(s):
    if s is None: return None
    s = s.replace(",", "").replace("$", "").replace("\\!", "").replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.replace(" ", "").replace("\\,", "").replace("\;", "").strip().rstrip(".")
    if s.startswith("x=") or s.startswith("y="): s = s[2:]
    try: return str(float(s)).rstrip("0").rstrip(".") if re.fullmatch(r"-?[0-9.]+", s) else s
    except ValueError: return s
_gold = {}
def gold(ds):
    if ds not in _gold:
        if ds == "gsm8k":
            import pandas as pd
            df = pd.read_parquet(snap("datasets--openai--gsm8k") + "/main/test-00000-of-00001.parquet")
            _gold[ds] = {i: a.split("####")[-1].strip() for i, a in enumerate(df.answer)}
        else:
            _gold[ds] = {i: json.loads(l)["answer"] for i, l in enumerate(open(snap("datasets--HuggingFaceH4--MATH-500") + "/test.jsonl"))}
    return _gold[ds]
out = []
for f in sys.argv[1:]:
    d = json.load(open(f)); s = d["summary"]; ds = s["dataset"]; G = gold(ds)
    ok = [norm(boxed(r["text"])) == norm(G[r["id"]]) for r in d["requests"]]
    out.append(dict(file=os.path.basename(f), method=s["method"], dataset=ds, T=s["temperature"], seed=s.get("seed"), n=len(ok), correct=sum(ok), acc=sum(ok) / len(ok)))
    print(f"{os.path.basename(f):55s} {ds:8s} T={s['temperature']} seed={s.get('seed')} acc={sum(ok)/len(ok):.3f} ({sum(ok)}/{len(ok)})")
if os.environ.get("P4_ACC_OUT"): json.dump(out, open(os.environ["P4_ACC_OUT"], "w"), indent=1)
