"""Teacher-forced acceptance-length (tau) of DFlash-family drafters on OUR prompt set, using DynaDFlash's own
score_context (external/dynadflash). Contexts = Qwen3-8B greedy continuations of bench/data/prompts at several offsets.
Usage (box): PYTHONPATH=/root/dynadflash python dynadflash_eval.py --out results/dynadflash_tau.json
"""
import argparse, json, random, sys, torch
sys.path.insert(0, "/root/dynadflash")
from transformers import AutoModelForCausalLM, AutoTokenizer
from dynamic_dflash.eval.paired import load_drafter
from dynamic_dflash.eval.acceptance import score_context
ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="/root/prompts"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--new", type=int, default=400); ap.add_argument("--offsets", default="1,16,64,128,256,380")
ap.add_argument("--out", required=True)
ap.add_argument("--drafters", default="baseline=midas-llm-research/qwen3-8b-dflash-baseline:0,dynadflash=midas-llm-research/qwen3-8b-dynadflash:1,zlab=z-lab/Qwen3-8B-DFlash-b16:0")
a = ap.parse_args()
dev = torch.device("cuda:0")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); tok.padding_side = "left"
tgt = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, attn_implementation="sdpa").eval().to(dev).requires_grad_(False)
eos = tok.convert_tokens_to_ids("<|im_end|>")
drafters = {}
for spec in a.drafters.split(","):
    name, rest = spec.split("="); path, dyn = rest.rsplit(":", 1)
    try: drafters[name] = load_drafter(path, bool(int(dyn)), dev)
    except Exception as e: print("SKIP drafter", name, repr(e)[:300], flush=True)
print("drafters:", list(drafters), flush=True)
seqs = []  # (dataset, id, prompt_len, ids)
for ds in ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]:
    rows = [json.loads(l) for l in open(f"{a.prompts}/{ds}.jsonl")][: a.n]
    for i in range(0, len(rows), 10):
        chunk = rows[i:i+10]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad(): out = tgt.generate(**enc, max_new_tokens=a.new, do_sample=False)
        for j, r in enumerate(chunk):
            p = enc.input_ids[j][enc.attention_mask[j].bool()].tolist()
            g = out[j, enc.input_ids.shape[1]:].tolist()
            if eos in g: g = g[: g.index(eos)+1]
            seqs.append((ds, r["id"], len(p), p + g))
    print("generated", ds, flush=True)
offs = [int(x) for x in a.offsets.split(",")]
rows = []
for ds, id_, pl, ids in seqs:
    for o in offs:
        pre = pl + o
        if not 0 < pre < len(ids) - 1: continue
        rec = {"dataset": ds, "id": id_, "offset": o}
        for n, m in drafters.items():
            rec[n] = score_context(tgt, m, ids, pre, eos)["accepted_length"]
        rows.append(rec)
res = {"n_contexts": len(rows), "drafters": list(drafters)}
for n in drafters:
    res[n] = {"all": sum(r[n] for r in rows) / len(rows)}
    for ds in ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]:
        s = [r[n] for r in rows if r["dataset"] == ds]
        if s: res[n][ds] = sum(s) / len(s)
if "baseline" in drafters and "dynadflash" in drafters:
    d = [r["dynadflash"] - r["baseline"] for r in rows]; rng = random.Random(0)
    bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(2000))
    res["dyna_minus_baseline"] = {"mean": sum(d)/len(d), "ci95": [bs[50], bs[1949]]}
json.dump({"summary": res, "rows": rows}, open(a.out, "w"))
print(json.dumps(res, indent=1))
