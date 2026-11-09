"""Phase 4 distribution test, sampling side: draw many short continuations (T=1, top_p=1, top_k off) of a few fixed prompts from a
running SGLang server and store the generated token ids. Run once against AR and once against the tree server (same prompts).
Usage: python p4_dist_sample.py --n 2500 --max-new 8 --conc 1 --out results/p4_dist_<tag>.json"""
import argparse, json, urllib.request, time
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30000")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--prompts-dir", default="/workspace/recursive-dflash2/bench/data/prompts")
ap.add_argument("--n", type=int, default=2500)
ap.add_argument("--max-new", type=int, default=8)
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--prompts", default="", help="comma-separated subset of the fixed prompt keys")
ap.add_argument("--out", required=True)
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
def first(ds, i=0): return [json.loads(l) for l in open(f"{a.prompts_dir}/{ds}.jsonl")][i]["prompt"]
# fixed prompts: open chat (high entropy), math, code, and an explicitly random-choice prompt
P = {"mtbench0": first("mtbench"), "gsm8k0": first("gsm8k"), "humaneval0": first("humaneval"),
     "story": "Write the opening line of a short story about a lighthouse keeper."}
if a.prompts: P = {k: v for k, v in P.items() if k in a.prompts.split(",")}
def build(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def post(body):
    r = urllib.request.Request(a.url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())
def one(text):
    o = post({"text": text, "sampling_params": {"temperature": a.temp, "top_p": 1.0, "top_k": -1, "max_new_tokens": a.max_new, "ignore_eos": True}})
    ids = o.get("output_ids")
    if ids is None: ids = tok(o["text"], add_special_tokens=False)["input_ids"]
    return ids, o["meta_info"].get("spec_verify_ct")
out = {"max_new": a.max_new, "temp": a.temp, "n": a.n, "samples": {}, "verify_ct": {}}
for k, p in P.items():
    text = build(p); t0 = time.time()
    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda _: one(text), range(a.n)))
    out["samples"][k] = [r[0] for r in res]; out["verify_ct"][k] = [r[1] for r in res]
    print(k, "done", round(time.time() - t0, 1), "s", flush=True)
json.dump(out, open(a.out, "w"))
