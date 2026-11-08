"""Run one benchmark cell against a running SGLang server (native /generate, thinking OFF).
Writes a JSON with per-request raw records (tokens, latency, acceptance histogram) and cell summary.
Usage: python run_cell.py --method dspine --dataset gsm8k --temp 0 --conc 1 --n 100 --out results/x.json
"""
import argparse, json, time, hashlib, threading, queue, urllib.request, os, sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30000")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--method", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--prompts-dir", default="/workspace/recursive-dflash2/bench/data/prompts")
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--warmup", type=int, default=4)
ap.add_argument("--block", type=int, default=1, help="min histogram length (draft tokens per round incl. anchor)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(f"{a.prompts_dir}/{a.dataset}.jsonl")][: a.n]

def build(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def post(path, body, timeout=3600):
    r = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def one(row, idx):
    text = build(row["prompt"])
    sp = {"temperature": a.temp, "max_new_tokens": a.max_new, "top_p": 1.0, "top_k": -1 if a.temp > 0 else 1,
          "sampling_seed": a.seed * 100003 + idx}
    t0 = time.time()
    out = post("/generate", {"text": text, "sampling_params": sp})
    t1 = time.time()
    mi = out["meta_info"]
    return {"id": row["id"], "n_in": mi.get("prompt_tokens"), "n_out": mi["completion_tokens"],
            "t_submit": t0, "t_done": t1, "latency": t1 - t0,
            "hit_cap": mi.get("finish_reason", {}).get("type") == "length",
            "spec_verify_ct": mi.get("spec_verify_ct"), "spec_accept_length": mi.get("spec_accept_length"),
            "spec_hist": mi.get("spec_correct_drafts_histogram"),
            "text_sha": hashlib.sha256(out["text"].encode()).hexdigest()[:16], "text": out["text"]}

# warmup (discarded)
for i, r in enumerate(rows[: a.warmup]):
    one(r, -1 - i)
t_start = time.time()
with ThreadPoolExecutor(a.conc) as ex:
    res = list(ex.map(lambda ir: one(ir[1], ir[0]), enumerate(rows)))
makespan = max(r["t_done"] for r in res) - min(r["t_submit"] for r in res)

tot_out = sum(r["n_out"] for r in res)
hist = None
L = max([len(r["spec_hist"] or []) for r in res] + [a.block])
for r in res:  # per-request histograms are trimmed of trailing zeros: pad before pooling
    h = r["spec_hist"]
    if h:
        h = list(h) + [0] * (L - len(h))
        hist = h if hist is None else [x + y for x, y in zip(hist, h)]
rounds = sum(hist) if hist else 0
summary = {"method": a.method, "dataset": a.dataset, "temperature": a.temp, "concurrency": a.conc, "n": len(res),
           "max_new_tokens": a.max_new, "seed": a.seed, "thinking": False, "tokens_out": tot_out,
           "makespan_s": makespan, "agg_tok_s": tot_out / makespan,
           "mean_req_tok_s": sum(r["n_out"] / r["latency"] for r in res) / len(res),
           "frac_hit_cap": sum(r["hit_cap"] for r in res) / len(res)}
if hist and rounds:
    # hist[k] = rounds with k accepted drafts (k=0..B-1); emitted per round = k+1 (bonus counted)
    B = len(hist)
    emitted = sum((k + 1) * c for k, c in enumerate(hist))
    surv = [sum(hist[i:]) / rounds for i in range(B)]  # P(>= i drafts accepted)
    summary.update({"block_size": B, "rounds": rounds, "accept_len_pooled": emitted / rounds,
                    "accept_len_mean_per_req": sum(r["spec_accept_length"] or 0 for r in res) / len(res),
                    "hist_pooled": hist, "survival": surv,
                    "cond_accept": [surv[i] / surv[i - 1] if surv[i - 1] > 0 else None for i in range(1, B)],
                    "frac_full_block": hist[-1] / rounds})
json.dump({"summary": summary, "requests": res}, open(a.out, "w"))
print(json.dumps({k: v for k, v in summary.items() if k not in ("hist_pooled",)}, indent=1))
