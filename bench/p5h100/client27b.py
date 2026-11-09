#!/usr/bin/env python3
"""Phase 5 27B client: same prompts as bench/qwen38_27b/client.py (z-lab formats, shuffle seed 42, thinking on, reasoning_effort=xhigh),
but through the native /generate endpoint with the chat template applied here, so output token ids are saved (for the teacher-forced
check, tf27b.py). c=1 only (sequential). Throughput = total completion tokens / wall time of the cell.
usage: client27b.py --method NAME --dataset gsm8k --num-prompts 8 --out DIR [--max-new-tokens 2048]"""
import argparse, json, time, os, sys, requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen38_27b"))
from client import load
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://127.0.0.1:30000"); p.add_argument("--method", required=True); p.add_argument("--dataset", required=True)
p.add_argument("--num-prompts", type=int, default=8); p.add_argument("--max-new-tokens", type=int, default=2048); p.add_argument("--warmup", type=int, default=1)
p.add_argument("--model", default="/root/w/models/Qwen/Qwen3.8-27B"); p.add_argument("--out", required=True)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
prompts = load(a.dataset)[: a.num_prompts]

def ids(pr):
    x = tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=True, add_generation_prompt=True, enable_thinking=True, reasoning_effort="xhigh")
    return list(x["input_ids"] if not isinstance(x, list) else x)

def gen(input_ids, max_new):
    body = {"input_ids": input_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "top_k": 1}}
    t0 = time.perf_counter(); r = requests.post(a.base_url + "/generate", json=body, timeout=3600); r.raise_for_status(); dt = time.perf_counter() - t0
    return r.json(), dt

for i in range(a.warmup): gen(ids("Say hello."), 64)
recs = []; t0 = time.perf_counter()
for pr in prompts:
    ii = ids(pr); j, dt = gen(ii, a.max_new_tokens); m = j["meta_info"]
    recs.append({"prompt_ids": ii, "output_ids": j.get("output_ids"), "completion_tokens": m["completion_tokens"], "latency_s": dt,
                 "finish": (m.get("finish_reason") or {}).get("type"), "spec_verify_ct": m.get("spec_verify_ct"),
                 "spec_accept_length": m.get("spec_accept_length"), "hist": m.get("spec_correct_drafts_histogram")})
wall = time.perf_counter() - t0
tot = sum(r["completion_tokens"] for r in recs); vct = sum(r["spec_verify_ct"] or 0 for r in recs)
out = {"method": a.method, "dataset": a.dataset, "temperature": 0, "concurrency": 1, "num_prompts": len(recs), "max_new_tokens": a.max_new_tokens,
       "thinking": "xhigh", "wall_s": wall, "output_tokens": tot, "tok_per_s": tot / wall, "pooled_accept_length": (tot / vct) if vct else None,
       "truncated_frac": sum(r["finish"] == "length" for r in recs) / len(recs), "requests": recs}
os.makedirs(a.out, exist_ok=True)
fn = f"{a.out}/{a.method}__{a.dataset}__T0__c1.json"; json.dump(out, open(fn, "w"))
print(f"{a.method} {a.dataset}: {out['tok_per_s']:.1f} tok/s, tau={out['pooled_accept_length']}, trunc={out['truncated_frac']:.2f}", flush=True)
