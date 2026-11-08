#!/usr/bin/env python3
"""Benchmark client for an SGLang server (OpenAI chat endpoint). Prompt formats follow z-lab/dflash benchmark.py.
Writes one JSON per (method, dataset, temp, concurrency) with per-request meta_info so acceptance can be re-derived.
usage: client.py --method NAME --dataset gsm8k --temperature 0 --concurrency 1 --num-prompts 32 --out DIR
"""
import argparse, json, random, statistics, time, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

DATASETS = {
 "gsm8k": (("openai/gsm8k", "main"), "test", lambda x: x["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."),
 "math500": (("HuggingFaceH4/MATH-500",), "test", lambda x: x["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."),
 "humaneval": (("openai/openai_humaneval",), "test", lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + x["prompt"] + "\n```"),
 "mbpp": (("google-research-datasets/mbpp", "sanitized"), "test", lambda x: x["prompt"]),
 "mt-bench": (("HuggingFaceH4/mt_bench_prompts",), "train", lambda x: x["prompt"][0]),  # first turn only (as z-lab)
}

def load(name):
    from datasets import load_dataset
    args, split, fmt = DATASETS[name]
    ds = load_dataset(*args, split=split)
    items = [fmt(r) for r in ds]
    order = list(range(len(items))); random.Random(42).shuffle(order)
    return [items[i] for i in order]

def send(a, prompt, max_new):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_new,
            "temperature": a.temperature, "top_p": a.top_p if a.temperature > 0 else 1.0,
            "chat_template_kwargs": {"enable_thinking": a.thinking != "off", **({"reasoning_effort": a.thinking} if a.thinking not in ("on", "off") else {})},
            "return_meta_info": True}
    if a.temperature > 0 and a.top_k > 0: body["top_k"] = a.top_k
    if a.seed is not None: body["seed"] = a.seed
    t0 = time.perf_counter()
    r = requests.post(a.base_url + "/v1/chat/completions", json=body, timeout=3600); r.raise_for_status()
    j = r.json(); dt = time.perf_counter() - t0
    m = j.get("meta_info") or j["choices"][0].get("meta_info") or {}
    return {"completion_tokens": j["usage"]["completion_tokens"], "prompt_tokens": j["usage"]["prompt_tokens"], "latency_s": dt,
            "finish": j["choices"][0].get("finish_reason"), "meta": m}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--method", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--temperature", type=float, default=0.0); p.add_argument("--top-p", type=float, default=0.95); p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1); p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--thinking", default="xhigh", help="on|off|<effort>; 'xhigh' = enable_thinking + reasoning_effort=xhigh (as model cards)")
    p.add_argument("--seed", type=int, default=None); p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    items = load(a.dataset); n = a.num_prompts
    prompts = [items[i % len(items)] for i in range(n)]
    warm = [items[(n + i) % len(items)] for i in range(a.concurrency)]
    with ThreadPoolExecutor(a.concurrency) as ex: list(ex.map(lambda q: send(a, q, 64), warm))
    t0 = time.perf_counter(); res = [None] * n
    with ThreadPoolExecutor(a.concurrency) as ex:
        fs = {ex.submit(send, a, q, a.max_new_tokens): i for i, q in enumerate(prompts)}
        for f in as_completed(fs): res[fs[f]] = f.result()
    wall = time.perf_counter() - t0
    tot = sum(r["completion_tokens"] for r in res)
    acc = [float(r["meta"]["spec_accept_length"]) for r in res if "spec_accept_length" in r["meta"]]
    verify = sum(int(r["meta"].get("spec_verify_ct", 0)) for r in res)
    out = {"method": a.method, "dataset": a.dataset, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "concurrency": a.concurrency,
           "num_prompts": n, "max_new_tokens": a.max_new_tokens, "thinking": a.thinking, "tag": a.tag, "wall_s": wall, "output_tokens": tot,
           "tok_per_s": tot / wall, "mean_req_accept_length": statistics.mean(acc) if acc else None,
           "pooled_accept_length": (tot / verify) if verify else None, "spec_verify_ct": verify,
           "truncated_frac": sum(r["finish"] == "length" for r in res) / n, "requests": res}
    os.makedirs(a.out, exist_ok=True)
    fn = f"{a.out}/{a.method}{('_' + a.tag) if a.tag else ''}__{a.dataset}__T{a.temperature:g}__c{a.concurrency}.json"
    json.dump(out, open(fn, "w"))
    print(f"{a.method} {a.dataset} T={a.temperature} c={a.concurrency}: {out['tok_per_s']:.1f} tok/s, accept(pooled)={out['pooled_accept_length']}, mean-req={out['mean_req_accept_length']}, trunc={out['truncated_frac']:.2f} -> {fn}")

if __name__ == "__main__": main()
