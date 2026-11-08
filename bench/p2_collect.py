"""Phase 2 collector: sequential (c=1, T=0) requests against a DSpine server started with RDF2_DUMP=<dumpfile>.
Per request records output token ids and the dump-file round range. Output: <out>.json (meta) alongside the dump.
Usage: python p2_collect.py --dump /workspace/p2/dump.bin --out /workspace/p2/meta.json --n 40 --datasets gsm8k,math500,humaneval,mbpp,mtbench"""
import argparse, json, os, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30000")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--prompts-dir", default="/workspace/recursive-dflash2/bench/data/prompts")
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--datasets", default="gsm8k,math500,humaneval,mbpp,mtbench")
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
REC = (1 + 15 + 15 * 16 + 15 * 256) * 8
def post(body):
    r = urllib.request.Request(a.url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())
def rounds_now():
    return os.path.getsize(a.dump) // REC if os.path.exists(a.dump) else 0
meta = []
for ds in a.datasets.split(","):
    rows = [json.loads(l) for l in open(f"{a.prompts_dir}/{ds}.jsonl")][: a.n]
    for i, row in enumerate(rows):
        text = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        r0 = rounds_now(); t0 = time.time()
        out = post({"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": a.max_new, "top_k": 1}})
        dt = time.time() - t0
        r1 = rounds_now()
        ids = out.get("output_ids")
        if ids is None:
            ids = tok(out["text"], add_special_tokens=False)["input_ids"]; src = "retok"
        else: src = "native"
        mi = out["meta_info"]
        meta.append({"ds": ds, "id": row["id"], "r0": r0, "r1": r1, "n_out": mi["completion_tokens"], "ids": ids, "ids_src": src,
                     "sglang_rounds": mi.get("spec_verify_ct"), "sglang_tau": mi.get("spec_accept_length"), "latency": dt})
        print(ds, i, mi["completion_tokens"], r1 - r0, mi.get("spec_verify_ct"), f"{dt:.1f}s", flush=True)
        json.dump(meta, open(a.out, "w"))
