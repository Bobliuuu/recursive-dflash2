#!/usr/bin/env python3
"""Build the tune_sps.py request JSONL (tokenized chat-template prefixes + sampling params).
Uses prompts at shuffled indices 100.. (disjoint from the benchmark prompts at indices 0..) so tuning does not see benchmark items.
usage: make_sps_requests.py MODEL_DIR TEMP OUT.jsonl [per_dataset=4]"""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from client import load, DATASETS
from transformers import AutoTokenizer
model, temp, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
per = int(sys.argv[4]) if len(sys.argv) > 4 else 4
tok = AutoTokenizer.from_pretrained(model)
sp = {"temperature": 0.0} if temp == 0 else {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
n = 0
with open(out, "w") as f:
    for d in DATASETS:
        items = load(d)
        for p in items[100:100 + per] if len(items) > 100 + per else items[-per:]:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True,
                                          enable_thinking=True, reasoning_effort="xhigh")
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            f.write(json.dumps({"input_ids": list(ids), "sampling_params": sp}) + "\n"); n += 1
print("wrote", n, "requests to", out)
