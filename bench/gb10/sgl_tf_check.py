"""Teacher-forced greedy-consistency check using SGLang's own prefill path (for targets HF cannot run here, e.g. NVFP4 ModelOpt).
For each output, send prompt+output as input_ids to a running AR server of the same target with return_logprob/top_logprobs_num=1 and
compare every generated token with the server's top-1 at that position. Reports mismatch rate and logprob margin (top1 - chosen) of
mismatches; near-ties are bf16/fp8/fp4 shape noise, large margins would indicate a bug. Same output format as bench/p2_tf_check.py.
Usage: python sgl_tf_check.py --url URL OUT.json gb10/<tag>:<dataset>[:n] ..."""
import sys, json, os, argparse, urllib.request
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("out"); ap.add_argument("specs", nargs="+")
a = ap.parse_args()
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results"); P = os.path.join(H, "..", "data", "prompts")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def post(body):
    r = urllib.request.Request(a.url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())
res = {}
for spec in a.specs:
    pre, ds, n = (spec.split(":") + ["8"])[:3]
    f = f"{R}/{pre}_{ds}_T0_c1.json"
    if not os.path.exists(f): print(spec, "missing, skipped", flush=True); continue
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(f"{P}/{ds}.jsonl")}
    tot = mism = misalign = 0; margins = []
    for r in json.load(open(f))["requests"][: int(n)]:
        p = tok.apply_chat_template([{"role": "user", "content": prompts[r["id"]]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; oi = tok(r["text"], add_special_tokens=False)["input_ids"][:1500]
        out = post({"input_ids": pi + oi, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True, "logprob_start_len": len(pi), "top_logprobs_num": 1})
        mi = out["meta_info"]; tl = mi["input_token_logprobs"]; top = mi["input_top_logprobs"]
        # entries cover input positions logprob_start_len .. end; entry j is the distribution for token pi+oi[len(pi)+j]
        for j, t in enumerate(oi):
            if j >= len(tl) or tl[j][0] is None or not top[j]: continue
            if tl[j][1] != t: misalign += 1; continue   # alignment self-check: entry j must be the logprob of token oi[j]
            tot += 1
            if top[j][0][1] != t:
                mism += 1; margins.append(top[j][0][0] - tl[j][0])
    margins.sort()
    res[spec] = {"tokens": tot, "mismatch": mism, "mismatch_rate": mism / max(tot, 1), "margin_max": margins[-1] if margins else 0.0,
                 "margin_median": margins[len(margins) // 2] if margins else 0.0, "n_margin_gt_0.5": sum(1 for x in margins if x > 0.5),
                 "n_margin_gt_2": sum(1 for x in margins if x > 2), "misaligned": misalign}
    print(spec, res[spec], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
