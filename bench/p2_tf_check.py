"""Teacher-forced check that engine outputs are greedy-consistent: for each output, run the HF target over prompt+output and compare each generated token with the
argmax; report mismatch count and logit margin (top1 - chosen) of mismatches. Near-ties (bf16 shape noise) are expected; large margins would indicate a bug.
Usage: python p2_tf_check.py OUT.json PREFIX:dataset[:n] ..."""
import sys, json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MID = os.environ.get("TF_MODEL", "Qwen/Qwen3-8B")   # e.g. Qwen/Qwen3-8B-FP8 to check cells served with a quantized target
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).cuda().eval()
pdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/prompts")
res = {}
for spec in sys.argv[2:]:
    pre, ds, n = (spec.split(":") + ["8"])[:3]
    prompts = {}
    for l in open(f"{pdir}/{ds}.jsonl"):
        j = json.loads(l); prompts[j["id"]] = j["prompt"]
    if not os.path.exists(f"{R}/{pre}_{ds}_T0_c1.json"): print(spec, "missing, skipped", flush=True); continue
    reqs = json.load(open(f"{R}/{pre}_{ds}_T0_c1.json"))["requests"][: int(n)]
    tot = mism = 0; margins = []
    for r in reqs:
        p = tok.apply_chat_template([{"role": "user", "content": prompts[r["id"]]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; oi = tok(r["text"], add_special_tokens=False)["input_ids"][:1500]
        x = torch.tensor([pi + oi], device="cuda")
        with torch.no_grad(): lg = m(x).logits[0, len(pi) - 1:-1].float()
        top = lg.argmax(-1); tgt = torch.tensor(oi, device="cuda")
        bad = top != tgt
        tot += len(oi); mism += int(bad.sum())
        if bad.any():
            idx = bad.nonzero().flatten()
            margins += (lg[idx, top[idx]] - lg[idx, tgt[idx]]).tolist()
    margins.sort()
    res[spec] = {"tokens": tot, "mismatch": mism, "mismatch_rate": mism / max(tot, 1), "margin_max": margins[-1] if margins else 0.0,
                 "margin_median": margins[len(margins) // 2] if margins else 0.0, "n_margin_gt_0.5": sum(1 for x in margins if x > 0.5)}
    print(spec, res[spec], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
