#!/usr/bin/env python3
"""Teacher-forced greedy check for 27B outputs using the AR SGLang server (no speculation) as the reference: for each saved request,
score prompt+output in one prefill and compare every generated token with the target's top-1 at that position.
Mismatches with small logprob margin are bf16 near-ties (batch-shape noise); large margins would indicate a bug.
usage (AR server running): tf27b.py OUT.json RESULT_JSON[:n] ..."""
import sys, os, json, requests
url = "http://127.0.0.1:30000/generate"; res = {}
for spec in sys.argv[2:]:
    fn, n = (spec.split(":") + ["8"])[:2]
    recs = json.load(open(fn))["requests"][: int(n)]
    tot = mism = 0; margins = []
    for r in recs:
        pi, oi = r["prompt_ids"], r["output_ids"][:1500]
        body = {"input_ids": pi + oi, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}, "return_logprob": True,
                "logprob_start_len": len(pi) - 1, "top_logprobs_num": 1}
        m = requests.post(url, json=body, timeout=3600).json()["meta_info"]
        chosen = m["input_token_logprobs"][1:]            # entry k: logprob of input token at position len(pi)-1+k given its prefix
        top = m["input_top_logprobs"][1:]
        for k, t in enumerate(oi):
            lp_c, tid = chosen[k][0], chosen[k][1]; lp_t, ttop = top[k][0][0], top[k][0][1]
            assert tid == t, (tid, t)
            tot += 1
            if ttop != t:
                mism += 1; margins.append(lp_t - lp_c)
                if os.environ.get('TF_VERBOSE'): print('mismatch at', k, 'margin', round(lp_t - lp_c, 3), flush=True)
    margins.sort()
    res[spec] = {"tokens": tot, "mismatch": mism, "mismatch_rate": mism / max(tot, 1), "margin_max": margins[-1] if margins else 0.0,
                 "margin_median": margins[len(margins) // 2] if margins else 0.0, "n_margin_gt_0.5": sum(1 for x in margins if x > 0.5)}
    print(spec.split("/")[-1], res[spec], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
