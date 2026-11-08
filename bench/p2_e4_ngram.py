"""E4: prompt-lookup / n-gram tail extension of the 16-token block. Only rounds where all 15 drafts are accepted can use the tail.
Extra tokens = number of lookup tokens (appended after d15) that match the target's greedy continuation (the first must equal the bonus token).
Usage: python p2_e4_ngram.py DUMP META OUT.json"""
import sys, json, numpy as np
from p2_common import *
from transformers import AutoTokenizer

def lookup(seq, n, m):
    """most recent earlier occurrence of the last-n-gram of seq; return next m tokens after it (may be shorter)."""
    if len(seq) <= n: return []
    key = seq[-n:]
    for i in range(len(seq) - n - 1, -1, -1):
        if seq[i:i + n] == key:
            return seq[i + n:i + n + m]
    return []

if __name__ == "__main__":
    reqs = load(sys.argv[1], sys.argv[2])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    pdir = "/workspace/recursive-dflash2/bench/data/prompts"
    prompts = {}
    for ds in sorted(set(r["ds"] for r in reqs)):
        for l in open(f"{pdir}/{ds}.jsonl"):
            j = json.loads(l); prompts[(ds, j["id"])] = j["prompt"]
    cfgs = [(n, m) for n in (2, 3, 4) for m in (4, 8, 16)]
    stats = {c: [] for c in cfgs}  # extra accepted per full-accept round
    nfull = 0; nall = 0; emit_all = []
    for r in reqs:
        text = tok.apply_chat_template([{"role": "user", "content": prompts[(r["ds"], r["id"])]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(text, add_special_tokens=False)["input_ids"]
        ids = r["ids"]
        for rd in r["rounds"]:
            nall += 1; emit_all.append(rd["L"] + 1)
            if rd["L"] < T1: continue
            nfull += 1
            p = rd["p"]
            seq = pids + ids[:p + 1 + T1]        # context + anchor + 15 accepted drafts
            future = ids[p + 1 + T1:]             # target greedy continuation after d15 (bonus first)
            for (n, m) in cfgs:
                prop = lookup(seq, n, m)
                k = 0
                while k < len(prop) and k < len(future) and prop[k] == future[k]: k += 1
                stats[(n, m)].append((k, len(prop)))
    res = {"n_rounds": nall, "n_full_accept": nfull, "full_accept_frac": nfull / nall, "chain_emit": float(np.mean(emit_all)), "cfg": {}}
    # cost: extra verify tokens = proposal length (padded to m); ms from the HF 8B ctx512 curve increments (16->16+m), added to a 14.0 ms round
    curve = json.load(open("/workspace/p2/sweep/vg8b_ctx512.json"))
    def vcost(w):
        ks = sorted(int(k) for k in curve); import bisect
        xs = [float(curve[str(k)]) for k in ks]
        return float(np.interp(w, ks, xs))
    for (n, m), v in stats.items():
        ks = np.array([a for a, _ in v]); lens = np.array([b for _, b in v])
        emit_gain = ks.sum() / nall
        width = 16 + m   # fixed padded width each round (cheap static graph); alternative: only when a match is found
        dcost = vcost(width) - vcost(16)
        base_t = 14.0
        sp = ((res["chain_emit"] + emit_gain) / (base_t + dcost)) / (res["chain_emit"] / base_t)
        res["cfg"][f"n{n}_m{m}"] = {"match_found_frac": float((lens > 0).mean()), "mean_extra_per_full": float(ks.mean()), "mean_extra_per_round": float(emit_gain),
                                    "extra_verify_ms": dcost, "speedup_fixed_width": float(sp)}
        print(n, m, res["cfg"][f"n{n}_m{m}"], flush=True)
    json.dump(res, open(sys.argv[3], "w"), indent=1)
