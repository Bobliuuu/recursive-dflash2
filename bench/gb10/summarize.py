"""Summarize GB10 c=1 T=0 cells: per-dataset tok/s, tau, x AR, and per-round ms split (draft / verify from the CUDA-event timer, other = wall round - both).
Usage: python summarize.py AR_TAG TAG [TAG ...]  (reads results/gb10/<tag>_<ds>_T0_c1.json and logs/gb10/timing_<tag>.jsonl) -> prints markdown, writes results/gb10/summary.json (merged)"""
import json, os, sys, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results", "gb10"); L = os.path.join(H, "..", "..", "logs", "gb10")
DS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
def cell(tag, ds):
    p = f"{R}/{tag}_{ds}_T0_c1.json"
    return json.load(open(p))["summary"] if os.path.exists(p) else None
ar_tag = sys.argv[1]
ar = {d: cell(ar_tag, d) for d in DS}
out_p = f"{R}/summary.json"; out = json.load(open(out_p)) if os.path.exists(out_p) else {}
print("| config | tau | tok/s | x AR | " + " / ".join(DS) + " tok/s | draft ms | verify ms | other ms | round ms |")
print("|---|---|---|---|---|---|---|---|---|")
for tag in sys.argv[1:]:
    c = {d: cell(tag, d) for d in DS}; ds = [d for d in DS if c[d]]
    if not ds: continue
    tps = {d: c[d]["agg_tok_s"] for d in ds}; tau = {d: c[d].get("accept_len_pooled") or 1.0 for d in ds}
    xar = {d: tps[d] / ar[d]["agg_tok_s"] for d in ds if ar.get(d)}
    rounds = sum(c[d].get("rounds") or c[d]["tokens_out"] for d in ds); wall = sum(c[d]["makespan_s"] for d in ds)
    round_ms = 1000 * wall / rounds
    tf = f"{L}/timing_{tag}.jsonl"; dm = vm = None
    if os.path.exists(tf):
        rows = [json.loads(l) for l in open(tf) if l.strip()]; rows = [r for r in rows if r.get("bs") == 1]
        if rows: dm = st.median(r["draft_ms"] for r in rows); vm = st.median(r["verify_ms"] for r in rows)
    rec = {"datasets": ds, "tau_mean": st.mean(tau.values()), "tok_s_mean": st.mean(tps.values()), "x_ar_mean": st.mean(xar.values()) if xar else None,
           "tok_s": tps, "tau": tau, "x_ar": xar, "round_ms": round_ms, "draft_ms_median": dm, "verify_ms_median": vm,
           "other_ms": (round_ms - dm - vm) if dm is not None else None, "n": {d: c[d]["n"] for d in ds}}
    out[tag] = rec
    f = lambda x: "-" if x is None else f"{x:.2f}"
    print(f"| {tag} | {rec['tau_mean']:.2f} | {rec['tok_s_mean']:.1f} | {f(rec['x_ar_mean'])}x | " + " / ".join(f"{tps[d]:.0f}" for d in ds)
          + f" | {f(dm)} | {f(vm)} | {f(rec['other_ms'])} | {round_ms:.2f} |")
json.dump(out, open(out_p, "w"), indent=1)
