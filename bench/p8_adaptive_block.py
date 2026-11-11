"""Phase 8: does a content-adaptive block length (or drafter) beat the best fixed one? Offline, CPU only.

Part A (measured, no model): per-dataset tok/s of committed engine cells -> best single config on the mixed workload vs a
  per-dataset and a per-domain (code / math / prose) oracle choice. Families: A100 8B chain blocks c=1 and c=8, A100 8B chain+tree
  c=1, H100 8B chain+tree c=1, H100 Qwen3.8-27B drafters (DFlash2 vs DSpine variants) c=1/c=8, T=0/1.
Part B (simulation, A100 8B DSpine chain, c=1): per-request accepted-draft histograms of the block-48 chain run
  (p2_blk48_*_T0_c1.json) replayed at width K by truncation (block-causal drafter: a width-K verify accepts min(a, K-1)),
  calibrated per (dataset, K) to the measured fixed-K acceptance, rounds drawn i.i.d. per request, round time from the measured
  per-dataset makespan / rounds. Policies: fixed K, per-domain oracle, per-request oracle, online EMA-threshold and hazard
  (bench/adaptive_k.py), per-round clairvoyant (reference only).
Part C (simulation, c=8): 8 always-busy slots over the mixed workload, block-32 c=8 histograms (p2_blk32_*_T0_c8.json), step time
  as a function of total verify tokens from the measured c=8 cells; per-request widths via adaptive_k.allocate_batch.

Usage: python bench/p8_adaptive_block.py [--seeds 5] [--out bench/results/p8_adaptive_block.json]
"""
import argparse, glob, json, math, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adaptive_k as ak  # noqa: E402

R = os.path.join(HERE, "results")
DSS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
DOMAIN = {"gsm8k": "math", "math500": "math", "humaneval": "code", "mbpp": "code", "mtbench": "prose"}
MIXES = {"mixed (all 5)": DSS, "code+prose": ["humaneval", "mbpp", "mtbench"]}


def load(path):
    return json.load(open(path))


# ---------------------------------------------------------------- Part A: measured cells
def cell_run_cell(path):
    s = load(path)["summary"]
    return {"tokens": s["tokens_out"], "time": s["makespan_s"], "tau": s.get("accept_len_pooled")}


def cell_27b(path):
    d = load(path)
    return {"tokens": d["output_tokens"], "time": d["wall_s"], "tau": d.get("pooled_accept_length")}


def oracle_table(family):
    """family: {config: {dataset: cell}} -> best fixed, per-dataset oracle, per-domain oracle on each mix."""
    cfgs = [c for c in family if all(d in family[c] for d in DSS)]
    out = {"configs": cfgs, "tok_s": {c: {d: family[c][d]["tokens"] / family[c][d]["time"] for d in DSS} for c in cfgs}}
    for mix, dss in MIXES.items():
        def rate(choice):  # choice: dataset -> config
            return sum(family[choice[d]][d]["tokens"] for d in dss) / sum(family[choice[d]][d]["time"] for d in dss)
        fixed = {c: rate({d: c for d in dss}) for c in cfgs}
        best = max(fixed, key=fixed.get)
        per_ds = {d: max(cfgs, key=lambda c: family[c][d]["tokens"] / family[c][d]["time"]) for d in dss}
        doms = sorted({DOMAIN[d] for d in dss})
        per_dom = {}
        for g in doms:
            gd = [d for d in dss if DOMAIN[d] == g]
            per_dom[g] = max(cfgs, key=lambda c: sum(family[c][d]["tokens"] for d in gd) / sum(family[c][d]["time"] for d in gd))
        r_ds, r_dom = rate(per_ds), rate({d: per_dom[DOMAIN[d]] for d in dss})
        out[mix] = {"fixed": fixed, "best_fixed": best, "best_fixed_tok_s": fixed[best],
                    "per_dataset_choice": per_ds, "per_dataset_tok_s": r_ds, "per_dataset_gain": r_ds / fixed[best] - 1,
                    "per_domain_choice": per_dom, "per_domain_tok_s": r_dom, "per_domain_gain": r_dom / fixed[best] - 1}
    return out


def part_a():
    fams = {}
    fams["A100 8B DSpine chain, c=1, T=0 (block K)"] = {
        f"K{k}": {d: cell_run_cell(f"{R}/p2_blk{k}_{d}_T0_c1.json") for d in DSS} for k in (16, 24, 32, 48, 64)}
    fams["A100 8B DSpine chain, c=8, T=0 (block K)"] = {
        f"K{k}": {d: cell_run_cell(f"{R}/p2_blk{k}_{d}_T0_c8.json") for d in DSS} for k in (8, 12, 16, 32)}
    f = {f"chain K{k}": {d: cell_run_cell(f"{R}/p2_blk{k}_{d}_T0_c1.json") for d in DSS} for k in (16, 32)}
    for n in (32, 40, 48, 56, 64):
        f[f"tree N{n}"] = {d: cell_run_cell(f"{R}/p2_tree{n}_d31_{d}_T0_c1.json") for d in DSS}
    fams["A100 8B DSpine chain + tree (first glue), c=1, T=0"] = f
    f = {}
    for name in ("chain16", "chain32", "chain48", "tree32_d31", "tree48_d31", "tree64_d31", "tree96_d31"):
        if all(os.path.exists(f"{R}/p5h100/{name}_{d}_T0_c1.json") for d in DSS):
            f[name] = {d: cell_run_cell(f"{R}/p5h100/{name}_{d}_T0_c1.json") for d in DSS}
    fams["H100 8B DSpine chain + tree, c=1, T=0"] = f
    q = f"{R}/qwen38_27b"
    ds27 = {d: ("mt-bench" if d == "mtbench" else d) for d in DSS}
    for c in (1, 8):
        for T in (0, 1):
            f = {}
            for m in ("dflash2", "dspine16", "dspine16sps", "dspine_w8", "dflash2b16_b16"):
                mm = m + "_n64" if c == 8 and all(os.path.exists(f"{q}/{m}_n64__{ds27[d]}__T{T}__c8.json") for d in DSS) else m
                paths = {d: f"{q}/{mm}__{ds27[d]}__T{T}__c{c}.json" for d in DSS}
                if all(os.path.exists(p) for p in paths.values()):
                    f[mm] = {d: cell_27b(p) for d, p in paths.items()}
            if len(f) >= 2:
                fams[f"H100 Qwen3.8-27B drafters, c={c}, T={T}"] = f
    return {k: oracle_table(v) for k, v in fams.items()}


# ---------------------------------------------------------------- Part B/C: simulation helpers
def req_hists(path, L):
    out = []
    for q in load(path)["requests"]:
        h = [0] * L
        for i, c in enumerate(q["spec_hist"] or []):
            h[i] += c
        if sum(h):
            out.append({"id": q["id"], "n_out": q["n_out"], "hist": h})
    return out


def trunc_tau(hists, K):
    n = sum(sum(r["hist"]) for r in hists)
    return sum((min(a, K - 1) + 1) * c for r in hists for a, c in enumerate(r["hist"])) / n


def curve(points):
    """Piecewise-linear lookup over {K: value}, memoised per K."""
    c, memo = ak.CostCurve(points), {}
    def f(k):
        if k not in memo:
            memo[k] = c(k)
        return memo[k]
    return f


class Env:
    """Per-dataset environment: request histograms, calibration factor(K), round time T(K)."""

    def __init__(self, reqs, factor, round_ms):
        self.reqs, self.factor, self.T = reqs, factor, round_ms
        for r in reqs:
            r["cum"] = [0] * len(r["hist"])
            s = 0
            for i, c in enumerate(r["hist"]):
                s += c
                r["cum"][i] = s
            r["n"] = s

    def draw(self, r, rng):
        u = rng.random() * r["n"]
        lo, hi = 0, len(r["cum"]) - 1
        while lo < hi:
            m = (lo + hi) // 2
            if r["cum"][m] > u:
                hi = m
            else:
                lo = m + 1
        return lo

    def exp_emit(self, r, K):
        return self.factor(K) * sum((min(a, K - 1) + 1) * c for a, c in enumerate(r["hist"])) / r["n"]


def build_c1_envs():
    envs, calib = {}, {}
    for d in DSS:
        src = req_hists(f"{R}/p2_blk48_{d}_T0_c1.json", 48)
        meas = {k: load(f"{R}/p2_blk{k}_{d}_T0_c1.json")["summary"]["accept_len_pooled"] for k in (16, 24, 32, 48)}
        fac = {k: meas[k] / trunc_tau(src, k) for k in meas}
        c8src = req_hists(f"{R}/p2_blk32_{d}_T0_c8.json", 32)
        for k in (8, 12):  # no c=1 run below 16: use the c=8 measured / truncated ratio (same mechanism, other prompt sample)
            fac[k] = load(f"{R}/p2_blk{k}_{d}_T0_c8.json")["summary"]["accept_len_pooled"] / trunc_tau(c8src, k)
        rt = {}
        for k in (16, 24, 32, 48):
            s = load(f"{R}/p2_blk{k}_{d}_T0_c1.json")["summary"]
            rt[k] = s["makespan_s"] / s["rounds"] * 1e3
        for k in (8, 12):
            rt[k] = rt[16] - (ak.A100_8B_C1_ROUND_MS[16] - ak.A100_8B_C1_ROUND_MS[k])
        calib[d] = {"factor": fac, "round_ms": rt, "trunc_tau": {k: trunc_tau(src, k) for k in (8, 12, 16, 24, 32, 48)}}
        envs[d] = Env(src, curve(fac), curve(rt))
    return envs, calib


def agg(per_req, dss):
    """per_req: {dataset: [(tokens, time_ms)]} -> tok/s over the given datasets."""
    tok = sum(t for d in dss for t, _ in per_req[d])
    ms = sum(m for d in dss for _, m in per_req[d])
    return tok / ms * 1e3


def summarize(per_req):
    out = {d: agg(per_req, [d]) for d in DSS}
    for g in ("code", "math", "prose"):
        out[g] = agg(per_req, [d for d in DSS if DOMAIN[d] == g])
    for mix, dss in MIXES.items():
        out[mix] = agg(per_req, dss)
    return out


def analytic(envs, choose_k):
    """Expected-value run: choose_k(dataset, request) -> K for all its rounds."""
    pr = {}
    for d, e in envs.items():
        pr[d] = []
        for r in e.reqs:
            K = choose_k(d, r)
            pr[d].append((r["n_out"], r["n_out"] / e.exp_emit(r, K) * e.T(K)))
    return pr


def simulate(envs, factory, seed):
    rng = random.Random(seed)
    pr, kcount = {}, defaultdict(int)
    for d, e in envs.items():
        pr[d] = []
        for r in e.reqs:
            pol = factory()
            pol.reset()
            got, ms = 0.0, 0.0
            while got < r["n_out"]:
                K = pol.choose()
                a = min(e.draw(r, rng), K - 1)
                got += (a + 1) * e.factor(K)
                ms += e.T(K)
                kcount[K] += 1
                pol.observe(a, K)
            pr[d].append((r["n_out"], ms * r["n_out"] / got))  # credit exactly n_out tokens (pro-rata last round)
    return pr, dict(kcount)


def clairvoyant(envs, menu):
    """Per-round reference: knows a before choosing K; picks max emitted/T per round (not a policy, not an exact bound)."""
    pr = {}
    for d, e in envs.items():
        pr[d] = []
        for r in e.reqs:
            emit = ms = 0.0
            for a, c in enumerate(r["hist"]):
                if c:
                    K = max(menu, key=lambda k: (min(a, k - 1) + 1) * e.factor(k) / e.T(k))
                    emit += c * (min(a, K - 1) + 1) * e.factor(K)
                    ms += c * e.T(K)
            pr[d].append((r["n_out"], r["n_out"] / (emit / ms)))
    return pr


def part_b(seeds):
    envs, calib = build_c1_envs()
    menus = {"{8,16,32}": [8, 16, 32], "{16,24,32,48}": [16, 24, 32, 48], "{8,12,16,24,32,48}": [8, 12, 16, 24, 32, 48]}
    cost = ak.CostCurve(ak.A100_8B_C1_ROUND_MS)
    res = {"calibration": calib, "fixed": {}, "menus": {}}
    for K in (8, 12, 16, 24, 32, 48):
        res["fixed"][f"K{K}"] = summarize(analytic(envs, lambda d, r, K=K: K))
    best_fixed = max(res["fixed"], key=lambda k: res["fixed"][k]["mixed (all 5)"])
    res["best_fixed"] = best_fixed
    for mname, menu in menus.items():
        m = {}
        # per-domain oracle (code / math / prose share one K, chosen on the expected-value model = measured-calibrated)
        dom_k = {}
        for g in ("code", "math", "prose"):
            dom_k[g] = max(menu, key=lambda K: summarize(analytic(envs, lambda d, r, K=K: K))[g])
        m["per-domain oracle"] = summarize(analytic(envs, lambda d, r: dom_k[DOMAIN[d]]))
        m["per-domain oracle"]["choice"] = dom_k
        m["per-request oracle"] = summarize(analytic(
            envs, lambda d, r: min(menu, key=lambda K: envs[d].T(K) / envs[d].exp_emit(r, K))))
        m["per-round clairvoyant (reference)"] = summarize(clairvoyant(envs, menu))
        for label, fac in [("fixed K32 (sim, check)", lambda: ak.FixedK(32))] if 32 in menu else []:
            runs = [simulate(envs, fac, s)[0] for s in range(seeds)]
            m[label] = {k: sum(summarize(x)[k] for x in runs) / seeds for k in summarize(runs[0])}
        for spec in ("ema:1.5", "ema:2.0", "ema:3.0", "hazard:0.8", "hazard:0.9", "hazard:0.97"):
            fac = (lambda spec=spec, menu=menu: ak.make_policy(spec, menu, cost))
            runs = [simulate(envs, fac, s) for s in range(seeds)]
            sm = [summarize(x[0]) for x in runs]
            row = {k: sum(x[k] for x in sm) / seeds for k in sm[0]}
            row["mixed_min"] = min(x["mixed (all 5)"] for x in sm)
            row["mixed_max"] = max(x["mixed (all 5)"] for x in sm)
            kc = defaultdict(int)
            for x in runs:
                for k, v in x[1].items():
                    kc[k] += v
            tot = sum(kc.values())
            row["k_share"] = {k: kc[k] / tot for k in sorted(kc)}
            m[f"online {spec}"] = row
        res["menus"][mname] = m
    return res


# ---------------------------------------------------------------- Part C: c=8, per-request widths in one batched verify
def part_c(seeds, total_tokens=200000):
    step_pts = {}
    for k in (8, 12, 16, 32):
        v = []
        for d in DSS:
            s = load(f"{R}/p2_blk{k}_{d}_T0_c8.json")["summary"]
            v.append(s["accept_len_pooled"] * 8 / s["agg_tok_s"] * 1e3)
        step_pts[8 * k] = sum(v) / len(v)
    step = ak.CostCurve(step_pts)
    menu = [8, 12, 16, 32]
    envs = {}
    calib = {}
    for d in DSS:
        src = req_hists(f"{R}/p2_blk32_{d}_T0_c8.json", 32)
        fac = {k: load(f"{R}/p2_blk{k}_{d}_T0_c8.json")["summary"]["accept_len_pooled"] / trunc_tau(src, k) for k in menu}
        calib[d] = fac
        envs[d] = Env(src, curve(fac), None)
    pool = [(d, r) for d in DSS for r in envs[d].reqs]

    def run(mode, seed, K=None, dom_k=None, decay=0.9):
        rng = random.Random(seed)
        order = pool[:]
        rng.shuffle(order)
        qi = 0
        slots = []
        def new_slot():
            nonlocal qi
            d, r = order[qi % len(order)]
            qi += 1
            pol = ak.HazardK(menu, cost=lambda k: step(8 * k), decay=decay) if mode == "hazard" else None
            return {"d": d, "r": r, "left": r["n_out"], "pol": pol}
        slots = [new_slot() for _ in range(8)]
        tok = defaultdict(float)
        ms = 0.0
        done = 0.0
        while done < total_tokens:
            if mode == "fixed":
                ks = [K] * 8
            elif mode == "domain":
                ks = [dom_k[DOMAIN[s["d"]]] for s in slots]
            else:
                ks = ak.allocate_batch([s["pol"].expected_emitted() for s in slots], menu, step)
            ms += step(sum(ks))
            for i, s in enumerate(slots):
                a = min(envs[s["d"]].draw(s["r"], rng), ks[i] - 1)
                e = (a + 1) * envs[s["d"]].factor(ks[i])
                tok[DOMAIN[s["d"]]] += e
                done += e
                s["left"] -= e
                if s["pol"] is not None:
                    s["pol"].observe(a, ks[i])
                if s["left"] <= 0:
                    slots[i] = new_slot()
        return done / ms * 1e3

    res = {"step_ms_by_total_tokens": step_pts, "calibration": calib, "fixed": {}}
    for K in menu:
        res["fixed"][f"K{K}"] = sum(run("fixed", s, K=K) for s in range(seeds)) / seeds
    best = max(res["fixed"], key=res["fixed"].get)
    res["best_fixed"] = best
    # per-domain widths: search all 4^3 assignments on the simulator (in-sample oracle)
    best_dom, best_v = None, -1
    for kc in menu:
        for km in menu:
            for kp in menu:
                dk = {"code": kc, "math": km, "prose": kp}
                v = sum(run("domain", s, dom_k=dk) for s in range(2))
                if v > best_v:
                    best_v, best_dom = v, dk
    res["per_domain_choice"] = best_dom
    res["per_domain"] = sum(run("domain", s, dom_k=best_dom) for s in range(seeds)) / seeds
    for dec in (0.8, 0.9):
        res[f"online hazard:{dec} + allocate_batch"] = sum(run("hazard", s, decay=dec) for s in range(seeds)) / seeds
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(R, "p8_adaptive_block.json"))
    ap.add_argument("--skip-c8", action="store_true")
    a = ap.parse_args()
    out = {"part_a_measured": part_a(), "part_b_sim_c1": part_b(a.seeds)}
    if not a.skip_c8:
        out["part_c_sim_c8"] = part_c(a.seeds)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    # console summary
    for fam, t in out["part_a_measured"].items():
        m = t["mixed (all 5)"]
        cp = t["code+prose"]
        print(f"[A] {fam}: best fixed {m['best_fixed']} {m['best_fixed_tok_s']:.1f} tok/s; per-domain {m['per_domain_gain']*100:+.1f}% "
              f"{m['per_domain_choice']}; per-dataset {m['per_dataset_gain']*100:+.1f}% | code+prose: best {cp['best_fixed']}, per-domain {cp['per_domain_gain']*100:+.1f}%")
    b = out["part_b_sim_c1"]
    bf = b["fixed"][b["best_fixed"]]["mixed (all 5)"]
    print(f"[B] best fixed {b['best_fixed']} mixed {bf:.1f} tok/s; fixed: " + ", ".join(f"{k} {v['mixed (all 5)']:.1f}" for k, v in b["fixed"].items()))
    for mname, m in b["menus"].items():
        for pol, row in m.items():
            print(f"[B] {mname:>20} {pol:<36} mixed {row['mixed (all 5)']:.1f} ({(row['mixed (all 5)']/bf-1)*100:+.2f}%)"
                  f"  code {row['code']:.0f} math {row['math']:.0f} prose {row['prose']:.0f}" + (f"  K-share {({k: round(v, 2) for k, v in row['k_share'].items()})}" if "k_share" in row else ""))
    if "part_c_sim_c8" in out:
        c = out["part_c_sim_c8"]
        bf = c["fixed"][c["best_fixed"]]
        print(f"[C] c=8 fixed: " + ", ".join(f"{k} {v:.0f}" for k, v in c["fixed"].items()))
        for k in [k for k in c if k.startswith("per_domain") or k.startswith("online")]:
            if isinstance(c[k], float):
                print(f"[C] {k}: {c[k]:.0f} tok/s ({(c[k]/bf-1)*100:+.2f}% vs {c['best_fixed']})")
        print(f"[C] per-domain choice {c['per_domain_choice']}")


if __name__ == "__main__":
    main()
