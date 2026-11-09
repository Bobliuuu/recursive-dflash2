"""Phase 4 coverage experiment: why does more coverage not raise the N=48 tree tau? Per round, find where the best-first tree (N=48)
loses the true path and classify the loss: (a) coverage miss = true token not among the K candidates at that depth, (b) budget miss = true
token is a candidate but its node was not among the 47 selected. Also: rank of the true token in its conditional row, the joint-prob
rank of the true node, and two extra score policies over the same tables (marginal-logit blend, per-depth temperature).
Usage: python analyze.py DUMP META OUT.json"""
import sys, os, json, numpy as np
from multiprocessing import Pool
import sim
from sim import D, softmax, sub, best_first, accept_by_n

BLENDS = (0.25, 0.5)
DTEMP = {"lin_1.0_2.0": lambda d: 1.0 + d / 30.0, "lin_1.0_1.5": lambda d: 1.0 + 0.5 * d / 30.0}


def blend_sub(cand, marg, tab, k, beta):
    sel = np.argsort(-marg, axis=1, kind="stable")[:, :k]
    c, t = sub(cand, marg, tab, k)
    m = np.take_along_axis(marg, sel, 1)               # [D, k] marginal logit of each kept candidate
    return c, t + beta * m[:, None, :]


def work(args):
    dump, items = args
    raw = np.memmap(dump, dtype=np.float32, mode="r").reshape(-1, sim.REC)
    out = []
    for (ds, qi, ri, p, truth, L) in items:
        cand, marg, tab = sim.unpack(raw[ri])
        rec = {"ds": ds}
        for k in (16, 64):
            c, t = sub(cand, marg, tab, k)
            P = softmax(t)
            nodes = best_first(P, 48, 8)
            acc = accept_by_n(nodes, c, truth, (48,))[0]
            # classify the loss at depth acc (0-based position of the first true token not in the tree)
            if acc >= min(D, len(truth)):
                kind = "full"; rank = -1
            else:
                # true predecessor lane at depth acc-1
                j = 0
                if acc > 0:
                    j = int(np.nonzero(c[acc - 1] == truth[acc - 1])[0][0])
                w = np.nonzero(c[acc] == truth[acc])[0]
                if len(w) == 0:
                    kind = "coverage"; rank = -1
                else:
                    row = P[acc, j if acc > 0 else 0]
                    rank = int((row > row[w[0]]).sum())
                    kind = "budget"
            rec[f"K{k}"] = (acc, kind, rank)
        for beta in BLENDS:
            c, t = blend_sub(cand, marg, tab, 16, beta)
            rec[f"K16_blend{beta}"] = accept_by_n(best_first(softmax(t), 48, 8), c, truth, (48,))[0]
            c, t = blend_sub(cand, marg, tab, 64, beta)
            rec[f"K64_blend{beta}"] = accept_by_n(best_first(softmax(t), 48, 8), c, truth, (48,))[0]
        for name, f in DTEMP.items():
            for k in (16, 64):
                c, t = sub(cand, marg, tab, k)
                tt = np.stack([t[d] / f(d) for d in range(D)])
                rec[f"K{k}_dtemp_{name}"] = accept_by_n(best_first(softmax(tt), 48, 8), c, truth, (48,))[0]
        out.append(rec)
    return out


if __name__ == "__main__":
    dump, meta, outp = sys.argv[1:4]
    nproc = int(os.environ.get("P4COV_NPROC", os.cpu_count()))
    raw, items = sim.load_rounds(dump, meta)
    chunks = [items[i::nproc * 4] for i in range(nproc * 4)]
    with Pool(nproc) as pool:
        rs = [r for o in pool.map(work, [(dump, ch) for ch in chunks]) for r in o]
    res = {"n_rounds": len(rs)}
    for k in (16, 64):
        a = [r[f"K{k}"] for r in rs]
        res[f"K{k}_N48_emit"] = float(np.mean([x[0] + 1 for x in a]))
        for kind in ("full", "coverage", "budget"):
            res[f"K{k}_loss_{kind}_frac"] = float(np.mean([x[1] == kind for x in a]))
        rk = np.array([x[2] for x in a if x[1] == "budget"])
        res[f"K{k}_budget_miss_rank_hist"] = {str(i): float(np.mean(rk == i)) for i in range(8)} | {"8+": float(np.mean(rk >= 8))}
        res[f"K{k}_loss_depth_mean"] = float(np.mean([x[0] for x in a if x[1] != "full"]))
    for key in rs[0]:
        if key.startswith("K") and ("blend" in key or "dtemp" in key):
            res[key + "_N48_emit"] = float(np.mean([r[key] + 1 for r in rs]))
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(res, indent=1))
