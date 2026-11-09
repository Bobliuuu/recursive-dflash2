"""Phase 4 coverage experiment, offline: candidate coverage and best-first tree tau vs candidate width K, from a K=64 dump
(bench/p4cov/apply_cov_dump_hook.py, collected with bench/p4cov/collect.py on a block-48 chain run).
The K'=16/32 candidate sets are the top-K' of the K=64 set by marginal draft logit (= what DSpine's topk_ids would pick at K'),
and their table is the matching sub-block of the exact [D,64,64] table (each entry depends only on its (predecessor, candidate) pair).
Paired per-round replay at the logged chain anchors against the target's greedy output (as in bench/p2_e1_tree.py).
Usage: python sim.py DUMP META OUT.json   env: P2_T1 (47), P4COV_K (64), P4COV_D (31), P4COV_NPROC, P4COV_MAXREQ"""
import sys, os, json, heapq, numpy as np
from multiprocessing import Pool

T1 = int(os.environ.get("P2_T1", 47)); K = int(os.environ.get("P4COV_K", 64)); D = int(os.environ.get("P4COV_D", 31))
REC = 1 + T1 + 2 * D * K + D * K * K
NS = (48, 64, 96)
NMAX = max(NS)


def load_rounds(dump, meta):
    raw = np.memmap(dump, dtype=np.float32, mode="r")
    assert raw.size % REC == 0, (raw.size, REC)
    raw = raw.reshape(-1, REC)
    out = []
    for qi, m in enumerate(json.load(open(meta))):
        ids = m["ids"]; p = 0
        for ri in range(m["r0"], m["r1"]):
            r = raw[ri]
            if p >= len(ids) - 1: break
            if int(r[0]) != ids[p]: break
            greedy = r[1:1 + T1].astype(np.int64)
            truth = np.array(ids[p + 1:p + 1 + T1], dtype=np.int64)
            L = 0
            while L < len(truth) and greedy[L] == truth[L]: L += 1
            out.append((m["ds"], qi, ri, p, truth, L))
            p += L + 1
    return raw, out


def unpack(r):
    o = 1 + T1
    cand = r[o:o + D * K].reshape(D, K).astype(np.int64); o += D * K
    marg = r[o:o + D * K].reshape(D, K).astype(np.float64); o += D * K
    tab = r[o:o + D * K * K].reshape(D, K, K).astype(np.float64)
    return cand, marg, tab


def softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def sub(cand, marg, tab, k):
    """top-k of the marginal per position -> (cand [D,k], prob [D,k(pred),k])."""
    sel = np.argsort(-marg, axis=1, kind="stable")[:, :k]
    c = np.take_along_axis(cand, sel, 1)
    t = np.empty((D, k, k))
    t[0] = tab[0, 0][sel[0]][None, :]
    for d in range(1, D):
        t[d] = tab[d][sel[d - 1]][:, sel[d]]
    return c, t


def best_first(score, nmax, expand):
    """score [D, j, c] conditional node scores in (0,1]. Returns (depth, lane, parent) per popped node, pop order (excl. root)."""
    nodes = []; heap = []; cnt = 0
    row = score[0, 0]
    for c in np.argsort(-row, kind="stable")[:expand]:
        heapq.heappush(heap, (-row[c], cnt, 1, int(c), -1)); cnt += 1
    while heap and len(nodes) < nmax - 1:
        negp, _, d, c, par = heapq.heappop(heap)
        idx = len(nodes); nodes.append((d, c, par))
        if d < D:
            row = score[d, c]
            for c2 in np.argsort(-row, kind="stable")[:expand]:
                heapq.heappush(heap, (negp * row[c2], cnt, d + 1, int(c2), idx)); cnt += 1
    return nodes


def accept_by_n(nodes, cand, truth, ns):
    """Accepted draft length for each budget N (tree = first N-1 pops)."""
    ch = {}
    for i, (d, c, par) in enumerate(nodes):
        ch.setdefault(par, {})[int(cand[d - 1, c])] = i
    path = []; cur = -1
    while len(path) < len(truth) and len(path) < D:
        nxt = ch.get(cur, {}).get(int(truth[len(path)]))
        if nxt is None: break
        path.append(nxt); cur = nxt
    return [sum(1 for i in path if i < n - 1) for n in ns]


# calibration (fit on even requests, applied to odd requests): per (depth bucket, prob bin) empirical P(candidate is the true token |
# predecessor on the true path). Bins on log10 p.
PBINS = np.array([-9, -4, -3, -2.5, -2, -1.5, -1.25, -1, -0.75, -0.5, -0.3, -0.15, -0.05, 0.01])
DB = np.array([0, 1, 2, 4, 8, 12, 16, 20, 24, 31])


def dbucket(d): return int(np.searchsorted(DB, d, side="right") - 1)


def calib_counts(prob, cand, truth):
    hit = np.zeros((len(DB), len(PBINS))); tot = np.zeros_like(hit)
    j = 0
    for d in range(min(D, len(truth))):
        row = prob[d, j if d > 0 else 0]
        b = np.clip(np.searchsorted(PBINS, np.log10(np.maximum(row, 1e-12)), side="right") - 1, 0, len(PBINS) - 2)
        db = dbucket(d)
        np.add.at(tot[db], b, 1)
        w = np.nonzero(cand[d] == truth[d])[0]
        if len(w) == 0: break
        np.add.at(hit[db], b[w], 1)
        j = int(w[0])
    return hit, tot


CFG = None  # set in worker: list of (name, k, kind, param, expand)


def work(args):
    dump, items, cal = args
    raw = np.memmap(dump, dtype=np.float32, mode="r").reshape(-1, REC)
    res = []
    for (ds, qi, ri, p, truth, L) in items:
        cand, marg, tab = unpack(raw[ri])
        rec = {"ds": ds, "qi": qi, "chain": L, "cov": {}, "covpos": {}, "tree": {}, "calib": {}}
        subs = {}
        for k in (16, 32, 64):
            c, t = sub(cand, marg, tab, k); subs[k] = (c, t)
            inset = [bool(d < len(truth) and truth[d] in set(c[d].tolist())) for d in range(D)]
            pre = 0
            while pre < D and inset[pre]: pre += 1
            rec["cov"][k] = pre; rec["covpos"][k] = inset
        for (name, k, kind, param, expand) in CFG:
            c, t = subs[k]
            if kind == "temp":
                score = softmax(t / param)
            elif kind == "calib":
                prob = softmax(t)
                if cal is None:
                    h, tt = calib_counts(prob, c, truth); rec["calib"][k] = (h, tt); continue
                lp = np.log10(np.maximum(prob, 1e-12))
                b = np.clip(np.searchsorted(PBINS, lp, side="right") - 1, 0, len(PBINS) - 2)
                tabc = cal[k]  # [len(DB), len(PBINS)]
                score = np.empty_like(prob)
                for d in range(D):
                    score[d] = tabc[dbucket(d)][b[d]]
                # tie-break inside a bin by the raw prob (keeps the within-row order of the model)
                score = score * (1 + 1e-6 * prob)
            nodes = best_first(score, NMAX, expand)
            rec["tree"][name] = accept_by_n(nodes, c, truth, NS)
        res.append(rec)
    return res


def run(dump, items, cal, nproc):
    chunks = [items[i::nproc * 4] for i in range(nproc * 4)]
    with Pool(nproc, initializer=_init, initargs=(CFG,)) as pool:
        out = pool.map(work, [(dump, ch, cal) for ch in chunks])
    return [r for o in out for r in o]


def _init(cfg):
    global CFG
    CFG = cfg


if __name__ == "__main__":
    dump, meta, outp = sys.argv[1:4]
    nproc = int(os.environ.get("P4COV_NPROC", os.cpu_count()))
    raw, items = load_rounds(dump, meta)
    if os.environ.get("P4COV_MAXREQ"):
        mq = int(os.environ["P4COV_MAXREQ"]); items = [it for it in items if it[1] % (200 // mq) == 0]
    print("rounds", len(items), flush=True)
    dss = sorted(set(it[0] for it in items))
    # pass 1: calibration counts on even requests
    CFG = [("cal", k, "calib", None, 0) for k in (16, 32, 64)]
    r1 = run(dump, [it for it in items if it[1] % 2 == 0], None, nproc)
    cal = {}
    for k in (16, 32, 64):
        h = sum(r["calib"][k][0] for r in r1); t = sum(r["calib"][k][1] for r in r1)
        # pooled fallback for empty cells: per prob-bin over all depths
        pooled = (h.sum(0) + 1) / (t.sum(0) + 2)
        c = np.where(t >= 20, (h + 1) / (t + 2), pooled[None, :])
        cal[k] = np.minimum(c, 1.0)
    CFG = []
    for k in (16, 32, 64):
        for e in ((8,) if k == 16 else (8, 16)):
            CFG.append((f"K{k}_e{e}_t1.0", k, "temp", 1.0, e))
        CFG.append((f"K{k}_e8_cal", k, "calib", None, 8))
    for k in (16, 64):
        for tp in (0.7, 1.5):
            CFG.append((f"K{k}_e8_t{tp}", k, "temp", tp, 8))
    rs = run(dump, items, cal, nproc)
    odd = [r for r in rs if r["qi"] % 2 == 1]

    def summ(sel):
        o = {"n_rounds": len(sel), "chain_emit": float(np.mean([r["chain"] + 1 for r in sel]))}
        for k in (16, 32, 64):
            o[f"cov_prefix_K{k}"] = float(np.mean([r["cov"][k] for r in sel]))
            o[f"cov_prefix15_K{k}"] = float(np.mean([min(r["cov"][k], 15) for r in sel]))
            o[f"cov_pos_K{k}"] = [float(np.mean([r["covpos"][k][d] for r in sel])) for d in range(D)]
            o[f"cov_first15_count_K{k}"] = float(np.mean([sum(r["covpos"][k][:15]) for r in sel]))
        o["tree_emit"] = {name: {str(n): float(np.mean([r["tree"][name][i] + 1 for r in sel])) for i, n in enumerate(NS)} for name in sel[0]["tree"]}
        o["tree_full_frac"] = {name: {str(n): float(np.mean([r["tree"][name][i] == D for r in sel])) for i, n in enumerate(NS)} for name in sel[0]["tree"]}
        return o

    res = {"T1": T1, "K": K, "D": D, "Ns": NS, "calibration_fit": "even requests; *_cal rows are only meaningful on 'odd'",
           "all": summ(rs), "odd": summ(odd), "by_ds": {d: summ([r for r in rs if r["ds"] == d]) for d in dss},
           "calib_tables": {k: v.tolist() for k, v in cal.items()}}
    json.dump(res, open(outp, "w"), indent=1)
    a = res["all"]
    print("chain", a["chain_emit"])
    for k in (16, 32, 64):
        print(f"K{k} cov prefix {a[f'cov_prefix_K{k}']:.2f} (first15-prefix {a[f'cov_prefix15_K{k}']:.2f}, count in first 15 {a[f'cov_first15_count_K{k}']:.2f})")
    for name, v in a["tree_emit"].items():
        print(name, v, "odd:", res["odd"]["tree_emit"][name])
