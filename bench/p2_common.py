"""Phase 2 shared loaders: parse the RDF2_DUMP rounds file + collector meta, align rounds to the true greedy output."""
import json, os, numpy as np
T1 = int(os.environ.get('P2_T1', 15)); K = 16  # T1 = drafts per block (block_size-1)
REC = 1 + T1 + T1 * K + T1 * K * K

def load(dump, meta):
    raw = np.fromfile(dump, dtype=np.float64)
    assert raw.size % REC == 0
    raw = raw.reshape(-1, REC)
    meta = json.load(open(meta))
    reqs = []
    for m in meta:
        recs = raw[m["r0"]:m["r1"]]
        ids = m["ids"]
        p = 0; rounds = []; bad = 0
        for r in recs:
            anchor = int(r[0])
            if p >= len(ids) - 1: break
            if anchor != ids[p]:
                bad += 1; break
            greedy = r[1:1 + T1].astype(np.int64)
            cand = r[1 + T1:1 + T1 + T1 * K].reshape(T1, K).astype(np.int64)
            tab = r[1 + T1 + T1 * K:].reshape(T1, K, K).astype(np.float32)
            truth = ids[p + 1:p + 1 + T1]
            L = 0
            while L < len(truth) and L < T1 and greedy[L] == truth[L]: L += 1
            rounds.append(dict(p=p, greedy=greedy, cand=cand, tab=tab, truth=np.array(truth), L=L))
            p += L + 1
        reqs.append(dict(ds=m["ds"], id=m["id"], ids=ids, rounds=rounds, bad=bad, n_out=m["n_out"]))
    return reqs
