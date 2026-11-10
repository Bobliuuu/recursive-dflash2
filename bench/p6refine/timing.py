"""Phase 6 cost estimate for a second draft pass (bs=1, Qwen3-8B DSpine geometry, CUDA graphs, bf16).
Plain-torch replica of one draft round's GPU work; only RATIOS are used: the engine-measured draft time (Phase 2) is scaled by
t(pass 2) / t(pass 1).  Pass-1 = fc on new context rows + 5 layers (q over the 48-token block, K/V over cached ctx + block,
draft-KV append of the new context rows) + DSpine injection per layer + lm_head/top-16 on 47 rows + transition table on 31 rows.
Usage: python timing.py out.json
"""
import sys, json, torch, torch.nn.functional as F
dev = "cuda"; BF = torch.bfloat16
H, I, NH, NKV, HD, V, R, MS = 4096, 12288, 32, 8, 128, 151936, 1024, 512
Q, CTX, NEW, DT, KC = 48, 1024, 12, 31, 16
g = torch.Generator(device=dev).manual_seed(0)
def W(*s): return (torch.randn(*s, device=dev, generator=g, dtype=torch.float32) * 0.02).to(BF)
class Layer:
    def __init__(s):
        s.q, s.k, s.v, s.o = W(NH * HD, H), W(NKV * HD, H), W(NKV * HD, H), W(H, NH * HD)
        s.gu, s.d = W(2 * I, H), W(H, I)
        s.rd, s.wr = W(R, H), W(H, MS)
        s.kc = torch.zeros(1, NKV, CTX + Q, HD, device=dev, dtype=BF); s.vc = torch.zeros_like(s.kc)
    def __call__(s, x, ctx_new):
        # draft-KV append for the new context rows
        s.kc[:, :, :NEW] = (ctx_new @ s.k.T).view(1, NEW, NKV, HD).transpose(1, 2)
        s.vc[:, :, :NEW] = (ctx_new @ s.v.T).view(1, NEW, NKV, HD).transpose(1, 2)
        h = F.rms_norm(x, (H,))
        q = (h @ s.q.T).view(1, Q, NH, HD).transpose(1, 2)
        s.kc[:, :, CTX:] = (h @ s.k.T).view(1, Q, NKV, HD).transpose(1, 2); s.vc[:, :, CTX:] = (h @ s.v.T).view(1, Q, NKV, HD).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, s.kc, s.vc, enable_gqa=True).transpose(1, 2).reshape(Q, NH * HD)
        x = x + a @ s.o.T
        gu = F.rms_norm(x, (H,)) @ s.gu.T
        x = x + (F.silu(gu[:, :I]) * gu[:, I:]) @ s.d.T
        # DSpine injection: readout, gate/message (shared small mats), write
        z = F.normalize((x @ s.rd.T).float(), dim=-1).to(BF)
        msg = z[:-1] @ MSG.T
        x = torch.cat((x[:1], x[1:] + (torch.sigmoid(msg) * msg) @ s.wr.T), 0)
        return x
MSG = W(MS, R)
FC = W(H, 5 * H); LM = W(V, H); CODES = W(V, R)
layers = [Layer() for _ in range(5)]
featin = [W(H, H) for _ in range(5)]
x0 = W(Q, H); aux = W(NEW, 5 * H); anchor = torch.zeros(1, dtype=torch.long, device=dev)

def head(x, cand=None):
    h = F.rms_norm(x[1:], (H,))
    if cand is None:
        cand = (h @ LM.T).topk(KC, -1).indices                 # [47, 16]
    c = cand[:DT]
    # transition table: refine with each predecessor candidate code, score against candidate head rows
    pred = torch.cat((anchor.expand(1, KC), c[:-1]), 0)
    m = (CODES[pred] @ MSG.T)                                   # [31, 16, MS]
    refined = h[:DT, None, :] + (torch.sigmoid(m) * m) @ layers[-1].wr.T
    tab = torch.einsum("ikh,ijh->ikj", F.rms_norm(refined, (H,)), LM[c])
    return cand, tab

def pass1():
    ctx = F.rms_norm(aux @ FC.T, (H,)); x = x0
    for L in layers: x = L(x, ctx)
    return head(x)
def pass2(nl, nfeat, own_head, own_fc):
    ctx = F.rms_norm(aux @ FC.T, (H,)) if own_fc else CTX_SHARED
    x = x0
    for f in featin[:nfeat]: x = x + x0 @ f.T
    for L in layers[:nl]: x = L(x, ctx)
    return head(x, None if own_head else CAND1)
CTX_SHARED = F.rms_norm(aux @ FC.T, (H,))
CAND1 = pass1()[0]

def bench(fn, iters=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): fn()
    for _ in range(20): gr.replay()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    for _ in range(iters): gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

res = {"gpu": torch.cuda.get_device_name(), "Q": Q, "ctx": CTX}
res["pass1_ms"] = bench(pass1)
res["v1_pass2_ms"] = bench(lambda: pass2(5, 5, True, True))
res["v2_pass2_owncand_ms"] = bench(lambda: pass2(2, 4, True, False))
res["v2_pass2_p1cand_ms"] = bench(lambda: pass2(2, 4, False, False))
res["head_only_ms"] = bench(lambda: head(x0))
ENGINE_DRAFT_MS = 3.15     # Phase 2 engine draft time at block 48-64 (docs/phase2-findings.md, verify cost vs width)
for k in ("v1_pass2", "v2_pass2_owncand", "v2_pass2_p1cand"):
    res[k + "_engine_est_ms"] = ENGINE_DRAFT_MS * res[k + "_ms"] / res["pass1_ms"]
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], "w"), indent=1)
