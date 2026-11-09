"""DSpine's own greedy-chain table kernel (sglang_dspine.kernels.dspine._serial_table, Triton) with one extra output: the full
[n, K, K] predecessor-conditioned logit table it already computes in registers (s[j, c] = logit of candidate c given the previous row chose
lane j) before taking its row-wise argmax. DSpine's greedy chain is the argmax walk of exactly this table, so the tree can be built from it
for the cost of a store instead of re-deriving the fp32 refinement (bench/p2tree/glue.py `table`, ~0.4 ms per round on A100).
Note this is DSpine's linearised greedy table (bf16 TG/TSV factor tables), not the fp32 serial_rs_table used for sampling; it changes tree
probabilities slightly (not losslessness: the accept walk follows the target argmax)."""
import torch, triton, triton.language as tl
from sglang_dspine.kernels import dspine as dk

TABLES = {}   # n -> static [n, K, K] fp32 buffer (allocated outside graph capture, on the eager warmup call)


@triton.jit
def _serial_table_s(Raw, stride_raw, NormW, Cand, InputIds, LmW, stride_w, TG, TSV, Query, stride_q, Rms, GateB, Out, SOut,
                    H: tl.constexpr, RK: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, TM1: tl.constexpr, BH: tl.constexpr, BR: tl.constexpr):
    n = tl.program_id(0)
    b = n // TM1
    r = n % TM1
    row = b * (TM1 + 1) + 1 + r
    kp = tl.arange(0, KP)
    live = kp < K
    ids = tl.load(Cand + n * K + kp, mask=live, other=0)
    prev = tl.load(Cand + tl.maximum(n - 1, 0) * K + kp, mask=live, other=0)
    anchor = tl.load(InputIds + b * (TM1 + 1)).to(tl.int64)
    sid = tl.where(r == 0, anchor, prev)
    base = tl.zeros([KP], dtype=tl.float32)
    for h0 in tl.static_range(0, H, BH):
        off = h0 + tl.arange(0, BH)
        u = tl.load(Raw + row * stride_raw + off).to(tl.float32) * tl.load(NormW + off).to(tl.float32)
        w = tl.load(LmW + ids[:, None] * stride_w + off[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        base += tl.sum(w * u[None, :], axis=1)
    s = tl.zeros([KP, KP], dtype=tl.float32)
    for r0 in tl.static_range(0, RK, BR):
        rr = r0 + tl.arange(0, BR)
        gate = tl.load(TSV + sid[:, None] * (2 * RK) + rr[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        val = tl.load(TSV + sid[:, None] * (2 * RK) + RK + rr[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        q = tl.load(Query + row * stride_q + rr).to(tl.float32)
        gb = tl.load(GateB + rr).to(tl.float32)
        m = tl.sigmoid(q[None, :] + gate + gb[None, :]) * val
        g = tl.load(TG + ids[:, None] * RK + rr[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        s += tl.dot(m, tl.trans(g), input_precision="ieee")
    base = tl.where(live, base, -1e30)
    s = base[None, :] + tl.load(Rms + row).to(tl.float32) * s
    tl.store(SOut + n * K * K + kp[:, None] * K + kp[None, :], s, mask=live[:, None] & live[None, :])
    (best_value, best) = tl.max(s, axis=1, return_indices=True, return_indices_tie_break_left=True)
    tl.store(Out + n * K + kp, best, mask=live)


def serial_table(raw, norm_w, cand, input_ids, lm_head_w, tg, tsv, packed, rms, gate_b, block_size: int, confidence=None):
    """Drop-in for dk.serial_table (same idx output) that also fills TABLES[n]. Falls back to the stock kernel with confidence."""
    if confidence is not None:
        return dk._rdf2_orig_serial_table(raw, norm_w, cand, input_ids, lm_head_w, tg, tsv, packed, rms, gate_b, block_size, confidence=confidence)
    t = int(block_size)
    (m, h) = raw.shape
    (n, k) = cand.shape
    rk = tg.shape[1]
    assert m % t == 0 and n == m // t * (t - 1) and raw.dtype == torch.bfloat16 and cand.dtype == torch.int64 and cand.is_contiguous()
    buf = TABLES.get(n)
    if buf is None:
        assert not torch.cuda.is_current_stream_capturing(), "table buffer must be allocated before capture"
        buf = TABLES[n] = torch.zeros((n, k, k), dtype=torch.float32, device=raw.device)
    idx = torch.empty((n, k), dtype=torch.int64, device=raw.device)
    if n:
        _serial_table_s[n,](raw, raw.stride(0), norm_w, cand, input_ids, lm_head_w, lm_head_w.stride(0), tg, tsv, packed, packed.stride(0),
                            rms, gate_b, idx, buf, H=h, RK=rk, K=k, KP=max(k, 16), TM1=t - 1, BH=512, BR=128, num_warps=8, num_stages=2)
    return idx


def install():
    if getattr(dk.serial_table, "_rdf2", False): return
    dk._rdf2_orig_serial_table = dk.serial_table
    serial_table._rdf2 = True
    dk.serial_table = serial_table
