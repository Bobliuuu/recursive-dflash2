"""Tree ReplaySSM (TRS) for the 27B GDN tree verify. Instead of snapshotting the fp32 recurrent state of every tree node (151 MB per node
for Qwen3.8-27B: the dominant per-node verify cost), the verify kernel
  * keeps the state in registers when a node's parent is the previous node (skip-reload),
  * stores a snapshot only for nodes that have a child which is not the next node (flag buffer, needed to restart a branch),
  * writes each node's raw (k, v, g, beta) into a small ring (SGLang's existing ReplaySSM ring-write in the same kernel),
and after acceptance the last accepted node's snapshot is rebuilt (before SGLang's commit copies it) by replaying the accepted path's ring entries from the pre-verify state with the same
fp32 update as the kernel. Buffers live here (module `rdf2_trs`, looked up by the patched GDN backend). Enabled by RDF2_TRS=1."""
import os, torch, triton, triton.language as tl

BUF = None


def alloc(temporal_all, n_k_heads, N, device):
    """temporal_all: all-layer fp32 state [Lm, S, HV, V, K]."""
    global BUF
    if BUF is not None:
        return BUF
    Lm, S, HV, V, K = temporal_all.shape
    f32 = torch.float32
    BUF = dict(rawv=torch.zeros(Lm, S, HV, N, V, dtype=f32, device=device), rawk=torch.zeros(Lm, S, n_k_heads, N, K, dtype=f32, device=device),
               g=torch.zeros(Lm, S, HV, N, dtype=f32, device=device), beta=torch.zeros(Lm, S, HV, N, dtype=f32, device=device),
               flag=torch.ones(N, dtype=torch.int32, device=device), N=N, H=n_k_heads)
    return BUF


def kwargs_for(layer_index):
    if BUF is None:
        return {}
    return dict(cache_ring=True, replayssm_rawv=BUF["rawv"][layer_index], replayssm_rawk=BUF["rawk"][layer_index],
                replayssm_g=BUF["g"][layer_index], replayssm_beta=BUF["beta"][layer_index], rdf2_store_flag=BUF["flag"])


@triton.jit
def _replay_kernel(h_ptr, s_layer, s_slot, o_ptr, so_layer, so_slot, so_step, slot_ptr, row_ptr, path_ptr, len_ptr, rawv, rawk, rg, rb,
                   HV: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, NMAX: tl.constexpr, S: tl.constexpr, BV: tl.constexpr):
    l = tl.program_id(0); i_hv = tl.program_id(1); i_v = tl.program_id(2)
    i_h = i_hv // (HV // H)
    slot = tl.load(slot_ptr).to(tl.int64)
    L = tl.load(len_ptr)
    o_k = tl.arange(0, K); o_v = i_v * BV + tl.arange(0, BV)
    p_h = h_ptr + l * s_layer + slot * s_slot + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    b_h = tl.load(p_h)
    base = (l * S + slot)
    for t in range(0, L):
        node = tl.load(path_ptr + t)
        b_k = tl.load(rawk + ((base * H + i_h) * NMAX + node) * K + o_k)
        b_v = tl.load(rawv + ((base * HV + i_hv) * NMAX + node) * V + o_v)
        b_g = tl.load(rg + (base * HV + i_hv) * NMAX + node)
        b_beta = tl.load(rb + (base * HV + i_hv) * NMAX + node)
        k_sum_sq = tl.sum(b_k * b_k)
        k_rsqrt = 1.0 / tl.sqrt(k_sum_sq + 1e-6)
        b_k *= k_rsqrt
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
    if L > 0:                                   # idle / dummy rounds commit nothing
        last = tl.minimum(tl.maximum(tl.load(path_ptr + L - 1).to(tl.int64), 0), NMAX - 1)
        row = tl.load(row_ptr).to(tl.int64)
        tl.store(o_ptr + l * so_layer + row * so_slot + last * so_step + i_hv * K * V + o_v[None, :] * K + o_k[:, None], b_h)


def replay(temporal_all, inter_all, slot, row, path, commit_len):
    """Replay the accepted path (path[0:commit_len], tree node ids) from the pre-verify state temporal_all[:, slot] and write the result into
    the snapshot of the last accepted node, inter_all[:, row, path[commit_len-1]] (row = the request's scratch row, 0 at bs=1), which SGLang's normal commit then copies to the state.
    Must run BEFORE that commit (it reads the pre-verify state)."""
    Lm, S, HV, V, K = temporal_all.shape
    assert temporal_all.dtype == torch.float32 and temporal_all.stride(-1) == 1 and temporal_all.stride(2) == V * K
    assert inter_all.dtype == torch.float32 and inter_all.shape[-3:] == (HV, V, K) and inter_all.stride(-1) == 1 and inter_all.stride(3) == V * K
    BV = 32
    _replay_kernel[(Lm, HV, V // BV)](temporal_all, temporal_all.stride(0), temporal_all.stride(1),
                                     inter_all, inter_all.stride(0), inter_all.stride(1), inter_all.stride(2), slot, row, path, commit_len,
                                     BUF["rawv"], BUF["rawk"], BUF["g"], BUF["beta"], HV=HV, H=BUF["H"], K=K, V=V, NMAX=BUF["N"], S=S, BV=BV,
                                     num_warps=4)


def set_flags(parent):
    """flag[j] = 1 iff node j has a child c != j+1 (that child restarts from j's snapshot)."""
    if os.environ.get("RDF2_TRS_ALL"):        # debug: store every snapshot (replay still used)
        BUF["flag"].fill_(1); return
    N = parent.shape[0]
    idx = torch.arange(N, device=parent.device, dtype=torch.int32)
    need = ((parent[None, :] == idx[:, None]) & (idx[None, :] != idx[:, None] + 1)).any(1)
    BUF["flag"].copy_(need.to(torch.int32))


def set_all_flags():
    BUF["flag"].fill_(1)
