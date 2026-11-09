"""Unit test (box GPU): SGLang's GDN tree-verify kernels with the tree links bench/p5h100/p4tree27b/glue_fast.py writes.
For a random best-first-shaped tree (parent < child), the parent-linked causal conv + gated-delta recurrence must give, at every node,
the same output (and cached state) as the chain kernels run along that node's root path. Qwen3.8-27B GDN shapes, fp32 states."""
import torch, random, sys
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
torch.manual_seed(0); random.seed(0); dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
H, HV, K, V, W = 16, 48, 128, 128, 4
D = 2 * H * K + HV * V
par = [-1] + [random.randrange(0, i) if random.random() < 0.4 else i - 1 for i in range(1, N)]
p = torch.tensor(par, dtype=torch.int32, device=dev); idx = torch.arange(N, dtype=torch.int32, device=dev)
big = torch.full((N, N), N, dtype=torch.int32, device=dev)
child = (p[None, :] == idx[:, None]) & (idx[None, :] > 0)
nxt = torch.where(child, idx[None, :].expand(N, N), big).amin(1)
sib = (p[None, :] == p[:, None]) & (idx[None, :] > idx[:, None]) & (idx[:, None] > 0)
nsb = torch.where(sib, idx[None, :].expand(N, N), big).amin(1)
nxt = torch.where(nxt >= N, -1, nxt).view(1, N).int(); nsb = torch.where(nsb >= N, -1, nsb).view(1, N).int()

def path(i):
    r = [i]
    while par[r[-1]] >= 0: r.append(par[r[-1]])
    return r[::-1]

conv_w = torch.randn(D, W, device=dev, dtype=torch.bfloat16) * 0.3
x = torch.randn(N, D, device=dev, dtype=torch.bfloat16)
conv0 = torch.randn(2, D, W - 1, device=dev, dtype=torch.bfloat16)
A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.5; dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.5
a = torch.randn(N, HV, device=dev, dtype=torch.bfloat16); b = torch.randn(N, HV, device=dev, dtype=torch.bfloat16)
h0 = torch.randn(2, HV, V, K, device=dev, dtype=torch.float32) * 0.1

def run(xs, a_, b_, tree):
    T = xs.shape[0]
    cs = conv0.clone(); icw = torch.zeros(2, N, D, W - 1, device=dev, dtype=torch.bfloat16)
    rpt = torch.zeros(1, T, dtype=torch.int32, device=dev)
    y = causal_conv1d_update(xs.view(1, T, D).transpose(1, 2), cs, conv_w, None, "silu", conv_state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
                             intermediate_conv_window=icw, intermediate_state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
                             retrieve_next_token=nxt if tree else None, retrieve_next_sibling=nsb if tree else None,
                             retrieve_parent_token=rpt if tree else None)
    y = y.transpose(1, 2).reshape(T, D)
    q, k, v = y.split([H * K, H * K, HV * V], -1)
    ist = torch.zeros(2, N, HV, V, K, device=dev, dtype=torch.float32)
    o = fused_sigmoid_gating_delta_rule_update(A_log=A_log, dt_bias=dt_bias, q=q.reshape(1, T, H, K), k=k.reshape(1, T, H, K), v=v.reshape(1, T, HV, V),
          a=a_, b=b_, initial_state_source=h0.clone(), initial_state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
          cu_seqlens=torch.tensor([0, T], device=dev, dtype=torch.int32), use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
          is_kda=False, disable_state_update=True, intermediate_states_buffer=ist, intermediate_state_indices=torch.tensor([0], device=dev, dtype=torch.int32),
          cache_steps=N, retrieve_parent_token=rpt if tree else None)
    return o.reshape(T, HV * V).float(), ist[0, :T], icw[0, :T], rpt

ot, st, cw, rpt = run(x, a, b, True)
print("kernel parents ok:", rpt[0, 1:].tolist() == par[1:])
eo = es = ec = 0.0
for i in range(N):
    pi = path(i); oc, sc, cc, _ = run(x[pi], a[pi], b[pi], False)
    eo = max(eo, (ot[i] - oc[-1]).abs().max().item()); es = max(es, (st[i] - sc[-1]).abs().max().item()); ec = max(ec, (cw[i].float() - cc[-1].float()).abs().max().item())
print(f"N={N}: max |out diff| {eo:.3g}, max |state diff| {es:.3g}, max |conv window diff| {ec:.3g}")
