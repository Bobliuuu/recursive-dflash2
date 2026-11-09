"""T2: recursive second pass for DSpine (Phase 3), kill-test version (teacher-forced, offline; no engine integration).

Pass 1 = a frozen block-16 DSpine drafter (normal MASK-conditioned parallel draft; greedy backbone tokens d_1..d_15).
Pass 2 = a trainable block-B2 (32) DSpine drafter (same architecture, init from a checkpoint) whose block input is
         [anchor, d_1..d_15, MASK x (B2-16)]  -- i.e. conditioned on the pass-1 draft TOKENS instead of MASKs -- and which emits B2-1 tokens.
Training is on-policy w.r.t. pass 1 (d_i are the frozen pass-1 model's own argmax drafts for the same anchors/contexts).
Metrics (held-out, teacher-forced, greedy chain acceptance, same anchors for every variant):
  tau_p1  : pass-1 chain (16 block)
  tau_p2  : pass-2 chain (B2 block)           -> kill if tau_p2 / tau_p1 < 1.05
  tau_p2_15: pass-2 chain truncated to 15 draft tokens (isolates the refinement gain on equal length)
  tau_b32 : plain MASK-conditioned B2-block chain from the pass-2 init (no pass-1 tokens)  (the "just train longer" reference)
tau = mean over blocks of 1 + sum_k prod_{j<=k} correct_j  (bonus token counted, like the engine's accept length).
"""
import argparse, json, math, os, random, time, copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--init1", required=True, help="dir with sf_state.pt of the pass-1 (b16) model")
ap.add_argument("--init2", required=True, help="dir with sf_state.pt used to init pass 2 (b32 weights)")
ap.add_argument("--out", required=True)
ap.add_argument("--b1", type=int, default=16)
ap.add_argument("--b2", type=int, default=32)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--anchors", type=int, default=64)
ap.add_argument("--eval-anchors", type=int, default=128)
ap.add_argument("--holdout", type=int, default=256)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--target", default="Qwen/Qwen3-8B")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev = torch.device("cuda"); BF = torch.bfloat16
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "train.log"), "a")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()

samples = torch.load(a.data + f".tok{a.max_len}.pt")
hold = samples[-a.holdout:]; train = samples[:-a.holdout]
def make_batches(data, bs, rng):
    idx = list(range(len(data))); rng.shuffle(idx)
    chunks = [idx[i:i + bs * 64] for i in range(0, len(idx), bs * 64)]
    out = []
    for c in chunks:
        c.sort(key=lambda i: len(data[i][0])); out += [c[i:i + bs] for i in range(0, len(c) - bs + 1, bs)]
    rng.shuffle(out); return out
def collate(data, ids_list):
    L = max(len(data[i][0]) for i in ids_list)
    x = torch.zeros(len(ids_list), L, dtype=torch.long); m = torch.zeros_like(x); lm = torch.zeros_like(x)
    for j, i in enumerate(ids_list):
        ids, l = data[i]; x[j, :len(ids)] = torch.tensor(ids); m[j, :len(ids)] = 1; lm[j, :len(ids)] = torch.tensor(l)
    return x, m, lm

from specforge.modeling.draft.dspine import DSpineDraftModel
from specforge.algorithms.dspine.model import OnlineDSpineModel
target = AutoModelForCausalLM.from_pretrained(a.target, dtype=BF).to(dev).eval(); target.requires_grad_(False)
TL = [1, 9, 17, 25, 33]
sf_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dspine_sf_config.json")))
sf_cfg["dspine_config"]["replacement_probability"] = 0.0

class RecWrapper(OnlineDSpineModel):
    """Same objective; optionally replaces MASK inputs at block positions 1..K with given token ids."""
    noise_tokens = None   # [B, n_blocks, K] or None
    def _create_noise_embed(self, input_ids, anchors, keep):
        e = super()._create_noise_embed(input_ids, anchors, keep)
        if self.noise_tokens is None: return e
        B, n, bs = anchors.shape[0], anchors.shape[1], self.block_size
        e = e.view(B, n, bs, -1).clone()
        K = self.noise_tokens.shape[-1]
        e[:, :, 1:1 + K] = self.embed_tokens(self.noise_tokens)
        return e.view(B, n * bs, -1)

def build(block, state_dir):
    c = copy.deepcopy(sf_cfg); c["block_size"] = block
    cfg = Qwen3Config.from_dict(c); cfg._attn_implementation = "flex_attention"
    with torch.device(dev):
        d = DSpineDraftModel(cfg).to(BF)
    sd = torch.load(state_dir + "/sf_state.pt", map_location="cpu")["draft"]
    d.load_state_dict(sd, strict=True)
    w = RecWrapper(draft_model=d, target_lm_head=target.lm_head, target_embed_tokens=target.model.embed_tokens, mask_token_id=151669,
                   block_size=block, attention_backend="flex_attention", num_anchors=a.anchors, loss_decay_gamma=7.0,
                   objective_chunk_blocks=16, fused_logits_loss=True, warmup_ratio=0.04).to(dev)
    return w

w1 = build(a.b1, a.init1).eval(); w1.requires_grad_(False)
w2 = build(a.b2, a.init2)          # trainable pass 2
w3 = build(a.b2, a.init2).eval(); w3.requires_grad_(False)  # plain b32 reference (frozen init; MASK inputs)
log("built; params(M)", sum(p.numel() for p in w2.draft_model.parameters() if p.requires_grad) / 1e6)

@torch.no_grad()
def feats(x, m):
    out = target.model(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False)
    return torch.cat([out.hidden_states[k + 1] for k in TL], dim=-1), out.hidden_states[-1]

def blocks_forward(w, x, aux, lm, anchors, keep, noise_tokens=None, grad=False):
    """Replicates OnlineDSpineModel.forward up to the draft output for GIVEN anchors."""
    B, L = x.shape
    offs = torch.arange(w.block_size, device=dev)
    idx = anchors.unsqueeze(-1) + offs
    safe = idx.clamp(max=L - 1)
    labels = x.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
    sup = (idx < L) & keep.unsqueeze(-1)
    sup = sup & (lm.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe) > 0.5)
    sup = sup[..., 1:].int().cumprod(-1).bool()
    pos = torch.cat((torch.arange(L, device=dev).expand(B, -1), w._create_position_ids(anchors)), dim=1)
    w.noise_tokens = noise_tokens
    with torch.set_grad_enabled(grad):
        out = w.draft_model(position_ids=pos, attention_mask=w._attention_mask(anchors, keep, L),
                            noise_embedding=w._create_noise_embed(x, anchors, keep), target_hidden=aux.detach(),
                            block_token_ids=labels, replace_mask=None, kernel_options={"BACKEND": "TRITON"})
    w.noise_tokens = None
    return out, labels, sup

@torch.no_grad()
def greedy(w, out):
    h = out.hidden_states[..., 1:, :]
    B, n, K, H = h.shape
    ids = torch.cat([w.lm_head(c).argmax(-1) for c in h.reshape(-1, K, H).split(2048)])
    return ids.view(B, n, K)

def tau_from(pred, labels, sup, K=None):
    c = (pred == labels[..., 1:]) & sup
    if K: c = c[..., :K]
    run = c.int().cumprod(-1)
    return (1 + run.sum(-1)).float()   # per block

def sample_anchors(w, lm, L, nanch):
    w.num_anchors = nanch
    nv = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
    return type(w)._sample_anchor_positions(w, L, lm, dev, nv)

def pass1_tokens(x, aux, lm, anchors, keep):
    out, labels, sup = blocks_forward(w1, x, aux, lm, anchors, keep)
    return greedy(w1, out), labels, sup

def evaluate(nb=48):
    w2.eval()
    hb = make_batches(hold, a.micro_bs, random.Random(0))[:nb]
    acc = {"p1": [0., 0], "p2": [0., 0], "p2_15": [0., 0], "b32": [0., 0], "b32_15": [0., 0]}
    g = torch.Generator(device=dev);
    with torch.no_grad():
        for b in hb:
            x, m, lm = collate(hold, b); x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
            aux, _ = feats(x, m)
            torch.manual_seed(1234 + b[0])
            anchors, keep = sample_anchors(w1, lm, x.shape[1], a.eval_anchors)
            d1, lab1, sup1 = pass1_tokens(x, aux, lm, anchors, keep)
            t1 = tau_from(d1, lab1, sup1)[keep]
            nt = torch.cat([d1], -1)
            out2, lab2, sup2 = blocks_forward(w2, x, aux, lm, anchors, keep, noise_tokens=nt)
            p2 = greedy(w2, out2)
            t2 = tau_from(p2, lab2, sup2)[keep]; t2_15 = tau_from(p2, lab2, sup2, a.b1 - 1)[keep]
            out3, lab3, sup3 = blocks_forward(w3, x, aux, lm, anchors, keep)
            p3 = greedy(w3, out3)
            t3 = tau_from(p3, lab3, sup3)[keep]; t3_15 = tau_from(p3, lab3, sup3, a.b1 - 1)[keep]
            for k, t in (("p1", t1), ("p2", t2), ("p2_15", t2_15), ("b32", t3), ("b32_15", t3_15)):
                acc[k][0] += float(t.sum()); acc[k][1] += int(t.numel())
    w2.train()
    r = {k: v[0] / v[1] for k, v in acc.items()}
    r["ratio_p2_over_p1"] = r["p2"] / r["p1"]; r["ratio_p2_over_b32"] = r["p2"] / r["b32"]
    return r

log("EVAL step0", json.dumps(evaluate()))
if a.eval_only: raise SystemExit

params = [p for p in w2.draft_model.parameters() if p.requires_grad]
master = [p.detach().clone().float().requires_grad_(True) for p in params]
opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
warm = max(1, int(0.05 * a.steps))
lr_at = lambda s: a.lr * (s + 1) / warm if s < warm else a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))
rng = random.Random(a.seed); step = 0; t0 = time.time(); agg = [0., 0]
w2.train(); done = False
while not done:
    batches = make_batches(train, a.micro_bs, rng)
    for bi in range(0, len(batches) - a.accum + 1, a.accum):
        for gq in opt.param_groups: gq["lr"] = lr_at(step)
        for b in batches[bi:bi + a.accum]:
            x, m, lm = collate(train, b); x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
            aux, last = feats(x, m)
            anchors, keep = sample_anchors(w2, lm, x.shape[1], a.anchors)
            d1, _, _ = pass1_tokens(x, aux, lm, anchors, keep)   # on-policy pass-1 drafts for these exact anchors
            # pass-2 loss through the standard objective, with the anchors forced and pass-1 tokens as block input
            w2._sample_anchor_positions = lambda *args, **kw: (anchors, keep)
            w2.noise_tokens = d1
            nvalid = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
            loss, acc_, met = w2(input_ids=x, hidden_states=aux, loss_mask=lm, target_last_hidden_states=last, max_valid_anchors=nvalid,
                                 global_step=10**9, total_steps=10**9 + 1)
            w2.noise_tokens = None
            cnt = met["accuracy_denom"].clamp_min(1)
            (loss * x.shape[0] / cnt / a.accum).backward()
            agg[0] += float(loss * x.shape[0] / cnt) / a.accum; agg[1] += 1.0 / a.accum
        for mp, p in zip(master, params): mp.grad = p.grad.float() if p.grad is not None else torch.zeros_like(mp)
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        for mp, p in zip(master, params): p.data.copy_(mp.data); p.grad = None
        step += 1
        if step % 10 == 0:
            el = time.time() - t0
            log(f"step {step}/{a.steps} loss {agg[0]/agg[1]:.4f} gn {float(gn):.2f} {el/step:.1f}s/step eta {(a.steps-step)*el/step/3600:.2f}h"); agg = [0., 0]
        if step % a.eval_every == 0 or step >= a.steps:
            log("EVAL step", step, json.dumps(evaluate()))
        if step >= a.steps: done = True; break
torch.save({"draft": w2.draft_model.state_dict(), "step": step}, a.out + "/sf_state.pt")
log("saved", step)
