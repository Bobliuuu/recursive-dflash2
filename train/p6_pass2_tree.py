"""Phase 6: recursive pass 2 that sees pass-1 hidden FEATURES and outputs a TREE (kill test, offline teacher-forced proxy).

Pass 1 = frozen official tencent/Qwen3-8B-DSpine-b16, run at block 36 (block-causal, so positions 1..31 equal the b16/b32/b48 runs).
Variants (both on-policy w.r.t. pass 1: their inputs are pass 1's own outputs for the same anchors/contexts; warm-started so that
step 0 reproduces pass 1 exactly):
  V1 "full":  pass 2 = trainable copy of the official 5-layer drafter at block 32; block input at position t>=1 is
              MASK-embedding + sum_{o in 0..4} W_o h1[t+o]  (h1 = pass-1 normed final hidden; W_o zero-init, so step 0 == pass 1).
              Position t sees pass-1 beliefs up to t+4 (lookahead pass 1 never had: it is block-causal).
  V2 "small": pass 2 = 2 new DSpine layers (init from official layers 3,4 with o_proj/down_proj zeroed, first injection write zeroed,
              last injection = official layer-4 injection) on top of pass-1's last-layer PRE-injection residual, plus
              sum_{o in 1..4} W_o h1[t+o] (zero-init). Context projection (fc, hidden_norm) shared with pass 1 (frozen).
              Step 0 == pass 1 exactly. Cost ~2/5 of the backbone layers.
Proxy per block (held-out T=0 greedy target sequences, 128 seeded anchors each, identical anchors for every model):
  candidates = top-16 of lm_head(hidden) per position, table = DSpine transition scores [31, 16 pred, 16 cand],
  chain tau (refined greedy selection, DSpine's own chain) at 15 and 31 drafts, backbone-argmax chain,
  best-first tree tau (same algorithm as bench/p2tree tree_build: joint prob, expand top-8, N nodes incl. anchor, depth cap 31),
  walked along the data (= target greedy) path; tau counts the bonus token.  V2 also: tree over pass-1 candidates rescored by its table.
"""
import argparse, json, math, os, random, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="jsonl of {prompt,response,truncated} (T=0.7 regen)")
ap.add_argument("--eval", required=True, help="jsonl held-out, T=0 greedy regen")
ap.add_argument("--out", required=True)
ap.add_argument("--variants", default="v1,v2")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr1", type=float, default=1e-4)
ap.add_argument("--lr2", type=float, default=2e-4)
ap.add_argument("--anchors", type=int, default=64)
ap.add_argument("--eval-anchors", type=int, default=128)
ap.add_argument("--eval-seqs", type=int, default=256)
ap.add_argument("--eval-at", default="0,150,300")
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--target", default="Qwen/Qwen3-8B")
ap.add_argument("--official", default="tencent/Qwen3-8B-DSpine-b16")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--grad-ckpt", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev = torch.device("cuda"); BF = torch.bfloat16
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "train.log"), "a")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()

B1, B2, D, K = 36, 32, 31, 16          # pass-1 block, pass-2 block, tree depth cap, candidates
OFF1, OFF2 = (0, 1, 2, 3, 4), (1, 2, 3, 4)
VARS = [v for v in a.variants.split(",") if v]

# ---------------- data ----------------
tok = AutoTokenizer.from_pretrained(a.target)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
def load_samples(path):
    cache = path + f".tok{a.max_len}.pt"
    if os.path.exists(cache): return torch.load(cache)
    out = []
    for l in open(path):
        r = json.loads(l)
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]; ri = tok(r["response"], add_special_tokens=False)["input_ids"]
        if not r.get("truncated"): ri = ri + [IM_END]
        ids = (pi + ri)[: a.max_len]; lm = ([0] * len(pi) + [1] * len(ri))[: a.max_len]
        if sum(lm) < 4: continue
        out.append((ids, lm))
    torch.save(out, cache); return out
train = load_samples(a.train); hold = load_samples(a.eval)[: a.eval_seqs]
log(f"train {len(train)} hold {len(hold)} mean_len_hold {sum(len(s[0]) for s in hold)/len(hold):.0f}")

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
    return x.to(dev), m.to(dev), lm.to(dev)

# ---------------- models ----------------
from specforge.modeling.draft.dspine import DSpineDraftModel
from specforge.algorithms.dspine.model import OnlineDSpineModel
target = AutoModelForCausalLM.from_pretrained(a.target, dtype=BF).to(dev).eval(); target.requires_grad_(False)
TL = [1, 9, 17, 25, 33]
sf_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dspine_sf_config.json")))
sf_cfg["dspine_config"]["replacement_probability"] = 0.0
sf_cfg["dspine_config"]["alignment_warmup_steps"] = 0

def tencent_to_sf(sd):
    P = "selected_prefix_bridge."
    out = {k: v for k, v in sd.items() if k.startswith(("layers.", "fc.", "hidden_norm.", "norm."))}
    for i in range(4): out[f"injection.readouts.{i}.weight"] = sd[P + "early_read_weights"][i]
    out["injection.readouts.4.weight"] = sd[P + "read_weight"]
    for i in range(5): out[f"injection.writes.{i}.weight"] = sd[P + "feature_write_weights"][i]
    out["injection.message.weight"] = sd[P + "prev.weight"]
    out["injection.receiver_gate.weight"] = sd[P + "gate.weight"]
    out["injection.bias"] = sd[P + "gate.bias"]
    out["injection.predecessor_gate.weight"] = sd[P + "predecessor_gate.weight"]
    out["transfer_codes"] = sd[P + "codebook"].float()
    return out
OFF_SD = tencent_to_sf(load_file(snapshot_download(a.official) + "/model.safetensors"))

class Wrap(OnlineDSpineModel):
    """Standard DSpine objective with externally forced anchors and an externally supplied block-input embedding."""
    forced = None; emb = None
    def _sample_anchor_positions(self, *args, **kw):
        if self.forced is not None: return self.forced
        return super()._sample_anchor_positions(*args, **kw)
    def _create_noise_embed(self, input_ids, anchors, keep):
        if self.emb is not None: return self.emb
        return super()._create_noise_embed(input_ids, anchors, keep)

def build(block, nlayers, sd):
    c = copy.deepcopy(sf_cfg); c["block_size"] = block; c["num_hidden_layers"] = nlayers
    c["layer_types"] = ["full_attention"] * nlayers; c["max_window_layers"] = nlayers
    cfg = Qwen3Config.from_dict(c); cfg._attn_implementation = "flex_attention"
    with torch.device(dev):
        d = DSpineDraftModel(cfg).to(BF)
    missing, unexpected = d.load_state_dict({k: v.to(BF) for k, v in sd.items() if k != "transfer_codes"}, strict=False)
    missing = [m for m in missing if m not in ("transfer_codes", "transfer_ready")]
    assert not missing and not unexpected, (missing, unexpected)
    d.transfer_codes.copy_(sd["transfer_codes"].to(d.transfer_codes.dtype)); d.transfer_ready.fill_(True)
    w = Wrap(draft_model=d, target_lm_head=target.lm_head, target_embed_tokens=target.model.embed_tokens, mask_token_id=151669,
             block_size=block, attention_backend="flex_attention", num_anchors=a.anchors, loss_decay_gamma=7.0,
             objective_chunk_blocks=16, fused_logits_loss=True).to(dev)
    return w

def v2_state():
    sd = {}
    for k, v in OFF_SD.items():
        if k.startswith("layers."):
            i = int(k.split(".")[1]); rest = k.split(".", 2)[2]
            if i in (3, 4): sd[f"layers.{i - 3}.{rest}"] = v.clone()
        elif k.startswith("injection.readouts."):
            i = int(k.split(".")[2])
            if i in (3, 4): sd[f"injection.readouts.{i - 3}.weight"] = v.clone()
        elif k.startswith("injection.writes."):
            i = int(k.split(".")[2])
            if i == 4: sd["injection.writes.1.weight"] = v.clone()
            if i == 3: sd["injection.writes.0.weight"] = torch.zeros_like(v)
        else:
            sd[k] = v.clone()
    for i in (0, 1):
        sd[f"layers.{i}.self_attn.o_proj.weight"].zero_(); sd[f"layers.{i}.mlp.down_proj.weight"].zero_()
    return sd

class FeatIn(nn.Module):
    """sum_o W_o h1[t+o] for block positions t = 1..B2-1 (anchor row untouched). Zero-init."""
    def __init__(self, H, offsets):
        super().__init__(); self.offsets = offsets
        self.proj = nn.ModuleList(nn.Linear(H, H, bias=False) for _ in offsets)
        for p in self.proj: nn.init.zeros_(p.weight)
    def forward(self, h1):                      # h1 [B, n, B1, H] -> [B, n, B2, H]
        out = torch.zeros(*h1.shape[:2], B2, h1.shape[-1], device=h1.device, dtype=h1.dtype)
        acc = 0
        for o, p in zip(self.offsets, self.proj):
            acc = acc + p(h1[:, :, 1 + o:B2 + o])
        return torch.cat((out[:, :, :1], out[:, :, 1:] + acc), dim=2)

w1 = build(B1, 5, OFF_SD).eval(); w1.requires_grad_(False)
models = {}
if "v1" in VARS:
    w = build(B2, 5, OFF_SD); fi = FeatIn(4096, OFF1).to(dev, BF); models["v1"] = (w, fi, a.lr1)
if "v2" in VARS:
    w = build(B2, 2, v2_state()); w.draft_model.fc.requires_grad_(False); w.draft_model.hidden_norm.requires_grad_(False)
    fi = FeatIn(4096, OFF2).to(dev, BF); models["v2"] = (w, fi, a.lr2)
for k, (w, fi, _) in models.items():
    if a.grad_ckpt: w.draft_model.gradient_checkpointing_enable()
    log(k, "trainable params (M)", (sum(p.numel() for p in w.draft_model.parameters() if p.requires_grad) + sum(p.numel() for p in fi.parameters())) / 1e6)

@torch.no_grad()
def feats(x, m):
    out = target.model(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False)
    return torch.cat([out.hidden_states[k + 1] for k in TL], dim=-1), out.hidden_states[-1]

def labels_for(x, lm, anchors, keep, bs):
    L = x.shape[1]
    idx = anchors.unsqueeze(-1) + torch.arange(bs, device=dev)
    safe = idx.clamp(max=L - 1)
    labels = x.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
    sup = (idx < L) & keep.unsqueeze(-1)
    sup = sup & (lm.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe) > 0.5)
    sup = sup[..., 1:].int().cumprod(-1).bool()
    return labels, sup

def draft_forward(w, x, aux, lm, anchors, keep, emb=None, grad=False):
    labels, _ = labels_for(x, lm, anchors, keep, w.block_size)
    pos = torch.cat((torch.arange(x.shape[1], device=dev).expand(x.shape[0], -1), w._create_position_ids(anchors)), dim=1)
    if emb is None: emb = w._create_noise_embed(x, anchors, keep)
    with torch.set_grad_enabled(grad):
        return w.draft_model(position_ids=pos, attention_mask=w._attention_mask(anchors, keep, x.shape[1]), noise_embedding=emb,
                             target_hidden=aux.detach(), block_token_ids=labels, replace_mask=None, kernel_options={"BACKEND": "TRITON"})

def pass2_emb(name, w, fi, x, anchors, keep, o1):
    """block input embedding for pass 2, flattened [B, n*B2, H]"""
    h1 = o1.hidden_states                                   # [B, n, B1, H] normed
    if name == "v1":
        base = OnlineDSpineModel._create_noise_embed(w, x, anchors, keep)
        base = base.view(x.shape[0], anchors.shape[1], B2, -1)
    else:
        base = o1.pre_injection[:, :, :B2]
    e = base + fi(h1)
    return e.reshape(x.shape[0], anchors.shape[1] * B2, -1)

# ---------------- proxy ----------------
def tree_tau(table, cand, truth, vlen, N, E=8):
    """Batched best-first tree (same rule as bench/p2tree tree_build) + walk along truth. table [M,D,16,16] logits (pred, cand)."""
    M = table.shape[0]; ar = torch.arange(M, device=dev)
    P = table.float().softmax(-1)
    Fn = N * E
    fs = torch.full((M, Fn), -1.0, device=dev); fd = torch.zeros(M, Fn, dtype=torch.long, device=dev)
    fl = torch.zeros_like(fd); fp = torch.zeros_like(fd)
    nd = torch.zeros(M, N, dtype=torch.long, device=dev); nl = torch.zeros_like(nd); npar = torch.full_like(nd, -1)
    ok = torch.zeros(M, N, dtype=torch.bool, device=dev); ok[:, 0] = True
    def expand(it, d, lane, score, valid):
        row = P[ar, d.clamp(max=D - 1), lane]                # [M,16]
        tv, ti = row.topk(E, -1)
        valid = valid & (d < D)
        s = slice(it * E, it * E + E)
        fs[:, s] = torch.where(valid[:, None], score[:, None] * tv, torch.full_like(tv, -1.0))
        fd[:, s] = (d + 1)[:, None]; fl[:, s] = ti; fp[:, s] = it
    expand(0, torch.zeros(M, dtype=torch.long, device=dev), torch.zeros(M, dtype=torch.long, device=dev), torch.ones(M, device=dev),
           torch.ones(M, dtype=torch.bool, device=dev))
    for it in range(1, N):
        v, idx = fs.max(1)
        val = v > 0
        nd[:, it] = fd[ar, idx]; nl[:, it] = fl[ar, idx]; npar[:, it] = torch.where(val, fp[ar, idx], torch.full_like(idx, -9)); ok[:, it] = val
        fs[ar, idx] = -1.0
        expand(it, nd[:, it], nl[:, it], v.clamp_min(0), val)
    ntok = cand[ar[:, None], (nd - 1).clamp(min=0), nl]
    ntok = torch.where(ok, ntok, torch.full_like(ntok, -2)); ntok[:, 0] = -1
    cur = torch.zeros(M, dtype=torch.long, device=dev); acc = torch.zeros(M, device=dev); alive = torch.ones(M, dtype=torch.bool, device=dev)
    for L in range(D):
        mt = (npar == cur[:, None]) & (ntok == truth[:, L:L + 1])
        f = mt.any(1) & alive & (L < vlen)
        cur = torch.where(f, mt.float().argmax(1), cur); acc += f.float(); alive = f
    return 1 + acc

def chain_tau(ids, truth, vlen, n):
    c = (ids[:, :n] == truth[:, :n]) & (torch.arange(n, device=dev)[None] < vlen[:, None])
    return 1 + c.int().cumprod(-1).sum(-1).float()

@torch.no_grad()
def proxy(draft, out, labels, sup, keep, cand_override=None, chunk=64):
    """per kept block metrics dict of 1-D tensors."""
    h = out.hidden_states[:, :, 1:D + 1]; pre = out.pre_injection[:, :, 1:D + 1]
    B, n = h.shape[:2]
    res = {}
    for s in range(0, n, chunk):
        hh, pp = h[:, s:s + chunk].flatten(0, 1), pre[:, s:s + chunk].flatten(0, 1)
        kk = keep[:, s:s + chunk].flatten()
        hh, pp = hh[kk], pp[kk]
        if hh.shape[0] == 0: continue
        lab = labels[:, s:s + chunk].flatten(0, 1)[kk]; su = sup[:, s:s + chunk].flatten(0, 1)[kk]
        truth = lab[:, 1:D + 1]; vlen = su[:, :D].sum(-1); anc = lab[:, 0]
        logits = target.lm_head(hh)                          # [m, D, V]
        bb = logits.argmax(-1); cand = logits.topk(K, -1).indices; del logits
        cands = {"own": cand}
        if cand_override is not None: cands["p1cand"] = cand_override[s]
        r = {"chain_bb15": chain_tau(bb, truth, vlen, 15), "chain_bb31": chain_tau(bb, truth, vlen, 31)}
        incov = (cand == truth[..., None]).any(-1) & (torch.arange(D, device=dev)[None] < vlen[:, None])
        r["cov_prefix"] = incov.int().cumprod(-1).sum(-1).float()
        for cn, cd in cands.items():
            tab = draft.transition_scores(pp, cd, anc, target.lm_head.weight)     # [m, D, 16, 16]
            ref = DSpineDraftModel.select_cached(tab, cd)
            sfx = "" if cn == "own" else "_p1cand"
            r["chain_ref15" + sfx] = chain_tau(ref, truth, vlen, 15); r["chain_ref31" + sfx] = chain_tau(ref, truth, vlen, 31)
            for N in (16, 32, 48, 64):
                r[f"tree{N}" + sfx] = tree_tau(tab, cd, truth, vlen, N)
        for k, v in r.items(): res.setdefault(k, []).append(v)
        res.setdefault("_cand", {})[s] = cand
    return res

def evaluate(step):
    for w, fi, _ in models.values(): w.eval()
    hb = [list(range(i, min(i + a.micro_bs, len(hold)))) for i in range(0, len(hold), a.micro_bs)]
    agg = {}; seqid = []
    t0 = time.time()
    with torch.no_grad():
        for b in hb:
            x, m, lm = collate(hold, b); aux, _ = feats(x, m)
            torch.manual_seed(1234 + b[0])
            w1.num_anchors = a.eval_anchors
            nv = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
            anchors, keep = OnlineDSpineModel._sample_anchor_positions(w1, x.shape[1], lm, dev, nv)
            o1 = draft_forward(w1, x, aux, lm, anchors, keep)
            lab, sup = labels_for(x, lm, anchors, keep, B2)
            sidm = torch.tensor(b, device=dev)[:, None].expand_as(keep)
            for s0 in range(0, keep.shape[1], 64):      # same chunk order as proxy()
                seqid.append(sidm[:, s0:s0 + 64].flatten()[keep[:, s0:s0 + 64].flatten()])
            r1 = proxy(w1.draft_model, o1, lab, sup, keep)
            p1c_by_start = r1.pop("_cand")
            for k, v in r1.items(): agg.setdefault("official/" + k, []).extend(v)
            for name, (w, fi, _) in models.items():
                e = pass2_emb(name, w, fi, x, anchors, keep, o1)
                o2 = draft_forward(w, x, aux, lm, anchors, keep, emb=e)
                r2 = proxy(w.draft_model, o2, lab, sup, keep, cand_override=p1c_by_start if name == "v2" else None)
                r2.pop("_cand")
                for k, v in r2.items(): agg.setdefault(name + "/" + k, []).extend(v)
    for w, fi, _ in models.values(): w.train()
    per = {k: torch.cat(v).cpu() for k, v in agg.items()}
    sid = torch.cat(seqid).cpu()
    summ = {k: float(v.mean()) for k, v in per.items()}
    summ["n_blocks"] = int(sid.numel()); summ["eval_s"] = time.time() - t0
    # paired bootstrap over sequences: ratio variant tree48 / official tree48
    gen = torch.Generator().manual_seed(0); S = len(hold)
    seq_sum = lambda v: torch.zeros(S).index_add_(0, sid, v.float())
    cnt = seq_sum(torch.ones_like(sid, dtype=torch.float))
    for name in models:
        for key in ("tree48", "tree48_p1cand", "chain_ref15", "chain_ref31"):
            if f"{name}/{key}" not in per: continue
            base = "tree48" if key.startswith("tree") else key
            a_ = seq_sum(per[f"{name}/{key}"]); b_ = seq_sum(per[f"official/{base}"])
            rs = []
            for _ in range(1000):
                i = torch.randint(0, S, (S,), generator=gen)
                rs.append(float(a_[i].sum() / b_[i].sum()))
            rs.sort()
            summ[f"{name}/{key}_over_official"] = float(a_.sum() / b_.sum())
            summ[f"{name}/{key}_over_official_ci95"] = [rs[25], rs[974]]
    torch.save({"per": per, "sid": sid}, os.path.join(a.out, f"per_block_step{step}.pt"))
    json.dump(summ, open(os.path.join(a.out, f"eval_step{step}.json"), "w"), indent=1)
    log("EVAL", step, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}))
    return summ

eval_at = {int(s) for s in a.eval_at.split(",") if s}
if 0 in eval_at or a.eval_only: evaluate(0)
if a.eval_only: raise SystemExit

# ---------------- training (both variants share the target and pass-1 forwards) ----------------
opts = {}
for name, (w, fi, lr) in models.items():
    params = [p for p in list(w.draft_model.parameters()) + list(fi.parameters()) if p.requires_grad]
    master = [p.detach().clone().float().requires_grad_(True) for p in params]
    opts[name] = (params, master, torch.optim.AdamW(master, lr=lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True), lr)
    w.train()
warm = max(1, int(0.05 * a.steps))
lr_at = lambda s, lr: lr * (s + 1) / warm if s < warm else lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm)))
rng = random.Random(a.seed); step = 0; t0 = time.time(); agg = {k: [0., 0.] for k in models}; done = False
while not done:
    batches = make_batches(train, a.micro_bs, rng)
    for bi in range(0, len(batches) - a.accum + 1, a.accum):
        for name, (_, _, opt, lr) in opts.items():
            for gq in opt.param_groups: gq["lr"] = lr_at(step, lr)
        for b in batches[bi:bi + a.accum]:
            x, m, lm = collate(train, b)
            aux, last = feats(x, m)
            nv = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
            w1.num_anchors = a.anchors
            anchors, keep = OnlineDSpineModel._sample_anchor_positions(w1, x.shape[1], lm, dev, nv)
            with torch.no_grad():
                o1 = draft_forward(w1, x, aux, lm, anchors, keep)
            for name, (w, fi, _) in models.items():
                w.forced = (anchors, keep); w.emb = pass2_emb(name, w, fi, x, anchors, keep, o1)
                loss, acc_, met = w(input_ids=x, hidden_states=aux, loss_mask=lm, target_last_hidden_states=last, max_valid_anchors=nv,
                                    global_step=10**9, total_steps=10**9 + 1)
                w.forced = None; w.emb = None
                (loss / a.accum).backward()
                agg[name][0] += float(loss) / a.accum; agg[name][1] += float(acc_) / a.accum
        gns = {}
        for name, (params, master, opt, _) in opts.items():
            for mp, p in zip(master, params): mp.grad = p.grad.float() if p.grad is not None else torch.zeros_like(mp)
            gns[name] = float(torch.nn.utils.clip_grad_norm_(master, 1.0)); opt.step(); opt.zero_grad(set_to_none=True)
            for mp, p in zip(master, params): p.data.copy_(mp.data); p.grad = None
        step += 1
        if step % 10 == 0:
            el = time.time() - t0
            log(f"step {step}/{a.steps} " + " ".join(f"{k}: loss {v[0]/10:.4f} acc {v[1]/10:.4f} gn {gns[k]:.2f}" for k, v in agg.items())
                + f" {el/step:.1f}s/step eta {(a.steps-step)*el/step/3600:.2f}h mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")
            agg = {k: [0., 0.] for k in models}
        if step in eval_at or step >= a.steps:
            evaluate(step)
        if step >= a.steps: done = True; break
for name, (w, fi, _) in models.items():
    torch.save({"draft": w.draft_model.state_dict(), "featin": fi.state_dict(), "step": step}, os.path.join(a.out, f"{name}_state.pt"))
log("saved", step)
