"""Phase 6: tree-aware DSpine fine-tune + offline tree-acceptance proxy (kill test). See bench/p6treetrain/PREREG.md.

Warm start from the official tencent/Qwen3-8B-DSpine-b16 weights (SpecForge DSpine model + objective, single-GPU colocated
HF Qwen3-8B target, like train/train_dspine.py). Adds L_tree (refined CE of the target's argmax given the true predecessor,
softmax over the 16 candidates, + 0.5 x backbone full-vocab CE of the argmax), weights exp(-(i-1)/gamma_tree).

Proxy (`proxy_eval`): teacher-forced held-out sequences x fixed random anchors; chain = DSpine refined greedy chain (15 tok);
tree = best-first over the exact [31,16,16] transition table (N nodes incl. anchor, 8 children per pop, depth <= 31).

Usage (box):
  python train_treeaware.py --data D.jsonl --out OUT --eval-only            # official weights proxy
  python train_treeaware.py --data D.jsonl --out OUT --tree-lambda 1.0      # V1
"""
import argparse, json, math, os, random, time, copy
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--block", type=int, default=32)
ap.add_argument("--steps", type=int, default=925)
ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup-ratio", type=float, default=0.04)
ap.add_argument("--anchors", type=int, default=64)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--gamma", type=float, default=7.0)
ap.add_argument("--tree-lambda", type=float, default=0.0)
ap.add_argument("--tree-gamma", type=float, default=14.0)
ap.add_argument("--tree-cov-mu", type=float, default=0.5)
ap.add_argument("--dspine-scale", type=float, default=1.0)
ap.add_argument("--init", default="tencent/Qwen3-8B-DSpine-b16", help="HF id/dir (Tencent format) or dir with sf_state.pt")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--eval-every", type=int, default=300)
ap.add_argument("--eval-anchors", type=int, default=128)
ap.add_argument("--eval-seqs", type=int, default=256)
ap.add_argument("--holdout", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--target", default="Qwen/Qwen3-8B")
ap.add_argument("--official-config", default="tencent/Qwen3-8B-DSpine-b16")
a = ap.parse_args()

torch.manual_seed(a.seed); random.seed(a.seed)
dev = torch.device("cuda"); BF = torch.bfloat16
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "train.log"), "a")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
log("ARGS", json.dumps(vars(a)))

from huggingface_hub import snapshot_download
off_dir = snapshot_download(a.official_config)
off_cfg = json.load(open(off_dir + "/config.json"))

# ---------------- data (same tokenisation as train_dspine.py) ----------------
tok = AutoTokenizer.from_pretrained(a.target)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
cache = a.data + f".tok{a.max_len}.pt"
if os.path.exists(cache):
    samples = torch.load(cache)
else:
    samples = []
    for l in open(a.data):
        r = json.loads(l)
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ri = tok(r["response"], add_special_tokens=False)["input_ids"]
        if not r.get("truncated"): ri = ri + [IM_END]
        ids = (pi + ri)[: a.max_len]
        lm = ([0] * len(pi) + [1] * len(ri))[: a.max_len]
        if sum(lm) < 4: continue
        samples.append((ids, lm))
    torch.save(samples, cache)
hold = samples[-a.holdout:]
train = samples[: len(samples) - a.holdout]
log(f"samples train={len(train)} hold={len(hold)} mean_len={sum(len(s[0]) for s in train)/len(train):.0f}")

def make_batches(data, bs, rng):
    idx = list(range(len(data))); rng.shuffle(idx)
    chunks = [idx[i:i + bs * 64] for i in range(0, len(idx), bs * 64)]
    out = []
    for c in chunks:
        c.sort(key=lambda i: len(data[i][0]))
        out += [c[i:i + bs] for i in range(0, len(c) - bs + 1, bs)]
    rng.shuffle(out); return out

def collate(data, ids_list):
    L = max(len(data[i][0]) for i in ids_list)
    x = torch.zeros(len(ids_list), L, dtype=torch.long); m = torch.zeros_like(x); lm = torch.zeros_like(x)
    for j, i in enumerate(ids_list):
        ids, l = data[i]; x[j, :len(ids)] = torch.tensor(ids); m[j, :len(ids)] = 1; lm[j, :len(ids)] = torch.tensor(l)
    return x, m, lm

# ---------------- models ----------------
from specforge.modeling.draft.dspine import DSpineDraftModel
from specforge.algorithms.dspine.model import OnlineDSpineModel
from specforge.core.fused_loss import ce_l1_loss

target = AutoModelForCausalLM.from_pretrained(a.target, dtype=BF).to(dev).eval(); target.requires_grad_(False)
TL = off_cfg["dflash_config"]["target_layer_ids"]

sf_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dspine_sf_config.json")))
sf_cfg["block_size"] = a.block
sf_cfg["dspine_config"]["replacement_probability"] = 0.0
sf_cfg["dspine_config"]["alignment_warmup_steps"] = max(1, int(a.warmup_ratio * a.steps))
cfg = Qwen3Config.from_dict(sf_cfg); cfg._attn_implementation = "flex_attention"
with torch.device(dev):
    draft = DSpineDraftModel(cfg).to(BF)

def sf_to_tencent(sd, codes_bf16):
    out = {k: v.to(BF) for k, v in sd.items() if k.startswith(("layers.", "fc.", "hidden_norm.", "norm."))}
    P = "selected_prefix_bridge."
    out[P + "early_read_weights"] = torch.stack([sd[f"injection.readouts.{i}.weight"] for i in range(4)]).to(BF)
    out[P + "read_weight"] = sd["injection.readouts.4.weight"].to(BF)
    out[P + "feature_write_weights"] = torch.stack([sd[f"injection.writes.{i}.weight"] for i in range(5)]).to(BF)
    out[P + "prev.weight"] = sd["injection.message.weight"].to(BF)
    out[P + "gate.weight"] = sd["injection.receiver_gate.weight"].to(BF)
    out[P + "gate.bias"] = sd["injection.bias"].to(BF)
    out[P + "predecessor_gate.weight"] = sd["injection.predecessor_gate.weight"].to(BF)
    out[P + "out.weight"] = torch.zeros(1024, 512, dtype=BF)
    out[P + "codebook"] = codes_bf16
    out[P + "independent_write_version"] = torch.tensor(1, dtype=torch.int64)
    out[P + "terminal_disabled_version"] = torch.tensor(1, dtype=torch.int64)
    out["draft_block_causal_version"] = torch.tensor(1, dtype=torch.int64)
    out["last_write_refine_version"] = torch.tensor(1, dtype=torch.int64)
    return {k: v.contiguous().cpu() for k, v in out.items()}

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

if os.path.exists(os.path.join(a.init, "sf_state.pt")):
    draft.load_state_dict(torch.load(a.init + "/sf_state.pt", map_location="cpu")["draft"], strict=True)
    log("loaded sf_state", a.init)
else:
    d = a.init if os.path.isdir(a.init) else snapshot_download(a.init)
    sd = tencent_to_sf(load_file(d + "/model.safetensors"))
    missing, unexpected = draft.load_state_dict({k: v.to(BF) for k, v in sd.items() if k != "transfer_codes"}, strict=False)
    log("loaded official; missing", [m for m in missing if m not in ("transfer_codes", "transfer_ready")], "unexpected", unexpected)
    draft.transfer_codes.copy_(sd["transfer_codes"].to(draft.transfer_codes.dtype))
    draft.transfer_ready.fill_(True)


class TreeDSpine(OnlineDSpineModel):
    """DSpine objective (copied from SpecForge OnlineDSpineModel._objective_terms) + optional tree loss folded into the
    refinement term (so SpecForge's forward/normalisation are unchanged)."""
    tree_lambda = 0.0; tree_gamma = 14.0; tree_mu = 0.5; dspine_scale = 1.0

    def _objective_terms(self, hidden, pre_injection, layer_features, labels, predecessors, teacher_hidden, weights, supervised):
        config = self.dspine_config
        with torch.no_grad():
            teacher_logits = self.lm_head(teacher_hidden)
        logits = self.lm_head(hidden)
        terms = self._logits_loss_terms(logits, labels, teacher_logits=teacher_logits, top_k=config.top_k)
        candidates = terms.topk_ids
        refined = self.draft_model.refine(pre_injection, predecessors)
        scores = torch.einsum("bmh,bmkh->bmk", refined, F.embedding(candidates, self.lm_head.weight)).float()
        target_candidates = teacher_logits.softmax(dim=-1, dtype=torch.float32).gather(-1, candidates)
        mass = target_candidates.sum(dim=-1)
        covered = candidates == labels.unsqueeze(-1)
        refinement = ce_l1_loss(scores, covered.long().argmax(dim=-1), teacher_logits=teacher_logits.gather(-1, candidates))
        refined_ce = refinement.cross_entropy * covered.any(dim=-1)
        refined_l1 = refinement.teacher_l1 * (mass > config.candidate_mass_threshold)
        reference_codes = F.embedding(labels, self.draft_model.transfer_codes).float()
        alignment = 1 - F.cosine_similarity(layer_features.float(), reference_codes.unsqueeze(1), dim=-1)
        s = self.dspine_scale
        ref_total = s * ((config.ce_weight * refined_ce + (1 - config.ce_weight) * refined_l1) * weights).sum()
        if self.tree_lambda > 0:
            with torch.no_grad():
                greedy = teacher_logits.argmax(-1)
            cov_g = candidates == greedy.unsqueeze(-1)
            has_g = cov_g.any(-1)
            logp = scores.log_softmax(-1)
            tree_ce = -logp.gather(-1, cov_g.long().argmax(-1, keepdim=True)).squeeze(-1) * has_g
            lf = logits.float()
            bb_ce = torch.logsumexp(lf, -1) - lf.gather(-1, greedy.unsqueeze(-1)).squeeze(-1)
            pos = torch.arange(labels.shape[-1], device=labels.device, dtype=torch.float32)
            wt = supervised.float() * torch.exp(-pos / self.tree_gamma)
            ref_total = ref_total + self.tree_lambda * ((tree_ce + self.tree_mu * bb_ce) * wt).sum()
        with torch.no_grad():
            correct = ((terms.argmax_ids == labels) * supervised).sum()
            coverage = (covered.any(dim=-1) * supervised).sum()
        return (s * (terms.cross_entropy * weights).sum(), s * (terms.teacher_l1 * weights).sum(), ref_total,
                s * (alignment * weights.unsqueeze(1)).sum(), correct.float(), coverage.float())


TreeDSpine.tree_lambda = a.tree_lambda; TreeDSpine.tree_gamma = a.tree_gamma
TreeDSpine.tree_mu = a.tree_cov_mu; TreeDSpine.dspine_scale = a.dspine_scale
wrapper = TreeDSpine(draft_model=draft, target_lm_head=target.lm_head, target_embed_tokens=target.model.embed_tokens,
                     mask_token_id=151669, block_size=a.block, attention_backend="flex_attention", num_anchors=a.anchors,
                     loss_decay_gamma=a.gamma, objective_chunk_blocks=16, fused_logits_loss=True).to(dev)
wrapper.draft_model.to(BF)
log("draft params (M):", sum(p.numel() for p in draft.parameters() if p.requires_grad) / 1e6, "codes", draft.transfer_codes.dtype)

@torch.no_grad()
def target_feats(x, m):
    out = target.model(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return torch.cat([hs[k + 1] for k in TL], dim=-1), hs[-1]

# ---------------- proxy eval ----------------
EXPAND = 8; NS = (16, 32, 48, 64); D = a.block - 1; CH = 15

@torch.no_grad()
def tree_accept(cand, logp, truth, tvalid, N, Dcap):
    """Batched best-first tree. cand [M,D,K] ids, logp [M,D,K,K] log-softmax rows (row d, pred j, cand c; row 0 uses j=0),
    truth [M,D] target greedy ids, tvalid [M,D] bool (prefix-valid). Returns accepted draft length [M]."""
    M, Dd, K = cand.shape
    F_ = EXPAND * N + EXPAND
    fs = torch.full((M, F_), -float("inf"), device=dev); fd = torch.zeros((M, F_), dtype=torch.long, device=dev)
    fl = torch.zeros_like(fd); fp = torch.full_like(fd, -1)
    top = logp[:, 0, 0].topk(EXPAND, -1)
    fs[:, :EXPAND] = top.values; fd[:, :EXPAND] = 0; fl[:, :EXPAND] = top.indices
    nfill = EXPAND
    nd = torch.zeros((M, N - 1), dtype=torch.long, device=dev); nl = torch.zeros_like(nd); npar = torch.zeros_like(nd)
    nok = torch.zeros((M, N - 1), dtype=torch.bool, device=dev)
    ar = torch.arange(M, device=dev)
    for i in range(N - 1):
        best = fs.argmax(-1); sc = fs[ar, best]; ok = torch.isfinite(sc)
        d = fd[ar, best]; l = fl[ar, best]; p = fp[ar, best]
        nd[:, i] = d; nl[:, i] = l; npar[:, i] = p; nok[:, i] = ok
        fs[ar, best] = -float("inf")
        if i == N - 2: break
        dn = (d + 1).clamp(max=Dd - 1)
        row = logp[ar, dn, l]                       # [M, K] children of (d, l) live at depth d+1
        t = row.topk(EXPAND, -1)
        can = ok & (d + 1 < Dcap)
        fs[:, nfill:nfill + EXPAND] = torch.where(can[:, None], sc[:, None] + t.values, torch.full_like(t.values, -float("inf")))
        fd[:, nfill:nfill + EXPAND] = (d + 1)[:, None]; fl[:, nfill:nfill + EXPAND] = t.indices
        fp[:, nfill:nfill + EXPAND] = i
        nfill += EXPAND
    ntok = cand[ar[:, None], nd, nl]                # [M, N-1]
    cur = torch.full((M,), -1, dtype=torch.long, device=dev); L = torch.zeros(M, dtype=torch.long, device=dev)
    alive = torch.ones(M, dtype=torch.bool, device=dev)
    for k in range(Dcap):
        match = nok & (npar == cur[:, None]) & (nd == k) & (ntok == truth[:, k:k + 1]) & tvalid[:, k:k + 1]
        f = match.any(-1) & alive
        cur = torch.where(f, match.float().argmax(-1), cur); L += f.long(); alive = f
    return L

def run_blocks(x, aux, lm, anchors, keep):
    B, L = x.shape
    offs = torch.arange(a.block, device=dev)
    idx = anchors.unsqueeze(-1) + offs
    safe = idx.clamp(max=L - 1)
    labels = x.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
    pos = torch.cat((torch.arange(L, device=dev).expand(B, -1), wrapper._create_position_ids(anchors)), dim=1)
    out = draft(position_ids=pos, attention_mask=wrapper._attention_mask(anchors, keep, L),
                noise_embedding=wrapper._create_noise_embed(x, anchors, keep), target_hidden=aux,
                block_token_ids=labels, replace_mask=None, kernel_options={"BACKEND": "TRITON"})
    return out, labels, idx

@torch.no_grad()
def proxy_eval():
    wrapper.eval()
    hb = make_batches(hold[:a.eval_seqs], a.micro_bs, random.Random(0))
    keys = ["chain15", "chain31", "bb_chain15"] + [f"tree{n}" for n in NS] + [f"tree{n}_d15" for n in (48,)]
    acc = {k: 0.0 for k in keys}; nblk = 0; cov15 = 0.0; tvlen = 0.0
    for b in hb:
        x, m, lm = collate(hold, b); x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
        aux, last = target_feats(x, m)
        B, L = x.shape
        g = torch.cat([target.lm_head(c).argmax(-1) for c in last.split(512, dim=1)], dim=1)   # g[t] = greedy next after x<=t
        torch.manual_seed(1234 + b[0])
        wrapper.num_anchors = a.eval_anchors
        nv = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
        anchors, keep = wrapper._sample_anchor_positions(L, lm, dev, nv)
        wrapper.num_anchors = a.anchors
        out, labels, idx = run_blocks(x, aux, lm, anchors, keep)
        n = anchors.shape[1]
        # truth: g at positions anchor+k-1 (k=1..D); valid while in range, supervised, and data token == greedy before it
        tidx = (idx[..., 1:] - 1).clamp(max=L - 1)                       # [B,n,D]
        truth = g.unsqueeze(1).expand(-1, n, -1).gather(2, tidx)
        inr = (idx[..., 1:] < L) & keep.unsqueeze(-1)
        sup = lm.unsqueeze(1).expand(-1, n, -1).gather(2, idx[..., 1:].clamp(max=L - 1)) > 0
        dataeq = labels[..., 1:] == truth                                  # x[a+k] == g[a+k-1]
        # position k is evaluable if k in range/supervised and all previous data tokens were greedy
        prev_ok = torch.cat([torch.ones_like(dataeq[..., :1]), dataeq[..., :-1]], -1).int().cumprod(-1).bool()
        tvalid = inr & sup & prev_ok
        hs = out.hidden_states[..., 1:, :]; pre = out.pre_injection[..., 1:, :]
        kf = keep.reshape(-1)
        hs = hs.reshape(B * n, D, -1)[kf]; pre = pre.reshape(B * n, D, -1)[kf]
        truth = truth.reshape(B * n, D)[kf]; tvalid = tvalid.reshape(B * n, D)[kf]
        anc = labels[..., 0].reshape(-1)[kf]
        for s in range(0, hs.shape[0], 64):
            h_, p_, t_, v_, a_ = hs[s:s + 64], pre[s:s + 64], truth[s:s + 64], tvalid[s:s + 64], anc[s:s + 64]
            lg = target.lm_head(h_)
            cand = lg.topk(16, -1).indices                                  # [m,D,16]
            bbg = lg.argmax(-1)
            tab = draft.transition_scores(p_, cand, a_, target.lm_head.weight)   # [m,D,16(pred),16(cand)]
            logp = tab.float().log_softmax(-1)
            ch = DSpineDraftModel.select_cached(tab, cand)                  # [m,D]
            ok = (ch == t_) & v_; run = ok.int().cumprod(-1)
            acc["chain15"] += float((1 + run[:, :CH].sum(-1)).sum()); acc["chain31"] += float((1 + run.sum(-1)).sum())
            okb = ((bbg == t_) & v_)[:, :CH].int().cumprod(-1); acc["bb_chain15"] += float((1 + okb.sum(-1)).sum())
            covr = ((cand == t_.unsqueeze(-1)).any(-1) & v_)[:, :CH].int().cumprod(-1); cov15 += float(covr.sum(-1).sum())
            tvlen += float(v_.int().cumprod(-1).sum(-1).sum())
            for N_ in NS:
                acc[f"tree{N_}"] += float((1 + tree_accept(cand, logp, t_, v_, N_, D)).sum())
            acc["tree48_d15"] += float((1 + tree_accept(cand, logp, t_, v_, 48, CH)).sum())
            nblk += h_.shape[0]
    wrapper.train()
    r = {k: v / nblk for k, v in acc.items()}
    r.update(n_blocks=nblk, cand_prefix_cov15=cov15 / nblk, truth_valid_len=tvlen / nblk)
    r["ratio_tree48_over_chain15"] = r["tree48"] / r["chain15"]
    return r

def save_ckpt(path, step):
    os.makedirs(path, exist_ok=True)
    sd = {k: v.detach() for k, v in draft.state_dict().items()}
    save_file(sf_to_tencent(sd, draft.transfer_codes.detach().to(BF)), path + "/model.safetensors")
    c = copy.deepcopy(off_cfg); c["block_size"] = 16
    json.dump(c, open(path + "/config.json", "w"), indent=1)
    json.dump({"step": step}, open(path + "/step.json", "w"))
    torch.save({"draft": draft.state_dict(), "step": step}, path + "/sf_state.pt")

results = {"args": vars(a), "evals": []}
def do_eval(step):
    t = time.time(); r = proxy_eval(); r["step"] = step; r["eval_s"] = round(time.time() - t, 1)
    results["evals"].append(r); log("PROXY", step, json.dumps(r))
    json.dump(results, open(os.path.join(a.out, "proxy.json"), "w"), indent=1)

draft.eval(); do_eval(0)
if a.eval_only: raise SystemExit

# ---------------- training ----------------
def step_batch(x, m, lm, gstep, total):
    x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
    aux, last = target_feats(x, m)
    n_valid = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
    loss, acc, met = wrapper(input_ids=x, hidden_states=aux, loss_mask=lm, target_last_hidden_states=last,
                             max_valid_anchors=n_valid, global_step=gstep, total_steps=total)
    # current SpecForge (b15ad38) already returns numerator / sum(position weights); no extra normalisation
    return loss, acc, met

params = [p for p in draft.parameters() if p.requires_grad]
master = [p.detach().clone().float().requires_grad_(True) for p in params]
opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
total = a.steps; warm = max(1, int(a.warmup_ratio * total))
def lr_at(s):
    if s < warm: return a.lr * (s + 1) / warm
    return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
log(f"plan: total_steps={total} seqs/step={a.micro_bs * a.accum} steps/epoch={len(train) // (a.micro_bs * a.accum)}")
rng = random.Random(a.seed)
wrapper.train(); step, t0 = 0, time.time(); agg = {"loss": 0., "acc": 0., "n": 0}; done = False
while not done:
    batches = make_batches(train, a.micro_bs, rng)
    for bi in range(0, len(batches) - a.accum + 1, a.accum):
        for gq in opt.param_groups: gq["lr"] = lr_at(step)
        for b in batches[bi:bi + a.accum]:
            x, m, lm = collate(train, b)
            loss, acc, met = step_batch(x, m, lm, step, total)
            (loss / a.accum).backward()
            agg["loss"] += float(loss) / a.accum; agg["acc"] += float(acc) / a.accum
        for mp, p in zip(master, params):
            mp.grad = p.grad.float() if p.grad is not None else torch.zeros_like(mp)
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        for mp, p in zip(master, params): p.data.copy_(mp.data); p.grad = None
        step += 1; agg["n"] += 1
        if step % 10 == 0:
            el = time.time() - t0
            log(f"step {step}/{total} loss {agg['loss']/agg['n']:.4f} acc {agg['acc']/agg['n']:.4f} gn {float(gn):.2f} "
                f"lr {lr_at(step):.2e} {el/step:.2f}s/step eta {(total-step)*el/step/3600:.2f}h")
            agg = {"loss": 0., "acc": 0., "n": 0}
        if step % a.eval_every == 0 and step < total:
            do_eval(step)
        if step >= total: done = True; break
do_eval(step)
save_ckpt(a.out + "/final", step)
log("saved final", step)
