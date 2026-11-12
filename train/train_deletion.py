"""Phase 9: deletion tuning of the DSpine drafter's gate_proj (see bench/p9deletion/PREREG.md, train/deletion_mask.py).

The official tencent/Qwen3-8B-DSpine-b16 weights are frozen. A deletion mask over the 5 MLP gate_proj matrices
(5 x 12288 x 4096) is learned with the DSpine training objective (SpecForge model + loss, in-process HF Qwen3-8B
target, as train/train_dspine.py and train/train_treeaware.py). The target is never changed.

Train a mask:   python train_deletion.py --train-data D.jsonl --out OUT            -> OUT/mask.dmask, OUT/wanda_init.dmask
Proxy eval:     python train_deletion.py --out OUT --eval gsm8k=E1.jsonl ... --masks gen=G.dmask gen_rand=random:G.dmask
  Variants are evaluated on identical teacher-forced blocks (shared target features and anchors); "official" is always
  included. `random:<mask>` = per-matrix matched-count uniform random deletion (seeded).
Proxy (block 16, as the Phase 6 proxy): truth = target greedy argmax, truncated where the data leaves the greedy path;
chain15 = 1 + accepted length of DSpine's refined greedy chain (`select_cached`) over 15 drafts (the engine's chain
b16); also the backbone-argmax chain, best-first tree (N=48, depth <= 15) and top-16 candidate prefix coverage.
"""
import argparse, json, math, os, random, sys, time
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deletion_mask as dm

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--train-data", default="")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--anchors", type=int, default=64)
ap.add_argument("--block", type=int, default=16)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--gamma", type=float, default=7.0)
ap.add_argument("--init-sparsity", type=float, default=1 / 2048)
ap.add_argument("--prior-strength", type=float, default=3.0)
ap.add_argument("--sharpness", type=float, nargs=2, default=[100.0, 1000.0])
ap.add_argument("--weight-reg", type=float, default=1e-5)
ap.add_argument("--calib-seqs", type=int, default=64)
ap.add_argument("--eval", nargs="*", default=[], help="name=jsonl eval sets")
ap.add_argument("--masks", nargs="*", default=[], help="name=path.dmask or name=random:path.dmask")
ap.add_argument("--eval-seqs", type=int, default=64)
ap.add_argument("--eval-anchors", type=int, default=128)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--target", default="Qwen/Qwen3-8B")
ap.add_argument("--official", default="tencent/Qwen3-8B-DSpine-b16")
a = ap.parse_args()

torch.manual_seed(a.seed); random.seed(a.seed)
dev = torch.device("cuda"); BF = torch.bfloat16
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "run.log"), "a")
def log(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
log("ARGS", json.dumps(vars(a)))

from huggingface_hub import snapshot_download
off_dir = snapshot_download(a.official)
off_cfg = json.load(open(off_dir + "/config.json"))

# ---------------- data (same tokenisation as train_dspine.py / train_treeaware.py) ----------------
tok = AutoTokenizer.from_pretrained(a.target)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
def load_samples(path):
    cache = path + f".tok{a.max_len}.pt"
    if os.path.exists(cache): return torch.load(cache)
    samples = []
    for l in open(path):
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
    return samples

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

target = AutoModelForCausalLM.from_pretrained(a.target, dtype=BF).to(dev).eval(); target.requires_grad_(False)
TL = off_cfg["dflash_config"]["target_layer_ids"]
sf_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dspine_sf_config.json")))
sf_cfg["block_size"] = a.block
sf_cfg["dspine_config"]["replacement_probability"] = 0.0   # warm-start style: curriculum off
sf_cfg["dspine_config"]["alignment_warmup_steps"] = 0         # trained drafter: full alignment weight from step 0
cfg = Qwen3Config.from_dict(sf_cfg); cfg._attn_implementation = "flex_attention"
with torch.device(dev):
    draft = DSpineDraftModel(cfg).to(BF)

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

off_sd = load_file(off_dir + "/model.safetensors")
sd = tencent_to_sf(off_sd)
missing, unexpected = draft.load_state_dict({k: v.to(BF) for k, v in sd.items() if k != "transfer_codes"}, strict=False)
log("loaded official; missing", [m for m in missing if m not in ("transfer_codes", "transfer_ready")], "unexpected", unexpected)
draft.transfer_codes.copy_(sd["transfer_codes"].to(draft.transfer_codes.dtype)); draft.transfer_ready.fill_(True)
draft.requires_grad_(False)
layers = dm.attach_masks(draft)                       # keys: layers.{i}.mlp.gate_proj.weight == Tencent checkpoint keys
for n, l in layers.items():
    assert n in off_sd and torch.equal(l.weight.cpu(), off_sd[n]), n
log("masked:", {n: list(l.weight.shape) for n, l in layers.items()})
base_w = {n: off_sd[n] for n in layers}
del off_sd, sd

wrapper = OnlineDSpineModel(draft_model=draft, target_lm_head=target.lm_head, target_embed_tokens=target.model.embed_tokens,
                            mask_token_id=151669, block_size=a.block, attention_backend="flex_attention", num_anchors=a.anchors,
                            loss_decay_gamma=a.gamma, objective_chunk_blocks=16, fused_logits_loss=True).to(dev)
wrapper.draft_model.to(BF)
for l in layers.values():   # .to(BF) above must not touch the fp32 logits
    l.logits.data = l.logits.data.float()

@torch.no_grad()
def target_feats(x, m):
    out = target.model(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return torch.cat([hs[k + 1] for k in TL], dim=-1), hs[-1]

def step_batch(x, m, lm, gstep, total):
    x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
    aux, last = target_feats(x, m)
    n_valid = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
    loss, acc, met = wrapper(input_ids=x, hidden_states=aux, loss_mask=lm, target_last_hidden_states=last,
                             max_valid_anchors=n_valid, global_step=gstep, total_steps=total)
    return loss, acc, met   # SpecForge b15ad38 already normalises by the position weights

# ---------------- proxy eval ----------------
EXPAND = 8; D = a.block - 1; CH = min(15, D)

@torch.no_grad()
def tree_accept(cand, logp, truth, tvalid, N, Dcap):
    """Batched best-first tree (copied from train_treeaware.py)."""
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
        row = logp[ar, dn, l]
        t = row.topk(EXPAND, -1)
        can = ok & (d + 1 < Dcap)
        fs[:, nfill:nfill + EXPAND] = torch.where(can[:, None], sc[:, None] + t.values, torch.full_like(t.values, -float("inf")))
        fd[:, nfill:nfill + EXPAND] = (d + 1)[:, None]; fl[:, nfill:nfill + EXPAND] = t.indices
        fp[:, nfill:nfill + EXPAND] = i
        nfill += EXPAND
    ntok = cand[ar[:, None], nd, nl]
    cur = torch.full((M,), -1, dtype=torch.long, device=dev); L = torch.zeros(M, dtype=torch.long, device=dev)
    alive = torch.ones(M, dtype=torch.bool, device=dev)
    for k in range(Dcap):
        match = nok & (npar == cur[:, None]) & (nd == k) & (ntok == truth[:, k:k + 1]) & tvalid[:, k:k + 1]
        f = match.any(-1) & alive
        cur = torch.where(f, match.float().argmax(-1), cur); L += f.long(); alive = f
    return L

def run_blocks(x, aux, anchors, keep):
    B, L = x.shape
    idx = anchors.unsqueeze(-1) + torch.arange(a.block, device=dev)
    labels = x.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, idx.clamp(max=L - 1))
    pos = torch.cat((torch.arange(L, device=dev).expand(B, -1), wrapper._create_position_ids(anchors)), dim=1)
    out = draft(position_ids=pos, attention_mask=wrapper._attention_mask(anchors, keep, L),
                noise_embedding=wrapper._create_noise_embed(x, anchors, keep), target_hidden=aux,
                block_token_ids=labels, replace_mask=None, kernel_options={"BACKEND": "TRITON"})
    return out, labels, idx

KEYS = ("chain15", "bb_chain15", "tree48", "cov15")

@torch.no_grad()
def proxy_eval(data, variants):
    """variants: {name: keeps dict or None (official)}. Returns {variant: metrics} plus per-sequence sums for CIs."""
    wrapper.eval()
    hb = make_batches(data[:a.eval_seqs], a.micro_bs, random.Random(0))
    tot = {v: {k: 0.0 for k in KEYS} for v in variants}
    perseq = {v: {} for v in variants}; nblk = 0; seqblk = {}; tvlen = 0.0
    for b in hb:
        x, m, lm = collate(data, b); x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
        aux, last = target_feats(x, m)
        B, L = x.shape
        g = torch.cat([target.lm_head(c).argmax(-1) for c in last.split(512, dim=1)], dim=1)
        torch.manual_seed(1234 + b[0])
        wrapper.num_anchors = a.eval_anchors
        nv = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
        anchors, keep = wrapper._sample_anchor_positions(L, lm, dev, nv)
        wrapper.num_anchors = a.anchors
        n = anchors.shape[1]
        kf = keep.reshape(-1)
        seq_id = torch.tensor(b, device=dev).unsqueeze(1).expand(-1, n).reshape(-1)[kf]
        first = True
        for vname, keeps in variants.items():
            if keeps is None: dm.set_mode(layers, "off")
            else: dm.set_mode(layers, "fixed", keeps)
            out, labels, idx = run_blocks(x, aux, anchors, keep)
            tidx = (idx[..., 1:] - 1).clamp(max=L - 1)
            truth = g.unsqueeze(1).expand(-1, n, -1).gather(2, tidx)
            inr = (idx[..., 1:] < L) & keep.unsqueeze(-1)
            sup = lm.unsqueeze(1).expand(-1, n, -1).gather(2, idx[..., 1:].clamp(max=L - 1)) > 0
            dataeq = labels[..., 1:] == truth
            prev_ok = torch.cat([torch.ones_like(dataeq[..., :1]), dataeq[..., :-1]], -1).int().cumprod(-1).bool()
            tvalid = (inr & sup & prev_ok).reshape(B * n, D)[kf]
            truth = truth.reshape(B * n, D)[kf]
            hs = out.hidden_states[..., 1:, :].reshape(B * n, D, -1)[kf]
            pre = out.pre_injection[..., 1:, :].reshape(B * n, D, -1)[kf]
            anc = labels[..., 0].reshape(-1)[kf]
            per_block = {k: [] for k in KEYS}
            for s in range(0, hs.shape[0], 64):
                h_, p_, t_, v_, a_ = hs[s:s + 64], pre[s:s + 64], truth[s:s + 64], tvalid[s:s + 64], anc[s:s + 64]
                lg = target.lm_head(h_)
                cand = lg.topk(16, -1).indices
                tab = draft.transition_scores(p_, cand, a_, target.lm_head.weight)
                ch = DSpineDraftModel.select_cached(tab, cand)
                run = ((ch == t_) & v_).int().cumprod(-1)
                per_block["chain15"].append(1 + run[:, :CH].sum(-1))
                okb = ((lg.argmax(-1) == t_) & v_)[:, :CH].int().cumprod(-1)
                per_block["bb_chain15"].append(1 + okb.sum(-1))
                per_block["tree48"].append(1 + tree_accept(cand, tab.float().log_softmax(-1), t_, v_, 48, D))
                per_block["cov15"].append(((cand == t_.unsqueeze(-1)).any(-1) & v_)[:, :CH].int().cumprod(-1).sum(-1))
                if first: tvlen += float(v_.int().cumprod(-1).sum())
            for k in KEYS:
                pb = torch.cat(per_block[k]).float()
                tot[vname][k] += float(pb.sum())
                sums = torch.zeros(int(seq_id.max()) + 1, device=dev).index_add_(0, seq_id, pb)
                for sidx in b:
                    perseq[vname].setdefault(k, {})[sidx] = perseq[vname].get(k, {}).get(sidx, 0.0) + float(sums[sidx])
            if first:
                nblk += int(kf.sum())
                for sidx in b: seqblk[sidx] = seqblk.get(sidx, 0) + int((seq_id == sidx).sum())
            first = False
    dm.set_mode(layers, "off")
    res = {v: {k: tot[v][k] / nblk for k in KEYS} for v in variants}
    return res, perseq, seqblk, {"n_blocks": nblk, "n_seqs": len(seqblk), "truth_valid_len": tvlen / nblk}

def paired_ci(perseq, v, ref="official", key="chain15", nboot=2000):
    """Bootstrap (over sequences) 95% CI of sum(v)/sum(ref) for one metric."""
    ids = sorted(perseq[ref][key]); g = torch.Generator().manual_seed(0)
    r = torch.tensor([perseq[ref][key][i] for i in ids]); x = torch.tensor([perseq[v][key][i] for i in ids])
    samp = torch.randint(0, len(ids), (nboot, len(ids)), generator=g)
    ratios = x[samp].sum(1) / r[samp].sum(1)
    return [float(ratios.quantile(0.025)), float(ratios.quantile(0.975))]

# ---------------- train a mask ----------------
def train_mask():
    data = load_samples(a.train_data)
    log(f"train samples={len(data)} mean_len={sum(len(s[0]) for s in data)/len(data):.0f}")
    rng = random.Random(a.seed)
    # Wanda calibration on the drafter's own gate_proj inputs, through the training forward (unmasked)
    dm.set_mode(layers, "off")
    col = dm.WandaCollector(layers)
    wrapper.eval()
    with torch.no_grad():
        for b in make_batches(data, a.micro_bs, random.Random(1))[: max(1, a.calib_seqs // a.micro_bs)]:
            step_batch(*collate(data, b), 0, a.steps)
    col.close()
    init_counts = dm.wanda_init(layers, col.norms, a.init_sparsity, a.prior_strength, seed=a.seed)
    log("wanda init deletions", json.dumps(init_counts))
    dm.save(os.path.join(a.out, "wanda_init.dmask"), dm.hard_masks(layers), base_w,
            meta={"kind": "wanda_init", "init_sparsity": a.init_sparsity, "prior_strength": a.prior_strength})
    params = [l.logits for l in layers.values()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    hist = []
    dm.set_mode(layers, "soft")
    wrapper.train()
    step, t0 = 0, time.time(); agg = {"loss": 0., "reg": 0., "acc": 0.}
    batches = []
    while step < a.steps:
        if len(batches) < a.accum: batches += make_batches(data, a.micro_bs, rng)
        prog = min(step / a.steps, 1.0)
        sharp = a.sharpness[0] + (a.sharpness[1] - a.sharpness[0]) * prog
        for l in layers.values(): l.sharpness = sharp
        wreg = a.weight_reg * (1.0 - prog)
        for _ in range(a.accum):
            bb = batches.pop()
            loss, acc, met = step_batch(*collate(data, bb), step, a.steps)
            reg = dm.reg_sum(layers)
            ((loss - wreg * reg) / a.accum).backward()
            agg["loss"] += float(loss) / a.accum; agg["reg"] += float(reg) / a.accum; agg["acc"] += float(acc) / a.accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            k = 1 if step == 1 else 10
            deleted = {n.split(".")[1]: int((l.logits <= 0).sum()) for n, l in layers.items()}
            rec = {"step": step, "loss": agg["loss"] / k, "reg": agg["reg"] / k, "acc": agg["acc"] / k, "gn": float(gn),
                   "sharpness": sharp, "deleted": deleted, "deleted_total": sum(deleted.values()),
                   "s_per_step": (time.time() - t0) / step}
            hist.append(rec); log("STEP", json.dumps(rec))
            agg = {"loss": 0., "reg": 0., "acc": 0.}
    keeps = dm.hard_masks(layers)
    meta = {"kind": "deletion_tuned", "train_data": os.path.basename(a.train_data), "steps": a.steps, "lr": a.lr,
            "seqs_per_step": a.micro_bs * a.accum, "anchors": a.anchors, "block": a.block, "sharpness": a.sharpness,
            "weight_reg": a.weight_reg, "init_sparsity": a.init_sparsity, "prior_strength": a.prior_strength,
            "base": a.official, "masked": "layers.*.mlp.gate_proj.weight"}
    nbytes = dm.save(os.path.join(a.out, "mask.dmask"), keeps, base_w, meta=meta)
    init_keep = dm.decode(open(os.path.join(a.out, "wanda_init.dmask"), "rb").read())[0]
    init_set = {n: set(v.tolist()) for n, v in init_keep.items()}
    fin = {n: set(dm.deleted_indices(k).tolist()) for n, k in keeps.items()}
    summ = {"deleted_total": sum(len(v) for v in fin.values()), "masked_params": sum(k.numel() for k in keeps.values()),
            "wanda_init_total": sum(len(v) for v in init_set.values()),
            "kept_from_init": sum(len(fin[n] & init_set[n]) for n in fin), "mask_bytes": nbytes,
            "train_s": time.time() - t0, "hist": hist}
    summ["frac_deleted"] = summ["deleted_total"] / summ["masked_params"]
    json.dump(summ, open(os.path.join(a.out, "train.json"), "w"), indent=1)
    log("SAVED mask", json.dumps({k: v for k, v in summ.items() if k != "hist"}))
    dm.set_mode(layers, "off")

# ---------------- main ----------------
if a.train_data:
    train_mask()
if a.eval:
    variants = {"official": None}
    for spec in a.masks:
        name, path = spec.split("=", 1)
        if path.startswith("random:"):
            deleted, _ = dm.load(path[len("random:"):])
            ref = {n: dm.keep_from_indices(i, layers[n].weight.shape) for n, i in deleted.items()}
            variants[name] = dm.random_matched(ref, seed=1234)
        else:
            deleted, _ = dm.load(path)
            variants[name] = {n: dm.keep_from_indices(i, layers[n].weight.shape) for n, i in deleted.items()}
        log("variant", name, "deleted", sum(int((~k).sum()) for k in variants[name].values()))
    results = {"args": vars(a), "domains": {}}
    for spec in a.eval:
        dname, path = spec.split("=", 1)
        data = load_samples(path)
        t = time.time()
        res, perseq, seqblk, info = proxy_eval(data, variants)
        info["eval_s"] = round(time.time() - t, 1)
        for v in variants:
            if v != "official":
                for k in ("chain15", "tree48"):
                    res[v][f"{k}_ratio"] = res[v][k] / res["official"][k]
                    res[v][f"{k}_ci95"] = paired_ci(perseq, v, key=k)
        results["domains"][dname] = {"info": info, "variants": res}
        log("PROXY", dname, json.dumps(results["domains"][dname]))
        json.dump(results, open(os.path.join(a.out, "proxy.json"), "w"), indent=1)
    log("EVAL DONE")
