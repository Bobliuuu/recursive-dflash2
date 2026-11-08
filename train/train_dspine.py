"""Colocated single-GPU DSpine trainer (Phase 3). Uses SpecForge's DSpine draft model + training objective (PR #1038 applied)
but replaces its Mooncake/SGLang-capture data plane with an in-process HF Qwen3-8B target forward.

Usage (box):  python train_dspine.py --data data/regen.jsonl --block 16 --out ckpt/b16 --epochs 4 ...
Checkpoints are written in the *Tencent/SGLang* DSpine format (selected_prefix_bridge.*) so SGLang's DSPINE plugin loads them.
"""
import argparse, json, math, os, random, time, glob, copy
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--block", type=int, default=16)
ap.add_argument("--epochs", type=float, default=4)
ap.add_argument("--max-steps", type=int, default=0, help="stop early (pilot); schedule still spans the full plan unless --plan-steps")
ap.add_argument("--plan-steps", type=int, default=0, help="override total optimizer steps for the LR/curriculum schedule")
ap.add_argument("--limit", type=int, default=0, help="use only first N samples")
ap.add_argument("--micro-bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--warmup-ratio", type=float, default=0.04)
ap.add_argument("--anchors", type=int, default=512)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--gamma", type=float, default=7.0)
ap.add_argument("--init-official", default="", help="HF id/path of an official DSpine ckpt to load (sanity eval / warm start)")
ap.add_argument("--eval-only", type=int, default=0, help="run N eval micro-batches with the loaded weights and exit")
ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--target", default="Qwen/Qwen3-8B")
ap.add_argument("--official-config", default="tencent/Qwen3-8B-DSpine-b16")
ap.add_argument("--holdout", type=int, default=256, help="last N samples held out for teacher-forced eval")
ap.add_argument("--log", default="")
a = ap.parse_args()

torch.manual_seed(a.seed); random.seed(a.seed)
dev = torch.device("cuda")
BF = torch.bfloat16
os.makedirs(a.out, exist_ok=True)
logf = open(a.log or os.path.join(a.out, "train.log"), "a")
def log(*x):
    s = " ".join(str(i) for i in x)
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

from huggingface_hub import snapshot_download
off_dir = snapshot_download(a.official_config)
off_cfg = json.load(open(off_dir + "/config.json"))

# ---------------- data ----------------
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
if a.limit: samples = samples[: a.limit]
hold = samples[-a.holdout:] if a.holdout else []
train = samples[: len(samples) - len(hold)]
log(f"samples train={len(train)} hold={len(hold)} mean_len={sum(len(s[0]) for s in train)/len(train):.0f} "
    f"mean_sup={sum(sum(s[1]) for s in train)/len(train):.0f}")

def make_batches(data, bs, rng):
    idx = list(range(len(data))); rng.shuffle(idx)
    chunks = [idx[i:i + bs * 64] for i in range(0, len(idx), bs * 64)]
    out = []
    for c in chunks:
        c.sort(key=lambda i: len(data[i][0]))
        out += [c[i:i + bs] for i in range(0, len(c) - bs + 1, bs)]
    rng.shuffle(out)
    return out

def collate(data, ids_list):
    L = max(len(data[i][0]) for i in ids_list)
    x = torch.zeros(len(ids_list), L, dtype=torch.long)
    m = torch.zeros(len(ids_list), L, dtype=torch.long)
    lm = torch.zeros(len(ids_list), L, dtype=torch.long)
    for j, i in enumerate(ids_list):
        ids, l = data[i]
        x[j, :len(ids)] = torch.tensor(ids); m[j, :len(ids)] = 1; lm[j, :len(ids)] = torch.tensor(l)
    return x, m, lm

# ---------------- models ----------------
from specforge.modeling.draft.dspine import DSpineDraftModel
from specforge.algorithms.dspine.model import OnlineDSpineModel

target = AutoModelForCausalLM.from_pretrained(a.target, dtype=BF).to(dev).eval()
target.requires_grad_(False)
TL = off_cfg["dflash_config"]["target_layer_ids"]  # [1,9,17,25,33]; hidden_states index k+1 = output of layer k

sf_cfg = json.load(open(os.path.join(os.path.dirname(__file__), "dspine_sf_config.json")))
sf_cfg["block_size"] = a.block
cfg = Qwen3Config.from_dict(sf_cfg)
cfg._attn_implementation = "flex_attention"
with torch.device(dev):
    draft = DSpineDraftModel(cfg).to(BF)

# --- format conversion SpecForge <-> Tencent/SGLang ---
def sf_to_tencent(sd, codes_bf16):
    out = {}
    for k, v in sd.items():
        if k.startswith(("layers.", "fc.", "hidden_norm.", "norm.")): out[k] = v.to(BF)
    P = "selected_prefix_bridge."
    for i in range(4): pass
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

def save_ckpt(path, step):
    os.makedirs(path, exist_ok=True)
    sd = {k: v.detach() for k, v in draft.state_dict().items()}
    save_file(sf_to_tencent(sd, draft.transfer_codes.detach().to(BF)), path + "/model.safetensors")
    c = copy.deepcopy(off_cfg); c["block_size"] = a.block
    json.dump(c, open(path + "/config.json", "w"), indent=1)
    json.dump({"step": step}, open(path + "/step.json", "w"))
    # SpecForge-native resumable copy
    torch.save({"draft": draft.state_dict(), "step": step}, path + "/sf_state.pt")

if a.init_official:
    d = snapshot_download(a.init_official)
    sd = tencent_to_sf(load_file(d + "/model.safetensors"))
    missing, unexpected = draft.load_state_dict({k: v.to(BF) for k, v in sd.items() if k != "transfer_codes"}, strict=False)
    log("loaded official; missing", [m for m in missing if m not in ("transfer_codes", "transfer_ready")], "unexpected", unexpected)
    draft.transfer_codes.copy_(sd["transfer_codes"].to(draft.transfer_codes.dtype))
    draft.transfer_ready.fill_(True)

wrapper = OnlineDSpineModel(
    draft_model=draft, target_lm_head=target.lm_head, target_embed_tokens=target.model.embed_tokens,
    mask_token_id=151669, block_size=a.block, attention_backend="flex_attention", num_anchors=a.anchors,
    loss_decay_gamma=a.gamma, objective_chunk_blocks=16, fused_logits_loss=True, warmup_ratio=a.warmup_ratio,
).to(dev)  # initialises transfer space from lm_head if not loaded
wrapper.draft_model.to(BF)
draft.transfer_codes = draft.transfer_codes.to(torch.float32) if False else draft.transfer_codes
log("draft params (M):", sum(p.numel() for p in draft.parameters() if p.requires_grad) / 1e6,
    "transfer dtype", draft.transfer_codes.dtype)

@torch.no_grad()
def target_feats(x, m):
    out = target.model(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    aux = torch.cat([hs[k + 1] for k in TL], dim=-1)
    return aux, hs[-1]

def step_batch(x, m, lm, gstep, total, alignment_ce=None):
    x, m, lm = x.to(dev), m.to(dev), lm.to(dev)
    aux, last = target_feats(x, m)
    n_valid = int(((lm[:, :-1] > 0) & (lm[:, 1:] > 0)).sum(1).max())
    loss, acc, met = wrapper(input_ids=x, hidden_states=aux, loss_mask=lm, target_last_hidden_states=last,
                             max_valid_anchors=n_valid, global_step=gstep, total_steps=total, alignment_ce=alignment_ce)
    return loss, acc, met

rng = random.Random(a.seed)

def evaluate(n_batches):
    wrapper.eval()
    hb = make_batches(hold, a.micro_bs, random.Random(0))[:n_batches]
    tot = {"acc": [0., 0.], "ce": [0., 0.], "l1": [0., 0.], "cov": [0., 0.], "ref": [0., 0.]}
    with torch.no_grad():
        for b in hb:
            x, m, lm = collate(hold, b)
            loss, acc, met = step_batch(x, m, lm, 10**9, 10**9 + 1)
            r = met["ratio_metrics"]
            for k, name in (("acc", "acc"), ("ce", "dspine/backbone_ce"), ("l1", "dspine/backbone_l1"),
                            ("cov", "dspine/candidate_coverage"), ("ref", "dspine/refinement_loss")):
                tot[k][0] += float(r[name][0]); tot[k][1] += float(r[name][1])
    wrapper.train()
    return {k: v[0] / max(v[1], 1e-9) for k, v in tot.items()}

if a.eval_only:
    log("EVAL", json.dumps(evaluate(a.eval_only)))
    raise SystemExit

# ---------------- optimiser (bf16 model, fp32 master) ----------------
params = [p for p in draft.parameters() if p.requires_grad]
master = [p.detach().clone().float().requires_grad_(True) for p in params]
opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
seqs_per_step = a.micro_bs * a.accum
steps_per_epoch = len(train) // seqs_per_step
total = a.plan_steps or int(a.epochs * steps_per_epoch)
warm = max(1, int(a.warmup_ratio * total))
def lr_at(s):
    if s < warm: return a.lr * (s + 1) / warm
    return a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
log(f"plan: total_steps={total} steps/epoch={steps_per_epoch} seqs/step={seqs_per_step} warmup={warm}")

wrapper.train()
step, t0, tok_seen = 0, time.time(), 0
agg = {"loss": 0., "acc": 0., "n": 0}
done = False
ep = 0
while not done:
    batches = make_batches(train, a.micro_bs, rng)
    for bi in range(0, len(batches) - a.accum + 1, a.accum):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        for b in batches[bi:bi + a.accum]:
            x, m, lm = collate(train, b)
            loss, acc, met = step_batch(x, m, lm, step, total)
            (loss / a.accum).backward()
            agg["loss"] += float(loss) / a.accum; agg["acc"] += float(acc) / a.accum
            tok_seen += int(m.sum())
        for mp, p in zip(master, params):
            mp.grad = p.grad.float() if p.grad is not None else torch.zeros_like(mp)
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        for mp, p in zip(master, params): p.data.copy_(mp.data); p.grad = None
        step += 1; agg["n"] += 1
        if step % 10 == 0:
            el = time.time() - t0
            log(f"step {step}/{total} ep {ep} loss {agg['loss']/agg['n']:.4f} acc {agg['acc']/agg['n']:.4f} gn {float(gn):.2f} "
                f"lr {lr_at(step):.2e} {el/step:.2f}s/step tok/s {tok_seen/el:.0f} eta {(total-step)*el/step/3600:.2f}h")
            agg = {"loss": 0., "acc": 0., "n": 0}
        if step % 100 == 0:
            log("EVAL", step, json.dumps(evaluate(32)))
        if step % a.save_every == 0: save_ckpt(a.out + "/last", step)
        if step >= total or (a.max_steps and step >= a.max_steps): done = True; break
    ep += 1
log("FINAL EVAL", json.dumps(evaluate(128)))
save_ckpt(a.out + "/final", step)
log("saved final", step)
