"""Deletion tuning (bit-mask tuning) for the DSpine drafter's MLP gate_proj weights (Phase 9).

Method (Lu et al., "The Surprising Effectiveness of Deleting Weights in LLM Reasoning and Adaptation", arXiv 2611.10866,
code github.com/agentic-learning-ai-lab/deleting-weights, read for this phase):
- every target weight W keeps its value and is frozen; it gets one fp32 logit G ~ N(0, 0.01^2);
- Wanda prior: score |W_ij| * a_j (a_j = per-batch L2 norm of input column j, summed over calibration batches); the
  lowest `init_sparsity` fraction per matrix gets G -= prior_strength * std(G), the rest G += prior_strength * std(G);
- training forward uses W * sigmoid(sharpness * G), sharpness linear from 100 to 1000 over the optimizer steps;
- loss = task loss - weight_reg * (1 - step / steps) * sum ||W * mask||^2 (the regulariser's weight decays to 0);
- deployment keeps exactly the weights with G > 0 and sets the others to 0. Nothing else changes.

This module is torch-only (no SpecForge, no CUDA) so it can be unit-tested on CPU with a tiny model.

Mask file (`.dmask`): the deleted weights of every masked matrix as sorted flat indices, delta-coded (first delta = first
index) as unsigned LEB128 varints, the concatenated streams zlib-compressed. Layout:
    b"DMASK1\\n" | u32 little-endian header length | JSON header | zlib(varint streams)
Header: {"format": "delta-varint-zlib", "tensors": [{"name", "shape", "count", "nbytes", "base_sha256"}], "meta": {...}}.
`base_sha256` (optional) is the SHA-256 of the base tensor's raw bytes; `apply` refuses a base that does not match.

CLI:
    python deletion_mask.py apply --base tencent/Qwen3-8B-DSpine-b16 --mask m.dmask --out /workspace/dspine_b16_masked
    python deletion_mask.py info m.dmask
`apply` writes model.safetensors with the deleted weights zeroed (every other tensor and every kept weight
bit-identical) and copies the remaining files (config.json, ...), so SGLang serves the directory like the official one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"DMASK1\n"
TARGET = "mlp.gate_proj"


# ---------------------------------------------------------------- masked layer
class MaskedLinear(nn.Module):
    """Frozen bias-free linear layer whose weight is multiplied by a learnable deletion mask.

    mode "soft": W * sigmoid(sharpness * logits) (training); "hard": W with logits <= 0 zeroed;
    "fixed": W with `fixed_keep == False` zeroed (evaluating an arbitrary mask); "off": W unchanged.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None, "gate_proj has no bias"
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        self.logits = nn.Parameter(torch.zeros(linear.weight.numel(), dtype=torch.float32, device=linear.weight.device))
        self.mode = "off"
        self.sharpness = 100.0
        self.fixed_keep = None
        self.reg_term = None     # ||W * soft_mask||^2 of the last soft forward with grad

    def keep(self) -> torch.Tensor:
        return (self.logits.detach() > 0).view(self.weight.shape)

    def effective_weight(self) -> torch.Tensor:
        w = self.weight
        if self.mode == "off":
            return w
        if self.mode == "hard":
            return torch.where(self.keep(), w, torch.zeros_like(w))
        if self.mode == "fixed":
            return torch.where(self.fixed_keep.view(w.shape), w, torch.zeros_like(w))
        assert self.mode == "soft", self.mode
        soft = torch.sigmoid(self.logits * self.sharpness).view(w.shape)
        wm = w.detach().float() * soft
        if torch.is_grad_enabled():
            self.reg_term = wm.pow(2).sum()
        return wm.to(w.dtype)

    def forward(self, x):
        return F.linear(x, self.effective_weight())


def attach_masks(model: nn.Module, target: str = TARGET) -> dict[str, MaskedLinear]:
    """Replace every nn.Linear whose qualified name ends with `target` by a MaskedLinear (in place).
    Returns {"<qualified name>.weight": layer}, i.e. keyed by the state-dict name of the masked weight."""
    found = {}
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and full.endswith(target):
                ml = MaskedLinear(child)
                setattr(mod, cname, ml)
                found[full + ".weight"] = ml
    assert found, f"no nn.Linear named *{target} found"
    return found


def set_mode(layers: dict[str, MaskedLinear], mode: str, keeps: dict[str, torch.Tensor] | None = None) -> None:
    for name, l in layers.items():
        l.mode = mode
        l.fixed_keep = keeps[name].to(l.weight.device) if mode == "fixed" else None


def reg_sum(layers: dict[str, MaskedLinear]) -> torch.Tensor:
    terms = [l.reg_term for l in layers.values() if l.reg_term is not None]
    return torch.stack(terms).sum() if terms else torch.zeros(())


# ---------------------------------------------------------------- Wanda prior
class WandaCollector:
    """Forward hooks that sum, per batch, the L2 norm of each input column of the masked layers (as the reference)."""

    def __init__(self, layers: dict[str, MaskedLinear]):
        self.norms = {}
        self.hooks = [l.register_forward_hook(self._hook(n)) for n, l in layers.items()]

    def _hook(self, name):
        def h(mod, inp, out):
            x = inp[0].detach()
            n = x.float().pow(2).sum(dim=tuple(range(x.dim() - 1))).sqrt()
            self.norms[name] = self.norms[name] + n if name in self.norms else n
        return h

    def close(self):
        for h in self.hooks: h.remove()


def wanda_init(layers: dict[str, MaskedLinear], norms: dict[str, torch.Tensor], init_sparsity: float,
               prior_strength: float, seed: int = 0) -> dict[str, int]:
    """Initialise logits ~ N(0, 0.01^2), then shift by +-prior_strength*std: the lowest-Wanda `init_sparsity` fraction of
    each matrix starts deleted. Returns the per-layer initial deletion count."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    counts = {}
    for name, l in layers.items():
        with torch.no_grad():
            init = torch.randn(l.logits.numel(), generator=g) * 0.01
            l.logits.copy_(init.to(l.logits.device))
            score = (l.weight.float().abs() * norms[name].to(l.weight.device).unsqueeze(0)).view(-1)
            k = max(1, int(score.numel() * init_sparsity))
            thr = score.kthvalue(k).values
            dele = score < thr
            sign = torch.where(dele, -1.0, 1.0)
            l.logits.add_(sign * l.logits.std() * prior_strength)
        counts[name] = int((l.logits <= 0).sum())
    return counts


# ---------------------------------------------------------------- masks <-> indices
def hard_masks(layers: dict[str, MaskedLinear]) -> dict[str, torch.Tensor]:
    """{name: bool keep mask (cpu)} from the current logits (keep iff logit > 0)."""
    return {n: l.keep().cpu() for n, l in layers.items()}


def deleted_indices(keep: torch.Tensor) -> torch.Tensor:
    return (~keep.reshape(-1)).nonzero().squeeze(-1).to(torch.int64)


def keep_from_indices(idx: torch.Tensor, shape) -> torch.Tensor:
    keep = torch.ones(int(torch.tensor(shape).prod()), dtype=torch.bool)
    keep[idx] = False
    return keep.view(*shape)


def random_matched(keeps: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Control: per matrix, delete the same number of weights uniformly at random."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, k in keeps.items():
        cnt = int((~k).sum())
        idx = torch.randperm(k.numel(), generator=g)[:cnt]
        out[n] = keep_from_indices(idx, k.shape)
    return out


# ---------------------------------------------------------------- varint codec
def _varint_encode(vals) -> bytes:
    out = bytearray()
    for v in vals:
        v = int(v)
        assert v >= 0
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def _varint_decode(buf: bytes, count: int) -> list[int]:
    vals, v, shift, i = [], 0, 0, 0
    while len(vals) < count:
        b = buf[i]; i += 1
        v |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            vals.append(v); v = 0; shift = 0
    assert i == len(buf), "trailing bytes in varint stream"
    return vals


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def encode(keeps: dict[str, torch.Tensor], base: dict[str, torch.Tensor] | None = None, meta: dict | None = None) -> bytes:
    tensors, streams = [], []
    for name in sorted(keeps):
        k = keeps[name]
        idx = deleted_indices(k)
        deltas = torch.diff(idx, prepend=torch.zeros(1, dtype=torch.int64)) if idx.numel() else idx
        s = _varint_encode(deltas.tolist())
        streams.append(s)
        tensors.append({"name": name, "shape": list(k.shape), "count": int(idx.numel()), "nbytes": len(s),
                        "base_sha256": tensor_sha256(base[name]) if base is not None else None})
    header = json.dumps({"format": "delta-varint-zlib", "tensors": tensors, "meta": meta or {}}).encode()
    return MAGIC + struct.pack("<I", len(header)) + header + zlib.compress(b"".join(streams), 9)


def decode(blob: bytes) -> tuple[dict[str, torch.Tensor], dict]:
    """Returns ({name: sorted int64 deleted flat indices}, header)."""
    assert blob[:len(MAGIC)] == MAGIC, "not a .dmask file"
    p = len(MAGIC)
    (hl,) = struct.unpack("<I", blob[p:p + 4]); p += 4
    header = json.loads(blob[p:p + hl]); p += hl
    assert header["format"] == "delta-varint-zlib"
    payload = zlib.decompress(blob[p:])
    out, off = {}, 0
    for t in header["tensors"]:
        s = payload[off:off + t["nbytes"]]; off += t["nbytes"]
        deltas = torch.tensor(_varint_decode(s, t["count"]), dtype=torch.int64)
        idx = torch.cumsum(deltas, 0)
        n = int(torch.tensor(t["shape"]).prod())
        assert idx.numel() == 0 or (int(idx[-1]) < n and bool((deltas[1:] > 0).all())), f"bad indices for {t['name']}"
        out[t["name"]] = idx
    assert off == len(payload)
    return out, header


def save(path: str, keeps, base=None, meta=None) -> int:
    blob = encode(keeps, base, meta)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def load(path: str):
    with open(path, "rb") as f:
        return decode(f.read())


def apply_to_state_dict(sd: dict[str, torch.Tensor], deleted: dict[str, torch.Tensor], header: dict | None = None,
                        verify: bool = True) -> dict[str, torch.Tensor]:
    """Return a new state dict: masked tensors are copies with the deleted flat indices set to exactly 0; every other
    tensor is the same object. Verifies the base SHA-256 recorded in the header when present."""
    shas = {t["name"]: t.get("base_sha256") for t in (header or {}).get("tensors", [])}
    shapes = {t["name"]: tuple(t["shape"]) for t in (header or {}).get("tensors", [])}
    out = dict(sd)
    for name, idx in deleted.items():
        assert name in sd, f"mask names {name}, not in the base checkpoint"
        w = sd[name]
        if name in shapes:
            assert tuple(w.shape) == shapes[name], f"{name}: shape {tuple(w.shape)} vs mask {shapes[name]}"
        if verify and shas.get(name):
            assert tensor_sha256(w) == shas[name], f"{name}: base weights differ from the ones the mask was trained on"
        nw = w.detach().clone().contiguous()
        nw.view(-1)[idx] = 0
        out[name] = nw
    return out


# ---------------------------------------------------------------- CLI
def _resolve(base: str) -> str:
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download
    return snapshot_download(base)


def apply_dir(base: str, mask_path: str, out: str) -> dict:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    src = _resolve(base)
    deleted, header = load(mask_path)
    st = os.path.join(src, "model.safetensors")
    with safe_open(st, "pt") as f:
        md = f.metadata()
    sd = load_file(st)
    new = apply_to_state_dict(sd, deleted, header)
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if fn != "model.safetensors" and os.path.isfile(p):
            shutil.copyfile(p, os.path.join(out, fn))
    save_file(new, os.path.join(out, "model.safetensors"), metadata=md)
    tot = sum(sd[n].numel() for n in deleted)
    cnt = sum(int(i.numel()) for i in deleted.values())
    return {"base": src, "out": out, "deleted": cnt, "masked_params": tot, "frac": cnt / max(tot, 1),
            "mask_bytes": os.path.getsize(mask_path)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("apply"); a1.add_argument("--base", required=True); a1.add_argument("--mask", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("info"); a2.add_argument("mask")
    a = ap.parse_args()
    if a.cmd == "apply":
        print(json.dumps(apply_dir(a.base, a.mask, a.out)))
    else:
        deleted, h = load(a.mask)
        for t in h["tensors"]:
            print(t["name"], t["shape"], "deleted", t["count"])
        print(json.dumps(h["meta"]))


if __name__ == "__main__":
    main()
