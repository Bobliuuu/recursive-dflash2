"""CPU tests for train/deletion_mask.py with a tiny random Qwen3 model (run: python -m pytest train/test_deletion_mask.py)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deletion_mask as dm  # noqa: E402

BF = torch.bfloat16


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).to(BF).eval()


def trained_like_masks(model, seed=1, frac=0.02):
    """Attach masks and give them random logits (a stand-in for a trained mask)."""
    layers = dm.attach_masks(model)
    g = torch.Generator().manual_seed(seed)
    for l in layers.values():
        with torch.no_grad():
            r = torch.rand(l.logits.numel(), generator=g)
            l.logits.copy_(torch.where(r < frac, -0.5, 0.5))
    return layers


def test_attach_targets_only_gate_proj():
    m = tiny_model()
    layers = dm.attach_masks(m)
    assert sorted(layers) == [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(3)]
    sd = m.state_dict()
    for n in layers:
        assert n in sd  # state-dict name of the masked weight is unchanged
    assert not any(isinstance(x, dm.MaskedLinear) for n, x in m.named_modules() if not n.endswith("gate_proj"))


def test_off_mode_matches_unmasked_model():
    m = tiny_model(); ref = tiny_model()
    dm.attach_masks(m)
    x = torch.randint(0, 97, (2, 9))
    with torch.no_grad():
        assert torch.equal(m(x).logits, ref(x).logits)


def test_varint_roundtrip_edge_values():
    vals = [0, 1, 127, 128, 255, 16383, 16384, 2**31, 2**40 + 7]
    assert dm._varint_decode(dm._varint_encode(vals), len(vals)) == vals


def test_mask_file_roundtrip(tmp_path):
    m = tiny_model()
    layers = trained_like_masks(m)
    keeps = dm.hard_masks(layers)
    base = {n: l.weight.detach() for n, l in layers.items()}
    p = str(tmp_path / "m.dmask")
    nbytes = dm.save(p, keeps, base, meta={"note": "t"})
    deleted, header = dm.load(p)
    assert header["meta"] == {"note": "t"}
    assert set(deleted) == set(keeps)
    for n, k in keeps.items():
        assert torch.equal(dm.keep_from_indices(deleted[n], k.shape), k)
        assert torch.equal(deleted[n], dm.deleted_indices(k))
        assert bool((deleted[n][1:] > deleted[n][:-1]).all())  # sorted, unique
    assert nbytes == os.path.getsize(p)


def test_mask_file_is_compact(tmp_path):
    # 0.05% of a 12288x4096 matrix: delta-varint is ~2 bytes per deleted weight, far below one bit per weight
    g = torch.Generator().manual_seed(0)
    shape = (12288, 4096); n = shape[0] * shape[1]; cnt = n // 2048
    keep = dm.keep_from_indices(torch.randperm(n, generator=g)[:cnt], shape)
    p = str(tmp_path / "big.dmask")
    nbytes = dm.save(p, {"layers.0.mlp.gate_proj.weight": keep})
    assert nbytes < 3 * cnt and nbytes < n / 8 / 100
    deleted, _ = dm.load(p)
    assert torch.equal(deleted["layers.0.mlp.gate_proj.weight"], dm.deleted_indices(keep))


def test_empty_and_full_masks_roundtrip(tmp_path):
    keeps = {"a.weight": torch.ones(4, 5, dtype=torch.bool), "b.weight": torch.zeros(3, 3, dtype=torch.bool)}
    p = str(tmp_path / "e.dmask")
    dm.save(p, keeps)
    deleted, _ = dm.load(p)
    assert deleted["a.weight"].numel() == 0 and deleted["b.weight"].tolist() == list(range(9))


def test_apply_zeroes_only_deleted_and_keeps_rest_bit_identical(tmp_path):
    m = tiny_model()
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    layers = trained_like_masks(m)
    keeps = dm.hard_masks(layers)
    p = str(tmp_path / "m.dmask")
    dm.save(p, keeps, base={n: sd0[n] for n in keeps})
    deleted, header = dm.load(p)
    sd1 = dm.apply_to_state_dict(sd0, deleted, header)
    n_zeroed = 0
    for k, v in sd0.items():
        if k in keeps:
            keep = keeps[k]
            assert torch.equal(sd1[k][keep].view(torch.int16), v[keep].view(torch.int16))  # kept: bit-identical
            assert bool((sd1[k][~keep].view(torch.int16) == 0).all())  # deleted: exactly +0.0
            n_zeroed += int((~keep).sum())
        else:
            assert sd1[k] is v  # untouched tensors are passed through as-is
    assert n_zeroed == sum(int(i.numel()) for i in deleted.values()) > 0
    # the original state dict is untouched
    for k in keeps:
        assert torch.equal(sd0[k], m.state_dict()[k])


def test_apply_rejects_wrong_base(tmp_path):
    m = tiny_model()
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    layers = trained_like_masks(m)
    keeps = dm.hard_masks(layers)
    p = str(tmp_path / "m.dmask")
    dm.save(p, keeps, base={n: sd0[n] for n in keeps})
    deleted, header = dm.load(p)
    bad = dict(sd0); n0 = next(iter(keeps)); bad[n0] = sd0[n0] + 1
    with pytest.raises(AssertionError):
        dm.apply_to_state_dict(bad, deleted, header)


def test_hard_mode_equals_applied_checkpoint():
    """The trainer's hard-mask forward is the same model as the deployed (zeroed) checkpoint."""
    m = tiny_model()
    sd0 = {k: v.clone() for k, v in m.state_dict().items()}
    layers = trained_like_masks(m)
    keeps = dm.hard_masks(layers)
    dm.set_mode(layers, "hard")
    x = torch.randint(0, 97, (2, 11))
    with torch.no_grad():
        y_hard = m(x).logits
    deployed = tiny_model()
    deleted = {n: dm.deleted_indices(k) for n, k in keeps.items()}
    deployed.load_state_dict(dm.apply_to_state_dict(sd0, deleted))
    with torch.no_grad():
        assert torch.equal(deployed(x).logits, y_hard)
    dm.set_mode(layers, "fixed", keeps)
    with torch.no_grad():
        assert torch.equal(m(x).logits, y_hard)


def test_wanda_init_and_soft_training_step():
    m = tiny_model()
    for p in m.parameters():
        p.requires_grad_(False)
    layers = dm.attach_masks(m)
    col = dm.WandaCollector(layers)
    x = torch.randint(0, 97, (4, 16))
    with torch.no_grad():
        m(x)
    col.close()
    assert set(col.norms) == set(layers)
    counts = dm.wanda_init(layers, col.norms, init_sparsity=0.05, prior_strength=3.0)
    for n, l in layers.items():
        g = l.logits.detach()
        score = (l.weight.float().abs() * col.norms[n].unsqueeze(0)).view(-1)
        prior = score < score.kthvalue(int(score.numel() * 0.05)).values   # lowest-Wanda 5%
        # logits ~ N(-0.03, 0.01^2) on the prior deletions and N(+0.03, 0.01^2) elsewhere
        assert abs(float(g[prior].mean()) + 0.03) < 0.005 and abs(float(g[~prior].mean()) - 0.03) < 0.005
        # the hard mask at init is the prior up to the ~0.1% 3-sigma tail of the N(0, 0.01^2) noise
        assert float(((g <= 0) != prior).float().mean()) < 0.01
        assert abs(counts[n] - int(prior.sum())) <= 0.01 * score.numel()
    dm.set_mode(layers, "soft")
    logits0 = {n: l.logits.detach().clone() for n, l in layers.items()}
    opt = torch.optim.AdamW([l.logits for l in layers.values()], lr=1e-2, weight_decay=0.0)
    out = m(x, labels=x)
    loss = out.loss - 1e-5 * dm.reg_sum(layers)
    loss.backward()
    for l in layers.values():
        assert l.logits.grad is not None and float(l.logits.grad.abs().sum()) > 0
    assert all(p.grad is None for n, p in m.named_parameters() if not n.endswith("logits"))
    opt.step()
    assert any(not torch.equal(l.logits, logits0[n]) for n, l in layers.items())


def test_random_matched_control_counts():
    m = tiny_model()
    keeps = dm.hard_masks(trained_like_masks(m, frac=0.1))
    r = dm.random_matched(keeps, seed=3)
    for n in keeps:
        assert int((~r[n]).sum()) == int((~keeps[n]).sum())
        assert not torch.equal(r[n], keeps[n])


def test_apply_dir_cli(tmp_path):
    from safetensors.torch import load_file, save_file
    m = tiny_model()
    base = tmp_path / "base"; base.mkdir()
    sd0 = {k: v.clone().contiguous() for k, v in m.state_dict().items() if k != "lm_head.weight"}
    save_file(sd0, str(base / "model.safetensors"), metadata={"format": "pt"})
    (base / "config.json").write_text('{"block_size": 16}')
    layers = trained_like_masks(m)
    keeps = dm.hard_masks(layers)
    mp = str(tmp_path / "m.dmask")
    dm.save(mp, keeps, base={n: sd0[n] for n in keeps})
    info = dm.apply_dir(str(base), mp, str(tmp_path / "out"))
    assert info["deleted"] == sum(int((~k).sum()) for k in keeps.values())
    assert (tmp_path / "out" / "config.json").read_text() == '{"block_size": 16}'
    sd1 = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert set(sd1) == set(sd0)
    for k, v in sd0.items():
        if k in keeps:
            assert torch.equal(sd1[k], torch.where(keeps[k], v, torch.zeros_like(v)))
        else:
            assert torch.equal(sd1[k].view(torch.int16), v.view(torch.int16))
