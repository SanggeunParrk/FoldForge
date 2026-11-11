"""``foldforge.train``: LR schedule, clipping, EMA, parameter groups, the batch-source plug-in and the checkpoint round trip (CPU, tiny model)."""

import math

import pytest
import torch
from torch import nn

from foldforge.models.checkpoints import miniworld
from foldforge.train import checkpoint, data, optim
from foldforge.train.config import TrainConfig


class Tiny(nn.Module):
    """Parameter names follow the graph's prefixes: trunk, diffusion head, confidence head."""

    def __init__(self) -> None:
        super().__init__()
        self.evoformer = nn.Linear(3, 3)
        self.diffusion_head = nn.Linear(3, 3)
        self.confidence_head = nn.Linear(3, 3)


def test_lr_factor_warms_up_linearly_then_decays_in_steps():
    cfg = TrainConfig(warmup_steps=10, decay_steps=100, decay_factor=0.5)
    assert optim.lr_factor(0, cfg) == 0.0
    assert optim.lr_factor(5, cfg) == pytest.approx(0.5)
    assert optim.lr_factor(10, cfg) == 1.0
    assert optim.lr_factor(109, cfg) == 1.0
    assert optim.lr_factor(110, cfg) == pytest.approx(0.5)
    assert optim.lr_factor(310, cfg) == pytest.approx(0.125)


def test_set_lr_floors_at_min_lr_after_warmup_and_scales_the_trunk_group():
    cfg = TrainConfig(
        max_lr=1e-2,
        min_lr=1e-3,
        warmup_steps=4,
        decay_steps=1,
        decay_factor=0.1,
        trunk_lr_scale=0.1,
    )
    model = Tiny()
    opt = optim.build_optimizer(model, cfg)
    names = {g["name"] for g in opt.param_groups}
    assert names == {"heads", "trunk"}
    assert optim.set_lr(opt, 2, cfg) == pytest.approx(5e-3), "warmup: no floor"
    assert optim.set_lr(opt, 4, cfg) == pytest.approx(1e-2)
    assert optim.set_lr(opt, 40, cfg) == pytest.approx(1e-3), (
        "after warmup the schedule never goes below min_lr"
    )
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs["heads"] == pytest.approx(1e-3)
    assert lrs["trunk"] == pytest.approx(1e-4)


def test_every_parameter_is_in_the_optimizer_and_the_confidence_head_keeps_the_head_lr():
    model = Tiny()
    trunk, heads = optim.split_parameters(model)
    assert {n.split(".")[0] for n, _ in trunk} == {"evoformer"}
    assert {n.split(".")[0] for n, _ in heads} == {"diffusion_head", "confidence_head"}
    opt = optim.build_optimizer(model, TrainConfig(trunk_lr_scale=0.1))
    assert [g["name"] for g in opt.param_groups] == ["heads", "trunk"]
    in_optimizer = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in in_optimizer for p in model.parameters()), "single stage: nothing is left out"
    scale = {g["name"]: g["scale"] for g in opt.param_groups}
    assert scale == {"heads": 1.0, "trunk": 0.1}


def test_clip_grad_norm_returns_the_preclip_norm_and_clips_globally():
    a, b = torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])
    total = optim.clip_grad_norm([a, b], 1.0)
    assert total.item() == pytest.approx(5.0)
    assert math.isclose(
        torch.linalg.vector_norm(torch.cat([a.grad, b.grad])).item(), 1.0, rel_tol=1e-4
    )
    small = torch.zeros(1, requires_grad=True)
    small.grad = torch.tensor([0.1])
    optim.clip_grad_norm([small], 1.0)
    assert small.grad.item() == pytest.approx(0.1), "below the bound nothing changes"
    assert optim.clip_grad_norm([torch.zeros(1)], 1.0).item() == 0.0, (
        "no gradient, no norm"
    )


def test_ema_lerps_and_swap_restores_the_live_weights():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    ema = optim.EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    assert ema.shadow["weight"].mean().item() == pytest.approx(1.1)
    ema.swap_in(model)
    assert model.weight.mean().item() == pytest.approx(1.1)
    ema.swap_out(model)
    assert model.weight.mean().item() == pytest.approx(2.0)


def test_load_source_resolves_module_callable_and_rejects_anything_else():
    assert data.load_source("collections:OrderedDict").__name__ == "OrderedDict"
    for bad in ("collections", ":x", "collections:"):
        with pytest.raises(ValueError, match="module:callable"):
            data.load_source(bad)
    with pytest.raises(AttributeError):
        data.load_source("collections:NoSuchName")


def test_train_batch_defaults_mean_no_mask_no_bonds():
    batch = data.TrainBatch(features={}, atom_positions=torch.zeros(2, 3, 3))
    assert batch.atom_mask is None
    assert batch.bond_pairs is None
    assert batch.name == ""


def test_checkpoint_round_trip_restores_model_optimizer_ema_and_step(tmp_path):
    cfg = TrainConfig()
    model = Tiny()
    opt = optim.build_optimizer(model, cfg)
    ema = optim.EMA(model, decay=0.5)
    loss = (
        model.evoformer(torch.ones(1, 3)).sum()
        + model.diffusion_head(torch.ones(1, 3)).sum()
    )
    loss.backward()
    opt.step()
    ema.update(model)
    path = tmp_path / "run.miniworld.pt"
    checkpoint.save(
        model, path, family="protenix1", source="test", ema=ema, optimizer=opt, step=7
    )
    assert miniworld.read(path)["state_dict"].keys() == model.state_dict().keys()
    assert (tmp_path / "run.train.pt").exists()

    other = Tiny()
    other_opt = optim.build_optimizer(other, cfg)
    other_ema = optim.EMA(other, decay=0.5)
    step = checkpoint.resume(other, path, optimizer=other_opt, ema=other_ema)
    assert step == 7
    for key, value in model.state_dict().items():
        assert torch.equal(other.state_dict()[key], value)
    assert all(torch.equal(other_ema.shadow[k], ema.shadow[k]) for k in ema.shadow)
    assert len(other_opt.state_dict()["state"]) == len(opt.state_dict()["state"])


def test_checkpoint_can_save_the_ema_as_the_model_without_touching_the_live_weights(
    tmp_path,
):
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = optim.EMA(model, decay=0.0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.shadow["weight"].fill_(2.0)
    path = checkpoint.save(
        model,
        tmp_path / "ema.miniworld.pt",
        family="protenix1",
        source="test",
        ema=ema,
        ema_weights=True,
    )
    assert miniworld.read(path)["state_dict"]["weight"].mean().item() == pytest.approx(
        2.0
    )
    assert model.weight.mean().item() == pytest.approx(3.0)


def test_checkpoint_refuses_a_path_without_the_native_suffix(tmp_path):
    with pytest.raises(ValueError, match="miniworld.pt"):
        checkpoint.save(
            nn.Linear(1, 1), tmp_path / "bad.pt", family="protenix1", source="test"
        )


def test_resume_without_training_state_starts_at_step_zero(tmp_path):
    model = nn.Linear(2, 2)
    path = checkpoint.save(
        model, tmp_path / "w.miniworld.pt", family="protenix1", source="test"
    )
    assert checkpoint.resume(nn.Linear(2, 2), path) == 0
