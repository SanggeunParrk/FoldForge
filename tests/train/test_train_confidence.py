"""Confidence-head training glue: targets, loss and the mini-rollout step (tiny fakes, CPU)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foldforge.data.constants import atom_types, residue_names
from foldforge.train.config import TrainConfig
from foldforge.train.loss import confidence as conf
from foldforge.train.step import confidence_step, mini_rollout
from foldforge.utils.atom_layout import GatherInfo

RES, ATOMS = 6, 5
CODES = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP
ALA = CODES.index(residue_names.ALA)


class FakeHead(nn.Module):
    """The pieces of ConfidenceHead the loss reads (bins) plus a forward whose logits depend on the trunk inputs and on its own weights."""

    def __init__(self) -> None:
        super().__init__()
        self.num_plddt_bins = 50
        self.register_buffer("distance_breaks", torch.linspace(0.0, 31.0, 63))
        self.register_buffer("pae_breaks", torch.linspace(0.0, 31.0, 63))
        self.plddt = nn.Linear(4, ATOMS * 50)
        self.pde = nn.Linear(4, 64)
        self.pae = nn.Linear(4, 64)
        self.seen_positions = None

    def forward(self, dense_atom_positions, embeddings, **_):
        self.seen_positions = dense_atom_positions
        single, pair = embeddings["single"], embeddings["pair"]
        return {
            "plddt_logits": self.plddt(single).reshape(RES, ATOMS, 50),
            "pde_logits": self.pde(pair),
            "pae_logits": self.pae(pair),
        }


def _gather(slot: int = 4) -> GatherInfo:
    """Pseudo-beta = slot ``slot`` (CB for an alanine) of every token."""
    idx = torch.arange(RES) * ATOMS + slot
    return GatherInfo(
        gather_idxs=idx, gather_mask=torch.ones(RES, dtype=torch.bool), input_shape=(RES, ATOMS)
    )


def _structure(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(RES, ATOMS, 3, generator=generator) * 4.0


def _rigid(x: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(7)))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return x @ q.T + torch.tensor([3.0, -2.0, 5.0])


def _targets(pred, gt, head=None, aatype=None, mask=None):
    head = head or FakeHead()
    return conf.confidence_targets(
        pred,
        gt,
        torch.ones(RES, ATOMS, dtype=torch.bool) if mask is None else mask,
        _gather(),
        torch.ones(RES),
        torch.full((RES,), ALA) if aatype is None else aatype,
        torch.zeros(RES, dtype=torch.bool),
        torch.zeros(RES, dtype=torch.bool),
        head,
    )


def test_frame_slots_follow_the_dense_tables():
    nucleotide = CODES.index(residue_names.A)
    aatype = torch.tensor([ALA, nucleotide, len(CODES) + 3, 999, -1])
    slots, ok = conf.frame_slots(aatype)
    assert ok.tolist() == [True, True, False, False, False]
    assert slots[0].tolist() == [0, 1, 2], "protein frame = N, CA, C"
    order = atom_types.DENSE_ATOM_ORDER[residue_names.A]
    assert slots[1].tolist() == [order["C1'"], order["C3'"], order["C4'"]]


def test_a_perfect_structure_has_top_lddt_and_zero_errors():
    gt = _structure()
    targets = _targets(gt.clone(), gt)
    plddt, valid = targets["plddt"]
    assert valid.all()
    assert (plddt == 49).all(), "lDDT 1.0 is the last of the 50 bins"
    assert (targets["pde"][0] == 0).all() and targets["pde"][1].all()
    pae, pae_mask = targets["pae"]
    assert (pae[pae_mask] == 0).all() and pae_mask.any(), "a random frame can be near colinear and drop out; the rest are exact"


def test_the_targets_are_invariant_to_a_rigid_motion_of_the_prediction():
    """Frame-aligned error and lDDT do not see a global rotation / translation (the PAE transform must be R^T d)."""
    gt = _structure()
    moved = _targets(_rigid(gt), gt)
    reference = _targets(gt.clone(), gt)
    for name in ("plddt", "pde", "pae"):
        bins, mask = moved[name]
        assert mask.any()
        # float noise (1e-6) of a rigid motion moves an exact zero error from bin 0 to bin 1 (the first break is 0): one bin of slack
        assert ((bins[mask] - reference[name][0][mask]).abs() <= 1).all(), name
        assert torch.equal(mask, reference[name][1]), name


def test_a_perturbed_structure_moves_every_target_off_the_top_bin():
    gt = _structure()
    noisy = gt + torch.randn(gt.shape, generator=torch.Generator().manual_seed(1)) * 2.0
    targets = _targets(noisy, gt)
    assert (targets["plddt"][0] < 49).any()
    assert (targets["pde"][0] > 0).any()
    pae, pae_mask = targets["pae"]
    assert (pae[pae_mask] > 0).any()


def test_atoms_missing_in_the_ground_truth_drop_out_of_the_masks():
    gt = _structure()
    mask = torch.ones(RES, ATOMS, dtype=torch.bool)
    mask[2] = False  # token 2 unresolved
    targets = _targets(gt.clone(), gt, mask=mask)
    assert not targets["plddt"][1][2].any()
    assert not targets["pde"][1][2].any() and not targets["pde"][1][:, 2].any()
    assert not targets["pae"][1][2].any() and not targets["pae"][1][:, 2].any()


def test_a_token_without_a_frame_has_no_pae_rows_but_keeps_pde():
    gt = _structure()
    aatype = torch.full((RES,), ALA)
    aatype[1] = len(CODES) + 1  # a ligand-like token
    targets = _targets(gt.clone(), gt, aatype=aatype)
    assert not targets["pae"][1][1].any()
    assert targets["pde"][1][1, 1]


def test_masked_ce_is_the_log_of_the_class_count_for_flat_logits_and_ignores_masked_positions():
    logits = torch.zeros(4, 3, 8)
    target = torch.randint(0, 8, (4, 3))
    mask = torch.zeros(4, 3, dtype=torch.bool)
    mask[0, 0] = True
    assert float(conf.masked_ce(logits, target, mask)) == pytest.approx(math.log(8))
    logits[1, 1, 0] = 50.0  # a masked position cannot change the loss
    assert float(conf.masked_ce(logits, target, mask)) == pytest.approx(math.log(8))


def test_confidence_loss_weights_the_three_terms():
    outputs = {
        "plddt_logits": torch.zeros(RES, ATOMS, 50),
        "pde_logits": torch.zeros(RES, RES, 64),
        "pae_logits": torch.zeros(RES, RES, 64),
    }
    targets = _targets(_structure(), _structure())
    total, parts = conf.confidence_loss(outputs, targets, plddt_loss=1.0, pde_loss=2.0, pae_loss=0.0)
    assert float(parts["plddt"]) == pytest.approx(math.log(50))
    assert float(parts["pde"]) == pytest.approx(math.log(64))
    assert float(total) == pytest.approx(math.log(50) + 2.0 * math.log(64))


class FakeModel:
    """AlphaFold3 surface ``confidence_step`` uses: rollout knobs, a diffusion head with ``_fold``, ``_prepare_diffusion_inputs`` / ``_sample_diffusion`` /
    ``_head_batch`` and a confidence head."""

    def __init__(self, x_mini: torch.Tensor) -> None:
        self.num_samples, self.diffusion_steps = 5, 200
        self.diffusion_head = nn.Linear(1, 1)
        self.diffusion_head._fold = object()
        self.confidence_head = FakeHead()
        self.x_mini = x_mini
        self.calls: list[tuple[int, int, bool, bool]] = []

    def _prepare_diffusion_inputs(self, features, batch_data, embeddings):
        self.prepared_with = embeddings
        return SimpleNamespace(batch=batch_data, embeddings=embeddings)

    def _sample_diffusion(self, batch, embeddings):
        self.calls.append(
            (
                self.num_samples,
                self.diffusion_steps,
                self.diffusion_head.training,
                torch.is_grad_enabled(),
            )
        )
        return {"atom_positions": self.x_mini[None]}

    def _head_batch(self, batch_data):
        return batch_data


def _batch():
    tokens = SimpleNamespace(
        mask=torch.ones(RES),
        asym_id=torch.zeros(RES, dtype=torch.long),
        aatype=torch.full((RES,), ALA),
        is_dna=torch.zeros(RES, dtype=torch.bool),
        is_rna=torch.zeros(RES, dtype=torch.bool),
        is_ligand=torch.zeros(RES, dtype=torch.bool),
    )
    return SimpleNamespace(
        token_features=tokens,
        pseudo_beta_info=SimpleNamespace(token_atoms_to_pseudo_beta=_gather()),
    )


def test_confidence_step_detaches_the_trunk_rolls_out_without_grad_and_restores_the_model():
    gt = _structure()
    model = FakeModel(gt + 0.5)
    model.diffusion_head.train()
    trunk_single = torch.randn(RES, 4, requires_grad=True)
    trunk_pair = torch.randn(RES, RES, 4, requires_grad=True)
    embeddings = {"single": trunk_single, "pair": trunk_pair}
    cfg = TrainConfig(rollout_steps=20)

    detached = {k: v.detach() for k, v in embeddings.items()}
    x_mini = mini_rollout(model, {}, _batch(), detached, cfg.rollout_steps)
    total, parts = confidence_step(
        model, {}, _batch(), detached, x_mini, gt, torch.ones(RES, ATOMS, dtype=torch.bool), cfg
    )
    total.backward()

    assert set(parts) == {"plddt", "pde", "pae"}
    # rollout: one sample, rollout_steps steps, head in eval mode, no grad; everything restored afterwards
    assert model.calls == [(1, 20, False, False)]
    assert (model.num_samples, model.diffusion_steps) == (5, 200)
    assert model.diffusion_head.training and model.diffusion_head._fold is None
    # stop gradient: the confidence loss reaches the confidence head only
    assert trunk_single.grad is None and trunk_pair.grad is None
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.confidence_head.parameters())
    # the head reads the rollout structure, not the ground truth
    assert torch.equal(model.confidence_head.seen_positions, gt + 0.5)
    assert not model.prepared_with["single"].requires_grad


def test_the_phase_timer_is_inert_when_disabled_and_reports_nothing():
    from foldforge.train.profile import PhaseTimer, arm_trunk_grad_marker, tensor_bytes

    timer = PhaseTimer(False)
    with timer.phase("x"):
        pass
    timer.mark("a")
    timer.between("y", "a", "a")
    arm_trunk_grad_marker(timer, [torch.zeros(1, requires_grad=True)])
    assert timer.result() == {}
    assert tensor_bytes({"a": torch.zeros(4, dtype=torch.bfloat16), "b": [torch.zeros(2)]}) == 8 + 8


def test_the_recipe_trains_the_confidence_head_in_the_same_stage():
    cfg = TrainConfig()
    assert cfg.confidence is True
    assert (cfg.loss.plddt_loss, cfg.loss.pde_loss, cfg.loss.pae_loss) == (1.0, 1.0, 1.0)
    assert cfg.rollout_steps == 20
    assert not hasattr(cfg, "freeze_trunk"), "single-stage full finetuning has no frozen part"
