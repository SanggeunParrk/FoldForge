"""All checkpoint formats use one strict loading and precision lifecycle."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import foldforge
from foldforge.models import get_model, load, loading


class TinyModel(nn.Module):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.projection = nn.Linear(4, 4, bias=False)


@pytest.mark.parametrize(
    ("name", "container"),
    # OpenDDE is the last model on the flat checkpoint path; everything else
    # loads through the dense graph.
    [("opendde", "state_dict"), ("opendde", None)],
)
def test_flat_checkpoint_uses_strict_native_precision(
    tmp_path, monkeypatch, name, container
):
    reference = TinyModel()
    with torch.no_grad():
        reference.norm.weight.fill_(1.000123)
    state = {"module." + key: value for key, value in reference.state_dict().items()}
    checkpoint = tmp_path / "weights.pt"
    torch.save({container: state} if container else state, checkpoint)
    constructor = "OpenDDE"
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(**{constructor: TinyModel})
    )
    config = SimpleNamespace(model_name="opendde_v1", data={"msa": {}})
    model = load(name, checkpoint, configs=config, backend="pytorch", device="cpu")
    assert model.projection.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.float32
    torch.testing.assert_close(model.norm.weight, reference.norm.weight, atol=0, rtol=0)
    assert not model.training
    assert not model.projection.weight.requires_grad
    assert model.foldforge_load_report["strict"] is True
    assert model.foldforge_load_report["backend"] == "pytorch"
    state.pop("module.projection.weight")
    torch.save({container: state} if container else state, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load(name, checkpoint, configs=config, backend="pytorch", device="cpu")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("pytorch", "pytorch"),
        ("miniworld", "miniworld_engine"),
        ("miniworld_engine", "miniworld_engine"),
        ("cuequivariance", "cuequivariance"),
    ],
)
def test_sequence_checkpoint_uses_same_loader(tmp_path, monkeypatch, backend, expected):
    from safetensors.torch import save_file

    from foldforge.models.checkpoints import sequence
    from foldforge.models.config.sequence import ESMFold2Config

    reference = TinyModel()
    save_file(reference.state_dict(), tmp_path / "model.safetensors")
    monkeypatch.setattr(ESMFold2Config, "from_json", lambda _: SimpleNamespace())
    monkeypatch.setattr(sequence, "convert_model", lambda state, _config: state)
    selected = []

    def construct(config, implementation) -> TinyModel:
        from team_gm.modules.exceptions import ImplementationType

        assert isinstance(implementation, ImplementationType)
        selected.append(implementation.value)
        return TinyModel(config, implementation)

    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(ESMFold2Model=construct)
    )
    model = load("esmfold2", tmp_path, backend=backend, device="cpu")
    assert selected == [expected]
    assert model.foldforge_load_report["backend"] == (
        "miniworld" if expected == "miniworld_engine" else expected
    )
    assert model.projection.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.float32
    assert model.foldforge_load_report["state_entries"] == len(reference.state_dict())


def test_registry_returns_one_loader_and_retired_packages_are_absent():
    root = Path(foldforge.__file__).parent
    for name in ("af3", "protenix", "opendde", "esmfold2"):
        bound = get_model(name)
        assert bound.func is load
        assert bound.args == (name,)
        assert not (root / "models" / name).exists()
    assert not list(root.rglob("ported"))
    for path in (root / "models/architectures").glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(n, ast.FunctionDef) and n.name in {"load", "from_checkpoint"}
            for n in ast.walk(tree)
        )


def test_haiku_checkpoint_is_imported_before_precision(tmp_path, monkeypatch):
    from foldforge.models.checkpoints import haiku

    class DenseModel(TinyModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.diffusion_head = nn.Module()
            fourier = nn.Module()
            fourier.register_buffer("weight", torch.zeros(4))
            fourier.register_buffer("bias", torch.zeros(4))
            self.diffusion_head.fourier_embeddings = fourier

    def import_weights(model, checkpoint) -> dict[str, int]:
        assert checkpoint == tmp_path / "af3.bin.zst"
        assert model.projection.weight.dtype == torch.float32
        with torch.no_grad():
            model.norm.weight.fill_(1.000123)
        return {"state_entries": len(model.state_dict())}

    monkeypatch.setattr(haiku, "import_jax_weights_", import_weights)
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(AlphaFold3=DenseModel)
    )
    model = load("af3", tmp_path / "af3.bin.zst", backend="pytorch", device="cpu")
    assert model.projection.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.float32
    assert model.diffusion_head.fourier_embeddings.weight.dtype == torch.float32
    torch.testing.assert_close(
        model.norm.weight, torch.full((4,), 1.000123), atol=0, rtol=0
    )
    assert model.foldforge_load_report["strict"] is True
    assert model.foldforge_load_report["team_gm_pairformer_blocks"] == 0


def test_af3_default_keeps_released_mixed_parameters(tmp_path, monkeypatch):
    from foldforge.models.checkpoints import haiku

    class DenseModel(TinyModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.diffusion_head = nn.Module()
            self.diffusion_head.projection = nn.Linear(4, 4, bias=False)
            fourier = nn.Module()
            fourier.register_buffer("weight", torch.zeros(4))
            fourier.register_buffer("bias", torch.zeros(4))
            self.diffusion_head.fourier_embeddings = fourier

    def import_weights(model, checkpoint, *, preserve_dtype=False) -> dict[str, int]:
        assert preserve_dtype
        assert checkpoint == tmp_path / "af3.bin.zst"
        with torch.no_grad():
            model.projection.weight.data = model.projection.weight.bfloat16()
            model.diffusion_head.projection.weight.fill_(1.000123)
            model.norm.weight.fill_(1.000123)
        return {"state_entries": len(model.state_dict())}

    monkeypatch.setattr(haiku, "import_jax_weights_", import_weights)
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(AlphaFold3=DenseModel)
    )
    model = load(
        "af3",
        tmp_path / "af3.bin.zst",
        backend="pytorch",
        device="cpu",
        precision_policy="af3_default",
    )
    assert model.projection.weight.dtype == torch.bfloat16
    assert model.diffusion_head.projection.weight.dtype == torch.float32
    torch.testing.assert_close(
        model.diffusion_head.projection.weight,
        torch.full((4, 4), 1.000123),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        model.norm.weight, torch.full((4,), 1.000123), atol=0, rtol=0
    )
    assert model.reference_precision


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_haiku_assign_preserves_source_dtype_and_exact_values(dtype):
    from foldforge.models.checkpoints.haiku import Param, ParamType, assign

    source = torch.full((3, 4), 1.000123, dtype=dtype)
    target = nn.Parameter(torch.zeros((4, 3), dtype=torch.bfloat16))
    assign(
        {"weight": Param(target, ParamType.linear_weight)},
        {"weight": source},
        preserve_dtype=True,
    )
    assert target.dtype == dtype
    torch.testing.assert_close(target, source.T, atol=0, rtol=0)


@pytest.mark.parametrize("recycles", [0, 2])
@pytest.mark.parametrize("reference_precision", [False, True])
def test_af3_recycles_are_additional_trunk_passes(
    monkeypatch, recycles, reference_precision
):
    from foldforge.models.architectures import af3
    from foldforge.modules.dense.spec import ALPHAFOLD3

    batch = SimpleNamespace(
        num_res=2,
        token_features=SimpleNamespace(mask=torch.ones(2), asym_id=torch.ones(2)),
        pseudo_beta_info=SimpleNamespace(token_atoms_to_pseudo_beta=None),
    )
    monkeypatch.setattr(af3.feat_batch.Batch, "from_data_dict", lambda _: batch)
    calls = []
    expected_passes = recycles + 1

    class Trunk(nn.Module):
        def forward(
            self, *, batch, prev, target_feat, first_pass, last_pass
        ) -> dict[str, torch.Tensor]:
            assert batch.num_res == 2
            # The first pass is the one the trunk is told about; a family whose
            # recycle carry starts at its own initial representations needs it.
            assert first_pass == (len(calls) == 0)
            # And the last, for a family with a post-loop stage: running that
            # stage on every pass would be wasted work, and recycling its
            # output would feed back a differently scaled representation.
            assert last_pass == (len(calls) == expected_passes - 1)
            calls.append(prev["pair"].dtype)
            return {
                "pair": torch.ones(2, 2, 2, dtype=torch.bfloat16),
                "single": torch.ones(2, 2, dtype=torch.bfloat16),
                "target_feat": target_feat,
            }

    model = af3.AlphaFold3.__new__(af3.AlphaFold3)
    nn.Module.__init__(model)
    model.num_recycles = recycles
    model.spec = ALPHAFOLD3
    model.reference_precision = reference_precision
    model.evoformer_pair_channel = model.evoformer_seq_channel = 2
    model.evoformer = Trunk()
    monkeypatch.setattr(
        model, "create_target_feat_embedding", lambda _: torch.ones(2, 2)
    )
    monkeypatch.setattr(
        model, "_sample_diffusion", lambda *_: {"atom_positions": torch.zeros(1, 2, 3)}
    )
    model.confidence_head = lambda **_: {"score": torch.ones(1)}
    model.distogram_head = lambda *_: {}
    model({})
    assert len(calls) == recycles + 1
    expected = torch.float32 if reference_precision else torch.bfloat16
    assert calls == [torch.float32] + [expected] * recycles


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_projection_compute_override_follows_native_policy(
    tmp_path, monkeypatch, dtype
):
    class GeometryModel(TinyModel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.projection.compute_dtype = torch.float32

    checkpoint = tmp_path / "geometry.pt"
    torch.save(GeometryModel().state_dict(), checkpoint)
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(OpenDDE=GeometryModel)
    )
    model = load(
        "opendde",
        checkpoint,
        configs=SimpleNamespace(model_name="opendde_v1", data={"msa": {}}),
        backend="pytorch",
        device="cpu",
        dtype=dtype,
    )
    expected = None if dtype == torch.bfloat16 else torch.float32
    assert model.projection.compute_dtype == expected
    assert model.projection.weight.dtype == dtype
    assert model.norm.weight.dtype == torch.float32
