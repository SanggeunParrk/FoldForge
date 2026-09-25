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
    ("requested", "recorded"),
    [
        ("pytorch", "pytorch"),
        ("miniworld", "miniworld"),
        ("miniworld_engine", "miniworld"),
        ("cuequivariance", "cuequivariance"),
    ],
)
def test_every_backend_spelling_reaches_one_loading_lifecycle(
    tmp_path, monkeypatch, requested, recorded
):
    """One loader, whatever the backend is called and whoever asks for it.

    ``miniworld_engine`` is the engine package's own name for the backend that
    the CLI and the report call ``miniworld``; the loader normalises it once
    so a benchmark row never carries two spellings of one backend.
    """
    from foldforge.models.checkpoints import haiku

    class DenseModel(TinyModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.diffusion_head = nn.Module()
            fourier = nn.Module()
            fourier.register_buffer("weight", torch.zeros(4))
            fourier.register_buffer("bias", torch.zeros(4))
            self.diffusion_head.fourier_embeddings = fourier

    seen = []

    def import_weights(model, checkpoint) -> dict[str, int]:
        seen.append((checkpoint, len(model.state_dict())))
        return {"state_entries": len(model.state_dict())}

    monkeypatch.setattr(haiku, "import_jax_weights_", import_weights)
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(AlphaFold3=DenseModel)
    )
    model = load("af3", tmp_path / "af3.bin.zst", backend=requested, device="cpu")
    report = model.foldforge_load_report
    assert [path for path, _ in seen] == [tmp_path / "af3.bin.zst"]
    assert report["backend"] == recorded
    assert report["strict"] is True
    assert report["state_entries"] == seen[0][1]
    # The precision split is the loader's, not the backend's.
    assert model.projection.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.float32


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
            self, *, batch, prev, target_feat, first_pass, last_pass, recycle_dropout
        ) -> dict[str, torch.Tensor]:
            assert batch.num_res == 2
            # AF3 families recycle without the MC dropout Protenix samples with.
            assert recycle_dropout == 0.0
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
def test_native_bf16_clears_a_projection_s_fp32_compute_override(dtype):
    """Native BF16 has to reach the GEMM, not only the stored weights.

    A released geometry or conditioning projection may carry an FP32 compute
    override; under a BF16 policy that override would quietly keep the matmul
    in FP32 while the weights say otherwise. The rule is layout-independent, so
    it is asked directly rather than through whichever loader still exists.
    """
    from team_gm.modules.precision import NativeLinear

    layer = NativeLinear(4, 4, bias=False)
    layer.compute_dtype = torch.float32
    loading.clear_native_compute_override(nn.Sequential(nn.LayerNorm(4), layer), dtype)

    assert layer.compute_dtype == (None if dtype == torch.bfloat16 else torch.float32)


def test_every_registered_family_names_a_distinct_default_checkpoint():
    """A default filename is a claim about which blob holds that family.

    Two families sharing one file is almost always a mistake rather than a
    fact: ESMFold2-Fast is a separate 24-block release with its own blob, and
    pointing it at ESMFold2's would make the strict load fail at the first
    missing trunk block instead of at the table that lied.
    """
    from foldforge.models import registered_models
    from foldforge.models.checkpoints import DEFAULT_FILES

    # Protenix names its blob by release, so `--variant` supplies the name.
    named = {name for name in registered_models() if name != "protenix"}
    assert named <= set(DEFAULT_FILES)
    files = [DEFAULT_FILES[name] for name in named]
    assert len(set(files)) == len(files), sorted(files)


@pytest.mark.skipif(
    not (Path(foldforge.__file__).parents[2] / "model_checkpoints").is_dir(),
    reason="weights are not downloaded on this machine",
)
def test_every_registered_family_resolves_its_checkpoint():
    """A distinct filename is not enough; it has to be findable.

    Two releases of one project often land in one directory -- both OpenFold3
    blobs sit under `openfold3/` -- so `openfold3-preview2` named the right
    file in a directory that does not exist. Naming the file was checked;
    finding it was not, and the gap only showed up as a failed Slurm job.
    """
    from foldforge.models import registered_models
    from foldforge.models.checkpoints import DEFAULT_FILES, resolve

    missing = []
    for name in sorted(registered_models()):
        if name == "protenix":
            continue  # --variant supplies the filename
        try:
            resolve(name, DEFAULT_FILES[name])
        except FileNotFoundError as error:
            missing.append(f"{name}: {str(error).splitlines()[0]}")
    assert not missing, "\n".join(missing)
