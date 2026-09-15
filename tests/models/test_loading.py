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
    [("protenix", "model"), ("opendde", "state_dict"), ("opendde", None)],
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
    constructor = "Protenix" if name == "protenix" else "OpenDDE"
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(**{constructor: TinyModel})
    )
    config = SimpleNamespace(
        model_name="protenix-v2" if name == "protenix" else "opendde_v1"
    )
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


def test_sequence_checkpoint_uses_same_loader(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from foldforge.models.checkpoints import esmfold2
    from foldforge.models.config.esmfold2 import ESMFold2Config

    reference = TinyModel()
    save_file(reference.state_dict(), tmp_path / "model.safetensors")
    monkeypatch.setattr(ESMFold2Config, "from_json", lambda _: object())
    monkeypatch.setattr(esmfold2, "convert_model", lambda state, _config: state)
    monkeypatch.setattr(
        loading, "import_module", lambda _: SimpleNamespace(ESMFold2Model=TinyModel)
    )
    model = load("esmfold2", tmp_path, backend="pytorch", device="cpu")
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
            self.diffusion_head.fourier_embeddings = nn.Module()

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
