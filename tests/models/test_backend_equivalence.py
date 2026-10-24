"""A whole fold on each backend agrees with the PyTorch backend.

AlphaFold 3 on 5I28 (one chain, no MSA, no templates), the released weights, the same seeds: one fold per backend
through ``foldforge fold``'s own entry point. The diffusion sampler amplifies kernel rounding over 200 steps, so the
predicted coordinates are compared after the best rigid fit and by pLDDT, not elementwise. Skipped without a GPU, the
AF3 checkpoint or the CCD database; the anthropic backend also needs its payloads (``TRIMUL_NATIVE_BUILD_DIR``,
``OPT_CORE_DIR``, ``MINIWORLD_ANTHROPIC_ROOT``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from foldforge.modules.engine.backends import SERVES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PAYLOAD = {
    name: os.environ[name]
    for name in ("TRIMUL_NATIVE_BUILD_DIR", "OPT_CORE_DIR", "MINIWORLD_ANTHROPIC_ROOT")
    if os.environ.get(name)
}
FASTA = Path(__file__).parents[1] / "release_references/inputs/5i28/A.fasta"
#: Rigid-fit rms of the predicted atoms from the PyTorch backend's, in Angstrom. Without an MSA the prediction is
#: low-confidence and the 200 diffusion steps amplify kernel rounding: cuequivariance and anthropic land within 0.5, the
#: miniworld kernels at 1.3 here (0.1-0.2 on the real-MSA inputs of docs/guides/BACKENDS.md). A wrong kernel is at 5 and more.
FIT_RMS = 2.5
#: Mean pLDDT difference from the PyTorch backend's (about 63 here: 0.3, 0.4 and 1.8 on the three backends).
PLDDT = 3.0


def _checkpoint_or_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip without the validated AF3 blob and the CCD database.

    ``FOLDFORGE_TEST_ALLOW_SIZE_MISMATCH=1`` accepts a converted blob whose size differs from ``checkpoints.lock``
    (timing and equivalence runs on a machine that holds another converter's output).
    """
    from foldforge.data.ccd import default_path
    from foldforge.models.checkpoints import DEFAULT_FILES, lock, resolve

    try:
        path = resolve("af3", DEFAULT_FILES["af3"])
    except Exception:  # noqa: BLE001 - any resolution failure means the weights are not here
        pytest.skip("the AF3 checkpoint is not available")
    if os.environ.get("FOLDFORGE_TEST_ALLOW_SIZE_MISMATCH"):
        monkeypatch.setattr(lock, "check_size", lambda _path: None)
    else:
        try:
            lock.check_size(path)
        except lock.CheckpointMismatchError:
            pytest.skip("the AF3 checkpoint is not the validated blob")
    if not Path(default_path()).exists():
        pytest.skip("the CCD database is not available")


def superposed_rms(moving: np.ndarray, fixed: np.ndarray) -> float:
    a, b = moving - moving.mean(0), fixed - fixed.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((a @ rotation - b) ** 2).sum(-1).mean()))


@pytest.fixture(scope="module")
def folds(tmp_path_factory):
    """Run one fold per backend lazily and keep the results for the whole module."""
    root = tmp_path_factory.mktemp("equivalence")
    spec = root / "spec.yaml"
    spec.write_text(
        "name: 5i28\nchain_letters:\n  '0': A\nfasta:\n  A: "
        f"{FASTA}\nn_trunk_samples: 1\nn_diffusion_samples: 1\ndiffusion_batch_size: 1\n"
        "save_trajectory: false\n"
    )
    cache: dict[str, tuple[np.ndarray, float, dict]] = {}

    def fold(backend: str, monkeypatch: pytest.MonkeyPatch):
        if backend in cache:
            return cache[backend]
        from foldforge.models import inference
        from foldforge.models.io import paths

        _checkpoint_or_skip(monkeypatch)

        for name in PAYLOAD:
            if backend == "anthropic":
                monkeypatch.setenv(name, PAYLOAD[name])
            else:
                monkeypatch.delenv(name, raising=False)
        config = root / f"{backend}.yaml"
        config.write_text(
            f"backend: {backend}\nmode: exact\ntrunk:\n  recycles: 3\ndiffusion:\n  steps: 200\n"
        )
        monkeypatch.setattr(paths, "RUNS_ROOT", root / "runs")
        code = inference.run("af3", ["--spec", str(spec), "--config", str(config), "--out", f"run-{backend}"])
        assert code == 0
        report = json.loads((root / "runs" / f"run-{backend}" / "5i28-seed0.json").read_text())
        arrays = np.load(root / "runs" / f"run-{backend}" / "5i28-seed0.npz")
        cache[backend] = (arrays["coordinates"][0], float(report["mean_plddt"]), report["checkpoint"]["backends"])
        return cache[backend]

    return fold


@pytest.mark.parametrize(
    "backend",
    [
        "miniworld",
        "cuequivariance",
        pytest.param(
            "anthropic",
            marks=pytest.mark.skipif(len(PAYLOAD) < 3, reason="the Anthropic payloads are not named"),
        ),
    ],
)
def test_a_fold_on_each_backend_agrees_with_the_pytorch_backend(backend, folds, monkeypatch):
    reference, reference_plddt, reference_ops = folds("pytorch", monkeypatch)
    assert set(reference_ops["ops"].values()) == {"pytorch"}
    coordinates, plddt, ops = folds(backend, monkeypatch)
    assert ops["backend"] == backend
    for op, ran in ops["ops"].items():
        assert ran == (backend if backend in SERVES[op] else "pytorch"), op
    assert coordinates.shape == reference.shape
    assert np.isfinite(coordinates).all()
    fit = superposed_rms(coordinates, reference)
    assert fit <= FIT_RMS, f"rigid-fit rms {fit:.2f} A from the pytorch backend"
    assert abs(plddt - reference_plddt) <= PLDDT, f"pLDDT {plddt:.2f} against {reference_plddt:.2f}"
    print(f"{backend}: rigid-fit rms {fit:.2f} A, pLDDT {plddt:.2f} against {reference_plddt:.2f}")  # noqa: T201
