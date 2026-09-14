# GPU dependencies are imported at the model boundary so registry/help stay cheap.
# ruff: noqa: PLC0415
"""Complete AF3 PyTorch model, converted from the official Haiku checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from foldforge.models.af_conditioning import install_conditioning
from foldforge.models.af_pairformer import install_pairformers
from foldforge.modules.af_family import configure_model

from .ported.alphafold3 import AlphaFold3
from .ported.params import import_jax_weights_


def load(
    checkpoint: str | Path | None = None,
    *,
    backend: str = "miniworld",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    recycles: int = 10,
    samples: int = 5,
    steps: int = 200,
) -> torch.nn.Module:
    """Strictly load released weights and configure native parameter precision."""
    if checkpoint is None:
        from foldforge.checkpoints import resolve

        checkpoint = resolve("af3", "af3.bin.zst")
    checkpoint = Path(checkpoint)
    model = AlphaFold3(
        num_recycles=recycles, num_samples=samples, diffusion_steps=steps
    )
    report = import_jax_weights_(model, checkpoint)
    model.foldforge_load_report = {
        "checkpoint": str(checkpoint),
        "strict": True,
        **report,
    }
    model = configure_model(model, backend, dtype, device)
    # AF3 defines these fixed Fourier features in FP32, independent of parameter dtype.
    from .fourier import _BIAS, _WEIGHT

    model.diffusion_head.fourier_embeddings.weight = torch.tensor(
        _WEIGHT, device=device
    )
    model.diffusion_head.fourier_embeddings.bias = torch.tensor(_BIAS, device=device)
    return install_conditioning(install_pairformers(model))
