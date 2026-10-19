"""The order in which a loaded model gets its engine modules.

The pairformer wrappers come first (they create the modules the installer looks for), the engine
modules next (they read the released ``DiffusionTransition`` blocks, which the conditioning pass
turns into other classes), the conditioning pass last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from foldforge.modules.engine.conditioning import install_conditioning
from foldforge.modules.engine.modules import install_engine_modules
from foldforge.modules.engine.pairformer import install_pairformers

if TYPE_CHECKING:
    from torch import nn


def install_engine(
    model: nn.Module, dtype: torch.dtype, device: str | torch.device, *, modules: bool
) -> nn.Module:
    """Wrap the pairformers, attach the engine modules (``modules``), convert the conditioning blocks."""
    model = install_pairformers(model)
    if modules:
        model.foldforge_load_report.update(
            install_engine_modules(model, torch.empty(0, dtype=dtype, device=device))
        )
    return install_conditioning(model)
