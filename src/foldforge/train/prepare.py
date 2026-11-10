"""Switch a loaded (inference) FoldForge model to training mode.

``configure_model`` leaves the model in ``eval()`` with ``requires_grad_(False)`` and ``load_model`` marks dead (numerically zero)
modules so inference can skip them; both are undone here. Dead modules MUST run in training (their parameters train).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from foldforge.modules.dense.dead import clear_dead

if TYPE_CHECKING:
    from torch import nn

    from foldforge.train.config import TrainConfig


def enable_training(model: nn.Module, cfg: TrainConfig) -> dict[str, int]:
    """``train()``, gradients on (trunk frozen when ``cfg.freeze_trunk``), dead skip off, distogram logits kept for the loss."""
    cleared = clear_dead(model)
    model.train()
    model.requires_grad_(True)
    if cfg.freeze_trunk:
        for name, param in model.named_parameters():
            if not name.startswith("diffusion_head."):
                param.requires_grad_(False)
    model.distogram_head.save_distogram = (
        True  # the head returns its logits only when asked
    )
    return {
        "dead_cleared": cleared,
        "sampler_ops_off_modules": disable_inference_only_ops(model),
    }


def draw_passes(cfg: TrainConfig, rng: random.Random) -> int:
    """Total trunk passes of one step, uniform in [recycles_min, recycles_max]."""
    return rng.randint(cfg.recycles_min, cfg.recycles_max)


def seed_everything(seed: int) -> random.Random:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def disable_inference_only_ops(model: nn.Module) -> int:
    """The dense atom encoder pools through the engine's ``sampler_ops`` kernel (``PoolAtoms``), which has no backward: set the module-level
    ``foldforge_sampler_ops = False`` (the switch ``atom_cross_attention`` reads) so training uses the torch pooling. Returns the module count."""
    count = 0
    for module in model.modules():
        module.foldforge_sampler_ops = False
        count += 1
    return count
