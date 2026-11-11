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
    """``train()``, gradients on for EVERY parameter (single-stage full finetuning), dead skip off, distogram / confidence logits kept for the loss."""
    cleared = clear_dead(model)
    model.train()
    model.requires_grad_(True)
    model.foldforge_sampler_scope = None  # the mini-rollout samples eagerly (no sampler CUDA graph)
    model.confidence_head.save_logits = cfg.confidence  # the head returns its logits only when asked
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


#: Block classes whose forward is recomputed in the backward under activation checkpointing (the pair-stack blocks that dominate the activation
#: memory: the engine ``PairformerBlock`` of the trunk / templates / confidence head and FoldForge's ``EvoformerBlock``).
CHECKPOINTED_BLOCKS = ("PairformerBlock", "EvoformerBlock")


def enable_activation_checkpointing(model: nn.Module) -> int:
    """Wrap the forward of every pair-stack block in ``torch.utils.checkpoint`` (non-reentrant, RNG preserved) so its activations are recomputed in
    the backward instead of stored. Returns the number of blocks wrapped. Training only: ``torch.is_grad_enabled()`` is the guard."""
    from torch.utils.checkpoint import checkpoint

    count = 0
    for module in model.modules():
        if type(module).__name__ not in CHECKPOINTED_BLOCKS or getattr(
            module, "_ff_checkpointed", False
        ):
            continue
        forward = module.forward

        def wrapped(*args, _forward=forward, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        module.forward = wrapped
        module._ff_checkpointed = True
        count += 1
    return count
