"""Adam(0.9, 0.95) with MiniWorld's step-decay schedule, global grad-norm clip, fp32 EMA.

The parameters are the fp32 masters of the model (FoldForge keeps fp32 parameters and runs the activations in bf16 under autocast), so
Adam and the EMA work on fp32 directly. Phase 3 replaces the three loops below by the fused kernels of MiniWorld's
``benchmarks/protenix_finetune/pfx_fused_opt.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from foldforge.train.config import TrainConfig

#: Parameter-name prefixes of the heads (LR scale 1.0): the diffusion, confidence and distogram heads. Everything else is the trunk
#: (`trunk_lr_scale`). Single-stage finetuning: every parameter is in the optimizer.
HEAD_PREFIXES = ("diffusion_head.", "confidence_head.", "distogram_head.")


def split_parameters(
    model: nn.Module,
) -> tuple[
    list[tuple[str, nn.Parameter]],
    list[tuple[str, nn.Parameter]],
]:
    """``(trunk, heads)`` named parameters."""
    trunk, heads = [], []
    for name, param in model.named_parameters():
        if name.startswith(HEAD_PREFIXES):
            heads.append((name, param))
        else:
            trunk.append((name, param))
    return trunk, heads


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    trunk, heads = split_parameters(model)
    groups = [
        {
            "params": [p for _, p in heads],
            "lr": cfg.max_lr,
            "scale": 1.0,
            "name": "heads",
        },
        {
            "params": [p for _, p in trunk],
            "lr": cfg.max_lr * cfg.trunk_lr_scale,
            "scale": cfg.trunk_lr_scale,
            "name": "trunk",
        },
    ]
    return torch.optim.Adam(groups, betas=cfg.betas)


def lr_factor(step: int, cfg: TrainConfig) -> float:
    """MiniWorld ``get_step_decay_scheduler_with_warmup``: linear warmup 0 -> 1, then ``decay_factor ** ((step - warmup) // decay_steps)``."""
    if step < cfg.warmup_steps:
        return step / float(cfg.warmup_steps)
    return cfg.decay_factor ** ((step - cfg.warmup_steps) // cfg.decay_steps)


def set_lr(optimizer: torch.optim.Optimizer, step: int, cfg: TrainConfig) -> float:
    factor = lr_factor(step, cfg)
    base = (
        max(cfg.min_lr, cfg.max_lr * factor)
        if step >= cfg.warmup_steps
        else cfg.max_lr * factor
    )
    for group in optimizer.param_groups:
        group["lr"] = base * group["scale"]
    return base


@torch.no_grad()
def clip_grad_norm(params: list[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Global L2 clip; returns the pre-clip norm (a device tensor, no host sync)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    total = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.float()) for g in grads])
    )
    scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.to(g.dtype))
    return total


class EMA:
    """Exponential moving average of the parameters (fp32 shadow); ``swap_in`` / ``swap_out`` evaluate with the EMA weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            self._backup[n] = p.detach().clone()
            p.copy_(self.shadow[n])

    @torch.no_grad()
    def swap_out(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            p.copy_(self._backup[n])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {n: t.cpu() for n, t in self.shadow.items()}
