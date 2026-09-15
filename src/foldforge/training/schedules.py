from __future__ import annotations

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import warnings
from typing import TYPE_CHECKING, Any

from torch.optim.lr_scheduler import ConstantLR, LRScheduler

if TYPE_CHECKING:
    import torch

    from foldforge.models.config.types import ConfigNode


class CosineAnnealingWithWarmup(LRScheduler):
    """Represent cosine annealing with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        lr: float,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.lr = lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def _get_step_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return (step + 1) / (self.warmup_steps + 1) * self.lr
        if step >= self.decay_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (
            self.decay_steps - self.warmup_steps
        )
        if not (0 <= decay_ratio <= 1):
            message = "Invalid state: 0 <= decay_ratio <= 1"
            raise ValueError(message)
        coff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coff * (self.lr - self.min_lr)

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return lr."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        return [
            self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float | torch.Tensor]:
        return [self._get_step_lr(self.last_epoch) for base_lr in self.base_lrs]


# The Alphafold3 Learning Rate Scheduler As in 5.4
class AlphaFold3LRScheduler(LRScheduler):
    """Represent alpha fold3 l r scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        warmup_steps: int = 1000,
        lr: float = 1.8e-3,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_every_n_steps
        self.lr = lr
        self.decay_factor = decay_factor
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def _get_step_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            lr = step / self.warmup_steps * self.lr
        else:
            decay_count = step // self.decay_steps
            lr = self.lr * (self.decay_factor**decay_count)
        return lr

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return lr."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        return [
            self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
        ]


class ConstantLRScheduler(ConstantLR):
    """Represent constant l r scheduler."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, lr: float, last_epoch: int = -1
    ) -> None:
        self.lr = lr
        super().__init__(optimizer, factor=1.0, last_epoch=last_epoch)

    def _get_step_lr(self, step: int) -> float:  # noqa: ARG002 - shared callback or fixture signature
        return self.lr


class FinetuneLRScheduler(LRScheduler):
    """Represent finetune l r scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr_config: ConfigNode,
        finetune_lr_config: ConfigNode,
        last_epoch: int = -1,
    ) -> None:

        self.lr_scheduler = get_lr_scheduler(base_lr_config, optimizer)
        self.finetune_lr_scheduler = get_lr_scheduler(finetune_lr_config, optimizer)
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def _get_step_lr(self, step: int) -> list[float | torch.Tensor]:
        lr = self.lr_scheduler._get_step_lr(step)  # noqa: SLF001 - shared runtime integration hook
        ft_lr = self.finetune_lr_scheduler._get_step_lr(step)  # noqa: SLF001 - shared runtime integration hook
        # this order is same to the order in optimizer
        return [ft_lr, lr]

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return lr."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        return self._get_step_lr(self.last_epoch)


def get_lr_scheduler(
    configs: ConfigNode, optimizer: torch.optim.Optimizer, **kwargs: Any
) -> AlphaFold3LRScheduler | CosineAnnealingWithWarmup | ConstantLRScheduler:
    """Get the learning rate scheduler based on the configuration.

    Args:
        configs: Configuration object containing scheduler settings.
        optimizer (torch.optim.Optimizer): The optimizer to which the scheduler will be
            attached.
        **kwargs: Additional keyword arguments to be passed to the scheduler.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler.

    Raises:
        ValueError: If the specified learning rate scheduler is invalid.

    """
    if configs.lr_scheduler == "af3":
        lr_scheduler = AlphaFold3LRScheduler(
            optimizer, **configs.af3_lr_scheduler, **kwargs
        )
    elif configs.lr_scheduler == "cosine_annealing":
        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer,
            configs.warmup_steps,
            configs.max_steps,
            configs.lr,
            configs.lr * configs.min_lr_ratio,
            **kwargs,
        )
    elif configs.lr_scheduler == "constant":
        lr_scheduler = ConstantLRScheduler(
            optimizer,
            lr=configs.lr,
            **kwargs,
        )
    else:
        msg = f"Invalid lr scheduler: [{configs.lr_scheduler}]"
        raise ValueError(msg)
    return lr_scheduler
