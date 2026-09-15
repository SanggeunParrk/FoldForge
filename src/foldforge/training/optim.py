from __future__ import annotations

import inspect

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
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from foldforge.models.config.types import ConfigNode


logger = logging.getLogger(__name__)


def get_adamw(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.AdamW:
    """Create an AdamW optimizer for the given model with specified parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is created.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        betas (tuple): Coefficients used for computing running averages of gradient and
            its square.
        device_type (str): The device type ('cuda' or 'cpu') on which the optimizer
            will operate.

    Returns:
        torch.optim.AdamW: The AdamW optimizer configured with the specified
        parameters.

    """
    # start with all of the candidate parameters
    param_dict = dict(model.named_parameters())
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise
    # no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms
    # don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # noqa: PLR2004 - tensor rank or format cardinality
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # noqa: PLR2004 - tensor rank or format cardinality
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        "%s",
        (
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        ),
    )
    logger.info(
        "%s",
        (
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        ),
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info("%s", f"using fused AdamW: {use_fused}")

    return optimizer


def get_optimizer(
    configs: ConfigNode, model: torch.nn.Module, param_names: list[str] | None = None
) -> torch.optim.Optimizer:
    """Return optimizer."""
    logger.info(
        "%s",
        " ".join(
            str(value)
            for value in (
                "param_names: ",
                param_names,
            )
        ),
    )
    if param_names is None or len(param_names) == 0 or param_names[0] == "":
        param_names = None
    if configs.adam.use_adamw:
        optimizer = get_adamw(
            model=model,
            weight_decay=configs.adam.weight_decay,
            learning_rate=configs.adam.lr,
            betas=(configs.adam.beta1, configs.adam.beta2),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        if param_names is None or len(param_names) == 0:
            param_groups = [{"params": model.parameters(), "lr": configs.lr}]
        else:
            finetune_params = []
            finetune_params_names = []
            other_params = []
            for n, p in model.named_parameters():
                if any(name in n for name in param_names):
                    finetune_params.append(p)
                    finetune_params_names.append(n)
                else:
                    other_params.append(p)
            param_groups = [{"params": finetune_params, "lr": configs.finetune.lr}]
            logger.info(
                "%s",
                " ".join(
                    str(value)
                    for value in (
                        "Sample Finetune Params Names: ",
                        finetune_params_names[:5],
                    )
                ),
            )
            if configs.lr > 0.0:
                param_groups.append({"params": other_params, "lr": configs.lr})

        optimizer = torch.optim.Adam(
            param_groups,
            lr=configs.adam.lr,
            weight_decay=configs.adam.weight_decay,
            betas=(configs.adam.beta1, configs.adam.beta2),
        )
        for i, param_group in enumerate(optimizer.param_groups):
            logger.info(
                "%s",
                (
                    f"Parameter Group {i}: Learning Rate = {param_group['lr']} Num "
                    f"= {len(param_group['params'])}"
                ),
            )
    return optimizer


def is_loss_nan_check(loss: torch.Tensor) -> bool:
    """Check the validness of the current loss.

    Args:
        loss: the loss from the model

    Returns:
        bool: if True, loss is not nan or inf

    """

    def is_nan(x: torch.Tensor) -> bool:
        """Check whether nan."""
        return bool(torch.isnan(x).any() or torch.isinf(x).any())

    def all_reduce_tensor(
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """Compute all reduce tensor."""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    nan_flag = torch.tensor(
        1.0 if is_nan(loss) else 0.0,
        device=loss.device if torch.cuda.is_available() else None,
    )  # support cpu
    # avoid "Watchdog caught collective operation timeout" error
    all_reduce_tensor(nan_flag)
    return nan_flag.item() > 0.0
