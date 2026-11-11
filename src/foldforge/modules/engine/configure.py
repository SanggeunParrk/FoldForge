"""Precision and device of a loaded model: fp32 master parameters, bf16 (or fp32) activations.

FoldForge parameters are ``master`` (fp32 parameters, bf16 activations under autocast: the engine's ``MasterWeights``) or plain ``fp32``;
there are no native-bf16 parameters, and the trunk is always master. ``dtype`` is therefore the ACTIVATION dtype: bf16 means precision
``master`` (run the model under :func:`activation_autocast`), fp32 means precision ``fp32``. The parameters keep the loaded fp32 values
exactly (no rounding, no re-cast of norms: engine norms keep their own fp32 pin); the Fourier / geometry tables are not touched.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from foldforge.modules.engine.backends import BACKENDS

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn

_ACTIVATION_PRECISION = {torch.bfloat16: "master", torch.float32: "fp32"}


def precision_of(dtype: torch.dtype) -> str:
    """The parameter precision a given activation dtype runs with (``master`` for bf16, ``fp32`` for fp32)."""
    try:
        return _ACTIVATION_PRECISION[dtype]
    except KeyError:
        message = f"activation dtype {dtype} has no parameter precision (bfloat16 -> master, float32 -> fp32)"
        raise ValueError(message) from None


def configure_model(
    model: nn.Module,
    backend: str = "miniworld",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Move ``model`` to ``device`` keeping fp32 parameters and record its precision: ``model.foldforge_precision`` (``master`` /
    ``fp32``) and ``model.foldforge_activation_dtype``. Engine modules inside get their fp32 masters pinned (``set_master_weights``) for
    ``master``. Backends are applied separately (:func:`foldforge.modules.engine.install.install_engine`)."""
    if backend not in BACKENDS:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)
    precision = precision_of(dtype)
    fixed_buffers = [
        (m, name, value.detach().clone())
        for m in model.modules()
        for name, value in m.named_buffers(recurse=False)
        if value is not None and value.dtype == torch.float32
    ]
    model.to(device=device)
    for m, name, value in fixed_buffers:
        setattr(m, name, value.to(device))
    if precision == "master":
        from miniworld_engine.modules import MasterWeights

        for module in model.modules():
            if isinstance(module, MasterWeights):
                module.set_master_weights()
    model.foldforge_precision = precision
    model.foldforge_activation_dtype = dtype
    model.eval().requires_grad_(requires_grad=False)
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            msg = f"{name}: {param.dtype}; FoldForge parameters are fp32 (master or plain), never native low precision"
            raise TypeError(msg)
    return model


def activation_autocast(model: nn.Module) -> contextlib.AbstractContextManager[None]:
    """The autocast a ``master`` model runs under (bf16 on its device type); a no-op for ``fp32`` and for a model that was not configured."""
    if getattr(model, "foldforge_precision", None) != "master":
        return contextlib.nullcontext()
    device = next(model.parameters()).device.type
    return torch.autocast(device, dtype=model.foldforge_activation_dtype)


def engine_activation(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """``tensor`` in the dtype the engine's ``master`` modules are called with: inside an enabled autocast a floating fp32 activation becomes
    the autocast dtype (the engine's ``MasterWeights``: fp32 masters, bf16 activations); anything else is returned as it is (``fp32``
    precision runs without autocast, so its activations stay fp32)."""
    if tensor is None or tensor.dtype != torch.float32 or not torch.is_autocast_enabled(tensor.device.type):
        return tensor
    return tensor.to(torch.get_autocast_dtype(tensor.device.type))


def set_tf32_policy(model: nn.Module, *, tf32: bool, diffusion_tf32: bool = False) -> None:
    """Record whether TF32 tensor cores may serve this model's modules (``foldforge_tf32`` on every module).

    ``tf32`` is ``execution.tf32``; ``diffusion_tf32`` additionally allows it in the diffusion head (the ``af3_default`` policy) while the
    trunk stays exact. The engine's fp32 backends read the flag through the model's execution policy, not through the engine classes."""
    head = getattr(model, "diffusion_head", None)
    diffusion = set() if head is None else {id(m) for m in head.modules()}
    for module in model.modules():
        module.foldforge_tf32 = tf32 or (diffusion_tf32 and id(module) in diffusion)


@contextlib.contextmanager
def inference_settings() -> Iterator[None]:
    """Restore torch's process-wide matmul precision after library inference (the engine keeps no process-global state: each module
    carries its own backend, so this is the only shared setting)."""
    precision = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
