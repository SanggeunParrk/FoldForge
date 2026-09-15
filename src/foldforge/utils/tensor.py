from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 ByteDance and/or its affiliates.
# Copyright (c) 2026 Aureka AI Research
import gc
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _mps_autocast_available() -> bool:
    """Whether an MPS autocast state exists that a guard would have to clear."""
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - optional optimization retains the reference path
        return False


@contextmanager
def disabled_autocast():
    """Force FP32 inside the block on every accelerator that can autocast.

    Regions that must stay FP32 historically guarded only CUDA autocast. The
    Apple MPS backend keeps a separate autocast state, so a CUDA-only guard
    silently lets BF16 through there.
    """
    with ExitStack() as stack:
        stack.enter_context(torch.autocast("cuda", enabled=False))
        if _mps_autocast_available():
            stack.enter_context(torch.autocast("mps", enabled=False))
        yield


def to_device(obj, device, *, non_blocking: bool = False):
    """Return tensors on ``device`` without mutating the caller's container."""
    if isinstance(obj, dict):
        return {
            key: to_device(obj=value, device=device, non_blocking=non_blocking)
            if isinstance(value, (dict, torch.Tensor))
            else value
            for key, value in obj.items()
        }
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=non_blocking)
    msg = f"type {type(obj)} not supported"
    raise RuntimeError(msg)


def _clear_accelerator_cache(
    *,
    synchronize: Callable[[], None] | None,
    empty_cache: Callable[[], None],
    suppress_errors: bool,
) -> None:
    """Run cache cleanup without letting synchronization skip cache release."""
    synchronize_error: RuntimeError | None = None
    if synchronize is not None:
        try:
            synchronize()
        except RuntimeError as exc:
            if not suppress_errors:
                synchronize_error = exc
    try:
        empty_cache()
    except RuntimeError:
        if not suppress_errors and synchronize_error is None:
            raise
    if synchronize_error is not None:
        raise synchronize_error


def cleanup_device_memory(
    device: torch.device | str,
    *,
    collect_garbage: bool = True,
    synchronize: bool = False,
    suppress_errors: bool = False,
) -> None:
    """Collect garbage, optionally synchronize, and clear an accelerator cache.

    ``suppress_errors`` is reserved for best-effort cleanup after an operation
    has already failed. Normal synchronization must surface asynchronous device
    errors instead of allowing inference to report success.
    """
    selected_device = torch.device(device)
    if collect_garbage:
        gc.collect()

    if selected_device.type == "mps" and torch.backends.mps.is_available():
        _clear_accelerator_cache(
            synchronize=torch.mps.synchronize if synchronize else None,
            empty_cache=torch.mps.empty_cache,
            suppress_errors=suppress_errors,
        )
        return

    if selected_device.type == "cuda" and torch.cuda.is_available():
        # Cleanup may run after a failed CUDA operation, which leaves the
        # context in a sticky error state where every CUDA call re-reports it.
        # Keep the calls on separate error boundaries so a failed synchronize
        # still lets the allocator release its blocks, but suppress failures
        # only when the caller is already recovering from another error.
        _clear_accelerator_cache(
            synchronize=(
                partial(torch.cuda.synchronize, device=selected_device)
                if synchronize
                else None
            ),
            empty_cache=torch.cuda.empty_cache,
            suppress_errors=suppress_errors,
        )


@contextmanager
def disable_cudnn_benchmark(device: torch.device | str | None = None):
    """Temporarily disable cuDNN benchmark for the selected CUDA device."""
    device_type = torch.device(device).type if device is not None else None
    if device_type not in {None, "cuda"} or not torch.cuda.is_available():
        yield
        return

    benchmark_enabled = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark_enabled


def cdist(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
    *,
    compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
) -> torch.Tensor:
    """Pair distances with an explicit numerical/compute-mode policy."""
    return torch.cdist(a, b if b is not None else a, compute_mode=compute_mode)


def map_values_to_list(data, *, recursive: bool = True):
    """Map values to list."""
    converted = {}
    for k, raw_v in data.items():
        v = raw_v
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            converted[k] = v.cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            converted[k] = v.tolist()
        elif isinstance(v, dict) and recursive:
            converted[k] = map_values_to_list(data=v, recursive=recursive)
        else:
            converted[k] = v
    return converted


def round_values(data, *, recursive: bool = True):
    """Compute round values."""
    for k, raw_v in data.items():
        v = raw_v
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = np.round(v.cpu().numpy(), 2)
        elif isinstance(v, np.ndarray):
            data[k] = np.round(v, 2)
        elif isinstance(v, list):
            data[k] = list(np.round(np.array(v), 2))
        elif isinstance(v, dict) and recursive:
            data[k] = round_values(data=v, recursive=recursive)
    return data


def autocasting_disable_decorator(disable_casting):
    """Compute autocasting disable decorator."""

    def func_wrapper(func):
        """Compute func wrapper."""

        def new_func(*args: Any, **kwargs: Any):
            """Compute new func."""
            _amp_context = disabled_autocast() if disable_casting else nullcontext()

            # Helper function to conditionally cast tensors
            def conditioned_cast(tensor):
                """Compute conditioned cast."""
                if (
                    disable_casting
                    and isinstance(tensor, torch.Tensor)
                    and torch.is_floating_point(tensor)
                ):
                    return tensor.to(dtype=torch.float32)
                return tensor

            with _amp_context:
                return func(
                    *(conditioned_cast(v) for v in args),
                    **{k: conditioned_cast(v) for k, v in kwargs.items()},
                )

        return new_func

    return func_wrapper


def dict_to_tensor(feature_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compute dict to tensor."""
    for k, v in feature_dict.items():
        if not isinstance(v, torch.Tensor):
            dtype = v.dtype
            feature_dict[k] = torch.tensor(v)

            if dtype in [np.int64, np.int32]:
                feature_dict[k] = feature_dict[k].to(torch.int64)
            elif dtype in [np.float32, np.float64]:
                feature_dict[k] = feature_dict[k].to(torch.float32)

    return feature_dict


def collate_fn_identity(x):
    """Compute collate fn identity."""
    return x


def grad_norm(params):
    """Compute grad norm."""
    total_norm = 0.0
    for p in params:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1.0 / 2)


def detach_if(t: torch.Tensor, *, detach: bool):
    """Compute detach if."""
    if detach:
        return t.detach()
    return t


def batch_avg_with_mask(
    value: torch.Tensor,
    mask: torch.Tensor,
    avg_dim: int | tuple[int, ...] | None = None,
    batch_reduction: str = "mean",
    eps: float = 1e-12,
):
    """Average values with mask.

    Args:
        value: tensor of shape [BS, ...]
        mask: tensor with same shape and type of value, 1 means valid, 0 means maksed
        avg_dim: dimensions to apply average, if None, all dims excluding BS dim will
            be averaged
        batch_reduction: mean/sum/none, reduction operation applied on BS dim

    """
    if avg_dim is None:
        avg_dim = tuple(range(1, len(value.shape)))
    avg = (value * mask).sum(dim=avg_dim) / (mask.sum(dim=avg_dim) + eps)
    if batch_reduction == "mean":
        return avg.mean()
    if batch_reduction == "sum":
        return avg.sum()
    if batch_reduction == "none":
        return avg
    msg = f"Invalid batch_reduction: {batch_reduction}"
    raise RuntimeError(msg)


def eye_mask(L, device=None, *, opposite: bool = False):
    """Compute eye mask."""
    if opposite:
        return 1.0 - torch.eye(L, device=device)
    return torch.eye(L, device=device)


def glorot_uniform(t) -> None:
    """Compute glorot uniform."""
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m, bias: str = "zero") -> None:
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            if bias == "zero":
                m.bias.data.zero_()
            else:
                if bias != "normal":
                    message = "Invalid state: bias == 'normal'"
                    raise ValueError(message)
                m.bias.data.normal_()
        glorot_uniform(m.weight.data)


def weights_init(m, bias: str = "zero") -> None:
    """Compute weights init."""
    for p in m.modules():
        if isinstance(p, (nn.ParameterList, nn.ModuleList)):
            for pp in p:
                _param_init(pp, bias)
        else:
            _param_init(p, bias)

    for name, p in m.named_parameters():
        if "." not in name:  # top-level parameters
            _param_init(p, bias)


def permute_last_dims(t: torch.Tensor, dims: Sequence[int]):
    """Permute tensor on last dims, all other dims are kept unchanged.

    Args:
        t (torch.Tensor): Input tensor with at least len(dims) dimensions.
        dims: The desired ordering of dimensions, here all values should be < 0, i.e.
            (-1, -2) means permute last two dims.

    """
    num_dims = len(t.shape)
    prefix_dims = list(range(num_dims - len(dims)))
    last_dims = [num_dims + d for d in dims]
    return torch.permute(t, prefix_dims + last_dims)


def flatten_tensors(tensors) -> torch.Tensor:
    """Flatten a list of tensors into a single 1D tensor."""
    return torch.cat([t.view(-1) for t in tensors], dim=0)


def unflatten_tensors(flat_tensor, shapes):
    """Unflatten a 1D tensor into a list of tensors with given shapes."""
    tensors = []
    offset = 0
    for shape in shapes:
        numel = shape.numel()
        tensors.append(flat_tensor[offset : offset + numel].view(shape))
        offset += numel
    return tensors


def collate_fn_first(x):
    """Compute collate fn first."""
    return x[0]
