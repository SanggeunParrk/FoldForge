from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 ByteDance and/or its affiliates.
# Copyright (c) 2026 Aureka AI Research
import gc
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


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
