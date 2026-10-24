"""Constant tables placed on the GPU once, so a CUDA-graph capture never copies from the host.

A module-level table lives on the CPU; ``table.to(device)`` inside a forward is a host-to-device
copy on every call, and a copy from pageable host memory cannot be captured in a CUDA graph. The
graph's warmup calls run outside the capture, so the first call fills this cache and the captured
call only reads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

_CACHE: dict[tuple, tuple[object, torch.Tensor]] = {}


def on_device(table: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """``table`` on ``device``, copied once per (table, device)."""
    key = (id(table), str(device))
    hit = _CACHE.get(key)
    if hit is None:
        # The table is kept alive with its copy so its id cannot be reused.
        hit = _CACHE[key] = (table, table.to(device))
    return hit[1]


def built_on_device(name: str, device: torch.device | str, build: Callable[[], torch.Tensor]) -> torch.Tensor:
    """A table made by ``build()`` (on the host or the device), made once per (name, device)."""
    key = (name, str(device))
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = (None, build())
    return hit[1]
