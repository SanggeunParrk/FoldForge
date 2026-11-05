"""Scoped compatibility bridge for engine releases without a public cache setter."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def cache_directory(path: Path | None) -> Iterator[None]:
    """Restore the engine's process-local cache selection even when a fold fails."""
    if path is None:
        yield
        return
    from miniworld_engine.autotune import cache

    previous = cache._CACHE_ROOT
    previous_loaded = cache._load_cache.copy()
    try:
        cache._CACHE_ROOT = path
        cache._load_cache.clear()
        yield
    finally:
        cache._CACHE_ROOT = previous
        cache._load_cache.clear()
        cache._load_cache.update(previous_loaded)


@contextmanager
def inference_settings() -> Iterator[None]:
    """Restore process-local precision/engine policy after library inference."""
    import torch
    from miniworld_engine import settings

    precision = torch.get_float32_matmul_precision()
    previous = settings.current()
    try:
        yield
    finally:
        settings.configure(
            **{field.name: getattr(previous, field.name) for field in fields(previous)}
        )
        torch.set_float32_matmul_precision(precision)
