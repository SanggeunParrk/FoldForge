# GPU dependencies are imported at the model boundary so registry/help stay cheap.
# ruff: noqa: PLC0415
"""protenix: released-checkpoint model port using shared MiniWorld operations.

See SOURCE.json for the pinned origin and docs/MODEL-INTEGRATION.md for
validated settings and the exact engine coverage.
"""

from __future__ import annotations

from typing import Any


def load(*args: Any, **kwargs: Any) -> Any:
    """Load the model lazily; importing the package does not initialize Torch."""
    from .model import load as load_checkpoint

    return load_checkpoint(*args, **kwargs)
