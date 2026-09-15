"""Structural interface for dynamically keyed released-model configuration."""

from collections.abc import Iterable
from typing import Any, Protocol


class ConfigNode(Protocol):
    """Describe ConfigDict's dynamic keys without inferring a scalar-only union.

    Released configuration dictionaries may contain further config nodes, tensors,
    paths, callables, and scalars. Concrete validated configuration models retain
    their own field types; this interface is for the dynamic ConfigDict boundary.
    """

    def __getattr__(self, name: str, /) -> Any:
        """Read a dynamically named configuration field."""
        ...

    def __getitem__(self, name: str, /) -> Any:
        """Read a field by key."""
        ...

    def __setitem__(self, name: str, value: Any, /) -> None:
        """Assign a configuration field."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Read an optional field with its default."""
        ...

    def items(self) -> Iterable[tuple[str, Any]]:
        """Iterate over configuration fields."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Export configuration fields as a dictionary."""
        ...
