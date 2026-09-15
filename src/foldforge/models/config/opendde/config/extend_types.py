from __future__ import annotations

from typing import Any


# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
class DefaultNoneWithType:
    """Represent default none with type."""

    def __init__(self, dtype: type | None) -> None:
        self.dtype = dtype


class ValueMaybeNone:
    """Represent value maybe none."""

    def __init__(self, value: Any) -> None:
        if not (value is not None):
            message = "Invalid state: value is not None"
            raise ValueError(message)
        self.dtype = type(value)
        self.value = value


class GlobalConfigValue:
    """Represent global config value."""

    def __init__(self, global_key: str) -> None:
        self.global_key = global_key


class RequiredValue:
    """Represent required value."""

    def __init__(self, dtype: type | None) -> None:
        self.dtype = dtype


class ListValue:
    """Represent list value."""

    def __init__(self, value: Any, dtype: type | None = None) -> None:
        if value:
            self.value = value
            self.dtype = type(value[0])
        elif value is not None:
            # Empty list (e.g. an "unset" default); element type can't be
            # inferred, so fall back to the explicit dtype.
            self.value = value
            self.dtype = dtype
        else:
            self.value = None
            self.dtype = dtype


def get_bool_value(bool_str: bool | str) -> bool:  # noqa: FBT001 - parses a bool-or-string configuration value
    """Return bool value."""
    if isinstance(bool_str, bool):
        return bool_str
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ("false", "f", "no", "n", "0"):
        return False
    if bool_str_lower in ("true", "t", "yes", "y", "1"):
        return True
    msg = f"Cannot interpret {bool_str} as bool"
    raise ValueError(msg)
