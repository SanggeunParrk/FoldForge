"""Atomic object ownership and serialized access to process-wide inference policy."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from foldforge.errors import ConcurrentCallError

# Engine settings, chemistry scopes, Torch precision and RNGs are process-global.
# Distinct predictors may be used from threads, but their operations serialize.
PROCESS_LOCK = RLock()


class CallGuard:
    """Reject overlapping use of the same object before any input preparation."""

    def __init__(self) -> None:
        self._lock = Lock()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        """Hold an atomic, non-reentrant operation lease."""
        if not self._lock.acquire(blocking=False):
            message = "This object is already in use by another operation"
            raise ConcurrentCallError(message)
        try:
            yield
        finally:
            self._lock.release()
