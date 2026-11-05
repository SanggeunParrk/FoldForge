"""Public lifetime of a model-bound trunk result for test-time sampling."""

# ruff: noqa: TC001, TC002, TC003 - public runtime type introspection

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Self

import torch

from foldforge.errors import TrunkStateError
from foldforge.lifecycle import CallGuard
from foldforge.models.io.runtime import TrunkRecord


class TrunkState:
    """A retained trunk on its predictor's GPU; close it when sampling is finished.

    The embeddings are borrowed read-only tensors. Sampling returns independent
    CPU results. States cannot be moved to another predictor or used after close.
    """

    def __init__(
        self, owner: object, model: str, device: torch.device, record: TrunkRecord
    ) -> None:
        self._owner = owner
        self._model = model
        self._device = device
        self._record: TrunkRecord | None = record
        self._report = deepcopy(record.report)
        self._guard = CallGuard()

    def __repr__(self) -> str:
        return (
            f"TrunkState(model={self.model!r}, device={str(self.device)!r}, "
            f"closed={self.closed})"
        )

    @property
    def model(self) -> str:
        """Registered model that produced this state."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Device on which the retained tensors live."""
        return self._device

    @property
    def closed(self) -> bool:
        """Whether the retained trunk has been released."""
        return self._record is None

    @property
    def embeddings(self) -> Mapping[str, torch.Tensor]:
        """Read-only mapping of residue trunk tensors; do not mutate the tensors."""
        return MappingProxyType(self._require().value.embeddings)

    @property
    def report(self) -> dict[str, Any]:
        """Detached trunk execution metadata, including its seed and pass count."""
        return deepcopy(self._report)

    def _require(self) -> TrunkRecord:
        if self._record is None:
            message = "This trunk state is closed"
            raise TrunkStateError(message)
        return self._record

    def close(self) -> None:
        """Release retained GPU tensors and decoding features."""
        with self._guard.acquire():
            self._record = None

    def __enter__(self) -> Self:
        """Retain a trunk within an explicit sampling scope."""
        self._require()
        return self

    def __exit__(self, *_: object) -> None:
        """Release the trunk on context exit."""
        self.close()
