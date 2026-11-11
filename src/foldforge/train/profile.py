"""Per-phase GPU timing and memory of a training step (off unless enabled: no event is created, no sync is made).

``PhaseTimer.phase(name)`` times a block on the CURRENT stream (the mini-rollout of the side-stream mode records on its own stream) and, with
``memory=True``, records the peak allocated / reserved memory of the block (``reset_peak_memory_stats`` at its start); ``mark`` / ``between`` time
the gap between two recorded points (the backward is split at the moment the gradient reaches the trunk output) and ``mem_point`` records the peak
since the previous point. ``result()`` synchronises once and returns ``{name: ms}`` plus ``{name + ":peak_gib"}`` / ``":start_gib"`` entries.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

GIB = 2**30


class PhaseTimer:
    def __init__(self, enabled: bool = False, memory: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self.memory = memory and self.enabled
        self._spans: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._marks: dict[str, torch.cuda.Event] = {}
        self._mem: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # peak allocated, peak reserved, start allocated

    def _event(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def mem_reset(self) -> None:
        if self.memory:
            torch.cuda.reset_peak_memory_stats()

    def mem_point(self, name: str, start: float | None = None) -> None:
        """Record the peak allocated memory since the last reset under ``name`` (max over repeats) and reset the peak counters."""
        if not self.memory:
            return
        row = self._mem[name]
        row[0] = max(row[0], torch.cuda.max_memory_allocated() / GIB)
        row[1] = max(row[1], torch.cuda.memory_reserved() / GIB)
        if start is not None:
            row[2] = max(row[2], start)
        torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start_mem = torch.cuda.memory_allocated() / GIB if self.memory else 0.0
        self.mem_reset()
        start = self._event()
        try:
            yield
        finally:
            self._spans.append((name, start, self._event()))
            self.mem_point(name, start_mem)

    def mark(self, name: str) -> None:
        if self.enabled:
            self._marks[name] = self._event()

    def between(self, name: str, first: str, last: str) -> None:
        if self.enabled and first in self._marks and last in self._marks:
            self._spans.append((name, self._marks[first], self._marks[last]))

    def result(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        out: dict[str, float] = defaultdict(float)
        for name, start, end in self._spans:
            out[name] += start.elapsed_time(end)
        for name, (peak, reserved, start_mem) in self._mem.items():
            out[f"{name}:peak_gib"] = peak
            out[f"{name}:reserved_gib"] = reserved
            out[f"{name}:start_gib"] = start_mem
        return dict(out)


def arm_trunk_grad_marker(timer: PhaseTimer, tensors: list[torch.Tensor], name: str = "trunk_grad_ready") -> None:
    """Mark ``name`` (and close the memory segment ``diffusion_train_bwd``) when the gradient of every tensor in ``tensors`` (the trunk outputs the
    heads read) has been computed: the backward before it is the heads' (diffusion, distogram), the rest is the trunk's."""
    if not timer.enabled:
        return
    pending = {"n": len(tensors)}

    def hook(grad: torch.Tensor) -> None:
        pending["n"] -= 1
        if pending["n"] == 0:
            timer.mark(name)
            timer.mem_point("diffusion_train_bwd")

    for tensor in tensors:
        if tensor.requires_grad:
            tensor.register_hook(hook)
        else:
            pending["n"] -= 1


def tensor_bytes(tree) -> int:
    """Bytes of every tensor in a (nested) dict / list / tuple."""
    if torch.is_tensor(tree):
        return tree.numel() * tree.element_size()
    if isinstance(tree, dict):
        return sum(tensor_bytes(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(tensor_bytes(v) for v in tree)
    return 0
