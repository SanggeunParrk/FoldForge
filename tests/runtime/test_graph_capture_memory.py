"""A CUDA-graph capture that runs out of memory stays an OutOfMemoryError, and a capture makes room first."""

from types import SimpleNamespace

import pytest
import torch

from foldforge.models import execution
from foldforge.models.execution import ExecutedCallable


def _callable(max_graphs=2):
    config = SimpleNamespace(compile=False, cuda_graph=True, max_graphs=max_graphs)
    return ExecutedCallable(lambda x: x * 2, config, "test.site")


def test_an_out_of_memory_capture_stays_an_out_of_memory_error(monkeypatch):
    def fail(*args, **kwargs):
        message = "CUDA out of memory. Tried to allocate 512.00 MiB."
        raise torch.OutOfMemoryError(message)

    monkeypatch.setattr(execution, "_Graph", fail)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(ExecutedCallable, "_make_room", lambda self: None)
    graphed = _callable()
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError) as caught:
        graphed(torch.ones(4, dtype=torch.int64))  # a small integer CPU tensor is the one CPU input a graph signature accepts
    assert "test.site" in str(caught.value)
    assert isinstance(caught.value.__cause__, torch.OutOfMemoryError)


def test_another_capture_failure_is_still_a_runtime_error(monkeypatch):
    def fail(*args, **kwargs):
        message = "illegal capture"
        raise ValueError(message)

    monkeypatch.setattr(execution, "_Graph", fail)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(ExecutedCallable, "_make_room", lambda self: None)
    graphed = _callable()
    with torch.no_grad(), pytest.raises(RuntimeError, match="capture failed at test.site"):
        graphed(torch.ones(4, dtype=torch.int64))


def test_make_room_drops_the_oldest_graphs_until_enough_memory_is_free(monkeypatch):
    graphed = _callable(max_graphs=8)
    for key in ("a", "b", "c"):
        graphed.graphs[key] = object()
    graphed.last_capture_bytes = 100
    free = {"value": 10}

    def mem_get_info(*args, **kwargs):
        return free["value"], 1000

    def empty_cache():
        free["value"] += 80  # every dropped graph frees 80 bytes (empty_cache runs after each pop)

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    graphed._make_room()  # noqa: SLF001 - the unit under test
    assert list(graphed.graphs) == ["c"]  # needs 150: 10 -> 90 -> 170 after two pops
    assert graphed.evictions == 2


def test_make_room_does_nothing_before_the_first_capture_or_with_enough_memory(monkeypatch):
    graphed = _callable()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (0, 1000))
    graphed._make_room()  # noqa: SLF001 - no graphs, no capture size yet
    graphed.graphs["a"] = object()
    graphed._make_room()  # noqa: SLF001 - capture size still unknown
    graphed.last_capture_bytes = 100
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (1000, 1000))
    graphed._make_room()  # noqa: SLF001
    assert list(graphed.graphs) == ["a"]
    assert graphed.evictions == 0
