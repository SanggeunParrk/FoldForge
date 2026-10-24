"""The one serving gate every dense module with an engine twin goes through."""

from __future__ import annotations

import torch
from torch import nn

from foldforge.modules.engine.modules import attach
from foldforge.modules.engine.routing import served_by
from foldforge.modules.engine.transformers import serve_atom, serve_token


class _Twin(nn.Module):
    """A dense module with the attribute the gate reads, and nothing else."""

    foldforge_implementation = None


def test_a_module_without_an_engine_twin_is_never_served():
    assert served_by(_Twin(), torch.zeros(2)) is None


def test_the_gate_needs_a_bf16_cuda_activation_and_the_engine_backend():
    twin = _Twin()
    attach(twin, object())
    # No backend requested (the PyTorch reference path), and a CPU float activation.
    assert served_by(twin, torch.zeros(2)) is None
    assert served_by(twin, torch.zeros(2, dtype=torch.bfloat16)) is None


def test_an_unattached_transformer_hands_back_to_the_dense_blocks():
    """None is the dense blocks' cue."""
    twin = _Twin()
    twin.window_half = None
    act, mask, cond, pair = (torch.zeros(4, 8) for _ in range(4))
    assert serve_token(twin, act, mask, cond, pair, None, None) is None
    atoms = torch.zeros(1, 32, 8)
    args = (twin, atoms, atoms[..., 0], atoms, atoms, None, None, None, None)
    assert serve_atom(*args) is None


def test_the_token_gate_refuses_a_cpu_activation_even_with_a_twin():
    twin = _Twin()
    attach(twin, [object()])
    act, mask, cond, pair = (torch.zeros(4, 8) for _ in range(4))
    assert serve_token(twin, act, mask, cond, pair, None, None) is None
