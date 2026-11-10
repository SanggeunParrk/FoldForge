"""The batch protocol and the plug-in point for a data loader.

FoldForge does not import MiniWorld. A training data source is ``module:callable``; the callable receives ``(cfg, device)`` and returns an
iterator of :class:`TrainBatch` (one example = batch size 1, as MiniWorld's ``num_batch: 1``). MiniWorld (or any loader) provides the
adapter: it turns its crop / MSA / template sampling and ground-truth atoms into the dense layout of FoldForge's featurisation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import torch

    from foldforge.train.config import TrainConfig


@dataclass
class TrainBatch:
    """One training example on the GPU, in FoldForge's dense layout ``[res, atoms_per_token]``."""

    #: The model's feature dict, exactly what ``AlphaFold3.run_trunk`` takes (FoldForge featurisation; MSA pool, templates, ref conformers).
    features: dict[str, torch.Tensor]
    #: Ground-truth atom positions ``[res, atoms, 3]`` in the layout of ``features['pred_dense_atom_mask']``.
    atom_positions: torch.Tensor
    #: Resolved-atom mask ``[res, atoms]`` (bool); ``None`` = ``features['pred_dense_atom_mask']``.
    atom_mask: torch.Tensor | None = None
    #: Covalent bonds for the bond loss ``[P, 4]`` int ``(token_i, atom_i, token_j, atom_j)``, ``-1`` padded; ``None`` = no bond term.
    bond_pairs: torch.Tensor | None = None
    name: str = ""


BatchSource = "Callable[[TrainConfig, torch.device], Iterator[TrainBatch]]"


def load_source(
    spec: str,
) -> Callable[[TrainConfig, torch.device], Iterator[TrainBatch]]:
    """Resolve ``module:callable`` (the callable may be a generator function or return any iterator)."""
    module, _, name = spec.partition(":")
    if not module or not name:
        message = f"--batch-source must be 'module:callable', got {spec!r}"
        raise ValueError(message)
    return getattr(importlib.import_module(module), name)
