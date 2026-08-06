"""The one output type every predictor in this repo returns.

Each upstream predictor returns its own object with its own field names for the
same six quantities. Comparing two of them then means remembering which one
calls the confidence ``plddt`` and which nests it under ``confidence``, and every
downstream tool grows a branch per model. This type is the seam that stops that:
adapters live in each model package, and everything after the fold — writing a
CIF, computing RMSD, plotting, benchmarking — sees only this.

Deliberately a plain frozen dataclass and not a Protocol-with-methods: it is
data, and the operations on it (write, score, plot) differ per use, not per
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Prediction:
    """One predictor's output for one target.

    Only ``coords`` is required. Every other field is optional because the
    predictors genuinely differ in what they emit — Boltz-2 and Chai-1 have no
    distogram head worth reading, and a model run with its confidence head
    disabled has no pLDDT. ``None`` means "this model did not produce it", which
    is information; a zero tensor would be a lie.
    """

    #: (n_samples, n_atoms, 3) — always sample-major, even for a single sample.
    coords: torch.Tensor
    #: (n_samples, n_tokens) predicted lDDT in [0, 1].
    plddt: torch.Tensor | None = None
    #: (n_samples, n_tokens, n_tokens) predicted aligned error, angstroms.
    pae: torch.Tensor | None = None
    #: (n_samples,) predicted TM-score.
    ptm: torch.Tensor | None = None
    #: (n_samples,) interface predicted TM-score; None for a monomer.
    iptm: torch.Tensor | None = None
    #: (n_tokens, n_tokens, n_bins) raw distogram logits.
    distogram_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Reject the shape mistake that otherwise surfaces as a bad structure."""
        expected_dims = 3
        if self.coords.ndim != expected_dims:
            msg = (
                f"coords must be (n_samples, n_atoms, 3), got "
                f"{tuple(self.coords.shape)}. A single sample still needs its "
                f"leading axis — squeezing it here makes every downstream "
                f"consumer guess."
            )
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of diffusion samples in this prediction."""
        return int(self.coords.shape[0])
