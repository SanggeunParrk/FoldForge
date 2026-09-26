"""The language model reads its own alphabet, not the structure-side residue types."""

from __future__ import annotations

from typing import Any

import torch

from foldforge.models.io import language
from foldforge.modules import language_model as tower


class _Shim(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.sum(-2)


def test_esmc_pair_path_hands_the_tower_vocabulary_ids(tmp_path, monkeypatch):
    (tmp_path / "model.lm.npz").write_bytes(b"")
    seen: dict[str, torch.Tensor] = {}

    def hidden(
        _model: object, input_ids: torch.Tensor, *_: Any, **__: Any
    ) -> torch.Tensor:
        seen["ids"] = input_ids.clone()
        return torch.zeros(*input_ids.shape, 2, 4)

    monkeypatch.setattr(tower, "load_esmc", lambda *_, **__: object())
    monkeypatch.setattr(tower, "compute_lm_hidden_states", hidden)
    monkeypatch.setattr(tower, "load_pair_shim", lambda _: _Shim())
    # A, E, C, S, V in the structure-side (alphabetical three-letter) order.
    aatype = torch.tensor([0, 6, 4, 15, 19])
    ones = torch.ones(5, dtype=torch.bool)
    language.pair_representation(
        "esmc",
        tmp_path,
        aatype=aatype,
        asym_id=torch.zeros(5, dtype=torch.long),
        residue_index=torch.arange(5),
        is_protein=ones,
        mask=ones,
        dtype=torch.float32,
    )
    letters = [tower.ESMC_VOCAB[i] for i in seen["ids"].reshape(-1).tolist()]
    assert letters == list("AECSV")
