"""Locating and running the language model a family's token stream needs."""

from __future__ import annotations

from pathlib import Path

import torch

from foldforge.modules import language_model as lm

#: Vocabulary and released file name of each tower, by the name a spec declares.
_TOWERS = {
    "esm2": (lm.ESM2_VOCAB, "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"),
}


def tower_path(name: str, checkpoint: Path) -> Path:
    """Return the tower beside the model's own weights.

    It lives next to them rather than at a path of its own so that a checkpoint
    directory is self-contained: moving the weights moves the tower with them.
    """
    try:
        _, relative = _TOWERS[name]
    except KeyError:
        message = f"unknown language model {name!r}; known: {', '.join(_TOWERS)}"
        raise KeyError(message) from None
    root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    path = root / relative
    if not path.is_file():
        message = (
            f"{name} is most of this model's token stream and is not at {path}; "
            "folding without it is a different model"
        )
        raise FileNotFoundError(message)
    return path


def token_embeddings(
    name: str,
    checkpoint: Path | str | None,
    *,
    aatype: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the tower once and return one embedding per structure token."""
    if checkpoint is None:
        message = f"{name} lives beside the model's weights; none were given"
        raise ValueError(message)
    vocab, _ = _TOWERS[name]
    path = tower_path(name, Path(checkpoint))
    model = lm.load_traced(path, device=aatype.device)
    with torch.no_grad():
        out = lm.embed_chains(model, aatype, asym_id, mask, vocab)
    del model
    torch.cuda.empty_cache()
    return out.to(dtype)
