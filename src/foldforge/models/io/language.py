"""Locating and running the language model a family's token stream needs."""

from __future__ import annotations

from pathlib import Path

import torch

from foldforge.modules import language_model as lm

#: Vocabulary and released file name of each tower, by the name a spec declares.
_TOWERS = {
    "esm2": (lm.ESM2_VOCAB, "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"),
    # ESM-C is a released directory rather than one traced file, and it is
    # shared across a family's releases, so it resolves through the checkpoint
    # registry instead of sitting beside one model's weights.
    "esmc": (lm.ESMC_VOCAB, None),
}

#: Which families read the tower as a PAIR through a shim, rather than as one
#: embedding per token. The two are different graphs, not two settings.
_PAIR_TOWERS = frozenset({"esmc"})


def reads_pair(name: str) -> bool:
    """Whether ``name``'s output reaches the trunk as a pair representation."""
    return name in _PAIR_TOWERS


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
    if relative is None:
        message = f"{name} is not a file beside the weights; it has its own entry"
        raise ValueError(message)
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


def pair_representation(
    name: str,
    checkpoint: Path | str | None,
    *,
    aatype: torch.Tensor,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    mol_type: torch.Tensor,
    mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the tower and its shim once, returning the trunk's pair injection.

    The shim is trained PER MODEL where the tower is shared across a family's
    releases, so it is loaded from beside this model's own weights. Feeding one
    release another's reads correlation 0.03 against its native output, which
    is exactly the mistake one shared filename invites.
    """
    # The tower pulls in a pinned Transformers fork; import it only when a
    # family actually asks for one.
    from foldforge.models import checkpoints  # noqa: PLC0415
    from foldforge.modules import language_model as tower  # noqa: PLC0415

    if checkpoint is None:
        message = f"{name}'s shim lives beside the model's weights; none were given"
        raise ValueError(message)
    root = Path(checkpoint)
    root = root if root.is_dir() else root.parent
    shims = sorted(root.glob("*.lm.npz"))
    if len(shims) != 1:
        message = (
            f"expected exactly one {name} shim (*.lm.npz) in {root}, found "
            f"{len(shims)}; the shim is trained per release and cannot be shared"
        )
        raise FileNotFoundError(message)

    tower = tower.load_esmc(
        str(checkpoints.resolve("esmc-6b")), device=aatype.device, dtype=dtype
    )
    try:
        with torch.no_grad():
            hidden = tower.compute_lm_hidden_states(
                tower, aatype, asym_id, residue_index, mol_type, mask
            )
    finally:
        del tower
        torch.cuda.empty_cache()

    shim = lm.load_pair_shim(shims[0]).to(aatype.device)
    with torch.no_grad():
        pair = shim(hidden.squeeze(0) if hidden.ndim == 4 else hidden)  # noqa: PLR2004 - a leading batch axis or not
    return pair.to(dtype)
