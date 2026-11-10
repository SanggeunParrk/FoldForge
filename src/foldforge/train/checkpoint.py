"""Native (miniworld-format) checkpoint save / resume.

The model file is the format of :mod:`foldforge.models.checkpoints.miniworld` (``{"format": "miniworld", "version": 1, "family", "source",
"state_dict"}``, an fp32 state_dict with the graph's own keys), loadable by ``ff.load_model(checkpoint=...)``. The training state (optimizer,
EMA, step) is a separate ``<name>.train.pt`` next to it. Converting back to the released ``protenix1.bin.zst`` layout is NOT supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from foldforge.models.checkpoints import miniworld

if TYPE_CHECKING:
    from torch import nn

    from foldforge.train.optim import EMA


def save(
    model: nn.Module,
    path: Path,
    *,
    family: str,
    source: str,
    ema: EMA | None = None,
    optimizer: Any = None,
    step: int = 0,
    ema_weights: bool = False,
) -> Path:
    """Write ``path`` (``*.miniworld.pt``) and ``path.with_suffix('.train.pt')``-style training state. ``ema_weights`` saves the EMA as the model."""
    path = Path(path)
    if not path.name.endswith(miniworld.SUFFIX):
        message = f"{path} must end with {miniworld.SUFFIX}"
        raise ValueError(message)
    if ema_weights and ema is not None:
        ema.swap_in(model)
    try:
        state = {
            # copy=True: on a CPU fp32 model `.to` returns the parameter itself, and `ema.swap_out` below would then rewrite the saved tensor
            k: v.detach().to("cpu", torch.float32, copy=True)
            if v.is_floating_point()
            else v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
    finally:
        if ema_weights and ema is not None:
            ema.swap_out(model)
    torch.save(
        {
            "format": miniworld.FORMAT,
            "version": miniworld.VERSION,
            "family": family,
            "source": source,
            "state_dict": state,
        },
        path,
    )
    if optimizer is not None or ema is not None:
        train_state = {
            "step": step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
        }
        torch.save(train_state, str(path).removesuffix(miniworld.SUFFIX) + ".train.pt")
    return path


def resume(
    model: nn.Module, path: Path, *, optimizer: Any = None, ema: EMA | None = None
) -> int:
    """Load a file written by :func:`save` (model + the ``.train.pt`` beside it); returns the step."""
    blob = miniworld.read(Path(path))
    model.load_state_dict(blob["state_dict"], strict=True)
    train_path = Path(str(path).removesuffix(miniworld.SUFFIX) + ".train.pt")
    if not train_path.exists():
        return 0
    state = torch.load(train_path, map_location="cpu", weights_only=False)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if ema is not None and state.get("ema") is not None:
        for name, tensor in state["ema"].items():
            ema.shadow[name].copy_(tensor)
    return int(state.get("step", 0))
