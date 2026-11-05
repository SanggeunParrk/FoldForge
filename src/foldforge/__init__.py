"""FoldForge — many structure predictors, one environment.

FoldForge is a *terminal* repo in a two-layer stack: miniworld-engine holds the
fused kernels and the modules that wrap them (since 2.4.2 no architecture, diffusion
or checkpoint adapters: those are ``foldforge.diffusion`` / ``blocks`` / ``layers`` /
``checkpoints``); the full models are assembled here. Nothing in this package should define an op or a general-purpose block —
if you are writing one, it belongs in the engine.
The point of the repo is that AF3, Boltz-2, Chai-1, Protenix, ESMFold2 and
OpenDDE run off *one* set of blocks and one set of kernels, so a change to a
triangle multiplication is a change to all of them at once.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from foldforge.errors import (
    ConcurrentCallError,
    FoldForgeError,
    InputError,
    ModelClosedError,
    TrunkStateError,
)
from foldforge.models import describe, get_model, known_models, load, registered_models
from foldforge.models.checkpoints import available, resolve
from foldforge.models.config import Config

if TYPE_CHECKING:
    from foldforge.api import Predictor, load_model, predict
    from foldforge.inputs import Inputs, load_inputs, protein
    from foldforge.prediction import Prediction
    from foldforge.result import PredictionResult
    from foldforge.stages import TrunkState

_LAZY = {
    "Inputs": "foldforge.inputs",
    "load_inputs": "foldforge.inputs",
    "protein": "foldforge.inputs",
    "Predictor": "foldforge.api",
    "load_model": "foldforge.api",
    "predict": "foldforge.api",
    "Prediction": "foldforge.prediction",
    "PredictionResult": "foldforge.result",
    "TrunkState": "foldforge.stages",
}

__all__ = [
    "ConcurrentCallError",
    "Config",
    "FoldForgeError",
    "InputError",
    "Inputs",
    "ModelClosedError",
    "Prediction",
    "PredictionResult",
    "Predictor",
    "TrunkState",
    "TrunkStateError",
    "available",
    "describe",
    "get_model",
    "known_models",
    "load",
    "load_inputs",
    "load_model",
    "predict",
    "protein",
    "registered_models",
    "resolve",
]


def __getattr__(name: str) -> Any:
    """Import optional inference dependencies only when their public API is used."""
    if name not in _LAZY:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
