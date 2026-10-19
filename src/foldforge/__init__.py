"""FoldForge — many structure predictors, one environment.

FoldForge is a *terminal* repo in a two-layer stack: miniworld-engine holds the
fused ops and the representative AF3 blocks, diffusion and checkpoint-layout
primitives (``miniworld_engine.architecture``); the full models are assembled
here. Nothing in this package should define an op or a general-purpose block —
if you are writing one, it belongs in the engine.
The point of the repo is that AF3, Boltz-2, Chai-1, Protenix, ESMFold2 and
OpenDDE run off *one* set of blocks and one set of kernels, so a change to a
triangle multiplication is a change to all of them at once.
"""

from foldforge.models import describe, get_model, known_models, load, registered_models
from foldforge.models.checkpoints import available, resolve
from foldforge.prediction import Prediction

__all__ = [
    "Prediction",
    "available",
    "describe",
    "get_model",
    "known_models",
    "load",
    "registered_models",
    "resolve",
]
