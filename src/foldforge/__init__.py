"""FoldForge — many structure predictors, one environment.

FoldForge is a *terminal* repo in the three-layer stack described by
``libs/team-gm/docs/ARCHITECTURE.md``: fused ops come from miniworld-engine, the
representative AF3 blocks come from team-gm, and the full models are assembled
here. Nothing in this package should define an op or a general-purpose block —
if you are writing one, it belongs a layer down.

The point of the repo is that AF3, Boltz-2, Chai-1, Protenix, ESMFold2 and
OpenDDE run off *one* set of blocks and one set of kernels, so a change to a
triangle multiplication is a change to all of them at once.
"""

from foldforge.checkpoints import available, resolve
from foldforge.models import describe, get_model, known_models, registered_models
from foldforge.prediction import Prediction

__all__ = [
    "Prediction",
    "available",
    "describe",
    "get_model",
    "known_models",
    "registered_models",
    "resolve",
]
