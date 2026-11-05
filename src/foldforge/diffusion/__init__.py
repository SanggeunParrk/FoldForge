"""Diffusion processes (EDM, decoupled x-prediction), augmentation, schedules, samplers/solvers and the geometry they need.

Self-contained on purpose: nothing here imports another foldforge module, so the folder can be moved out into its own package.

base/             — abstract Diffuser / DiffusionScheduler / DiffusionSolver
edm/              — default Euclidean EDM (eps-prediction) with Karras/AF3
preconditioning (c_noise = log(sigma/sigma_data)/4)
decoupled_xpred/  — decoupled x-prediction (VE, separate R/T noise); the
diffuser used by the MiniWorld folding model + inference
"""

from foldforge.diffusion.base import (
    AtomChainMap,
    Diffuser,
    DiffusionScheduler,
    DiffusionSolver,
    ModelFn,
)
from foldforge.diffusion.decoupled_xpred import (
    DecoupledXPredScheduler,
    XPredDecoupledDiffuser,
    XPredDecoupledSolver,
)
from foldforge.diffusion.edm import (
    AF3Solver,
    EDMScheduler,
    EuclideanDiffuser,
)
from foldforge.diffusion.edm.preconditioned import (
    PreconditionedEulerSolver,
)
from foldforge.diffusion.edm.sampling import (
    EulerSampler,
    PreconditionedDenoiser,
)
from foldforge.diffusion.edm.schedules import (
    LogNormalNoise,
    PowerLawSchedule,
)

__all__ = [
    "AF3Solver",
    "AtomChainMap",
    "DecoupledXPredScheduler",
    "Diffuser",
    "DiffusionScheduler",
    "DiffusionSolver",
    "EDMScheduler",
    "EuclideanDiffuser",
    "EulerSampler",
    "LogNormalNoise",
    "ModelFn",
    "PowerLawSchedule",
    "PreconditionedDenoiser",
    "PreconditionedEulerSolver",
    "XPredDecoupledDiffuser",
    "XPredDecoupledSolver",
]
