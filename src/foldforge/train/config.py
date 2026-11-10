"""The finetuning recipe. Values are MiniWorld's (configs/miniworld/phase2a_diffusion_medium_v200_b200_joint.yaml chain,
loss/diffusion_v102_ft.yaml, train/default.yaml), not Protenix's finetune_demo.sh.

WARNING -- learning rate. ``max_lr=5e-3`` is MiniWorld's value for a FROZEN trunk plus a freshly initialised diffusion head
(docs/pipeline.md there). Applied to a pretrained, UNFROZEN Protenix trunk it can destroy the weights. Keep the recipe value as the
default, watch the first steps (``scripts/train_protenix1.py --smoke`` reports the loss curve) and lower ``trunk_lr_scale`` /
``max_lr`` if it diverges.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LossWeights:
    """loss/diffusion_v102_ft.yaml + ``loss.distogram_loss``."""

    diffusion_loss: float = 4.0
    alpha_dna: float = 5.0
    alpha_rna: float = 5.0
    alpha_ligand: float = 10.0
    #: 0.0 in the finetune loss file: not implemented in phase 1 (MiniWorld uses the engine's ``smooth_lddt`` kernel).
    smooth_lddt_loss: float = 0.0
    bond_loss: float = 1.0
    distogram_loss: float = 0.03


@dataclass(frozen=True)
class EDMNoise:
    """EDM AF3 noise distribution (MiniWorld: P_mean -1.2, P_std 1.5, sigma_data 16, sigma_max 160, sigma_min 4e-4)."""

    P_mean: float = -1.2
    P_std: float = 1.5
    sigma_data: float = 16.0
    sigma_max: float = 160.0
    sigma_min: float = 4e-4
    translation_noise: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    loss: LossWeights = field(default_factory=LossWeights)
    noise: EDMNoise = field(default_factory=EDMNoise)
    #: Diffusion samples (augmentations) per micro step.
    num_augment: int = 48
    #: Total trunk passes per step, drawn uniformly from this range; only the last pass carries gradient.
    recycles_min: int = 1
    recycles_max: int = 4
    #: Optimizer: torch Adam(betas=(0.9, 0.95)); LR = max_lr * lambda(step) (linear warmup, then x decay_factor every decay_steps), floor min_lr.
    max_lr: float = 5e-3
    min_lr: float = 1e-4
    warmup_steps: int = 1000
    decay_steps: int = 50_000
    decay_factor: float = 0.95
    betas: tuple[float, float] = (0.9, 0.95)
    #: Multiplier of the LR of the trunk (input embedder, template / MSA / pairformer stack); the diffusion head keeps 1.0.
    trunk_lr_scale: float = 1.0
    grad_clip_max_norm: float = 1.0
    #: Micro steps per optimizer step (effective batch = grad_accum_steps x world size).
    grad_accum_steps: int = 32
    ema_decay: float = 0.999
    use_ema: bool = True
    #: Freeze the trunk (MiniWorld: ``model.freeze_trunk: false``).
    freeze_trunk: bool = False
    seed: int = 0
