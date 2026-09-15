"""ESMFold2 configuration, mirroring the released ``config.json``.

Field names and defaults follow the ``biohub/ESMFold2`` release checkpoint, not
the upstream dataclass defaults — those two disagree in several places (trunk
depth 48 vs 24, ``msa_head_width`` 16 vs 32, 14 sampling steps vs 68). Load a
checkpoint's own ``config.json`` with :meth:`ESMFold2Config.from_json` rather
than relying on the defaults here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FoldingTrunkConfig(BaseModel):
    """Pair-track trunk depth.

    ``n_heads`` and ``dropout`` are carried for checkpoint fidelity: the release
    trunk is pair-only (no attention) and runs dropout-free at inference, so
    neither reaches the modules.
    """

    n_layers: int = 48
    n_heads: int = 8
    dropout: float = 0.25


class MSAEncoderConfig(BaseModel):
    """Optional MSA encoder that conditions the pair representation."""

    enabled: bool = True
    d_msa: int = 128
    d_hidden: int = 32
    n_layers: int = 4
    n_heads_msa: int = 8
    msa_head_width: int = 16


class LMEncoderConfig(BaseModel):
    """Pair-side encoder applied to the projected language-model features."""

    enabled: bool = True
    n_layers: int = 4
    lm_dropout: float = 0.25
    per_loop_lm_dropout: bool = True


class ParcaeConfig(BaseModel):
    """Linear-recurrent trunk scheduler.

    The Poisson fields describe the training-time loop-count sampler; the
    release inference path uses a deterministic ``num_loops + 1`` steps instead.
    """

    enabled: bool = True
    poisson_mean: float = 3.0
    min_steps: int = 1
    max_steps: int | None = 6
    coda_n_layers: int = 2


class AtomAttentionConfig(BaseModel):
    """Sliding-window atom encoder/decoder with 3D RoPE."""

    d_atom: int = 128
    d_token: int = 768
    n_blocks: int = 3
    n_heads: int = 4
    swa_window_size: int = 128
    expansion_ratio: int = 2
    spatial_rope_base_frequency: float = 20.0
    n_spatial_rope_pairs_per_axis: int = 2
    n_uid_rope_pairs: int = 10
    uid_rope_base_frequency: float = 10000.0


class InputsEmbedderConfig(BaseModel):
    """Token-level input featuriser."""

    d_inputs: int = 451
    atom_encoder: AtomAttentionConfig = Field(default_factory=AtomAttentionConfig)


class DiffusionModuleConfig(BaseModel):
    """Denoiser used by the structure head."""

    sigma_data: float = 16.0
    c_atom: int = 128
    c_token: int = 768
    c_z: int = 256
    c_s_inputs: int = 451
    fourier_dim: int = 256
    relpos_r_max: int = 32
    relpos_s_max: int = 2
    atom_num_blocks: int = 3
    atom_num_heads: int = 4
    token_num_blocks: int = 12
    token_num_heads: int = 16
    transition_multiplier: int = 2


class StructureHeadConfig(BaseModel):
    """Diffusion structure head, including the inference ODE schedule."""

    diffusion_module: DiffusionModuleConfig = Field(
        default_factory=DiffusionModuleConfig
    )
    distogram_bins: int = 64
    train_noise_log_mean: float = -1.2
    train_noise_log_std: float = 1.5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    inference_s_max: float = 160.0
    inference_s_min: float = 4e-4
    inference_p: float = 7.0
    inference_num_steps: int = 14


class ConfidenceHeadConfig(BaseModel):
    """pLDDT / PAE / PDE / distogram head."""

    enabled: bool = True
    distogram_bins: int = 39
    min_dist: float = 3.25
    max_dist: float = 50.75
    num_pae_bins: int = 64
    num_pde_bins: int = 64
    num_plddt_bins: int = 50
    folding_trunk: FoldingTrunkConfig = Field(
        default_factory=lambda: FoldingTrunkConfig(n_layers=4)
    )


class ESMFold2Config(BaseModel):
    """Top-level ESMFold2 configuration."""

    d_single: int = 384
    d_pair: int = 256
    n_relative_residx_bins: int = 32
    n_relative_chain_bins: int = 2
    num_loops: int = 3
    num_diffusion_samples: int = 32
    disable_msa_features: bool = False
    msa_encoder_overwrite: bool = True

    esmc_id: str = "biohub/ESMC-6B"
    lm_d_model: int = 2560
    lm_num_layers: int = 80
    lm_dropout: float = 0.0
    force_lm_dropout_during_inference: bool = False

    folding_trunk: FoldingTrunkConfig = Field(default_factory=FoldingTrunkConfig)
    lm_encoder: LMEncoderConfig = Field(default_factory=LMEncoderConfig)
    msa_encoder: MSAEncoderConfig = Field(default_factory=MSAEncoderConfig)
    parcae: ParcaeConfig = Field(default_factory=ParcaeConfig)
    inputs: InputsEmbedderConfig = Field(default_factory=InputsEmbedderConfig)
    structure_head: StructureHeadConfig = Field(default_factory=StructureHeadConfig)
    confidence_head: ConfidenceHeadConfig = Field(default_factory=ConfidenceHeadConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> ESMFold2Config:
        """Build a config from a released checkpoint's ``config.json``.

        Parameters
        ----------
        path : str or Path
            Path to ``config.json``, or to the directory containing it.

        Returns
        -------
        ESMFold2Config
            Config populated from the file; unknown keys (``architectures``,
            ``dtype``, ``transformers_version``, ...) are ignored.

        """
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw: dict[str, Any] = json.loads(path.read_text())
        known = set(cls.model_fields)
        return cls(**{k: v for k, v in raw.items() if k in known})
