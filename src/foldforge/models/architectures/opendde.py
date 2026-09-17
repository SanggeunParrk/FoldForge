from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import functools
import operator
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from team_gm.diffusion.edm.schedules import InferenceNoiseScheduler
from team_gm.modules.checkpoints.confidence import ConfidenceHead
from team_gm.modules.checkpoints.denoiser import DiffusionModule
from team_gm.modules.checkpoints.dense import LinearNoBias
from team_gm.modules.checkpoints.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from team_gm.modules.checkpoints.heads import DistogramHead
from team_gm.modules.checkpoints.layers import LayerNorm
from team_gm.modules.checkpoints.policy import BOUNDED_MEMORY_POLICY
from team_gm.modules.checkpoints.stacks import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from team_gm.modules.checkpoints.structural import StructuralTokenExpander
from team_gm.modules.checkpoints.structural_roles import STRUCTURAL_TOKEN_ROLES
from team_gm.modules.checkpoints.tensor_utils import simple_merge_dict_list
from team_gm.modules.checkpoints.trunk import RecycledTrunk, update_input_feature_dict

import foldforge.eval.confidence as sample_confidence
from foldforge.models.sampling import sample_diffusion
from foldforge.training.shape_complementarity import (
    build_shape_comp_pred_outputs,
    compute_shape_complementarity_fields,
    get_shape_comp_atom_mask,
)
from foldforge.utils.logging import get_logger
from foldforge.utils.tensor import autocasting_disable_decorator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldforge.models.config.opendde.config.schema import OpenDDEConfig

logger = get_logger(__name__)


@contextmanager
def _tf32_runtime_scope(*, enabled: bool) -> Iterator[None]:
    """Apply one model's TF32 policy without leaking it to the host process."""
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def _offload_prediction_tree_to_cpu(value: Any) -> Any:
    """Detach prediction leaves from the accelerator between model seeds."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            key: _offload_prediction_tree_to_cpu(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_offload_prediction_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_offload_prediction_tree_to_cpu(item) for item in value)
    return value


class OpenDDE(RecycledTrunk):
    """Implement the OpenDDE prediction loop."""

    def __init__(self, configs: OpenDDEConfig) -> None:
        super().__init__()
        self.configs = configs
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, policy=BOUNDED_MEMORY_POLICY
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(
            **configs.model.template_embedder, policy=BOUNDED_MEMORY_POLICY
        )
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data["msa"],
            policy=BOUNDED_MEMORY_POLICY,
        )
        self.pairformer_stack = PairformerStack(
            **configs.model.pairformer, policy=BOUNDED_MEMORY_POLICY
        )
        diffusion_module_configs = copy.deepcopy(
            configs.model.diffusion_module.to_dict()
        )
        self.diffusion_module = DiffusionModule(
            **diffusion_module_configs, policy=BOUNDED_MEMORY_POLICY
        )
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(
            **configs.model.confidence_head, policy=BOUNDED_MEMORY_POLICY
        )
        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)
        structural_token_expansion_configs = configs.model.structural_token_expansion
        self.enable_structural_token_expansion = (
            structural_token_expansion_configs.enable
        )
        self.pair_output_space = structural_token_expansion_configs.pair_output_space
        if self.pair_output_space not in {"residue", "structural"}:
            msg = (
                f"model.structural_token_expansion.pair_output_space "
                f"must be 'residue' or 'structural'; got {self.pair_output_space!r}"
            )
            raise ValueError(msg)
        structural_refiner_configs = (
            structural_token_expansion_configs.structural_refiner
        )
        self.enable_structural_token_refiner = (
            self.enable_structural_token_expansion and structural_refiner_configs.enable
        )
        if self.enable_structural_token_expansion:
            required_n_roles = max(STRUCTURAL_TOKEN_ROLES.values()) + 1
            configured_n_roles = structural_token_expansion_configs.n_roles
            if configured_n_roles < required_n_roles:
                msg = (
                    f"model.structural_token_expansion.n_roles={configured_n_roles} "
                    f"is too small; need at least {required_n_roles} for "
                    f"structural token roles {STRUCTURAL_TOKEN_ROLES}"
                )
                raise ValueError(msg)
            self.structural_token_expander = StructuralTokenExpander(
                c_s=self.c_s,
                c_z=self.c_z,
                c_s_inputs=self.c_s_inputs,
                n_roles=configured_n_roles,
                init_mode=structural_token_expansion_configs.init_mode,
                role_init_std=structural_token_expansion_configs.role_init_std,
                pair_feature_init_std=structural_token_expansion_configs.pair_feature_init_std,
                attention_bias_init=structural_token_expansion_configs.attention_bias_init,
                pair_projection_mode=structural_token_expansion_configs.pair_projection_mode,
                pair_chunk_size=structural_token_expansion_configs.pair_chunk_size,
            )
            if self.enable_structural_token_refiner:
                self.structural_token_refiner = PairformerStack(
                    n_blocks=structural_refiner_configs.n_blocks,
                    n_heads=structural_refiner_configs.n_heads,
                    c_z=self.c_z,
                    c_s=self.c_s,
                    num_intermediate_factor=structural_refiner_configs.num_intermediate_factor,
                    blocks_per_ckpt=structural_refiner_configs.blocks_per_ckpt,
                    hidden_scale_up=structural_refiner_configs.hidden_scale_up,
                    policy=BOUNDED_MEMORY_POLICY,
                )
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def expand_to_structural_tokens(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        *,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        lazy_relp: bool = False,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand to structural tokens."""
        if not self.enable_structural_token_expansion:
            return (input_feature_dict, s_inputs, s, z)
        required_features = [
            "parent_residue_idx",
            "subtoken_role_id",
            "structural_token_index",
            "atom_to_structural_token_idx",
            "atom_to_structural_tokatom_idx",
            "structural_distogram_rep_atom_mask",
            "structural_pae_rep_atom_mask",
            "structural_has_frame",
            "structural_frame_atom_index",
        ]
        missing_features = [
            key for key in required_features if key not in input_feature_dict
        ]
        if missing_features:
            raise KeyError(
                (
                    "Structural token expansion is enabled, but input_feature_dict is"
                    " missing required structural feature(s): "
                )
                + ", ".join(missing_features)
            )
        structural_feature_dict = dict(input_feature_dict)
        for residue_feature in [
            "token_index",
            "asym_id",
            "residue_index",
            "entity_id",
            "sym_id",
            "atom_to_token_idx",
            "atom_to_tokatom_idx",
            "has_frame",
            "frame_atom_index",
            "pae_rep_atom_mask",
            "distogram_rep_atom_mask",
        ]:
            structural_feature_dict[f"residue_level_{residue_feature}"] = (
                input_feature_dict[residue_feature]
            )
        parent = input_feature_dict["parent_residue_idx"].long()
        s_inputs, s, z, structural_pair_features = self.structural_token_expander(
            input_feature_dict=input_feature_dict,
            s_inputs_res=s_inputs,
            s_res=s,
            z_res=z,
        )
        structural_feature_dict["token_index"] = input_feature_dict[
            "structural_token_index"
        ].long()
        structural_feature_dict["atom_to_token_idx"] = input_feature_dict[
            "atom_to_structural_token_idx"
        ].long()
        structural_feature_dict["atom_to_tokatom_idx"] = input_feature_dict[
            "atom_to_structural_tokatom_idx"
        ].long()
        for token_feature in ["asym_id", "residue_index", "entity_id", "sym_id"]:
            structural_feature_dict[token_feature] = input_feature_dict[
                token_feature
            ].index_select(dim=-1, index=parent)
        structural_feature_dict["has_frame"] = input_feature_dict[
            "structural_has_frame"
        ]
        structural_feature_dict["frame_atom_index"] = input_feature_dict[
            "structural_frame_atom_index"
        ]
        structural_feature_dict["pae_rep_atom_mask"] = input_feature_dict[
            "structural_pae_rep_atom_mask"
        ].long()
        structural_feature_dict["distogram_rep_atom_mask"] = input_feature_dict[
            "structural_distogram_rep_atom_mask"
        ].long()
        structural_feature_dict.update(dict(structural_pair_features.items()))
        structural_feature_dict = self.relative_position_encoding.generate_relp(
            structural_feature_dict, lazy=lazy_relp
        )
        if self.enable_structural_token_refiner:
            s, z = self.structural_token_refiner(
                s=s,
                z=z,
                pair_mask=None,
                triangle_multiplicative=self.configs.triangle_multiplicative,
                triangle_attention=self.configs.triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                extra_attn_bias=structural_feature_dict.get(
                    "structural_pair_attn_bias", None
                ),
            )
        self.drop_residue_only_features_for_structural_branch(structural_feature_dict)
        return (structural_feature_dict, s_inputs, s, z)

    def select_pair_output_branch(
        self,
        residue_feature_dict: dict[str, Any],
        residue_s_inputs: torch.Tensor,
        residue_s: torch.Tensor,
        residue_z: torch.Tensor,
        structural_feature_dict: dict[str, Any],
        structural_s_inputs: torch.Tensor,
        structural_s: torch.Tensor,
        structural_z: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select pair output branch."""
        if (
            self.enable_structural_token_expansion
            and self.pair_output_space == "residue"
        ):
            return (residue_feature_dict, residue_s_inputs, residue_s, residue_z)
        return (
            structural_feature_dict,
            structural_s_inputs,
            structural_s,
            structural_z,
        )

    @staticmethod
    def drop_residue_only_features_for_structural_branch(
        input_feature_dict: dict[str, Any],
    ) -> None:
        """Keep MSA/template strictly residue-level.

        After trunk, structural-token branches consume expanded s/z plus structural
        metadata. Residue-column features are removed from this local branch dict so
        downstream code cannot accidentally treat them as structural-token features.
        """
        residue_only_keys = {
            "msa",
            "has_deletion",
            "deletion_value",
            "msa_mask",
            "profile",
            "deletion_mean",
            "token_bonds",
        }
        for key in list(input_feature_dict.keys()):
            if key in residue_only_keys or key.startswith("template_"):
                input_feature_dict.pop(key, None)

    @staticmethod
    def pool_pair_matrix_to_residue(
        values: torch.Tensor, parent: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        """Compute pool pair matrix to residue."""
        n_struct = parent.numel()
        pair_index = (
            (parent[:, None] * n_residue + parent[None, :])
            .reshape(-1)
            .to(device=values.device)
        )
        flat_values = values.reshape(*values.shape[:-2], n_struct * n_struct)
        prefix_shape = flat_values.shape[:-1]
        out = values.new_zeros(*prefix_shape, n_residue * n_residue)
        index = pair_index.reshape((1,) * len(prefix_shape) + (-1,)).expand(
            *prefix_shape, n_struct * n_struct
        )
        out.scatter_add_(dim=-1, index=index, src=flat_values)
        counts = values.new_zeros(n_residue * n_residue)
        counts.scatter_add_(
            dim=0, index=pair_index, src=values.new_ones(n_struct * n_struct)
        )
        out = out / counts.clamp_min(1).reshape((1,) * len(prefix_shape) + (-1,))
        return out.reshape(*prefix_shape, n_residue, n_residue)

    @staticmethod
    def pool_pair_matrix_to_residue_max(
        values: torch.Tensor, parent: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        """Compute pool pair matrix to residue max."""
        n_struct = parent.numel()
        pair_index = (
            (parent[:, None] * n_residue + parent[None, :])
            .reshape(-1)
            .to(device=values.device)
        )
        flat_values = values.reshape(*values.shape[:-2], n_struct * n_struct)
        prefix_shape = flat_values.shape[:-1]
        out = values.new_full(
            (*prefix_shape, n_residue * n_residue), torch.finfo(values.dtype).min
        )
        index = pair_index.reshape((1,) * len(prefix_shape) + (-1,)).expand(
            *prefix_shape, n_struct * n_struct
        )
        out.scatter_reduce_(
            dim=-1, index=index, src=flat_values, reduce="amax", include_self=True
        )
        return out.reshape(*prefix_shape, n_residue, n_residue)

    @staticmethod
    def pool_pair_distribution_to_residue(
        probs: torch.Tensor, parent: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        """Compute pool pair distribution to residue."""
        n_struct = parent.numel()
        n_bins = probs.shape[-1]
        pair_index = (
            (parent[:, None] * n_residue + parent[None, :])
            .reshape(-1)
            .to(device=probs.device)
        )
        flat_probs = probs.reshape(*probs.shape[:-3], n_struct * n_struct, n_bins)
        prefix_shape = flat_probs.shape[:-2]
        out = probs.new_zeros(*prefix_shape, n_residue * n_residue, n_bins)
        index = pair_index.reshape((1,) * len(prefix_shape) + (-1, 1)).expand(
            *prefix_shape, n_struct * n_struct, n_bins
        )
        out.scatter_add_(dim=-2, index=index, src=flat_probs)
        counts = probs.new_zeros(n_residue * n_residue)
        counts.scatter_add_(
            dim=0, index=pair_index, src=probs.new_ones(n_struct * n_struct)
        )
        out = out / counts.clamp_min(1).reshape((1,) * len(prefix_shape) + (-1, 1))
        return out.reshape(*prefix_shape, n_residue, n_residue, n_bins)

    @staticmethod
    def get_parent_representative_token_idx(
        parent: torch.Tensor, role: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        """Pick the public residue representative in structural-token space.

        Polymer residues prefer their BB token so public PAE/PDE preserve
        backbone-frame residue semantics. Non-polymer/atom-token parents fall
        back to their first structural token.
        """
        backbone_roles = {
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_bb"],
            STRUCTURAL_TOKEN_ROLES["rna_bb"],
        }
        parent_list = parent.detach().cpu().tolist()
        role_list = role.detach().cpu().tolist()
        representative_idx = [-1] * n_residue
        for token_idx, parent_idx in enumerate(parent_list):
            if representative_idx[parent_idx] < 0:
                representative_idx[parent_idx] = token_idx
        for token_idx, (parent_idx, role_id) in enumerate(
            zip(parent_list, role_list, strict=False)
        ):
            if role_id in backbone_roles:
                representative_idx[parent_idx] = token_idx
        if any(token_idx < 0 for token_idx in representative_idx):
            msg = (
                f"Could not find a structural representative "
                f"token for every parent residue: {representative_idx}"
            )
            raise ValueError(msg)
        return torch.tensor(representative_idx, dtype=torch.long, device=parent.device)

    def get_residue_level_confidence_inputs(
        self,
        input_feature_dict: dict[str, Any],
        pae_logits: torch.Tensor,
        pde_logits: torch.Tensor,
        contact_probs: torch.Tensor,
        target_device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return residue level confidence inputs."""
        target_device = target_device or pae_logits.device
        parent = input_feature_dict.get("parent_residue_idx")
        role = input_feature_dict.get("subtoken_role_id")
        residue_level_keys = (
            "residue_level_asym_id",
            "residue_level_has_frame",
            "residue_level_atom_to_token_idx",
        )
        has_residue_level_features = all(
            key in input_feature_dict for key in residue_level_keys
        )
        is_structural_pair_space = (
            parent is not None
            and role is not None
            and has_residue_level_features
            and (pae_logits.shape[-3] == parent.numel())
            and (pae_logits.shape[-2] == parent.numel())
            and (pde_logits.shape[-3] == parent.numel())
            and (pde_logits.shape[-2] == parent.numel())
            and (contact_probs.shape[-2] == parent.numel())
            and (contact_probs.shape[-1] == parent.numel())
        )
        if not is_structural_pair_space or parent is None or role is None:
            return {
                "pae_logits": pae_logits.to(device=target_device),
                "pde_logits": pde_logits.to(device=target_device),
                "contact_probs": contact_probs.to(device=target_device),
                "token_asym_id": input_feature_dict["asym_id"].to(device=target_device),
                "token_has_frame": input_feature_dict["has_frame"].to(
                    device=target_device
                ),
                "atom_to_token_idx": input_feature_dict["atom_to_token_idx"].to(
                    device=target_device
                ),
            }
        parent = parent.long().to(device=pae_logits.device)
        n_residue = int(parent.max().item()) + 1
        representative_idx = self.get_parent_representative_token_idx(
            parent=parent,
            role=role.long().to(device=pae_logits.device),
            n_residue=n_residue,
        )
        residue_pae_logits = pae_logits.index_select(
            dim=-3, index=representative_idx
        ).index_select(dim=-2, index=representative_idx)
        residue_pde_logits = pde_logits.index_select(
            dim=-3, index=representative_idx
        ).index_select(dim=-2, index=representative_idx)
        pooled_contact_probs = self.pool_pair_matrix_to_residue_max(
            values=contact_probs.to(device=pae_logits.device, dtype=torch.float32),
            parent=parent,
            n_residue=n_residue,
        )
        return {
            "pae_logits": residue_pae_logits.to(
                device=target_device, dtype=pae_logits.dtype
            ),
            "pde_logits": residue_pde_logits.to(
                device=target_device, dtype=pde_logits.dtype
            ),
            "contact_probs": pooled_contact_probs.to(
                device=target_device, dtype=contact_probs.dtype
            ),
            "token_asym_id": input_feature_dict["residue_level_asym_id"].to(
                device=target_device
            ),
            "token_has_frame": input_feature_dict["residue_level_has_frame"].to(
                device=target_device
            ),
            "atom_to_token_idx": input_feature_dict[
                "residue_level_atom_to_token_idx"
            ].to(device=target_device),
        }

    def _shape_comp_effective_weight(self, weight_name: str) -> float:
        shape_comp_configs = self.configs.confidence.shape_comp
        return float(self.configs.confidence.weight.alpha_shape_comp) * float(
            getattr(shape_comp_configs, weight_name)
        )

    def _should_store_shape_comp_pair_map(self) -> bool:
        shape_comp_configs = self.configs.confidence.shape_comp
        return bool(shape_comp_configs.debug_pair_map)

    def _should_compute_shape_comp(self) -> bool:
        shape_comp_configs = self.configs.confidence.shape_comp
        return bool(
            self._shape_comp_effective_weight("pair_weight") > 0
            or self._shape_comp_effective_weight("token_weight") > 0
            or self._shape_comp_effective_weight("global_weight") > 0
            or shape_comp_configs.debug_pair_map
        )

    def add_shape_complementarity_predictions(
        self,
        pred_dict: dict[str, torch.Tensor],
        input_feature_dict: dict[str, Any],
        coordinate: torch.Tensor,
        label_dict: dict[str, Any] | None = None,
    ) -> None:
        """Add shape complementarity predictions."""
        if not self._should_compute_shape_comp():
            return
        keep_pair_map = self._should_store_shape_comp_pair_map()
        pred_dict["shape_comp_uses_structural_tokens"] = torch.tensor(
            [int("residue_level_token_index" in input_feature_dict)],
            dtype=torch.long,
            device=coordinate.device,
        )
        shape_comp = autocasting_disable_decorator(disable_casting=True)(
            compute_shape_complementarity_fields
        )(
            coordinate=coordinate,
            feat_dict=input_feature_dict,
            atom_mask=get_shape_comp_atom_mask(
                feat_dict=input_feature_dict, label_dict=label_dict
            ),
            return_pair_map=keep_pair_map,
            **self.configs.confidence.shape_comp,
        )
        pred_dict.update(
            build_shape_comp_pred_outputs(
                shape_comp=shape_comp, keep_pair_map=keep_pair_map
            )
        )

    def run_sample_diffusion_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        cache: dict[str, Any],
        N_sample: int,  # noqa: N803 - checkpoint-compatible keyword
        noise_schedule: Any,
        chunk_size: int | None,
        inplace_safe: bool,
    ) -> torch.Tensor:
        """Compute run sample diffusion stage."""
        sample_pair_z = cache["pair_z"]
        sample_z_trunk = None if sample_pair_z is not None else z
        rollout_seed = input_feature_dict.get("inference_seed")
        if isinstance(rollout_seed, torch.Tensor):
            rollout_seed = int(rollout_seed.detach().cpu().item())
        elif rollout_seed is not None:
            rollout_seed = int(rollout_seed)
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=sample_z_trunk,
            pair_z=sample_pair_z,
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            attn_chunk_size=chunk_size,
            diffusion_chunk_size=self.configs.infer_setting.sample_diffusion_chunk_size,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
            rollout_seed=rollout_seed,
        )
        return pred_dict["coordinate"]

    def sample_diffusion(
        self,
        attn_chunk_size: int | None = None,
        diffusion_chunk_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Sample diffusion process based on the provided configurations.

        Args:
            **kwargs: Additional options forwarded to the selected implementation.
            attn_chunk_size (Optional[int]): Token chunk size used inside
                attention-style blocks.
            diffusion_chunk_size (Optional[int]): Chunk size used to split diffusion
                samples.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.

        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in ["gamma0", "gamma_min", "noise_scale_lambda", "step_scale_eta"]
        }
        sample_diffusion_configs = self.configs.sample_diffusion.to_dict()
        _configs.update(
            {
                "attn_chunk_size": attn_chunk_size,
                "diffusion_chunk_size": diffusion_chunk_size,
                "guidance_configs": sample_diffusion_configs.get("guidance"),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def prepare_diffusion_cache_for_sampling(
        self, *, input_feature_dict: dict[str, Any], z: torch.Tensor
    ) -> dict[str, Any]:
        """Prepare diffusion cache for sampling."""
        cache: dict[str, Any] = {}
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, inplace_safe=False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        return cache

    def compute_distogram_contact_probs(
        self, pair_z: torch.Tensor, pred_dict: dict[str, Any] | None = None
    ) -> torch.Tensor | None:
        """Compute distogram contact probs."""
        bin_params = sample_confidence.get_bin_params(self.configs.confidence.distogram)
        logits = self.distogram_head(pair_z)
        if getattr(self, "save_distogram", False) and pred_dict is not None:
            pred_dict["distogram_logits"] = logits
        return sample_confidence.compact_compute_contact_prob(
            distogram_logits=logits, **bin_params
        )

    def run_distogram_contact_stage(
        self, *, pred_dict: dict[str, Any], pair_z: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute run distogram contact stage."""
        pred_dict["contact_probs"] = autocasting_disable_decorator(
            disable_casting=True
        )(self.compute_distogram_contact_probs)(pair_z, pred_dict)
        return pred_dict["contact_probs"]

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """Compute run confidence head.

        Run the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.

        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def run_confidence_head_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor | None,
        x_pred_coords: torch.Tensor,
        triangle_multiplicative: str,
        triangle_attention: str,
        inplace_safe: bool,
        chunk_size: int | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute run confidence head stage."""
        plddt, pae, pde, resolved = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_mask=pair_mask,
            x_pred_coords=x_pred_coords,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        self.update_confidence_predictions(
            pred_dict=pred_dict, plddt=plddt, pae=pae, pde=pde, resolved=resolved
        )
        return (plddt, pae, pde, resolved)

    @staticmethod
    def update_confidence_predictions(
        pred_dict: dict[str, Any],
        plddt: torch.Tensor | None,
        pae: torch.Tensor | None,
        pde: torch.Tensor | None,
        resolved: torch.Tensor | None,
    ) -> None:
        """Update confidence predictions."""
        pred_dict.update(
            {
                key: value
                for (key, value) in {
                    "plddt": plddt,
                    "pae": pae,
                    "pde": pde,
                    "resolved": resolved,
                }.items()
                if value is not None
            }
        )

    @staticmethod
    def replace_public_pair_logits_with_residue_level(
        pred_dict: dict[str, Any], residue_confidence_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Replace public pair logits with residue level."""
        if (
            pred_dict["pae"].shape[-3]
            == residue_confidence_inputs["pae_logits"].shape[-3]
            and pred_dict["pae"].shape[-2]
            == residue_confidence_inputs["pae_logits"].shape[-2]
        ):
            pred_dict["pae"] = residue_confidence_inputs["pae_logits"]
            pred_dict["pde"] = residue_confidence_inputs["pde_logits"]
        else:
            pred_dict["structural_pae"] = pred_dict["pae"]
            pred_dict["structural_pde"] = pred_dict["pde"]
            pred_dict["pae"] = residue_confidence_inputs["pae_logits"]
            pred_dict["pde"] = residue_confidence_inputs["pde_logits"]
        pred_dict["contact_probs"] = residue_confidence_inputs["contact_probs"]
        pred_dict.pop("per_sample_contact_probs", None)

    def run_post_confidence_outputs_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        pair_input_feature_dict: dict[str, Any],
        N_cycle: int,  # noqa: N803 - checkpoint-compatible keyword
    ) -> dict[str, Any]:
        """Compute run post confidence outputs stage."""
        torch.cuda.empty_cache()
        self.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=pair_input_feature_dict,
            coordinate=pred_dict["coordinate"],
            label_dict=None,
        )
        residue_confidence_inputs = self.get_residue_level_confidence_inputs(
            input_feature_dict=pair_input_feature_dict,
            pae_logits=pred_dict["pae"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            target_device=pred_dict["pae"].device,
        )
        self.replace_public_pair_logits_with_residue_level(
            pred_dict=pred_dict, residue_confidence_inputs=residue_confidence_inputs
        )
        pred_dict["summary_confidence"], pred_dict["full_data"] = (
            autocasting_disable_decorator(disable_casting=True)(
                sample_confidence.compact_compute_full_data_and_summary
            )(
                configs=self.configs,
                pae_logits=residue_confidence_inputs["pae_logits"],
                plddt_logits=pred_dict["plddt"],
                pde_logits=residue_confidence_inputs["pde_logits"],
                contact_probs=residue_confidence_inputs["contact_probs"],
                token_asym_id=residue_confidence_inputs["token_asym_id"],
                token_has_frame=residue_confidence_inputs["token_has_frame"],
                atom_coordinate=pred_dict["coordinate"],
                atom_to_token_idx=residue_confidence_inputs["atom_to_token_idx"],
                atom_is_polymer=1 - input_feature_dict["is_ligand"],
                N_recycle=N_cycle,
                interested_atom_mask=None,
                return_full_data=bool(
                    getattr(self.configs, "need_atom_confidence", False)
                ),
                mol_id=None,
                elements_one_hot=None,
            )
        )
        return pred_dict

    def main_inference_loop(  # noqa: C901 - preserve model-seed tensor lifetime and offload order
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,  # noqa: N803 - checkpoint-compatible keyword
        *,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
        N_model_seed: int = 1,  # noqa: N803 - checkpoint-compatible keyword
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Run the main inference loop, optionally evaluating multiple model seeds.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            N_cycle (int): Number of cycles.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to
                True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations.
                Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction,
            log, and time dictionaries.

        """
        if N_model_seed > 1:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            non_output_rank = False
            last_non_output_prediction = None
            for model_seed_index in range(N_model_seed):
                if non_output_rank and model_seed_index > 0:
                    last_non_output_prediction = None
                seed_input_features = copy.deepcopy(input_feature_dict)
                if seed_input_features is None:
                    msg = f"model-seed {model_seed_index} input was not prepared."
                    raise RuntimeError(msg)
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=seed_input_features,
                    N_cycle=N_cycle,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
                del seed_input_features

                def _retain_model_seed_prediction(
                    *, pred_dict: dict[str, Any] = pred_dict
                ) -> dict[str, Any]:
                    if non_output_rank:
                        return pred_dict
                    return _offload_prediction_tree_to_cpu(pred_dict)

                retained_prediction = _retain_model_seed_prediction()
                if retained_prediction is None:
                    msg = f"model-seed {model_seed_index} prediction was not retained."
                    raise RuntimeError(msg)
                if non_output_rank:
                    last_non_output_prediction = retained_prediction
                else:
                    pred_dicts.append(retained_prediction)
                del retained_prediction, pred_dict
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)
            if non_output_rank:
                if last_non_output_prediction is None:
                    msg = "multi-seed inference produced no prediction."
                    raise RuntimeError(msg)
                return (
                    last_non_output_prediction,
                    simple_merge_dict_list(log_dicts),
                    simple_merge_dict_list(time_trackers),
                )

            def _cat(dict_list: list[dict[str, Any]], key: str) -> torch.Tensor:
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list: list[dict[str, Any]], key: str) -> list[Any]:
                return functools.reduce(operator.iadd, [x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }
            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return (all_pred_dict, all_log_dict, all_time_dict)
        return self._main_inference_loop(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

    def _get_dynamic_chunk_size(self, N_token: int) -> int | None:  # noqa: N803 - checkpoint-compatible keyword
        """Get dynamic chunk_size based on token count.

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count

        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size
        thresholds = self.configs.infer_setting.chunk_size_thresholds
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size
        return 32

    def _resolve_pairformer_chunk_size(
        self, n_token: int, chunk_size: int | None, *, dynamic_chunk_size: bool
    ) -> int | None:
        """Resolve the Pairformer attention chunk size for ``n_token`` tokens.

        With dynamic chunking enabled (the default), the threshold table picks
        the chunk and the N-squared score budget then bounds it so large
        inputs cannot allocate an oversized attention temporary. An explicitly
        configured fixed ``infer_setting.chunk_size`` with dynamic chunking
        disabled is honoured as given.
        """
        if not dynamic_chunk_size:
            return chunk_size
        return self._bound_pairformer_chunk_size(
            n_token, self._get_dynamic_chunk_size(n_token)
        )

    @staticmethod
    def _bound_pairformer_chunk_size(
        n_token: int, chunk_size: int | None
    ) -> int | None:
        """Bound attention score batches by a platform-neutral N-squared budget."""
        requested_chunk = chunk_size or n_token
        score_batch_budget = 450000000
        budget_chunk = max(1, score_batch_budget // max(1, n_token * n_token))
        power_of_two_chunk = 1 << budget_chunk.bit_length() - 1
        bounded_chunk = min(requested_chunk, power_of_two_chunk)
        if chunk_size is None and bounded_chunk >= n_token:
            return None
        return bounded_chunk

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,  # noqa: N803 - checkpoint-compatible keyword
        *,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Run the main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction,
            log, and time dictionaries.

        """
        step_st = time.time()
        n_token = input_feature_dict["residue_index"].shape[-1]
        dynamic_chunk_size = (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        )
        chunk_size = self._resolve_pairformer_chunk_size(
            n_token, chunk_size, dynamic_chunk_size=dynamic_chunk_size
        )
        log_dict = {}
        pred_dict = {}
        time_tracker = {}
        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        residue_input_feature_dict = input_feature_dict
        residue_s_inputs, residue_s, residue_z = (s_inputs, s, z)
        structural_chunk_size = chunk_size
        structural_refiner_chunk_size = structural_chunk_size
        if self.enable_structural_token_expansion:
            n_structural_token = input_feature_dict["parent_residue_idx"].shape[-1]
            structural_refiner_chunk_size = self._resolve_pairformer_chunk_size(
                n_structural_token,
                structural_chunk_size,
                dynamic_chunk_size=dynamic_chunk_size,
            )
            if dynamic_chunk_size:
                structural_chunk_size = structural_refiner_chunk_size
        input_feature_dict, s_inputs, s, z = self.expand_to_structural_tokens(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s=s,
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=structural_refiner_chunk_size,
            lazy_relp=True,
        )
        pair_input_feature_dict, pair_s_inputs, pair_s, pair_z = (
            self.select_pair_output_branch(
                residue_feature_dict=residue_input_feature_dict,
                residue_s_inputs=residue_s_inputs,
                residue_s=residue_s,
                residue_z=residue_z,
                structural_feature_dict=input_feature_dict,
                structural_s_inputs=s_inputs,
                structural_s=s,
                structural_z=z,
            )
        )
        del residue_input_feature_dict, residue_s_inputs, residue_s, residue_z
        if dynamic_chunk_size:
            chunk_size = structural_chunk_size
        keys_to_delete = []
        keys_to_delete.extend(
            key
            for key in input_feature_dict
            if "template_" in key
            or key
            in ["msa", "has_deletion", "deletion_value", "profile", "deletion_mean"]
        )
        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        n_sample = self.configs.sample_diffusion["N_sample"]
        n_step = self.configs.sample_diffusion["N_step"]
        noise_schedule = self.inference_noise_scheduler(
            N_step=n_step, device=str(s_inputs.device), dtype=s_inputs.dtype
        )
        if noise_schedule is None:
            msg = "inference noise schedule was not prepared."
            raise RuntimeError(msg)
        cache = self.prepare_diffusion_cache_for_sampling(
            input_feature_dict=input_feature_dict, z=z
        )
        self.run_sample_diffusion_stage(
            pred_dict=pred_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s=s,
            z=z,
            cache=cache,
            N_sample=n_sample,
            noise_schedule=noise_schedule,
            chunk_size=chunk_size,
            inplace_safe=inplace_safe,
        )
        del cache
        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        self.run_distogram_contact_stage(pred_dict=pred_dict, pair_z=pair_z)
        self.run_confidence_head_stage(
            pred_dict=pred_dict,
            input_feature_dict=pair_input_feature_dict,
            s_inputs=pair_s_inputs,
            s_trunk=pair_s,
            z_trunk=pair_z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})
        del pair_s_inputs, pair_s, pair_z, s_inputs, s, z, noise_schedule
        is_non_output_rank = False
        if not is_non_output_rank:
            self.run_post_confidence_outputs_stage(
                pred_dict=pred_dict,
                input_feature_dict=input_feature_dict,
                pair_input_feature_dict=pair_input_feature_dict,
                N_cycle=N_cycle,
            )
        if is_non_output_rank:
            return (pred_dict, log_dict, time_tracker)
        return (pred_dict, log_dict, time_tracker)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any] | None = None,
        label_dict: dict[str, Any] | None = None,
        mode: str = "inference",
        *,
        disable_inplace: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None, dict[str, Any]]:
        """Run one forward with this model's TF32 policy scoped to the call."""
        enable_tf32 = getattr(
            self.configs, "enable_tf32", torch.backends.cuda.matmul.allow_tf32
        )
        with _tf32_runtime_scope(enabled=bool(enable_tf32)):
            return self._forward_impl(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                mode=mode,
                disable_inplace=disable_inplace,
            )

    def _forward_impl(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any] | None = None,  # noqa: ARG002 - shared callback or fixture signature
        label_dict: dict[str, Any] | None = None,
        mode: str = "inference",
        *,
        disable_inplace: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None, dict[str, Any]]:
        """Forward pass for structure prediction.

        Args:
            disable_inplace: Whether to disable mutations of activation tensors.
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict: Kept for checkpoint/API compatibility; ignored.
            label_dict: Kept for checkpoint/API compatibility; ignored.
            mode: Only "inference" is supported.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, ignored label dictionary, and log dictionary.

        """
        if mode != "inference":
            msg = "OpenDDE only supports mode='inference'."
            raise ValueError(msg)
        not_use_gradient = not torch.is_grad_enabled()
        inplace_safe = not_use_gradient and (not disable_inplace)

        def _prepare_forward_input_features() -> dict[str, torch.Tensor]:
            prepared_input_features = self.relative_position_encoding.generate_relp(
                input_feature_dict, lazy=True
            )
            return update_input_feature_dict(prepared_input_features)

        input_feature_dict = _prepare_forward_input_features()
        if input_feature_dict is None:
            msg = "forward input features were not prepared."
            raise RuntimeError(msg)
        pred_dict, log_dict, time_tracker = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=self.configs.infer_setting.chunk_size,
            N_model_seed=self.N_model_seed,
        )
        log_dict.update({"time": time_tracker})
        return (pred_dict, label_dict, log_dict)
