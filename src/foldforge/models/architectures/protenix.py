from __future__ import annotations

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import functools
import operator
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from team_gm.diffusion.edm.schedules import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
)
from team_gm.modules.checkpoints.confidence import ConfidenceHead
from team_gm.modules.checkpoints.denoiser import DiffusionModule
from team_gm.modules.checkpoints.dense import LinearNoBias
from team_gm.modules.checkpoints.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from team_gm.modules.checkpoints.heads import DistogramHead
from team_gm.modules.checkpoints.layers import LayerNorm
from team_gm.modules.checkpoints.stacks import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from team_gm.modules.checkpoints.tensor_utils import simple_merge_dict_list
from team_gm.modules.checkpoints.trunk import RecycledTrunk, update_input_feature_dict

import foldforge.eval.confidence as sample_confidence
from foldforge.models.sampling import sample_diffusion, sample_diffusion_training
from foldforge.utils.logging import get_logger
from foldforge.utils.tensor import autocasting_disable_decorator

if TYPE_CHECKING:
    from foldforge.eval.permutation import SymmetricPermutation

logger = get_logger(__name__)


class Protenix(RecycledTrunk):
    """Implement Algorithm 1 [Main Inference/Train Loop] in AF3."""

    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            if configs.loss.weight.alpha_diffusion != 0.0:
                message = "Invalid state: configs.loss.weight.alpha_diffusion == 0.0"
                raise ValueError(message)
            if configs.loss.weight.alpha_distogram != 0.0:
                message = "Invalid state: configs.loss.weight.alpha_distogram == 0.0"
                raise ValueError(message)

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

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

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """Sample diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.

        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        _configs.update(
            {
                "guidance_configs": self.configs.sample_diffusion.to_dict().get(
                    "guidance"
                )
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """Compute run confidence head.

        Run the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.

        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any] | None,
        N_cycle: int,  # noqa: N803 - checkpoint-compatible keyword
        mode: str,
        *,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
        N_model_seed: int = 1,  # noqa: N803 - checkpoint-compatible keyword
        symmetric_permutation: SymmetricPermutation | None = None,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Compute main inference loop.

        Run the main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to
                True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations.
                Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
                Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction,
            log, and time dictionaries.

        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode == "inference":
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
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
            return all_pred_dict, all_log_dict, all_time_dict
        # Single seed inference - delegate to _main_inference_loop
        return self._main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=label_dict,
            N_cycle=N_cycle,
            mode=mode,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            symmetric_permutation=symmetric_permutation,
            mc_dropout=random.random() < mc_dropout_apply_rate,
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

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any] | None,
        N_cycle: int,  # noqa: N803 - checkpoint-compatible keyword
        mode: str,
        *,
        inplace_safe: bool = True,
        chunk_size: int | None = 4,
        symmetric_permutation: SymmetricPermutation | None = None,
        mc_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Run the main inference loop (single model seed) for the Alphafold3 model.

        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction,
            log, and time dictionaries.

        """
        step_st = time.time()
        n_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(n_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the
        # function parameter

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

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
        # Sample diffusion
        n_sample = self.configs.sample_diffusion["N_sample"]
        n_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=n_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        cache = {}
        if self.enable_diffusion_shared_vars_cache:
            # line 1-5 of algorithm 21 calculate z in diffusion conditioning
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
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=n_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        distogram_logits = self.distogram_head(z)
        if getattr(self, "save_distogram", False):
            pred_dict["distogram_logits"] = distogram_logits
        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(
            disable_casting=True
        )(sample_confidence.detailed_compute_contact_prob)(
            distogram_logits=distogram_logits,
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        del distogram_logits

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
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

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask")
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(disable_casting=True)(
            sample_confidence.detailed_compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any] | None,
        N_cycle: int,  # noqa: N803 - checkpoint-compatible keyword
        symmetric_permutation: SymmetricPermutation,
        *,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Run the main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to
                False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations.
                Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.

        """
        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        log_dict = {}
        pred_dict = {}

        cache = {}
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
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            n_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            n_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            self.diffusion_module.eval()  # use eval mode for mini-rollout
            cached_pair, cached_single = cache["p_lm/c_l"]
            if label_dict is None:
                message = "Training mini-rollout requires labels"
                raise ValueError(message)
            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(None if cached_pair is None else cached_pair.detach()),
                c_l=(None if cached_single is None else cached_single.detach()),
                N_sample=n_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=n_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
            self.diffusion_module.train()
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            (
                label_dict,
                perm_log_dict,
            ) = symmetric_permutation.permute_label_to_match_mini_rollout(
                coordinate_mini,
                input_feature_dict,
                label_dict,
                label_full_dict,
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        n_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=n_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(disable_casting=True)(
                    self.distogram_head
                )(z),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)
        log_dict.update({"noise_level": x_noise_level})

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: int | None = None,
        symmetric_permutation: SymmetricPermutation | None = None,
        *,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Forward pass of the Alphafold3 model.

        Args:
            disable_inplace: Whether to disable mutations of activation tensors.
            mc_dropout_apply_rate: Probability of applying inference-time dropout.
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to
                'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
                Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.

        """
        if mode not in ["train", "eval", "inference"]:
            message = "Invalid state: mode in ['train', 'eval', 'inference']"
            raise ValueError(message)
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            n_cycle = nc_rng.randint(1, self.N_cycle + 1)
            if not (self.training):
                message = "Invalid state: self.training"
                raise ValueError(message)
            if not (label_dict is not None):
                message = "Invalid state: label_dict is not None"
                raise ValueError(message)
            if not (symmetric_permutation is not None):
                message = "Invalid state: symmetric_permutation is not None"
                raise ValueError(message)

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=n_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = n_cycle
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                if (
                    label_dict["coordinate"].size()
                    != label_full_dict["coordinate"].size()
                ):
                    message = (
                        "Invalid state: label_dict['coordinate'].size() == "
                        "label_full_dict['coordinate'].size()"
                    )
                    raise ValueError(message)
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        else:
            message = f"Unsupported execution mode: {mode}"
            raise ValueError(message)

        return pred_dict, label_dict, log_dict
