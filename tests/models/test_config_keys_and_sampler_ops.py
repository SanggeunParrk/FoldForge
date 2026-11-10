"""The two switches that were environment variables (dead-module skip, graph pool snapshot) are config keys; the engine sampler ops are built in one place."""

import pytest
import torch

from foldforge.models.config import Config
from foldforge.modules.engine import sampler_ops


def test_dead_module_skip_is_on_by_default_and_a_config_key():
    assert Config().engine.skip_dead_modules is True
    assert (
        Config.model_validate(
            {"engine": {"skip_dead_modules": False}}
        ).engine.skip_dead_modules
        is False
    )


def test_graph_pool_snapshot_is_off_by_default_and_an_execution_key():
    assert Config().execution.graph_pool_snapshot is False
    assert (
        Config.model_validate(
            {"execution": {"graph_pool_snapshot": True}}
        ).execution.graph_pool_snapshot
        is True
    )


def test_unknown_keys_next_to_them_are_still_rejected():
    with pytest.raises(ValueError, match="extra|Extra"):
        Config.model_validate({"engine": {"skip_dead_module": False}})


def test_sampler_ops_are_not_built_without_a_cuda_device():
    cpu = torch.device("cpu")
    assert sampler_ops.pool_atoms(cpu) is None
    assert sampler_ops.augment_align(cpu, churn={}, step_scale=1.0) is None
