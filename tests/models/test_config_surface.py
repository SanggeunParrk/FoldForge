"""The config surface that replaced the behaviour-steering environment variables."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foldforge.models.config import Config
from foldforge.models.config.runtime import ENGINE_SKIP_KEYS, REMOVED_ENVIRONMENT


def test_defaults_are_the_behaviour_without_the_old_variables():
    config = Config()
    assert config.engine.skip == ()
    assert config.debug.dense_spec_override is None
    assert config.execution.confidence_sample_chunk is None


def test_engine_skip_override_and_validation():
    config = Config.model_validate({"engine": {"skip": ["engine_modules"]}})
    assert config.engine.skip == ("engine_modules",)
    assert set(config.engine.skip) <= set(ENGINE_SKIP_KEYS)
    with pytest.raises(ValidationError):
        Config.model_validate({"engine": {"skip": ["engine_nonsense"]}})


def test_other_overrides():
    config = Config.model_validate(
        {
            "execution": {"confidence_sample_chunk": 3},
            "debug": {"dense_spec_override": {"confidence_reembed_pair": True}},
        }
    )
    assert config.execution.confidence_sample_chunk == 3
    assert config.debug.dense_spec_override == {"confidence_reembed_pair": True}
    with pytest.raises(ValidationError):
        Config.model_validate({"execution": {"confidence_sample_chunk": 0}})


@pytest.mark.parametrize("name", sorted(REMOVED_ENVIRONMENT))
def test_a_removed_environment_variable_is_an_error(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match=name):
        Config()


def test_the_loader_rejects_a_removed_variable_too(monkeypatch):
    from foldforge.models.loading import load_checkpoint

    monkeypatch.setenv("FF_ENGINE_SKIP", "pairformers")
    with pytest.raises(RuntimeError, match="FF_ENGINE_SKIP"):
        load_checkpoint("af3")


def test_the_engine_skip_reaches_the_install():
    """``skip`` is the only route: the install reads no environment."""
    import inspect

    from foldforge.modules.engine import install, modules

    assert "environ" not in inspect.getsource(install)
    assert "environ" not in inspect.getsource(modules)
