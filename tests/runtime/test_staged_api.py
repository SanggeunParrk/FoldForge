"""Public staged inference preserves ownership, RNGs and trunk reuse."""
# ruff: noqa: SLF001 - instrument stage counts and ownership boundaries

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, get_type_hints
from unittest.mock import Mock

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
import torch

import foldforge as ff
from foldforge import api
from foldforge.models import execution, loading
from foldforge.models.io import runtime
from foldforge.models.io.output import Decoded
from foldforge.prediction import Prediction


class Network(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_recycles = config.trunk.recycles or 10
        self.num_samples = 1
        self.diffusion_steps = config.diffusion.steps or 200
        self.trunk_calls = 0

    def run_trunk(self, _features) -> SimpleNamespace:
        self.trunk_calls += 1
        single = torch.rand(3, 2) + self.num_recycles
        return SimpleNamespace(embeddings={"single": single, "pair": single[:, None]})

    def _sample_diffusion(self, trunk) -> torch.Tensor:
        return torch.rand(self.num_samples, 3, 3) + trunk.embeddings["single"].mean()

    def sample_diffusion(self, trunk) -> torch.Tensor:
        return self._sample_diffusion(trunk)

    def forward(self, features) -> torch.Tensor:
        return self.sample_diffusion(self.run_trunk(features))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    from foldforge.data import ccd

    checkpoint = tmp_path / "weights.bin"
    checkpoint.touch()
    database = tmp_path / "ccd"
    database.mkdir()
    active = [0]

    @contextmanager
    def activate() -> Iterator[None]:
        active[0] += 1
        try:
            yield
        finally:
            active[0] -= 1

    db = SimpleNamespace(root=database, lookup={}, activate=activate)
    monkeypatch.setattr(ccd, "CCDDatabase", lambda _: db)
    monkeypatch.setattr(runtime, "CCDDatabase", lambda _: db)
    monkeypatch.setattr(api, "_resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runtime, "measured_forward", lambda f, _: (f(), {}))
    load = Mock(side_effect=lambda _name, _path, config, **_kwargs: Network(config))
    monkeypatch.setattr(loading, "load_configured", load)

    def runner(model, _name, _config) -> SimpleNamespace:
        return SimpleNamespace(
            begin_request=lambda: None,
            report=lambda **_kwargs: {"trunk_passes": model.num_recycles},
        )

    monkeypatch.setattr(execution, "Execution", runner)
    monkeypatch.setattr(runtime, "Execution", runner)

    def prepare(_request, _database, session) -> Iterator[runtime.Case]:
        model = session.load_model()
        session.bind(model)
        features = {"x": torch.ones(3)}

        def decode(value, _measurements) -> Decoded:
            return Decoded(
                "toy",
                Prediction(value, plddt=torch.full((value.shape[0], 3), 0.9)),
                ["data_toy\n"] * value.shape[0],
            )

        yield runtime.Case(
            lambda: model(features),
            decode,
            inference_mode=False,
            run_trunk=lambda: model.run_trunk(features),
            sample_diffusion=model.sample_diffusion,
        )

    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda _: SimpleNamespace(prepare=prepare)
    )
    # Importing model registry modules is not intercepted; adapter selection only.
    config = {
        "trunk_seed": 13,
        "diffusion_seed": 7,
        "trunk": {"recycles": 2},
        "diffusion": {"steps": 4},
    }
    model = ff.load_model("boltz2", config, checkpoint=checkpoint)
    return model, ff.load_inputs(ff.protein("AG")), {"ccd_db": database}, load, active


def test_staged_matches_whole_and_reuses_trunk_with_independent_sampling(staged):
    model, inputs, options, load, active = staged
    original_rng = torch.random.get_rng_state()
    whole = model(inputs, **options)
    with model.run_trunk(inputs, **options) as state:
        assert active == [0]
        assert model._network.trunk_calls == 2
        held = state.embeddings["pair"].clone()
        first = model.sample_diffusion(state)
        again = model.sample_diffusion(state, seed=7)
        changed = model.sample_diffusion(state, samples=3, steps=9, seed=19)
        torch.testing.assert_close(first.coords, whole.coords, rtol=0, atol=0)
        torch.testing.assert_close(again.coords, first.coords, rtol=0, atol=0)
        assert not torch.equal(changed.coords[0], first.coords[0])
        assert changed.coords.shape[0] == 3
        assert changed.report["steps"] == 9
        assert changed.report["diffusion_seed"] == 19
        assert changed.report["trunk_reused"]
        assert model._network.trunk_calls == 2
        torch.testing.assert_close(state.embeddings["pair"], held, rtol=0, atol=0)
    assert state.closed
    assert state.report["trunk_seed"] == 13
    assert active == [0]
    assert load.call_count == 1
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    with pytest.raises(ff.TrunkStateError, match="closed"):
        model.sample_diffusion(state)
    assert torch.isfinite(first.coords).all()


def test_multiple_trunks_restore_schedule_and_close_with_model(staged):
    model, inputs, options, _, _ = staged
    first = model.run_trunk(inputs, **options)
    second = model.run_trunk(inputs, recycles=5, seed=3, **options)
    result = model.sample_diffusion(second, samples=2)
    assert result.report["trunk_passes"] == 5
    assert result.report["trunk_seed"] == 3
    assert model.config.trunk.recycles == 2
    assert model._network.num_recycles == 2
    before = model.sample_diffusion(first)
    model.sample_diffusion(second)
    after = model.sample_diffusion(first)
    torch.testing.assert_close(before.coords, after.coords, rtol=0, atol=0)
    model.close()
    assert first.closed
    assert second.closed
    assert model.closed


def test_trunk_is_bound_to_exact_predictor(staged):
    model, inputs, options, _, _ = staged
    state = model.run_trunk(inputs, **options)
    other = ff.load_model("boltz2", checkpoint=model._options.checkpoint)
    with pytest.raises(ff.TrunkStateError, match="exact"):
        other.sample_diffusion(state)
    state.close()
    other.close()


def test_concurrent_call_rejected_during_preparation(staged, monkeypatch):
    model, inputs, options, _, _ = staged
    entered, release = Event(), Event()
    original = api.plan_prediction

    def plan(*args: Any, **kwargs: Any) -> api.PredictionPlan:
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "plan_prediction", plan)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(model, inputs, **options)
        try:
            assert entered.wait(10)
            with pytest.raises(ff.ConcurrentCallError):
                model(inputs, **options)
            with pytest.raises(ff.ConcurrentCallError):
                model.close()
        finally:
            release.set()
        assert first.result(timeout=10).coords.shape[0] == 1


@pytest.mark.parametrize(
    "source",
    [{"sequences": []}, {"sequences": [{"protein": {"id": "A", "sequence": ""}}]}],
)
def test_bad_input_fails_before_loading_a_model(source):
    with pytest.raises(ff.InputError):
        ff.load_inputs(source)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_sampling_options_are_strict_positive_integers(staged, value):
    model, inputs, options, _, _ = staged
    with pytest.raises(ValueError, match="positive integer"):
        model.run_trunk(inputs, recycles=value, **options)
    state = model.run_trunk(inputs, **options)
    with pytest.raises(ValueError, match="positive integer"):
        model.sample_diffusion(state, samples=value)
    with pytest.raises(ValueError, match="positive integer"):
        model.sample_diffusion(state, steps=value)


def test_public_runtime_annotations_resolve():
    for function in (
        ff.load_model,
        ff.load_inputs,
        ff.predict,
        ff.Predictor.run_trunk,
        ff.Predictor.sample_diffusion,
        ff.PredictionResult.save,
    ):
        assert get_type_hints(function)
    assert get_type_hints(ff.Prediction)


def test_library_output_is_owned_by_the_caller(staged, tmp_path, monkeypatch):
    from foldforge.models.io import paths

    model, inputs, options, _, _ = staged
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "cli-runs")
    destination = tmp_path / "service-results"
    result = model(inputs, out=destination, **options)
    assert result.output_dir == destination
    state = model.run_trunk(inputs, **options)
    second = model.sample_diffusion(state, out=tmp_path / "samples")
    assert (second.output_dir / "model.config.json").is_file()
    assert (second.output_dir / "toy.prediction.pt").is_file()


def test_distinct_predictors_serialize_global_execution_policy(staged):
    model, inputs, options, _, _ = staged
    other = ff.load_model("boltz2", checkpoint=model._options.checkpoint)
    entered, attempted, second_entered, release = Event(), Event(), Event(), Event()
    original_first, original_second = model._network.forward, other._network.forward

    def first_forward(features) -> torch.Tensor:
        entered.set()
        assert release.wait(10)
        return original_first(features)

    def second_forward(features) -> torch.Tensor:
        second_entered.set()
        return original_second(features)

    model._network.forward = first_forward
    other._network.forward = second_forward

    def second_call() -> ff.PredictionResult:
        attempted.set()
        return other(inputs, **options)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(model, inputs, **options)
        try:
            assert entered.wait(10)
            second = pool.submit(second_call)
            assert attempted.wait(10)
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        assert first.result(timeout=10).coords.shape[0] == 1
        assert second.result(timeout=10).coords.shape[0] == 1
    assert second_entered.is_set()
    other.close()


def test_compatibility_parser_retains_explicit_device():
    from foldforge.models.io.cli import parse

    request = parse(
        "boltz2",
        ["--input", "input.json", "--checkpoint", "weights.bin", "--device", "cuda:1"],
    )
    assert request.device == "cuda:1"
