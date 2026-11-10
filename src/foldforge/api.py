"""Reusable prediction and staged sampling; the package root imports this lazily."""
# ruff: noqa: PLC0415, TC001, TC003, SLF001 - public runtime hints and private state ownership

from __future__ import annotations

import warnings
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from weakref import WeakSet

import torch

from foldforge.data.inputs.build import Input
from foldforge.errors import InputError, ModelClosedError, TrunkStateError
from foldforge.inputs import Inputs, load_inputs
from foldforge.inputs import normalize_payload as _normalize_payload
from foldforge.inputs import read_inputs as _read_inputs
from foldforge.lifecycle import PROCESS_LOCK, CallGuard
from foldforge.models.architectures.af3 import AlphaFold3
from foldforge.models.config import Config
from foldforge.models.execution import Execution
from foldforge.result import PredictionResult
from foldforge.stages import TrunkState

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldforge.models.io.request import Request


def resolve_config(
    config: Config | Mapping[str, Any] | str | Path | None = None,
) -> Config:
    """Validate a detached configuration from a model, mapping or YAML path."""
    import yaml

    from foldforge.models.config import Config

    if isinstance(config, Config):
        data = config.model_dump()
    elif isinstance(config, (str, Path)):
        data = yaml.safe_load(Path(config).expanduser().read_text()) or {}
    else:
        data = deepcopy(dict(config)) if config is not None else {}
    return Config.model_validate(data)


def _validate_target(model: str, target: Input) -> None:
    """Validate released-model capabilities without creating prediction artifacts."""
    from foldforge.models import entry
    from foldforge.modules.dense.spec import SPECS

    spec = SPECS[entry(model).family or "alphafold3"]
    if not spec.template_layers and target.spec.template:
        message = f"the {model} checkpoint has no template conditioning path"
        raise ValueError(message)
    if target.spec.save_trajectory:
        message = "Released adapters write final structures; set save_trajectory: false"
        raise ValueError(message)
    if target.spec.n_trunk_samples != 1:
        message = "Released adapters currently require n_trunk_samples=1"
        raise ValueError(message)
    if target.spec.diffusion_batch_size != target.spec.n_diffusion_samples:
        message = (
            "Set diffusion_batch_size=n_diffusion_samples; "
            "these adapters do not chunk sampling"
        )
        raise ValueError(message)


@dataclass(frozen=True)
class PredictionPlan:
    """A validated execution request; preparation is separate from persistence."""

    request: Request
    config: Config
    engine_cache_dir: Path | None


@dataclass(frozen=True)
class _ModelOptions:
    name: str
    checkpoint: Path
    config: Config
    cache: Path | None
    device: torch.device | None = None


_CACHE_DIR_WARNED = False


def _warn_engine_cache_dir() -> None:
    """Warn once: the engine has no autotune cache any more, the argument is ignored."""
    global _CACHE_DIR_WARNED  # noqa: PLW0603 - one-time process flag
    if _CACHE_DIR_WARNED:
        return
    _CACHE_DIR_WARNED = True
    message = (
        "engine_cache_dir is deprecated and ignored: miniworld-engine 2.5.0 has no "
        "autotune cache (launch configs come from the engine's pretuned tables)"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _model_options(
    model: str,
    config: Any,
    checkpoint: str | Path | None,
    engine_cache_dir: str | Path | None,
) -> _ModelOptions:
    """Validate model policy and checkpoint independently of any target."""
    from foldforge.models import registered_models
    from foldforge.models.checkpoints.lock import check_size
    from foldforge.models.checkpoints.miniworld import resolve_native

    if model not in registered_models():
        message = (
            f"Unsupported model {model!r}; choose from {', '.join(registered_models())}"
        )
        raise ValueError(message)
    resolved = resolve_config(config)
    weights = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else resolve_native(model)  # <model>.miniworld.pt; a missing file raises with the `foldforge weights convert` command
    )
    if not weights.exists():
        raise FileNotFoundError(weights)
    check_size(weights)
    if engine_cache_dir is not None:
        _warn_engine_cache_dir()
    return _ModelOptions(model, weights, resolved, None)


def _validate_payload_capabilities(model: str, data: dict[str, Any]) -> None:
    """Apply the same template restrictions to raw JSON as resolved YAML inputs."""
    from foldforge.models import entry
    from foldforge.modules.dense.spec import SPECS

    spec = SPECS[entry(model).family or "alphafold3"]
    if not spec.template_layers and any(
        chain.get("protein", {}).get("templates") for chain in data["sequences"]
    ):
        message = f"the {model} checkpoint has no template conditioning path"
        raise InputError(message)


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        message = f"{name} must be a positive integer"
        raise InputError(message)
    return value


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Pin the selected CUDA index at load time, independent of later contexts."""
    selected = torch.device("cuda" if device is None else device)
    if selected.type != "cuda":
        message = "FoldForge inference requires a CUDA device"
        raise InputError(message)
    if not torch.cuda.is_available():
        message = "CUDA is unavailable; load_model requires a CUDA runtime"
        raise RuntimeError(message)
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index >= torch.cuda.device_count():
        message = f"CUDA device index {index} is unavailable"
        raise InputError(message)
    return torch.device("cuda", index)


def _validate_chemistry(database: Any, target: Input | None) -> None:
    """Check referenced chemistry before creating artifacts or loading weights."""
    if target is not None:
        for chain in target.chains:
            for code in set(chain.ccds):
                database.lookup[code]


def plan_prediction(
    inputs: Inputs | Input | Mapping[str, Any] | str | Path,
    config: Config | Mapping[str, Any] | str | Path | None = None,
    *,
    model: str = "af3",
    checkpoint: str | Path | None = None,
    out: str | Path | None = None,
    samples: int | None = None,
    ccd_db: str | Path | None = None,
    base_dir: str | Path | None = None,
    engine_cache_dir: str | Path | None = None,
    verbose: bool = False,
    _options: _ModelOptions | None = None,
) -> PredictionPlan:
    """Resolve inputs/config/checkpoint before any output directory is created."""
    from foldforge.data.ccd import CCDDatabase, default_path
    from foldforge.models.io.paths import output_directory
    from foldforge.models.io.request import Request

    if samples is not None:
        _positive(samples, "samples")
    output = output_directory(out) if out is not None else None
    options = _options or _model_options(model, config, checkpoint, engine_cache_dir)
    resolved, weights, cache = options.config, options.checkpoint, options.cache
    target, data, origin = _read_inputs(
        load_inputs(inputs, base_dir=base_dir), base_dir
    )
    if target is not None:
        if samples is not None:
            target = replace(
                target,
                spec=target.spec.__class__.model_validate(
                    {
                        **target.spec.model_dump(),
                        "n_diffusion_samples": samples,
                        "diffusion_batch_size": samples,
                    }
                ),
            )
        _validate_target(model, target)
        database_path = target.spec.ccd_db
        if (
            ccd_db is not None
            and Path(ccd_db).expanduser().resolve() != database_path.resolve()
        ):
            message = "Spec and selected CCD database disagree"
            raise ValueError(message)
        count = target.spec.n_diffusion_samples
        name = target.spec.name or "prediction"
    else:
        data = _normalize_payload(data)
        _validate_payload_capabilities(model, data)
        database_path = (
            Path(ccd_db).expanduser() if ccd_db is not None else default_path()
        )
        count = samples if samples is not None else 1
        name = data["name"]
    if not database_path.exists():
        raise FileNotFoundError(database_path)
    database = CCDDatabase(database_path)
    _validate_chemistry(database, target)
    request = Request.from_config(
        resolved,
        model=model,
        ccd_db=database.root,
        checkpoint=weights,
        input=origin,
        samples=count,
        out=output,
        resolved_input=target,
        target=name,
        input_data=data,
        write_outputs=output is not None,
        verbose=verbose,
        output_policy="path",
        device=str(options.device) if options.device is not None else None,
    )
    return PredictionPlan(request, resolved, cache)


def prepare_prediction(plan: PredictionPlan) -> Request:
    """Prepare MSA rows in memory, or persist the established CLI input artifacts."""
    import json

    from foldforge.data.inputs.build import limit_msa, write_adapter_input

    request = plan.request
    target = request.resolved_input
    output = request.out
    if request.write_outputs and output is None:
        message = "An output run directory is required"
        raise ValueError(message)
    if target is not None:
        target = limit_msa(
            target,
            request.msa_depth,
            output / "prepared-msa" if output is not None else None,
        )
        request = replace(request, resolved_input=target)
    if not request.write_outputs or output is None:
        return request
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.config.json").write_text(
        plan.config.model_dump_json(indent=2) + "\n"
    )
    if target is not None:
        artifacts = {
            "msa-species-aliases.json": target.msa_species,
            "input.resources.json": target.resources(),
            "input.spec.json": target.spec.model_dump(mode="json"),
            "chain-map.json": [
                {"index": c.index, "id": c.id, "letter": c.letter}
                for c in target.chains
            ],
        }
        for name, value in artifacts.items():
            (output / name).write_text(json.dumps(value, indent=2) + "\n")
        path = output / "input.adapter.json"
        write_adapter_input(target, path, request.trunk_seed)
        request = replace(request, input=path)
    elif request.input_data is not None:
        path = output / "input.adapter.json"
        data = {**request.input_data, "modelSeeds": [request.trunk_seed]}
        path.write_text(json.dumps(data, indent=2) + "\n")
        request = replace(request, input=path)
    return request


class Predictor:
    """A model owning its CUDA device, execution wrappers and retained trunk states."""

    def __init__(
        self,
        options: _ModelOptions,
        network: AlphaFold3,
        execution: Execution,
    ) -> None:
        self._options = options
        self._network: Any = network
        self._execution: Any = execution
        self._language_models: dict[str, Any] = {}
        self._guard = CallGuard()
        self._token = object()
        self._states: WeakSet[TrunkState] = WeakSet()

    def __repr__(self) -> str:
        return (
            f"Predictor(model={self.name!r}, device={str(self.device)!r}, "
            f"closed={self.closed})"
        )

    @property
    def name(self) -> str:
        """Registered model name."""
        return self._options.name

    @property
    def device(self) -> torch.device:
        """The CUDA device fixed when this model was loaded."""
        if self._options.device is None:
            message = "This predictor has no configured device"
            raise RuntimeError(message)
        return self._options.device

    @property
    def closed(self) -> bool:
        """Whether weights and active trunk states have been released."""
        return self._network is None

    @property
    def config(self) -> Config:
        """Return a detached copy of the model's fixed inference policy."""
        return resolve_config(self._options.config)

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._guard.acquire(), PROCESS_LOCK:
            if self.closed:
                message = "This predictor is closed; load a new model"
                raise ModelClosedError(message)
            yield

    @contextmanager
    def _scope(self) -> Iterator[None]:
        from foldforge.modules.engine.configure import inference_settings

        device_scope = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_scope, inference_settings():
            yield

    def __call__(
        self,
        inputs: Inputs | Input | Mapping[str, Any] | str | Path,
        *,
        samples: int | None = None,
        out: str | Path | None = None,
        ccd_db: str | Path | None = None,
        base_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> PredictionResult:
        """Predict one target with the shared trunk/sampling network lifecycle."""
        from foldforge.models.io.runtime import predict as execute

        with self._operation():
            plan = plan_prediction(
                inputs,
                model=self.name,
                samples=samples,
                out=out,
                ccd_db=ccd_db,
                base_dir=base_dir,
                verbose=verbose,
                _options=self._options,
            )
            with self._scope():
                results = execute(
                    prepare_prediction(plan),
                    model=self._network,
                    execution=self._execution,
                    language_models=self._language_models,
                    device=self.device,
                )
            if len(results) != 1:
                message = f"Expected one target result, received {len(results)}"
                raise RuntimeError(message)
            return results[0]

    def run_trunk(
        self,
        inputs: Inputs | Input | Mapping[str, Any] | str | Path,
        *,
        recycles: int | None = None,
        seed: int | None = None,
        ccd_db: str | Path | None = None,
        base_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> TrunkState:
        """Prepare inputs and run total trunk passes once, retaining GPU state."""
        from foldforge.data.inputs.validation import validate_inference_seed
        from foldforge.models.io.runtime import run_trunk

        with self._operation():
            if self._options.config.execution.scope == "model":
                message = "Staged inference requires execution.scope=denoiser or trunk"
                raise InputError(message)
            data = self._options.config.model_dump()
            if recycles is not None:
                data["trunk"]["recycles"] = _positive(recycles, "recycles")
            if seed is not None:
                data["trunk_seed"] = validate_inference_seed(seed)
            options = replace(self._options, config=resolve_config(data))
            plan = plan_prediction(
                inputs,
                model=self.name,
                ccd_db=ccd_db,
                base_dir=base_dir,
                verbose=verbose,
                _options=options,
            )
            with self._scope():
                record = run_trunk(
                    prepare_prediction(plan),
                    model=self._network,
                    execution=self._execution,
                    language_models=self._language_models,
                    device=self.device,
                )
            state = TrunkState(self._token, self.name, self.device, record)
            self._states.add(state)
            return state

    def sample_diffusion(
        self,
        trunk: TrunkState,
        *,
        samples: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        out: str | Path | None = None,
    ) -> PredictionResult:
        """Sample fresh diffusion trajectories and heads without rerunning a trunk."""
        from foldforge.data.inputs.validation import validate_inference_seed
        from foldforge.models.io.paths import output_directory
        from foldforge.models.io.runtime import sample_diffusion

        with self._operation():
            if not isinstance(trunk, TrunkState) or trunk._owner is not self._token:
                message = "A trunk state must belong to this exact loaded predictor"
                raise TrunkStateError(message)
            with trunk._guard.acquire():
                record = trunk._require()
                request = replace(
                    record.request,
                    samples=_positive(samples, "samples")
                    if samples is not None
                    else record.request.samples,
                    steps=_positive(steps, "steps")
                    if steps is not None
                    else record.request.steps,
                    diffusion_seed=validate_inference_seed(seed)
                    if seed is not None
                    else record.request.diffusion_seed,
                    out=output_directory(out) if out is not None else None,
                    write_outputs=out is not None,
                )
                config = request.to_config()
                with self._scope():
                    request = prepare_prediction(
                        PredictionPlan(request, config, self._options.cache)
                    )
                    return sample_diffusion(
                        record, request, model=self._network, execution=self._execution
                    )

    def close(self) -> None:
        """Release weights, auxiliary models and all surviving trunk states."""
        import gc

        with self._guard.acquire(), PROCESS_LOCK:
            for state in self._states:
                state.close()
            self._states.clear()
            self._execution = None
            self._network = None
            self._language_models.clear()
            gc.collect()

    def __enter__(self) -> Self:
        """Own a predictor within a context manager."""
        if self.closed:
            message = "This predictor is closed"
            raise ModelClosedError(message)
        return self

    def __exit__(self, *_: object) -> None:
        """Close the predictor and its retained states on context exit."""
        self.close()


def load_model(
    model: str = "af3",
    config: Config | Mapping[str, Any] | str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    engine_cache_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> Predictor:
    """Load weights once and return a callable predictor, independent of inputs."""
    from foldforge.models.execution import Execution
    from foldforge.models.loading import load_configured
    from foldforge.modules.engine.configure import inference_settings
    from foldforge.utils.seed import seed_context

    options = _model_options(model, config, checkpoint, engine_cache_dir)
    selected_device = _resolve_device(device)
    options = replace(options, device=selected_device)
    device_scope = (
        torch.cuda.device(selected_device)
        if selected_device.type == "cuda"
        else nullcontext()
    )
    with (
        PROCESS_LOCK,
        device_scope,
        inference_settings(),
        seed_context(options.config.trunk_seed),
    ):
        network = load_configured(
            model, options.checkpoint, options.config, device=selected_device
        )
        execution = Execution(network, model, options.config.execution)
    return Predictor(options, network, execution)


def predict(
    inputs: Inputs | Input | Mapping[str, Any] | str | Path,
    config: Config | Mapping[str, Any] | str | Path | None = None,
    *,
    model: str = "af3",
    checkpoint: str | Path | None = None,
    out: str | Path | None = None,
    samples: int | None = None,
    ccd_db: str | Path | None = None,
    base_dir: str | Path | None = None,
    engine_cache_dir: str | Path | None = None,
    verbose: bool = False,
    device: str | torch.device | None = None,
) -> PredictionResult:
    """Predict one target on CUDA and return CPU tensors, CIFs and metadata.

    Inputs are a MiniWorld YAML path, resolved Input, or AlphaFold3 JSON mapping/path.
    Config accepts Config, a mapping or YAML. No prediction files are written unless
    out is supplied; the result can be saved later. Missing MSA fields mean no MSA
    search. Checkpoints must already be available through the registry or override.
    """
    inputs = load_inputs(inputs, base_dir=base_dir)
    plan = plan_prediction(
        inputs,
        config,
        model=model,
        checkpoint=checkpoint,
        out=out,
        samples=samples,
        ccd_db=ccd_db,
        base_dir=base_dir,
        engine_cache_dir=engine_cache_dir,
        verbose=verbose,
    )
    with load_model(
        model,
        plan.config,
        checkpoint=plan.request.checkpoint,
        engine_cache_dir=plan.engine_cache_dir,
        device=device,
    ) as predictor:
        return predictor(
            inputs,
            samples=samples,
            out=out,
            ccd_db=ccd_db,
            verbose=verbose,
        )
