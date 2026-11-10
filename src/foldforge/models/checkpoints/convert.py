"""Convert a released AF3-family checkpoint into the AF3 parameter blob.

Offline provenance step, not part of inference. The key maps are the Apache-2.0
converters of sokrypton/alphafold3 (branch af3-any-model); pass that checkout with
--reference. They are imported, never copied, and they write through FoldForge's
own record codec. Release dimensions and sampler settings are checked against
FoldForge's specs without importing JAX, Haiku or the reference graph.
"""

# ruff: noqa: PLC0415 - converter setup keeps optional imports lazy

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import sys
import types
from pathlib import Path

_RELEASE_CONFIG_TOLERANCE = 1e-9

MODELS = (
    "intellifold2",
    "openfold3",
    "openbind0",
    "boltz2",
    "rosettafold3",
    "chai1",
    "protenix1",
    "protenix2",
    "opendde",
    "esmfold2",
    "esmfold2_fast",
)


def _seed_reference_params(reference: Path) -> None:
    """Expose only the local record codec to the external parameter key maps."""
    from foldforge.models.checkpoints.legacy import haiku

    source = reference / "src" / "alphafold3"
    for name, path in {
        "alphafold3": [str(source)],
        "alphafold3.model": [str(source / "model")],
        "alphafold3.model.params": [],
    }.items():
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        module.__path__ = path  # type: ignore[attr-defined]
        sys.modules[name] = module
    params = sys.modules["alphafold3.model.params"]
    params.encode_record = haiku.encode_record  # type: ignore[attr-defined]
    params.read_records = haiku.read_records  # type: ignore[attr-defined]


def _unfold_chai1_chain_features(chai1: types.ModuleType) -> None:
    """Give Chai-1's relative encoding its chain and entity columns back.

    The reference converter folds RelativeChain and RelativeEntity into the
    projection's bias at their single-chain values (same chain, same entity),
    which is exact for one protein chain and wrong for every pair across
    chains in a complex. This wraps its pair-init mapping: the folded constant
    comes back out of the bias and both features become input columns, in the
    order ``featurization.chai_relative_encoding`` emits them.
    """
    import numpy as np

    original = chai1.map_pair_init

    def map_pair_init(sds: dict, params: dict) -> None:
        original(sds, params)
        fe, te = sds["feature_embedding"], sds["token_embedder"]
        wp = np.asarray(te["token_pair_proj_in_trunk.weight"])
        half = np.asarray(te["token_single_to_token_pair_outer_sum_proj.weight"])
        half = half.shape[0] // 2
        p = np.asarray(fe["input_projs.TOKEN_PAIR.0.weight"])[:half]
        chain = slice(*chai1.PAIR_COLS["RelativeChain"])
        entity = slice(*chai1.PAIR_COLS["RelativeEntity"])
        folded = p[:, chain.start + 2] + p[:, entity.start + 1]
        scope = params["diffuser/evoformer/~_relative_encoding/position_activations"]
        extra = wp @ np.concatenate([p[:, chain], p[:, entity]], axis=1)
        scope["weights"] = np.concatenate([scope["weights"], extra.T], axis=0)
        scope["bias"] = scope["bias"] - wp @ folded

    chai1.map_pair_init = map_pair_init


def _convert_protenix(
    converter: types.ModuleType, model: str, checkpoint: Path, output: Path
) -> Path:
    """Validate the PyTorch release against FoldForge's own graph dimensions.

    The external v1 entry point imports a JAX AF3 model just to construct these
    four constants. The v2 entry point has no equivalent check. Apply the same
    independent checkpoint-versus-graph check to both releases, without JAX.
    """
    from foldforge.modules.dense.spec import SPECS

    spec = SPECS[model]
    state = converter.load_protenix_checkpoint(checkpoint)
    dimensions = converter.derive_dims(state)
    expected = {
        "n_pairformer": spec.trunk_layers,
        "n_msa": spec.msa_layers,
        "n_diff_token": spec.diffusion_blocks,
        "c_z": spec.pair_channel,
    }
    differences = [
        f"{name} is {dimensions[name]}, expected {value}"
        for name, value in expected.items()
        if dimensions[name] != value
    ]
    if differences:
        message = f"Checkpoint does not match {model}: " + "; ".join(differences)
        raise ValueError(message)
    params = converter.map_protenix2_to_af3(state)
    return Path(
        converter.C.write_params_blob(output, f"{model}.bin.zst", params, add_meta=True)
    )


def _prepare_esmfold2(converter: types.ModuleType) -> None:
    """Keep both release guards while replacing reference JAX config imports."""
    from foldforge.modules.dense.spec import SPECS, DenseSpec

    def spec_for(model: str) -> DenseSpec:
        return SPECS[model.replace("_fast", "-fast")]

    def check_variant_table(state: dict, model: str) -> None:
        spec = spec_for(model)
        actual = converter.derive_dims(state)
        expected = {
            "n_trunk": spec.trunk_layers,
            "n_msa": spec.msa_layers,
            "n_coda": spec.coda_layers,
            "n_lm_encoder": spec.lm_encoder_layers,
            "msa_head_width": (
                (spec.msa_value_dim or spec.msa_channel // 8) if spec.msa_layers else 0
            ),
            "num_bins": spec.distogram_bins,
        }
        differences = {
            name: (actual[name], value)
            for name, value in expected.items()
            if actual[name] != value
        }
        if "confidence_head.boundaries" in state:
            bins = int(state["confidence_head.boundaries"].shape[0]) + 1
            if bins != spec.confidence_learned_bins:
                differences["conf_bins"] = (bins, spec.confidence_learned_bins)
        if differences:
            message = f"Checkpoint does not match {model}: {differences}"
            raise ValueError(message)

    def check_release_config(checkpoint: Path, model: str) -> dict | None:
        path = Path(checkpoint)
        config = path / "config.json"
        if not path.is_dir() or not config.is_file():
            return None
        spec = spec_for(model)
        data = json.loads(config.read_text())
        head = data.get("structure_head", {})
        expected = {
            "inference_num_steps": 14,  # Both supported original release profiles.
            "gamma_0": spec.gamma_0,
            "gamma_min": spec.gamma_min,
            "noise_scale": spec.noise_scale,
            "step_scale": spec.step_scale,
            "inference_s_min": spec.sigma_min,
            "inference_s_max": spec.sigma_max,
            "inference_p": spec.rho,
            "max_inference_sigma": 256.0,
        }
        differences = {
            name: (head[name], value)
            for name, value in expected.items()
            if head.get(name) is not None
            and abs(float(head[name]) - value) > _RELEASE_CONFIG_TOLERANCE
        }
        if differences:
            message = f"Release sampler does not match {model}: {differences}"
            raise ValueError(message)
        return data

    converter.check_variant_table = check_variant_table
    converter.check_release_config = check_release_config


def _conversion_profile() -> None:
    """Select the supported CPU conversion profile before importing NumPy.

    FP32 affine folding uses BLAS: backend versions, ISA dispatch and thread
    partitioning can alter its rounding. The inference process never calls this.
    """
    expected = {"numpy": "2.5.3", "zstandard": "0.25.0"}
    for package, version in expected.items():
        actual = importlib.metadata.version(package)
        if actual != version:
            message = (
                f"CPU conversion profile requires {package}=={version}, got {actual}. "
                "Install: pip install 'foldforge[weights]'"
            )
            raise RuntimeError(message)
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        message = "The qualified CPU conversion profile requires x86-64 with AVX2."
        raise RuntimeError(message)
    os.environ["OPENBLAS_CORETYPE"] = "Haswell"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _convert_intellifold2(
    converter: types.ModuleType, checkpoint: Path, output: Path
) -> Path:
    """Keep release tensors at FP32 in the shared, versioned blob container."""
    params = converter.map_intellifold2_to_af3(converter.load_pt(checkpoint))
    return Path(converter.C.write_params_blob(output, "intellifold2.bin.zst", params))


def main(argv: list[str] | None = None) -> int:
    """Run a selected release converter in this dedicated interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    _conversion_profile()
    # Full precision everywhere; the runtime's precision policy decides the rest.
    os.environ["IF2_FP32_BLOB"] = "1"
    _seed_reference_params(args.reference)
    sys.path.insert(0, str(args.reference))
    converters = importlib.import_module("converters")
    if args.model in {"esmfold2", "esmfold2_fast"}:
        _prepare_esmfold2(importlib.import_module("converters.esmfold2"))
    if args.model == "chai1":
        _unfold_chai1_chain_features(importlib.import_module("converters.chai1"))
    if args.model in {"protenix1", "protenix2"}:
        result = _convert_protenix(
            importlib.import_module("converters.protenix2"),
            args.model,
            args.checkpoint,
            args.out,
        )
    elif args.model == "intellifold2":
        result = _convert_intellifold2(
            importlib.import_module("converters.intellifold2"),
            args.checkpoint,
            args.out,
        )
    else:
        result = converters.CONVERTERS[args.model](args.checkpoint, args.out)
    print(f"{args.model}: {result}")  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
