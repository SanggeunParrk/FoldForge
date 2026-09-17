"""Independent trunk/diffusion streams at the actual inference boundaries."""

import json
import random
from types import SimpleNamespace
from typing import Any, Never

import numpy as np
import pytest
import torch
from team_gm.diffusion.edm.sampling import EulerSampler
from team_gm.modules.execution import ExecutedCallable
from torch import nn

from foldforge.models.config import Config, ExecutionConfig
from foldforge.models.io.cli import parse
from foldforge.models.io.request import Request
from foldforge.models.sampling import bind_sampling_seed
from foldforge.utils.seed import RNGState, conformer_seed, seed_all, seed_context


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
@pytest.mark.parametrize("field", ["trunk_seed", "diffusion_seed"])
def test_invalid_seed_rejected_before_execution(seed, field, tmp_path):
    with pytest.raises(ValueError, match="seed"):
        Config.model_validate({field: seed})
    with pytest.raises(ValueError, match="seed"):
        Request(model="esmfold2", ccd_db=tmp_path, **{field: seed})


def test_legacy_config_and_cli_are_unambiguous():
    config = Config.model_validate({"seed": 19})
    assert (config.trunk_seed, config.diffusion_seed) == (19, 19)
    assert "seed" not in config.model_dump()
    for options in ({"seed": 19, "trunk_seed": 7}, {"seed": 19, "diffusion_seed": 8}):
        with pytest.raises(ValueError, match="not both"):
            Config.model_validate(options)
    request = parse("esmfold2", ["--trunk-seed", "7", "--diffusion-seed", "19"])
    assert (request.trunk_seed, request.diffusion_seed) == (7, 19)
    request = parse("esmfold2", ["--seed", "19"])
    assert (request.trunk_seed, request.diffusion_seed) == (19, 19)
    with pytest.raises(SystemExit):
        parse("esmfold2", ["--seed", "19", "--trunk-seed", "7"])


def test_seed_context_restores_all_rngs_on_failure():
    seed_all(17)
    state = RNGState.capture()
    expected = (random.random(), np.random.random(), torch.rand(3))
    state.restore()

    def fail() -> Never:
        with seed_context(91):
            random.random()
            np.random.random()
            torch.rand(7)
            message = "probe"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="probe"):
        fail()
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize("family", ["residue", "structural"])
def test_smiles_conformer_follows_trunk_rng_and_retry(family, monkeypatch):
    from foldforge.data.inputs import entities

    build = getattr(entities, f"{family}_smiles_to_atom_info")
    # Keep this test focused on real RDKit embedding, before atom-layout conversion.
    monkeypatch.setattr(
        entities,
        f"{family}_rdkit_mol_to_atom_info",
        lambda mol: mol.GetConformer().GetPositions().copy(),
    )

    def run(seed) -> np.ndarray:
        with seed_context(seed):
            return build("CCCCCO")

    assert np.array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))
    calls = []
    embed = entities.rdDistGeom.EmbedMolecule

    def retry(mol, **kwargs: Any) -> int:
        calls.append(kwargs)
        return -1 if len(calls) == 1 else embed(mol, **kwargs)

    monkeypatch.setattr(entities.rdDistGeom, "EmbedMolecule", retry)
    run(7)
    assert calls[0]["randomSeed"] == calls[1]["randomSeed"]
    assert calls[1]["useRandomCoords"]


def test_rdkit_seed_covers_unsigned_input_range():
    assert conformer_seed(0) == 0
    assert conformer_seed(2**32 - 1) == 2**31 - 1


def test_af3_json_uses_requested_trunk_seed(tmp_path, monkeypatch):
    from alphafold3.data import featurisation

    from foldforge.models.io import dense_atoms

    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "name": "probe",
                "dialect": "alphafold3",
                "version": 1,
                "modelSeeds": [91, 92],
                "sequences": [{"protein": {"id": "A", "sequence": "AG"}}],
            }
        )
    )
    request = Request(
        model="af3",
        ccd_db=tmp_path,
        input=path,
        checkpoint=tmp_path / "weights",
        trunk_seed=7,
        diffusion_seed=19,
        recycles=1,
        steps=2,
    )

    class FinishedInputCheckError(Exception):
        pass

    def inspect(fold_input, *_args: Any, **_kwargs: Any) -> Never:
        assert tuple(fold_input.rng_seeds) == (7,)
        raise FinishedInputCheckError

    monkeypatch.setattr(featurisation, "featurise_input", inspect)
    database = SimpleNamespace(af3_ccd=lambda **_: {})
    with pytest.raises(FinishedInputCheckError):
        next(dense_atoms._prepare(request, database, None))  # noqa: SLF001 - stop before loading weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sampling boundaries")
@pytest.mark.parametrize("family", ["af3", "esmfold2", "protenix", "opendde"])
@pytest.mark.parametrize("mode", ["eager", "graph", "compile_graph"])
def test_diffusion_seed_is_independent_of_trunk_and_graph(family, mode):
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.zeros(1, device="cuda")))
    model.structure_head = nn.Module()
    policy = ExecutionConfig(
        compile=mode == "compile_graph", cuda_graph=mode != "eager"
    )
    denoise = ExecutedCallable(
        lambda x, sigma: x * 0.9 + sigma * 0.01, policy, "seed-test"
    )
    original_trunk_generator = torch.Generator(device="cuda").manual_seed(7)

    def sample(*, generator=None, rollout_seed=None) -> tuple:
        if family == "esmfold2":
            assert generator is not original_trunk_generator
        if family in {"protenix", "opendde"}:
            assert rollout_seed in {19, 20}
            generator = torch.Generator(device="cuda").manual_seed(rollout_seed)
        coords = EulerSampler().sample(
            denoise,
            (5, 8, 3),
            torch.tensor([3.0, 2.0, 1.0, 0.1], device="cuda"),
            device="cuda",
            generator=generator,
        )
        return coords, (random.random(), np.random.random())

    owner, method = (
        (model.structure_head, "sample")
        if family == "esmfold2"
        else (model, "_sample_diffusion" if family == "af3" else "sample_diffusion")
    )

    def run(trunk_seed, diffusion_seed) -> tuple:
        setattr(owner, method, sample)
        bind_sampling_seed(model, family, diffusion_seed)
        seed_all(trunk_seed)
        trunk = torch.randn(4, device="cuda")
        state = RNGState.capture()
        generator_state = original_trunk_generator.get_state()
        with torch.no_grad():
            result, host = getattr(owner, method)(
                **(
                    {"generator": original_trunk_generator}
                    if family == "esmfold2"
                    else {}
                )
            )
        after = torch.rand(4, device="cuda")
        state.restore()
        assert torch.equal(after, torch.rand(4, device="cuda"))
        assert torch.equal(generator_state, original_trunk_generator.get_state())
        return trunk, result, host

    first = run(7, 19)
    repeat = run(7, 19)
    other_trunk = run(8, 19)
    other_diffusion = run(7, 20)
    assert torch.equal(first[1], repeat[1])
    assert torch.equal(first[1], other_trunk[1])
    assert first[2] == repeat[2] == other_trunk[2]
    assert first[2] != other_diffusion[2]
    assert not torch.equal(first[0], other_trunk[0])
    assert torch.equal(first[0], other_diffusion[0])
    assert not torch.equal(first[1], other_diffusion[1])
    assert not torch.equal(first[1][0], first[1][1])
