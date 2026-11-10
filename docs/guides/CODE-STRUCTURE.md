# FoldForge code ownership

Every model is a row of one graph. `foldforge.load_model(name, config, **options)`
and `foldforge fold <model>` run the same lifecycle for all of them:
registry entry -> `DenseSpec` row (`spec_for(family, mode)`) -> the one
architecture (`models/architectures/af3.py`, built from miniworld-engine classes
plus FoldForge-own modules) -> strict load of the miniworld-format checkpoint ->
precision (`master` | `fp32`) -> backend plan applied to the engine modules
(`pytorch | cuda | triton | anthropic | cueq`, aliases `miniworld` / `cuequivariance`).
A new release is a registry entry and a spec row, not new code.

```text
src/foldforge/
  __init__.py          public load_model / load_inputs / predict, configuration and results
  api.py               device-bound predictor, whole and staged inference
  inputs.py            reusable input loading and early validation
  stages.py            model-bound trunk state and lifetime
  lifecycle.py         atomic object leases and process-wide policy serialization
  errors.py            public lifecycle and input exceptions
  result.py            CPU results, CIF strings and explicit saving
  cli.py               Rich/Click commands over the public Python API
  prediction.py        common prediction result type
  diffusion/          EDM / decoupled x-pred processes, augmentation, schedules, samplers, SE(3) helpers
                     (self-contained: imports no other foldforge module, shared with MiniWorld)
  models/
    __init__.py        the registry: one Entry per released model
    architectures/     af3.py, the only network
    checkpoints/       lookup, the miniworld format (miniworld.py: load, `convert`), checkpoints.lock, provenance
      legacy/          the released-layout dense graph, Haiku codec and released -> engine conversion rules (released.py): imported ONLY by the offline converter
    config/            the runtime configuration schema
    io/                request/runtime, input adapters, language-model glue, output
    loading.py         one strict loading lifecycle
    execution.py       bounded compile + CUDA-graph callables, complete-forward timing
    bucketing.py       inference padding ladders (tokens / atoms / MSA rows)
    sampling.py        engine sampler adaptation, sampling-seed binding
  modules/
    dense/             FoldForge-own pieces of the graph (featurization, atom cross attention, diffusion head + Fourier features, template embedders
                       + their mean, conditioned-norm / SwiGLU equations of ESMFold2's atom stack,
                       heads, ESMFold2's own atom stack) and DenseSpec (spec.py); confidence decoding
    engine/            the engine as FoldForge uses it
      graph.py         pairformer / MSA / DiT pieces from engine classes; the converter's single switch (use_dense_blocks)
      modules.py       factories of the engine classes, model keys, precision
      pairformer.py    the engine PairformerBlock as each family needs it
      transformers.py  TokenDiTStack / AtomDiTStack
      backends.py      backend names, aliases, ops, the plan; policy.py the per-module guard and fallback / error
      install.py       applies the plan to a loaded model; configure.py precision, device, TF32 policy, matmul-precision restore
      capture.py       the ONLY import of engine-private modules (graph-capture scopes, weight-cache clearing; public API requested in ENGINE-REQUESTS.md)
      serving.py       host-side observations of the guarded modules
    esmc/              the vendored ESM-C model
    language_model.py  ESM2 / ESM-C token and pair streams
  guidance/            TFG guidance (not wired into the AF3 sampler yet)
  data/
    ccd/               the MiniWorld BioMol CCD database
    constants/         chemical tables and per-frame reference geometry
    features/          dense featurization, family conventions, structural tokens
    inputs/            MiniWorld input specs, FASTA/LMDB resources, validation
  eval/                references.py (the judge), structure I/O
  utils/               atom_layout (gather tables), dense_geometry, seeding, tensor helpers
```

## Who owns what

| Concern | miniworld-engine | FoldForge |
|---|---|---|
| Modules (Transition, triangle ops, attention, PairformerBlock, token / atom DiT, sampler glue) | classes, parameters, `state_dict` keys | builds them per family (`modules/engine/modules.py`, `pairformer.py`, `transformers.py`), never re-implements them |
| Kernels and their backends (`pytorch`, `cuda`, `triton`, `anthropic`, `cueq`) | the kernels, pretuned launch configs, `module.unsupported(...)`, `BackendUnsupportedError` | none |
| Backend POLICY: request -> `unsupported()` -> fallback / error | none (the engine never falls back) | `modules/engine/backends.py`, `policy.py`, `install.py` |
| Models, spec rows, featurization, checkpoints (miniworld format, offline converter) | none | `models/`, `modules/dense/`, `data/` |
| Diffusion process, sampling, guidance, evaluation, API, CLI | none (sampler glue modules only) | `diffusion/`, `models/sampling.py`, `guidance/`, `eval/`, `api.py`, `cli.py` |
| CUDA-graph execution scope | private capture hooks | `models/execution.py`; the one private-import file is `modules/engine/capture.py` |

## Where to make a change

| Change | Owner |
|---|---|
| Add a release | `models/__init__.py` (entry), `modules/dense/spec.py` (row), `models/checkpoints/__init__.py` (default blob) |
| A release computes something differently | a named field on `DenseSpec`, and the one place in the graph that reads it |
| Public prediction API / result | `api.py`, `result.py`; CLI flags in `models/io/options.py` |
| Choose backend or precision at load | `models/loading.py`, `modules/engine/backends.py` ([backends](BACKENDS.md)) |
| Serialized weight names or packing | `models/checkpoints/miniworld.py` (the format), `models/checkpoints/legacy/` + `models/checkpoints/legacy/released.py` (the offline converter; [format](CHECKPOINT-FORMAT.md), [checkpoints](CHECKPOINTS.md)) |
| Input conventions a family needs | `data/features/dense_conventions.py` |
| Shared attention, triangle, MSA, transition equations | `miniworld_engine` (the classes), `modules/engine/pairformer.py` (how a family builds the engine block) |
| An accelerated kernel, or how a call picks it | `miniworld_engine`; FoldForge only chooses a backend per module (`modules/engine/backends.py`, `policy.py`) |
| Something the engine lacks | [ENGINE-UNSUPPORTED.md](../ENGINE-UNSUPPORTED.md), requests in [ENGINE-REQUESTS.md](../ENGINE-REQUESTS.md) |

## What is not here

As of 2026-09-27 the tree holds only code that some entry point runs. The
second, Protenix-style data pipeline (tokenizer, featurizer, parser, template
and MSA featurizers, dataset/loader/cropping), the training library, the
notebook web service, permutation matching, and the vendored Protenix/OpenDDE
configuration trees were removed: none was loaded by any fold, `validate`,
`checkpoints` or `ccd` run, and none was imported by a script. They remain in
git history (before this change) if a training stack is ever built on the one
graph; it should then read the same dense featurization inference does.
