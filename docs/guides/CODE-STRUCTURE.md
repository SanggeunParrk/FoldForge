# FoldForge code ownership

Every model is a row of one graph. `foldforge.load(name, checkpoint, **options)`
and `foldforge fold <model>` run the same lifecycle for all of them:
registry entry -> `DenseSpec` row (`spec_for(family, mode)`) -> the one
architecture (`models/architectures/af3.py`) -> strict checkpoint import ->
precision policy -> backend installation (team-gm / miniworld-engine blocks).
A new release is a registry entry and a spec row, not new code.

```text
src/foldforge/
  __init__.py          public load, checkpoint discovery and Prediction API
  cli.py               models / ccd / checkpoints / validate / fold
  prediction.py        common prediction result type
  models/
    __init__.py        the registry: one Entry per released model
    architectures/     af3.py, the only network
    checkpoints/       lookup, the blob codec, checkpoints.lock and provenance
    config/            the runtime configuration schema
    io/                request/runtime, input adapters, language-model glue, output
    loading.py         one strict loading lifecycle
    execution.py       compile and CUDA-graph ownership
    bucketing.py       inference padding policies
    sampling.py        team-gm sampler adaptation
    precision.py       parameter/normalization precision policy
  modules/
    dense/             the graph's blocks and DenseSpec (spec.py)
    ops/               shared operator entry points
    language_model.py  ESM2 / ESM-C token and pair streams
  data/
    ccd/               the MiniWorld BioMol CCD database
    constants/         chemical tables and per-frame reference geometry
    features/          dense featurization, family conventions, structural tokens
    inputs/            MiniWorld input specs, FASTA/LMDB resources, validation
  eval/                references.py (the judge), confidence decoding, structure I/O
  utils/               geometry, seeding, logging and tensor helpers
```

## Where to make a change

| Change | Owner |
|---|---|
| Add a release | `models/__init__.py` (entry), `modules/dense/spec.py` (row), `models/checkpoints/__init__.py` (default blob) |
| A release computes something differently | a named field on `DenseSpec`, and the one place in the graph that reads it |
| Choose backend or precision at load | `models/loading.py` |
| Serialized weight names or packing | `models/checkpoints/`, the converter (see [checkpoints](CHECKPOINTS.md)) |
| Input conventions a family needs | `data/features/dense_conventions.py` |
| Shared attention, triangle, MSA, transition equations | `team_gm.modules` |
| An accelerated kernel | `miniworld_engine` |

## What is not here

As of 2026-09-27 the tree holds only code that some entry point runs. The
second, Protenix-style data pipeline (tokenizer, featurizer, parser, template
and MSA featurizers, dataset/loader/cropping), the training library, the
notebook web service, permutation matching, and the vendored Protenix/OpenDDE
configuration trees were removed: none was loaded by any fold, `validate`,
`checkpoints` or `ccd` run, and none was imported by a script. They remain in
git history (before this change) if a training stack is ever built on the one
graph; it should then read the same dense featurization inference does.
