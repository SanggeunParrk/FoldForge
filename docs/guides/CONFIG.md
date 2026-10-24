# Configuration

Every option that changes what a fold computes or how it runs is a key of the nested Pydantic `Config`
(`src/foldforge/models/config/runtime.py`), given with `foldforge fold <model> --spec input.yaml --config runtime.yaml`.
A key left out has its default, and the defaults are the behaviour without any override. The config written next to a
run (`model.config.json`) records every key.

```yaml
backend: miniworld          # miniworld | pytorch | cuequivariance
precision: bf16             # bf16 | fp32 | af3_default | model_default
mode: fast                  # fast | exact
trunk: {recycles: null, msa_depth: null}
diffusion: {steps: null}
execution:
  compile: false
  cuda_graph: false
  scope: denoiser           # denoiser | trunk | model
  confidence_sample_chunk: null   # samples per confidence-head call; null = the head's rule, 1 = sequential
engine:
  skip: []                  # parts left dense, see below
debug:
  dense_spec_override: null # DenseSpec fields to replace, e.g. {confidence_reembed_pair: true}
```

## `engine.skip`

A debugging aid: the engine install leaves the named parts as the dense PyTorch modules, so one kernel family can be
priced or ruled out at a time. Allowed values: `pairformers`, `conditioning`, `engine_trimul`,
`engine_triangle_attention`, `engine_msa_pwa`, `engine_opm`, `engine_dit_blocks`, `engine_single_attention`,
`engine_atom_blocks`, `engine_dit_variant_blocks`, `engine_atom_variant_blocks` (the keys of the load report's engine
counts). An unknown value is a config error. From Python: `load(..., engine_skip=("engine_opm",))`.

## `debug.dense_spec_override`

Replaces fields of the family's `DenseSpec` after it is chosen, to flip one forward convention of a release and
measure it. A field that changes a parameter shape makes the strict weight load fail loudly.

## What stays an environment variable

Locations and the machine, not behaviour: `FOLDFORGE_HOME`, `FOLDFORGE_CHECKPOINT_DIR`, `FOLDFORGE_CCD_DB`,
`FOLDFORGE_RUNS`, `FOLDFORGE_RELEASE_ROOT`, `FOLDFORGE_BENCHMARKS`; `CUDA_VISIBLE_DEVICES`, `SLURM_*`, `XDG_CACHE_HOME`,
`TORCH_EXTENSIONS_DIR`, `CUTLASS_PATH`; `CUBLAS_WORKSPACE_CONFIG` (set by `utils.seed` for deterministic cuBLAS); the
test inputs `FOLDFORGE_TEST_CCD_DB` and `MINIWORLD_TEST_SOURCE`.

## Removed variables

These used to change behaviour and are now an error if they are set (`RuntimeError` from `Config(...)` and from the
loader), never silently ignored:

| removed | now |
|---|---|
| `FF_ENGINE_SKIP` | `engine.skip` |
| `FOLDFORGE_DENSE_SPEC_OVERRIDE` | `debug.dense_spec_override` |
| `FOLDFORGE_CONFIDENCE_SAMPLE_CHUNK` | `execution.confidence_sample_chunk` |
| `FF_TRUNK_BF16` | deleted: an experiment (bf16 trunk on an fp32 release under `af3_default`), concluded; the released dtypes stay |
