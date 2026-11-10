# Documentation

- [Weight downloads](guides/WEIGHTS.md): multi-model download, conversion, verification, language models and offline import.
- [Checkpoint format](guides/CHECKPOINT-FORMAT.md): the fp32 engine-key `state_dict` every model loads, how the default file is found, and the one-time `foldforge weights convert` from the released blob.
- [Python API and CLI](guides/PYTHON-API.md): `load_model()` / `load_inputs()`, reusable models, trunk/diffusion test-time scaling, device ownership, in-memory results, optional saving and shared command-line execution.
- [Benchmark results](benchmark_results.md): 5I28 replicated N times on A100 and A6000; five inference
  execution modes across four models at 512 tokens, and a length ladder to the out-of-memory point.
  Rendered by `scripts/render_length_benchmark.py`; [benchmark_notes.md](benchmark_notes.md) is inserted on
  every render. Earlier reports: [4YX2, 2026-09-18](archive/benchmark-20260918-4yx2/benchmark_results.md)
  and [2026-09-16](archive/benchmark-20260916/benchmark_results.md).
- [Code structure](guides/CODE-STRUCTURE.md): shared runtime and code ownership.
- [Backends](guides/BACKENDS.md): `pytorch | cuda | triton | anthropic | cueq` (aliases `miniworld`, `cuequivariance`), `fallback | error`, precision `master | fp32`, how to compare backends. Measurements of the pre-2.5.0 engine: [archive](archive/BACKENDS-engine-2.3.md).
- [What the engine does not support](ENGINE-UNSUPPORTED.md) (categorized: what, where, what FoldForge does meanwhile) and [requests to the engine team](ENGINE-REQUESTS.md) (numbered, prioritized).
- [Configuration](guides/CONFIG.md): the `Config` keys, `engine.skip`, the removed `engine_cache_dir` / autotune, and the environment variables that are now config.
- [Execution scopes](guides/EXECUTION-SCOPES.md): what `compile` and `cuda_graph` wrap (`denoiser`, `trunk`), the rules for captured code, and that ESM-C optimisation is out of scope (taken from Anthropic's kit).
- [Input and database formats](guides/MINIWORLD-FORMAT.md): CCD, MSA-LMDB and template-LMDB.
- [Data and evaluation](guides/DATA-AND-EVALUATION.md).
- [Model integration](guides/MODEL-INTEGRATION.md) and [checkpoint variants](guides/CHECKPOINT-VARIANTS.md).
- [Code quality](guides/CODE-QUALITY.md) and [validation](guides/QUALITY-VALIDATION.md).

`archive/` preserves dated audits, migration records and their raw JSON/CSV evidence.
These are historical snapshots, not the current benchmark table. New benchmark
figures and compact measurement data belong in `assets/`.

The recorded benchmarks (before the team-gm dependency was removed, commit `f3e6128`) pinned `libs/team-gm` to `92a50c9` on branch
`foldforge/msa-bucket-1024`: the validated `a823f69` runtime plus a 1024-row MSA
inference bucket. The 2026-09-16 release used `a823f69` itself, which is also
included in `exp/miniworld`.
To reproduce those releases, check out `f3e6128` and run `git submodule update --init --recursive`;
`git submodule update --remote` selects newer runtime and dependency changes
that are not covered by these recorded measurements.
