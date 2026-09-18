# Documentation

- [Benchmark results](benchmark_results.md): five inference execution modes across four models, latency tables and plots.
  Rendered by `scripts/render_benchmark_results.py`; the change summary after the
  headline table is [benchmark_notes.md](benchmark_notes.md), inserted on every render. The
  2026-09-16 report is preserved in [archive/benchmark-20260916](archive/benchmark-20260916/benchmark_results.md).
- [Code structure](guides/CODE-STRUCTURE.md): shared runtime and code ownership.
- [Input and database formats](guides/MINIWORLD-FORMAT.md): CCD, MSA-LMDB and template-LMDB.
- [Data and evaluation](guides/DATA-AND-EVALUATION.md).
- [Model integration](guides/MODEL-INTEGRATION.md) and [checkpoint variants](guides/CHECKPOINT-VARIANTS.md).
- [Kernel cache builds](guides/CACHE-BUILD.md).
- [Code quality](guides/CODE-QUALITY.md) and [validation](guides/QUALITY-VALIDATION.md).

`archive/` preserves dated audits, migration records and their raw JSON/CSV evidence.
These are historical snapshots, not the current benchmark table. New benchmark
figures and compact measurement data belong in `assets/`.

The current benchmark pins `libs/team-gm` to `92a50c9` on branch
`foldforge/msa-bucket-1024`: the validated `a823f69` runtime plus a 1024-row MSA
inference bucket. The 2026-09-16 release used `a823f69` itself, which is also
included in `exp/miniworld`.
Use `git submodule update --init --recursive` to reproduce this release;
`git submodule update --remote` selects newer runtime and dependency changes
that are not covered by these recorded measurements.
