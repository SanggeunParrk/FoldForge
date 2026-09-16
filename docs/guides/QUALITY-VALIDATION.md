# Quality validation — 2026-09-15

FoldForge uses team-gm's global Ruff `ALL` policy and numerical-code exceptions,
with Pyright in standard mode over every FoldForge source file. See
[CODE-QUALITY.md](CODE-QUALITY.md) for the coverage and documented exceptions.

| Check | Result |
| --- | --- |
| Ruff 0.15.22 lint | Pass, all `src`, `tests`, `scripts` |
| Ruff formatting | Pass, 232 Python files |
| Pyright 1.1.408 | 191 source files, 0 errors, 0 warnings |
| CPU regression suite | 245 passed, 159 skipped (CUDA cases and external fixtures) |
| A100 full regression run | 397 passed, 7 external-fixture cases skipped |
| CLI | `foldforge --help`, `foldforge models`, and help for all four model entrypoints passed |
| Packaging | Wheel build passed |
| Development setup | Lock consistency and pre-commit configuration validation passed |

Both local and cssb3 working trees passed the static checks after applying the
changes. Tests ran on allocated cssb3 compute nodes, including an A100 80GB PCIe;
GPU work was not run on login nodes. The final A100 full run
validates all 397 available cases in the 404-case suite. The seven skipped
cases require external MiniWorld integration fixtures. Versioned numerical
reference formulas and comparison tolerances remain unchanged.

This verifies regression behavior and quality checks, not full checkpoint
structure-prediction accuracy or new performance claims. Some A100 kernel shapes
used existing heuristic cache fallbacks; this change does not rebuild the engine
cache. CUDA/model dependency versions and source revisions in `uv.lock` are
unchanged; additions are development tooling and typing dependencies.

Run the static checks again with:

```bash
bash scripts/check_quality.sh
```

The final data layout has seven subpackages instead of thirteen. Imports, CLI
entrypoints, test lookup maps, provenance destination paths, and lint-path rules
follow the new layout. Comparing source ASTs before and after the relocation
found no changes outside imports and docstrings. Wheel contents contain only the
seven retained data subpackages; retired package directories are removed.
