# Tests

Run `pytest tests` from the repository root. Tests are grouped by responsibility:

| Directory | Scope |
|---|---|
| `data/` | CCD, MSA/template LMDB, input/output contracts |
| `models/` | Loading, prediction, composition and frozen forward equivalence |
| `kernels/` | Backend dispatch, fused operators and precision contracts |
| `runtime/` | Bucketing, execution, cache supervision and output destinations |
| `utils/` | Shared numerical, serialization and process-state utilities |
| `references/` | Frozen forward bodies and source identities used by regressions |

`support.py` owns repository and reference paths. `conftest.py` gives persistence
tests temporary run roots; tests never write into real prediction runs. GPU tests
skip when no GPU is allocated. Execute them on a Slurm compute node. Prepared
large input assets live under `validation/inputs`, not inside the test package.
