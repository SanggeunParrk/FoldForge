# Code quality

Run all static checks from an already installed FoldForge environment:

```bash
bash scripts/check_quality.sh
```

The script runs Ruff lint, Ruff formatting, and Pyright. It does not change the
CUDA environment or synchronize packages. Install the development group through
the normal environment setup before using it. Optional commit hooks use the same
commands (`uv run --no-sync pre-commit install`).

## Coverage and policy

- Ruff checks every Python file in `src`, `tests`, and `scripts`, selects `ALL`,
  targets Python 3.12, and uses an 88-character line length. Global exceptions
  match team-gm's Ruff policy.
- Pyright checks all `src/foldforge` with `typeCheckingMode = "standard"` and the
  project `.venv`. Tests and scripts remain included in Ruff.
- Vendored libraries, archived validation outputs, weights and generated runs
  are separate projects/data, not FoldForge source. No FoldForge source subtree
  is excluded from either source check.
- Numerical modules use the same notation/annotation exceptions as team-gm's
  diffusion and SE(3) code. Ordered chemistry and MSA routines retain explicit
  complexity exceptions while their types and other lint rules remain checked.
- Function-local comments document retained checkpoint keyword names, upstream
  extension hooks, date-only PDB cutoffs, and coherent algorithm/protocol stages.
  These exceptions do not disable checks for newly added functions. An unused
  suppression is itself a Ruff error.
- Heterogeneous configuration, serialized feature trees and provider boundaries
  use explicit `Any` where their schema is dynamic. Tensor and record contracts
  use concrete types; type-error reporting is not disabled globally.

`typings/biotite-stubs` is a partial overlay for the installed Biotite 1.7.1 stubs.
Its README and license describe its provenance and the dynamic atom annotation
API. It does not replace runtime Biotite code. Recheck it when upgrading Biotite.
Pyright's Node runtime is locked with its development dependencies so an old
system Node installation cannot change the checker used by the project.

## Runtime verification

Run `python -m pytest tests -q` on an allocated compute node after static checks.
CPU runs skip the CUDA backend and CUDA graph cases; an allocated GPU run is
needed for those. The versioned numerical oracles are kept unchanged. Test-only
adapters restore their original import namespace and checkpoint policy when
comparing them with shared modules.

Offline ESM2 embedding generation requires fair-esm, which shares an import name
with FoldForge's ESMC package. Use a separate fair-esm environment for that utility;
do not replace the pinned ESMC dependency in the prediction environment.

## What an import scan cannot tell you

A scan for "which modules does nothing import" finds ENTRY POINTS as readily as
dead code, and cannot tell them apart. Run against this tree on 2026-09-23 it
flagged five things, and every one of them was live:

| flagged | what it actually is |
|---|---|
| `data/web/viewer.py`, `data/web/visualization.py` | the remote service UI and visualization, named as such in [data and evaluation](DATA-AND-EVALUATION.md); consumed by notebooks, not by Python imports |
| `training/` (3,200 lines) | the loss, optimizer and schedule library, documented in three places; the inference CLI is not supposed to call it |
| `modules/flat/frames.py` | `training/loss.py`'s dependency |
| `data/constants/periodic_table.py` | vendored AF3 data under CC BY-NC-SA; removing it is a provenance decision, not a cleanup |
| `Input.af_family()` | writes the AF-family JSON that [model integration](MODEL-INTEGRATION.md) calls an explicit checkpoint-compatibility interface |

So the scan is a place to start asking, never a list to delete. What DID make a
safe deletion, twice, was the opposite evidence: a layout with zero registry
entries, whose every branch was unreachable by construction.
