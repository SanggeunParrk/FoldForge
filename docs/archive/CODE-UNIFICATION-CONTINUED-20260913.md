Current contract: [MiniWorld formats](../guides/MINIWORLD-FORMAT.md). Current execution qualification: [2026-09-14](QUALIFICATION-20260914.md). Dated results below are historical and do not override these documents.

# Code unification continuation — 2026-09-13

This pass follows [the first CCD/Pairformer pass](CODE-UNIFICATION-20260913.md).
Validation was performed in a separate candidate tree while the original dirty
FoldForge checkout was preserved. No checkpoint, CCD asset, dependency pin or
branch was replaced.

## CCD answer

All five qualified model variants select the same `CCDDatabase` and manifest.
This follows MiniWorld's explicit database selection and per-database cache
ownership, **not its physical BioMol LMDB schema**. FoldForge currently retains
the full CIF, prepared RDKit reference molecules and derived AF3 category index.
MiniWorld's processed molecule schema is not a substitute for all of those
source categories and conformer metadata. No LMDB interchange compatibility is
claimed or silently fabricated.

## Shared computation and explicit boundaries

| Component | Shared implementation | Model-specific boundary retained |
| --- | --- | --- |
| MSA block | `team_gm.modules.blocks.composition.msa_pair_update`, frozen `MSAUpdateConfig` | AF3/Protenix OPM-first; OpenDDE MSA-first, including last-block behavior |
| MSA row update | `msa_row_update` with chunking on the actual MSA axis | Released attention parameters, AF3 MSA mask and OPM denominator/projection convention |
| Token and atom diffusion ordering | `conditioned_residual`, also used by native team-gm diffusion | Pair super-block logits, query/key gathering, local windows, extra OpenDDE bias and residual-path dropout |
| AdaLN and conditioned transition | Loaded projections/norms mapped to engine `AdaptiveLayerNorm` and `ConditionedTransition` | AF3 packed SwiGLU, source epsilon and sigmoid placement; explicit non-batched and mixed-precision adapters |
| Template aggregation | `template_embedding_mean` after common Pairformer processing | Model-specific template features and asymmetric 64→128 template contractions |
| Projection precision | `team_gm.modules.precision.NativeLinear` | Optional FP32 geometry GEMM, with the original parameter-dtype input/output boundary |

Learned non-normalization parameters remain BF16 and normalization parameters
remain exact FP32. Fixed geometry/Fourier tables and coordinate-derived FP32
activations are preserved. All former Linear input-casting forward hooks are
removed. There is no implicit autocast in the new precision module.

Each AF-family model maps **75 AdaLN modules and 33 conditioned transitions**.
The optimized and reference engine modules share parameter storage. A mixed
FP32 activation/BF16 weight pair selects the explicit engine reference equation,
which normalizes before the projection cast; it is not sent to an unsupported
mixed-dtype fused GEMM. These counts measure converted modules, not the number
of fused GPU calls. Plain transition/TriMul retain their earlier compatibility
adapters; compatible conditioning now has converted engine module state.

Unbatched `[L,D]` inputs acquire a semantic batch axis before engine dispatch,
so cache-key construction receives `[B,L,D]`. Broadcasting conditioning does
not discard the token/atom length. Both shape handling and mixed precision were
caught by full-model candidate runs and added to regressions before promotion.
Failed candidate logs are retained.

Template accumulation stays sequential, preserving BF16 reduction order.
An explicitly empty template set now returns a shaped zero projection for
Protenix/OpenDDE instead of attempting a linear projection on a Python integer.
The existing disabled-template (`n_blocks=0`/missing feature) contract remains.

## Qualification

A6000 gpu03, Slurm **1683200**. AF3/Protenix v1/v2/OpenDDE used the actual strictly
loaded checkpoints on 1UBQ, 10 recycles, 200 diffusion steps, one sample and the
same prepared MSA. ESMFold2 used 3 loops, 14 steps, cached ESMC and MSA 512.
Every run used the same CCD hashes. Compile and CUDA graphs were **off** for this
correctness comparison; cold runtime is not a controlled speed benchmark.

| Model | CA RMSD vs preceding pass (Å) | CA RMSD vs deposit (Å) | Summary pLDDT (0–100) |
| --- | ---: | ---: | ---: |
| af3 | 0.043 | 0.697 | 91.220 |
| protenix | 0.104 | 2.584 | 94.066 |
| protenix-v2 | 0.204 | 2.748 | 93.331 |
| opendde | 0.055 | 1.042 | 94.816 |
| esmfold2 | 0.120 | 0.670 | 80.078 |

All comparisons match 76 CA identities with finite coordinates. This is a
structural regression, not bitwise equivalence. Confidence output units stay
unchanged from the corrected common prediction contract.

The tests compare nonzero released weights for 128/128 atom and 768/384 token
conditioning, BF16 and FP32 residual inputs, semantic batch axes, broadcasting,
MSA masks and last-block ordering. Additional template cases include zero,
one and two templates, interchain and pair masks; atom tests cover local query/
key windows. Frozen pre-change forward bodies are versioned under
`tests/references`, so applying the patch cannot turn the comparison
into a comparison of the new function against itself. Constructors and unchanged
primitive operators remain shared and are not an independent full-model oracle.

**127 FoldForge tests passed**, plus the root team-gm diffusion block forward/backward
comparison (**1 passed**, 51 other parameterizations deselected). New code and adapted
imports pass Ruff. The inventory contains 145 forward-bearing classes; this is
not a completion percentage.

Validation logs and structures currently live under root team-gm's ignored
`validation/unify2-*` paths, with `unify2-results/comparison.json`. The original
first-pass outputs remain in FoldForge's `validation/results/unify-20260913`.

## Remaining limits

This does not claim that all retained source classes have been rewritten with
nested Pydantic configurations. Released stack/checkpoint shells and some raw
attention equations remain as compatibility code. The updated class inventory
labels shared composition separately from complete engine operator conversion.
Atom gather/scatter and model-specific feature/head definitions remain at the
terminal-model boundary. ESMFold2's DiT modulation is intentionally distinct
from AF3 sigmoid-scale AdaLN.

Schedulers were inspected, not silently substituted: Protenix keeps a nonzero
last noise level, OpenDDE overwrites its last value with zero, and team-gm's EDM
scheduler uses a different sampling grid before its appended zero. RNG,
augmentation, guidance and distributed behavior likewise require separate
qualification. Full ligand/complex inference, Fold-CP/distributed execution,
training, compile/graph performance and complete cache coverage are not certified
by these single-GPU inference tests. Missing-shape heuristic warnings remain.

Root team-gm: `exp/miniworld-integrated`; FoldForge: `main`; its team-gm member:
`exp/miniworld`. No commit or push is included in this continuation.

The validated files were promoted with before/after SHA256 checks and backups
in `validation/unify2-before`. Actual-environment imports resolve to FoldForge
and its member team-gm without the candidate PYTHONPATH. Post-promotion smoke
checks passed **13 tests** (58 other cases deselected), and `uv pip check`
confirmed all 175 installed packages are compatible. The smoke log is
`/home/psk6950/practice/team-gm/validation/logs/unify2-installed-check.log`.
