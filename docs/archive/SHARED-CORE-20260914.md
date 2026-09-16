# Shared diffusion and neural core — 2026-09-14

Implementation ownership moved from FoldForge's model copies to team-gm. The
released checkpoint parameter names and input layouts remain at the model boundary.
This change preserves numerical conventions instead of forcing different learned
models to use one schedule, coordinate layout or augmentation distribution.

| Responsibility | Shared implementation in team-gm | Remaining model boundary |
| --- | --- | --- |
| Churn, initial noise, Euler loop, sample chunking | `diffusion/edm/sampling.py` | Bind a clean-coordinate denoiser |
| ESM/legacy AF3 solver | `diffusion/edm/preconditioned.py`, `solver.py` | Compatibility constructors and scheduler coefficients |
| Power-law schedule, log-normal training noise | `diffusion/edm/schedules.py` | Endpoint/grid choices in immutable Pydantic configuration |
| Coordinate preconditioning | `diffusion/edm/preconditioning.py` | Standard/ratio formulation and checkpoint dtype policy |
| Rigid augmentation, training corruption | `diffusion/augmentation.py`, `edm/training.py` | Coordinate layout, explicit Torch/NumPy generators and masks |
| Training-free guidance | `diffusion/guidance/` | Denoiser argument binding |
| Transition, AdaLN, conditioned transition | `modules/blocks/transition_math.py`, `modules/checkpoints/` | Packed/separate projection weights and memory policy |
| Triangle multiplication, including in-place path | `modules/blocks/triangle_math.py`, `triangle_inplace.py` | Direction, weight layout and residual ownership |
| Fourier features, local attention, triangle attention projections | `modules/blocks/fourier_math.py`, `local_attention.py`, `modules/checkpoints/attention.py` | Heads/window sizes and released chunk/backend calls |
| Activation checkpoint groups | `modules/checkpointing.py` | Released positional arguments; shared non-reentrant policy |
| PTM/iPTM and shared confidence calculations | `metrics/confidence.py` | Model-specific confidence head and output assembly |

The duplicate FoldForge ESMFold2 solver and augmentation modules were deleted.
The two AF-family `generator.py` files and six TFG modules are compatibility
imports/binders. They no longer own sampling or guidance algorithms. FoldForge's
`models/sampling.py` binds released argument names; the reverse-time loop lives
only in team-gm's Euler sampler. Existing decoupled translation/rotation diffusion
in team-gm remains a different algorithm.

## Numerical and compiler regressions caught during migration

- Native-BF16 preconditioning must preserve zero-dimensional scalar coefficients.
  Expanding a scalar coefficient can change Torch type promotion for mixed
  FP32 coordinates/BF16 updates. Both helper math and the actual patched model
  methods are compared against the original source.
- Constructing `TransitionConfig()` as `getattr`'s default inside forward executed
  Pydantic validation even when the attribute existed. OpenDDE's denoiser split
  from one Dynamo graph into seven. Eager outputs at 132 trace points matched,
  but compiled predictions changed (maximum coordinate difference 2.990234375).
  Moving immutable transition/triangle configurations out of forward restored
  one graph and exact coordinates, pLDDT, PAE and PTM. Four `fullgraph=True`
  regression tests cover these adapters. Neither compile nor CUDA graphs were
  disabled to obtain agreement.
- Batched masked augmentation used an incorrect center-denominator broadcast.
  The denominator now retains its last dimension; masked geometry is tested
  for single and multiple batch dimensions.

## Validation

Job 43697 on cssb3's allocated A100 gpu04 completed successfully in 5:29.
It passed 375 core tests, 29 backend/bucketing checks, and real checkpoint
compile+CUDA graph checks for all four models. The final activation-checkpoint
consolidation passed another 35 output/gradient/dropout-RNG tests in job 43699.
Job 43700 passed all 410 core tests, the 29 backend/bucketing checks, and the
affected Protenix/OpenDDE real-checkpoint runs. Final predictions still match
the frozen originals exactly for all four models.

| Model | Compared tensor/array fields, including saved features where available | Original vs common-core maximum absolute difference |
| --- | ---: | ---: |
| AF3 | 12 | 0 |
| ESMFold2 | 5 | 0 |
| Protenix | 59 | 0 |
| OpenDDE | 73 | 0 |

All comparisons use `atol=0, rtol=0`; tolerances were not relaxed. The output
report includes coordinates/confidence, and AF3's full confidence NPZ. Graph
replays also match the same compiled callable without capture at zero maximum
absolute difference. These checks establish migration equivalence, not predictive
accuracy with a two-step diffusion smoke profile.

The 125 changed source/test files match byte-for-byte between the local and
cssb3 workspaces. Source hashes and branch/base-commit identifiers are recorded in
`shared-core-source-manifest-20260914.json`; numerical results are in
`shared-core-prediction-equivalence-20260914.json`. Full logs, compiled/eager layer traces
and frozen sources remain under `validation/shared-core-20260914/` on cssb3.

Migration comparisons use the frozen pre-change source, retained outside Git at
`validation/shared-core-20260914/reference-src` and `reference-team-gm-src` on cssb3.
Set `FOLDFORGE_REFERENCE_SRC`, `TEAM_GM_REFERENCE_SRC`,
`FOLDFORGE_SAMPLING_ADAPTER`, `FOLDFORGE_PRECONDITIONED_CANDIDATE`, and
`TEAM_GM_SOLVER_CANDIDATE` as in the validation job to run all migration tests.
Without snapshots, the old-versus-new tests explicitly skip; standalone sampler,
geometry, schedule and fullgraph tests still run. Trace files and failed-before-
fix predictions are preserved under the same validation directory.

These are correctness smokes: native BF16 with FP32 norms, cuEquivariance backend,
1UBQ, one recycle, two diffusion steps and one sample. Compile and CUDA graph
capture cover the denoiser. AF3/ESMFold2 use token/atom/MSA buckets 128/1024/2048;
Protenix/OpenDDE remain unbucketed for this profile. No end-to-end speed claim or
full three-backend/model/training qualification follows from these small smokes.

## Follow-up: common checkpoint layers and FoldCP removal

The follow-up described in [MODEL-CORE-UNIFICATION-20260914.md](MODEL-CORE-UNIFICATION-20260914.md)
deletes the entire FoldCP package and the duplicated Protenix/OpenDDE operator
files. Layer construction, atom/token transformers, MSA/template stacks,
denoising, confidence heads, initialization, chunking and recycling now have
common implementations in `team_gm.modules.checkpoints`. Constructor settings
preserve the released checkpoint differences.

Model loading, feature/output adapters and AF3/ESMFold2 checkpoint layout assembly
still remain at the model boundary. Protenix/OpenDDE whole-input bucketing and
production-size end-to-end validation are separate unfinished integration work.
The older source/count reports above describe the earlier phase; the follow-up
report and manifest describe the final source state.

Working branches remain `team-gm: exp/miniworld`, `FoldForge: main`, and
`FoldForge/libs/team-gm: exp/miniworld`. This refactor has not been committed or
pushed. The existing engine cache build and its shards were not modified.
