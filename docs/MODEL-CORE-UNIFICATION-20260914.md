# Common checkpoint core and FoldCP removal — 2026-09-14

FoldCP execution, configuration and call arguments have been removed. The entire
21-file `opendde/ported/distributed` package is deleted. The two released model
copies now import one implementation of their dense layers, triangle layers,
atom/token transformers, MSA/template stacks, diffusion conditioning, denoiser,
confidence heads and tensor utilities from `team_gm.modules.checkpoints`.

The final follow-up also removes the former rank-synchronization callbacks from
the shared Euler sampler and guidance runtime, including their signature
arguments and local-action wrappers.

This is a source deletion and caller migration: the old operator paths do not
contain compatibility classes or another set of forward methods. Twenty-two
model-layer/utility files were deleted, along with six private triangle-attention
kernel files and the model-private fused-dropout kernel. Together with FoldCP,
this follow-up removes 50 Python files.

| Component | Owner |
| --- | --- |
| Dense projections, AdaLN, Transition | `team_gm.modules.checkpoints.dense` |
| Triangle layers and checkpoint initialization | `checkpoints.triangle`, `checkpoints.layers` |
| Atom encoder/decoder, token/atom transformer | `checkpoints.transformer` |
| Pairformer, MSA and template stacks | `checkpoints.stacks` |
| Relative/Fourier/input/constraint embeddings | `checkpoints.embedders` |
| Optional structural-token projection and shared role vocabulary | `checkpoints.structural`, `checkpoints.structural_roles` |
| Diffusion conditioning, compression and denoising | `checkpoints.denoiser` |
| Confidence and distogram heads | `checkpoints.confidence`, `checkpoints.heads` |
| Atom-window feature preparation and recycle loop | `checkpoints.trunk` |
| Chunking, tensor containers and MSA sampling | `checkpoints.tensor_utils` |
| Solver, sampling, schedules, augmentation, guidance | Existing `team_gm.diffusion` common runtime |

`CheckpointPolicy` is frozen and is bound when modules are constructed. It records
transition chunking, atom-window chunking, pair-transition chunking, dropout,
MSA update order and activation precision. The default and bounded-memory presets
preserve the released checkpoint conventions. There is no model-name branch in
the shared layer forward methods, and no FoldForge import in team-gm.

The last MSA block is intentionally different between the presets: MSA-first
checkpoints still update MSA before the final OPM, while OPM-first checkpoints omit
the final unused MSA update. This is part of the learned architecture, not an
implementation fallback. Optional pair-channel compression, structural pair
attention bias and language-model embeddings are constructor/feature options in
the common implementations.

## Validation

All GPU work used allocated A100 compute nodes on cssb3. The unrelated engine cache
job and its shards were preserved.

- Job 43701: removing FoldCP alone retained strict loading of 4,482 OpenDDE state
  entries and compiled CUDA-graph inference; coordinates/confidence matched exactly.
- Job 43705: shared layer constructors and forward methods loaded Protenix's
  4,174 and OpenDDE's 4,482 state entries strictly. Comparisons covered 59/73 saved
  tensor/array fields with `atol=0, rtol=0` and maximum difference 0.
- Job 43711: deleting the old import paths and sharing the recycle loop also passed
  both real checkpoints. Job 43714 confirmed the same exact 59/73-field equality
  and passed three source-ownership checks.
- Job 43713: 417 numerical, gradient, sampling, checkpointing and fullgraph tests
  passed. Frozen reference equations are retained outside Git; tests rebind only
  retired imports/control hooks, rather than replacing the reference math.
- Job 43715: the actual checkout (no staged import paths) passed 36 backend,
  bucketing, ownership and residual checks, then all four real checkpoint
  compile+CUDA graph runs. All 149 saved tensor/array fields matched exactly.
- Jobs 43716/43717 repeated the 417 core and 36 integration checks after the
  final sampler/guidance hook removal. All four checkpoints again matched all
  149 fields exactly. Job 43719 passed the four final ownership checks.
- The live-checkout validation and exact output report are recorded in
  `model-core-live-equivalence-20260914.json` and the source manifest beside it.

Runtime correctness profiles use native BF16 parameters with FP32 norms,
cuEquivariance, one recycle, two diffusion steps and one sample on 1UBQ.
The denoiser uses compile and CUDA graph capture. This is an equivalence check;
its wall time is not a production end-to-end speed benchmark. AF3/ESMFold2 retain
the existing bucketed smoke profile, while Protenix/OpenDDE use the existing
unbucketed profile.

## Boundary and remaining integration work

The shared numerical implementations replace the duplicated Protenix/OpenDDE
operator packages. Checkpoint loading, state-key conventions, feature preparation,
model assembly and public output formatting remain in FoldForge adapters.
AF3's packed projections and ESMFold2's learned topology still have checkpoint
layout/assembly code; this change does not claim that every model directory now
contains configuration alone. Their existing shared solver, kernel and composition
paths are preserved.

Whole-input bucketing for Protenix/OpenDDE and production-size end-to-end speed
qualification remain separate integration work. No changes were made to the
running MiniWorld-engine cache build.

Working branches remain `team-gm: exp/miniworld`, `FoldForge: main`, and
`FoldForge/libs/team-gm: exp/miniworld`. This work has not been committed or pushed.
