# PyTorch backend execution audit — 2026-09-15

> Historical audit before the attention wiring fix. The verified PyTorch baseline
> remains valid; MiniWorld triangle/token attention routing and new measurements
> are documented in [BACKEND-WIRING-20260915.md](BACKEND-WIRING-20260915.md).

The AF3 4YX2 baseline behind the reported 1.33× speedup executes PyTorch
operations with an Inductor-compiled, CUDA-graph-captured denoiser. No MiniWorld
or cuEquivariance computational kernel was observed in its runtime audit.
ESMFold2 uses the explicitly permitted FlashAttention exception, through a
MiniWorld-namespaced window-attention wrapper; calling that run ATen-only would
be inaccurate. The other three audited models had no such custom operator.

## Evidence

Allocated cssb3 A100 compute nodes, Slurm jobs 43899 and 43904. GPU work did
not execute on a login node. The runner:

1. Rejects MiniWorld/CuEq `CustomOpDef` calls, permitting only
   `miniworld_engine::swa_atom_attention_flash_window` for the FA exception.
2. Rejects hand-written Triton launches from MiniWorld kernels, CuEq and AF3's
   legacy fastnn package; compiler-generated kernels remain permitted.
3. Profiles CPU-dispatched operators across full forwards, including initial
   compilation/warmup/capture, and checks custom operator namespaces.
4. Records the installed modules' backend values and actual compiler/graph
   counters, rather than assuming execution from the requested flags.

The initial strict ESMFold2 check rejected the actual FlashAttention wrapper.
That diagnostic is preserved at
`validation/backend-purity-20260915/esmfold2/flash-wrapper-detected.json` on cssb3.
Inspection confirmed the wrapper invokes FlashAttention varlen plus layout/mask
handling; it does not fuse the MiniWorld QK-normalization/RoPE or output-gate
kernels. The final check permits only this wrapper. An earlier audit-instrument
error accessed an AF3-specific load-report attribute on ESMFold2; the helper was
fixed and rerun. No model implementation was changed by this audit.

| Model | Input / sampling | Compiled graphs | CUDA graph replays | Non-FA MiniWorld/CuEq calls observed |
| --- | --- | ---: | ---: | ---: |
| AF3 | 4YX2, 10 recycles, 200 steps, 5 samples | 1 | 2000 | 0 |
| ESMFold2 | 3PTB, 1 recycle, requested 2 steps, 1 sample | 1 | 4 | 0 |
| Protenix v1 | 3PTB, 1 recycle, 2 steps, 1 sample | 1 | 4 | 0 |
| OpenDDE | 3PTB, 1 recycle, 2 steps, 1 sample | 1 | 4 | 0 |

All runs use native BF16, no autocast, denoiser compilation/capture and bucketing.
Each executes an initial and a repeated complete forward. AF3's input token
bucket remains 640 under both the old and new padding rules. Other models are
shorter runtime smoke checks, not full resampling of their prior benchmark.
Protenix v2 was not requalified by this audit.

The CPU profile contains zero non-ATen custom operators for AF3. ESMFold2 records
73 calls to the window wrapper and 24 `flash_attn::_flash_attn_varlen_forward`
events; these include tracing/warmup and are not total GPU launch counts.
CUDA graph replay internals do not reappear individually as CPU operators;
the audited initial forward includes the graph's warmup and capture path.
No forbidden direct Triton launch was attempted. The modules' recorded
`implementation`, `_backend`, and `foldforge_implementation` values are PyTorch.

Full operator counts, settings, job IDs and observed execution counters:
[BACKEND-PURITY-20260915.json](BACKEND-PURITY-20260915.json).
Machine-specific runner: `validation/backend-purity-20260915/audit.py` on cssb3.
Instrumentation affects timing; its recorded times are not replacement benchmarks.

## What 1.33× does and does not establish

The previous warm whole-model measurements were 67.2016 s for PyTorch and
50.6747 s for MiniWorld (1.326×). Both had denoiser compile/CUDA graph enabled.
This was not a comparison against eager, uncompiled PyTorch. Inductor-generated
Triton is part of the PyTorch baseline, distinct from hand-written engine kernels.

The MiniWorld label does not mean every computation was replaced. AF3's
`GridSelfAttention._attention` selects the shared torch equation for both the
PyTorch and MiniWorld modes; its normalization can still differ. OPM's core
contraction/projection likewise remains shared PyTorch. The engine-backed
conditioning modules explicitly select their PyTorch reference branches for
the baseline, so class names such as `AdaptiveLayerNorm` from the engine package
are not evidence of a fused-kernel launch.

Relevant source:
- `libs/team-gm/src/team_gm/modules/checkpoints/af_family.py`: native norms and TriMul/Transition backend guards.
- `libs/team-gm/src/team_gm/modules/checkpoints/conditioning.py`: reference backend selection.
- `src/foldforge/models/af3/ported/fastnn/config.py`: legacy fastnn paths explicitly set to torch.
- `src/foldforge/models/af3/ported/nn/attention.py`: shared AF3 triangle attention core.
- `src/foldforge/models/af3/ported/nn/primitives.py`: shared OPM core.
- `libs/team-gm/src/team_gm/modules/execution.py`: actual Inductor/manual graph boundary.

The three main adapter files above match the prior benchmark's staged source.
Historical PyTorch logs also contain no MiniWorld-autotune warnings; that alone
would not have proved purity, hence the runtime checks.

The audit rules out the suspected contamination for the checked paths. It does
not establish a performance ceiling: partial engine coverage and known missing/
stale tuned caches remain. It also does not quantify how much time each shared
operation costs; that requires a separate stage/kernel timing profile.
