# Inference buckets and end-to-end measurements — 2026-09-15

> Token padding policy changed later on September 15: all token axes now round
> up to multiples of 128 (1140 → 1152). The measurements and bucket descriptions
> below record the earlier policy, including 1140 → 2048; their timings have not
> been remeasured for the new policy. See [current token policy](TOKEN-BUCKETS-20260915.md).

## Implemented execution contract

Execution and speed measurements completed, but full backend numerical parity is
**not qualified**. For 4YX2, AF3 sample 5 differs by 10.43 Å aligned all-atom RMSD
and OpenDDE sample 3 by 5.05 Å. Independent chain alignment still leaves AF3
chain-A CA RMSD 9.38 Å (chains B/C 3.20/2.88 Å) and OpenDDE chain-A CA RMSD
8.18 Å (B/C 0.14/0.13 Å). These differences cannot be explained solely by rigid
placement of the complex. Even the best one-to-one sample reassignment leaves AF3 RMSD up to 7.43 Å;
OpenDDE retains its identity sample assignment. Per-sample matrices and per-chain
measurements are in the JSON artifact. Their cause is not established by this benchmark; do not label
the speed results as full scientific or numerical qualification.

Protenix/OpenDDE now use the same team-gm inference bucket primitives and common
runtime as the other checkpoints. Model weights and learned feature vocabularies
are preserved; the padding masks are passed through the installed Pairformer
adapters, not just the original checkpoint block classes.

| Axis | Bucket lengths |
| --- | --- |
| Residue/input tokens | 128, 256, 384, 512, 640, 768 |
| Flat atoms | 1024, 2048, 4096, 8192 |
| Sampled MSA rows | 2048, 4096, 8192, 16384 |
| Internal expanded structural tokens | Existing engine union of token and atom ladders |

Lengths round up and never truncate. Trunk/template/confidence Pairformer inputs
are padded at their compute boundaries and sliced back afterwards. MSA sampling
happens before row padding, preserving the selected sequences and RNG state;
padded rows do not enter outer-product normalization. Padding is excluded from
single/token attention, local atom attention and atom-to-token pooling. The sampler,
random augmentation/centering, confidence decoding and output atom/chain identities
use real atoms. The denoiser wrapper pads before compilation/capture and unpads
before returning to the shared sampler.

The Protenix/OpenDDE top-level `buckets` report describes input ceilings; it is not
an allocation record for the unsampled MSA. `sampled_msa_buckets`, `pair_buckets`
and `denoiser_buckets` record the actual compute shapes (MSA values describe the
last recycle). The engine's token and atom ladders are imported directly.

## Defects exposed and corrected

- The installed Pairformer adapter did not pass the single-attention key mask.
  Both the original and converted blocks now receive it. Nonzero-weight tests
  cover both policies and both adapter states.
- The bounded atom encoder modified a reused pair cache with `inplace_safe=False`.
  It now preserves the cached conditions.
- Template frame masks have two token axes; padding treats them explicitly.
- Attention chunk sizes selected before padding could allocate a 24 GiB score
  tensor afterwards. Bucketed PyTorch triangle attention now bounds a score
  tensor to 512 MiB using the actual length, head count and element size (at least
  one row). This is a temporary-score bound, not a total model-memory guarantee.
- Repeated inference retained request conditions and a large denoiser graph pool
  while the next eager trunk ran. Request conditions are released before the next
  trunk. Graph caches larger than one third of device memory are evicted there;
  small graph caches stay reusable. Pool size comes from the allocator's actual
  private-pool snapshot, including inactive capture temporaries. Compilation is
  retained; each denoising sequence still uses graph replay. Evictions and captures
  are reported, and recapture is included in measured time.
- Repeated benchmarks no longer retain the previous complete prediction while
  measuring the next forward.
- Undefined chain-pair PAE (no local frames, such as an ion) is encoded as JSON
  `null` in those summary fields only. Other non-finite report values and canonical
  predictions still fail validation. A valid frame at index zero is no longer
  mistaken for an empty frame set. Raw checkpoint tensors remain unchanged except
  for that corrected single-frame score.

## Measurement setup

Allocated cssb3 A100 80 GB PCIe GPUs; no GPU work ran on a login node. Each case used
native BF16 learned parameters with FP32 normalization parameters, no autocast,
10 recycles, a requested 200 diffusion schedule steps, and five output samples.
`compile: true`, `cuda_graph: true`, `bucketing: true`, `scope: denoiser` were
observed at runtime. The trunk and host sampler remain outside the compiled/captured
boundary. Backend comparisons use identical settings within each model; unsupported
operations retain the common PyTorch implementation.

Inputs are prepared real MSA/template LMDBs for 3PTB (223-residue protein plus
benzamidine and calcium) and 4YX2 (three protein chains, 594 residues total).
The MSA input limit is 2048 rows per chain; checkpoint row sampling is preserved.
MSA search and LMDB creation are excluded. ESMFold2 uses its existing exact-input
ESMC embedding cache and its supported input without templates; ESMC computation
is excluded. Released ESMFold2 sigma capping reduces the requested 200-point
schedule to 134 actual denoiser steps. AF3 samples serially; the other models batch
samples according to their released execution policy. This table compares backends
within a model, not scientific performance between architectures.

Environment: torch 2.10.0+cu128, Triton 3.6.0, Quack 0.5.0, CUTLASS DSL 4.5.2,
cuEquivariance torch 0.11.1, miniworld-engine 1.0.0. Some A100 tuning records were
missing or stale; first requests performed additional kernel compilation/autotuning.
No cache grids were retuned as part of this task.

## Complete model execution after compilation

One cold model forward followed by one warm forward, with the same RNG state
restored before each. Times cover trunk, sampling and confidence; input preparation,
weight loading and output serialization are excluded here. Memory shows PyTorch peak allocated / peak reserved across both forwards.
Reserved memory includes graph-pool reservations and is also needed when assessing
whether another card can hold the workload; CUDA context and external libraries
can use additional memory. Complete process time is in the JSON artifact.
A single warm observation per case establishes a measured comparison, not a
latency distribution. Recapture, where needed for memory, remains in the warm time.

| Model | Input | PyTorch warm (s) | MiniWorld warm (s) | Speedup | PyTorch alloc / reserved (GiB) | MiniWorld alloc / reserved (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| af3 | 3ptb | 12.42 | 10.60 | 1.17× | 2.08 / 2.78 | 1.47 / 2.09 |
| af3 | 4yx2 | 67.20 | 50.67 | 1.33× | 6.44 / 7.71 | 6.72 / 10.39 |
| esmfold2 | 3ptb | 5.50 | 2.51 | 2.19× | 4.17 / 6.49 | 2.04 / 3.10 |
| esmfold2 | 4yx2 | 39.00 | 10.73 | 3.64× | 22.66 / 37.07 | 7.69 / 15.00 |
| protenix | 3ptb | 8.66 | 6.53 | 1.33× | 2.54 / 3.15 | 1.79 / 2.46 |
| protenix | 4yx2 | 65.91 | 49.11 | 1.34× | 7.37 / 10.17 | 6.11 / 9.02 |
| opendde | 3ptb | 20.54 | 15.76 | 1.30× | 5.09 / 8.99 | 5.30 / 8.19 |
| opendde | 4yx2 | 243.41 | 199.84 | 1.22× | 60.73 / 77.37 | 60.73 / 74.33 |

## First complete CLI requests

These earlier diagnostic trials include imports, input preparation, checkpoint
loading, first execution/compilation/capture and saving. Existing disk caches were
left in place, so these are fresh-process timings rather than empty-cache timings.
They are not used to calculate steady execution speedups. Protenix rows include
successful retries after the JSON fix. OpenDDE 4YX2 initially failed before the
attention and graph-pool lifecycle corrections; the final warm table above verifies
the repaired repeated-inference path.

| Model | Input | PyTorch first CLI (s) | MiniWorld first CLI (s) |
| --- | --- | ---: | ---: |
| af3 | 3ptb | 111.03 | 334.13 |
| af3 | 4yx2 | 172.97 | 214.63 |
| esmfold2 | 3ptb | 92.02 | 413.81 |
| esmfold2 | 4yx2 | 118.37 | 209.46 |
| protenix | 3ptb | 60.13 | 109.40 |
| protenix | 4yx2 | 211.14 | 244.60 |
| opendde | 3ptb | 141.50 | 291.65 |
| opendde | 4yx2 | OOM before chunk fix | OOM before chunk fix |

## Validation and numerical scope

- GPU job 43842: 66 focused tests plus both real-checkpoint bucket smoke runs passed.
- Actual installed worktree GPU job 43861: 72 tests passed.
- Final graph-pool lifecycle GPU job 43875: 75 tests passed, including a freed capture-temporary regression and eviction/recapture output checks.
- CPU shared-layer/attention/compiler regression: 131 passed; request-condition lifetime regression also passed (job 43866).
- Large OpenDDE repeated-request memory job 43876: same 594-residue / 4591-atom input, 1 recycle, 1 step, 5 samples, two complete forwards passed. Observed 2 graph captures, 1 eviction and 2 replay calls.
- Full final Protenix/OpenDDE array 43878 and prior unaffected AF3/ESMFold2 array tasks 43853_0/1 supply all 16 successful cases below; full numerical comparisons are recorded separately.

| Model | 3PTB bucket off/on aligned all-atom RMSD range (Å) | pLDDT RMS difference (0..1) |
| --- | ---: | ---: |
| protenix | 0.06059–0.09939 | 0.0001833 |
| opendde | 0.05302–0.06147 | 0.0002468 |

Full sampling backend comparisons (same ordered atoms; five samples):

| Model | Input | PyTorch/MiniWorld aligned all-atom RMSD range (Å) | pLDDT RMS difference (0..1) |
| --- | --- | ---: | ---: |
| af3 | 3ptb | 0.02273–0.12213 | 0.0005133 |
| af3 | 4yx2 | 0.25013–10.43230 | 0.0065716 |
| esmfold2 | 3ptb | 0.18118–0.59458 | 0.0067708 |
| esmfold2 | 4yx2 | 0.86869–1.86780 | 0.0073160 |
| protenix | 3ptb | 0.06269–0.10231 | 0.0001680 |
| protenix | 4yx2 | 0.21849–1.35535 | 0.0008694 |
| opendde | 3ptb | 0.05994–0.06300 | 0.0002256 |
| opendde | 4yx2 | 0.17201–5.05267 | 0.0025001 |

Full sampling comparisons preserve atom identities, finite canonical outputs and
all five CIF samples. Padding and changing BF16 kernels need not be bitwise equal.
The numerical artifact reports per-head differences and per-sample rigidly aligned
all-atom RMSD. These are execution regressions, not a scientific accuracy benchmark.
Full eager-vs-compiled scientific parity, all-backend parity (including full-model
cuEquivariance), other checkpoint variants, larger inputs and other GPU models are
not qualified by this run. The previously recorded AF3 compile-vs-eager threshold
failure is not waived by these backend measurements.

## Reproduction and source state

Run `scripts/benchmark_end_to_end.py` inside a Slurm GPU allocation after sourcing
`scripts/activate_env.sh`. The script requires `SLURM_JOB_ID`, validates observed
compile/graph/native-BF16 fields and sample counts, records incremental results,
and terminates a timed-out subprocess group. `--resume` preserves successful
cases and archives failed attempts before retrying. Use a fresh `--root` directory for an independent run. The prepared inputs/checkpoints
and detailed logs are in cssb3 `validation/bucket-e2e-20260915/` and the referenced
qualification-input directories.

The source is installed in local and cssb3 FoldForge `main`, root team-gm
`exp/miniworld`, and FoldForge's team-gm checkout `exp/miniworld`. Existing changes
were preserved through source-hash checks and backups. No commit or push was made.
The unrelated engine cache job 43215 and its shards were not modified.

[Machine-readable measurements and numerical comparisons](inference-e2e-results-20260915.json).
