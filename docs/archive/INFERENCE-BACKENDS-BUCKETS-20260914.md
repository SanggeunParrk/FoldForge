# Inference backends and bucket integration (2026-09-14)

> Token padding policy changed later on September 15: all token axes now round
> up to multiples of 128 (1140 → 1152). The measurements and bucket descriptions
> below record the earlier policy, including 1140 → 2048; their timings have not
> been remeasured for the new policy. See [current token policy](TOKEN-BUCKETS-20260915.md).

Status: bucket integration implemented; see the September 15 report for current
qualification. Full three-backend scientific parity is not established; uncommitted.

The requested inference profile is `configs/inference/compile-graph-buckets-bf16.yaml`:
`compile: true`, `cuda_graph: true`, `bucketing: true`, native BF16 learned
parameters and FP32 normalization, with `scope: denoiser`. Host sampling stays
outside the graph. Existing qualification/cache-build profiles retain their
execution settings. The profile is explicit; legacy defaults have not changed.

## Bucket contract

The shared policy lives in `team_gm.modules.bucketing` and imports
`TOKEN_SHAPES` and `ATOM_SHAPES` from the installed pinned miniworld-engine.

| Axis | Allocated lengths |
| --- | --- |
| Tokens | 128, 256, 384, 512, 640, 768 |
| Flat atoms | 1024, 2048, 4096, 8192 |
| MSA rows | 2048, 4096, 8192, 16384 |

Physical tensors round **up**. The engine's autotune floor/clamping function
selects tuning configurations, and must not truncate inference inputs. Overflow
raises an error. Missing MSA stays missing; bucketing does not invent sequences.

ESMFold2 pads explicit semantic axes after the exact-input ESMC cache is checked.
Token/atom/MSA masks are false on padding. Public coordinates and every
per-token/per-atom/pair confidence field are cropped back to the real input.
AF3 uses the official token featurizer and rebuilds flat cross-attention gathers
with the official atom-layout API. Dense token-by-24 geometry retains its
checkpoint layout. Its released 1024-row trunk sampling cap is preserved;
sampled MSA tensors are masked and padded to 2048 instead of changing which
sequences the model selects. Input MSA storage can occupy a larger bucket.

Protenix/OpenDDE now pad compute boundaries in shared team-gm code. Token masks
reach trunk/template/confidence Pairformers and denoiser attention; sampled MSA
masks exclude padded rows from outer-product normalization. Atom validity reaches
local attention and atom-to-token pooling. Sampling, centering, confidence decoding
and output identity stay on real atoms. Expanded structural-token compute axes use
the engine's existing union of token and atom bucket ladders. Triangle score chunks
are bounded using the actual padded length and head count. See
[the September 15 implementation and measurements](INFERENCE-BUCKETS-E2E-20260915.md)
for the measured scope, numerical differences and remaining qualification limits.

## Backend changes

All common configurations and AF-family model loaders now accept `pytorch`,
`cuequivariance`, and `miniworld`; ESMFold2 maps each value explicitly.
AF-family TriMul uses the same checkpoint layout mapping for both optimized
backends, adding the residual exactly once for the vendor delta. AF3 grid
attention and Protenix/OpenDDE triangle attention dispatch to the vendor API.
Unsupported vendor operations retain PyTorch math.

Fixed two existing bugs: Protenix/OpenDDE vendor attention sliced a tensor with
`[0]`, dropping a batch axis; Protenix chose the small-length reference fallback
after Q scaling. ESMFold2's SWA Config previously discarded `implementation`,
so the PyTorch selection could still execute MiniWorld SWA preprocessing and
gating. SWA now receives the selected backend (cueq uses its PyTorch+FA path).
Vendor Triton caches initialize outside compiled regions. The Pairformer adapter
lets the vendor tile attention rows instead of using upstream row chunking, which
would expand the shared bias and violate the vendor layout contract.

## Evidence and remaining checks

- FoldForge CPU suite: 121 passed, 103 GPU tests skipped.
- AF3 real 1UBQ input: 76 tokens -> 128, 602 atoms -> 1024,
  129 MSA rows -> 2048; official gather round-trip preserves every valid atom.
- CPU tests cover semantic-axis collisions, masks, output cropping and bounds.
- A5000 job 1689406: 17 tests passed in 41.52 seconds, including six vendor
  TriMul direction/mask/residual cases and actual compile+graph reuse across
  lengths 129 and 177 in bucket 256. Short vendor attention fell back to the
  vendor's PyTorch reference; this is not evidence of a fast attention kernel.
- A5000 job 1689434: 21 passed, 2 AF3 grid attention cases failed because the
  vendor's return had an extra batch axis. The real-checkpoint stage did not run
  after this test failure. AF3 now supplies an explicit leading batch axis to
  the vendor and removes that singleton from the result; coverage includes
  both attention directions at L=32 and L=128.
- cssb3 A100 80GB job 43660 on gpu06: 25 focused tests passed in 63.79 seconds,
  including the AF3 batch-axis fix and real compile/graph bucket reuse. Runtime
  version checks, BF16 FlashAttention and SM80 mm/addmm/bmm policy also passed.
  The installed cuEquivariance triangle-attention build uses its reference path
  on this GPU; these tests establish correctness, not fast vendor-attention
  kernel performance. AF3 1UBQ checkpoint inference also passed: native BF16,
  token/atom/MSA buckets 128/1024/2048, one compiled denoiser graph, one CUDA
  graph capture, four replay calls, and zero max-absolute error on both checked
  denoiser calls against the same compiled callable without capture.
- ESMFold2 in job 43660 initially failed because team-gm's standalone LayerNorm
  rejected cuEquivariance while the engine already falls back to PyTorch for
  that unsupported op. Aligned team-gm LayerNorm with that policy in both
  checkouts, preserving FP32 affine parameters and the input activation dtype.
- A100 retry 43664 on gpu04 completed successfully in 2:01: 29 focused tests
  passed in 16.03 seconds, including four new BF16/FP32 affine/non-affine
  LayerNorm output/gradient cases. ESMFold2 real 1UBQ checkpoint inference passed
  with native BF16 and buckets 128/1024/2048: one compiled denoiser graph, one
  CUDA graph capture, four replay calls, and zero max-absolute error in both
  checked denoiser calls. Results are under `validation/cssb3-runtime/`.
- These real-checkpoint smoke tests used cuEquivariance, one recycle, two
  diffusion steps and one sample. They establish execution and graph replay
  consistency, not unbucketed-vs-bucketed parity, compiler-vs-eager parity,
  scientific accuracy or full-model three-backend qualification.
- Redundant A6000 job 1689304 was canceled after the A5000 module run passed.
- The new focused files pass Ruff. The existing common inference CLI has
  pre-existing style violations; its entire-file lint is not clean.
- Complete-model three-backend scientific parity remains unqualified. Bucket
  execution and full-sampling measurements are updated in the September 15 report.
- Prior AF3 4YX2 compile-vs-eager numerical threshold failure remains unresolved;
  this change does not waive or loosen that threshold.

Branches remain team-gm `exp/miniworld` and FoldForge `main`, including the
FoldForge team-gm checkout. No new commit or push was performed for this
unqualified work.
