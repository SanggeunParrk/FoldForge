## Changes since the 2026-09-16 report

The previous table measured each checkpoint with its own released MSA consumption,
a 2048-row input cap, no templates and one coupled seed. This table applies one
input policy to all four models, taken from the AF3 pipeline. The runs are new;
the reports record the difference.

| Input | 2026-09-16 report | This report |
|---|---|---|
| Seeds | coupled `seed: 0` | `trunk_seed: 0`, `diffusion_seed: 0` (split-v1) |
| Prepared MSA rows | capped at 2048 | up to 16384; 4YX2 chains have 429, 9772 and 9043 |
| Rows the MSA module embeds per recycle | AF3 1024; OpenDDE 1280; Protenix v2 a random 1 to n (559 in that run); ESMFold2 every row, embedded once | 1024 sampled rows, re-drawn every recycle, in every model |
| MSA padding inside the model | up to the 2048-row bucket | none: 1024-row bucket |
| Templates | none | four per chain for AF3, Protenix v2 and OpenDDE; ESMFold2 has no template path |

The policy is recorded as `msa_policy: af3-msa-v1` in every run report, and the
flat adapters record `sampled_msa_buckets: [1024, 1024]`.

| Model | MiniWorld 2026-09-16 (s) | MiniWorld now (s) | Reference compile 2026-09-16 (s) | Reference compile now (s) |
|---|---:|---:|---:|---:|
| AF3 | 15.49 | 15.30 | 87.41 | 87.25 |
| OpenDDE | 46.94 | 47.53 | 223.75 | 225.87 |
| ESMFold2 | 10.81 | 11.86 | 41.96 | 45.30 |
| Protenix v2 | 27.53 | 27.20 | 128.25 | 128.08 |

Latency barely moves because the changed work is a small share of a forward.
AF3 already sampled 1024 rows per recycle; only its sampling pool and the
templates changed, and its MSA stack and template embedding were 7.6% and 3.5%
of a warm forward in the stage profile. OpenDDE and Protenix v2 previously
padded their rows to 2048 inside the MSA stack, so removing that padding offsets
the added template embedding. ESMFold2 is the one model whose computation
changed shape: its MSA encoder now runs on every recurrence loop, as its paper
specifies, instead of once, and it pays about 10%. The speed ratios between
backends are therefore unchanged by the policy; that is the result.

**1024-row MSA bucket.** Measured against the same policy with the previous
smallest bucket of 2048 rows, the 1024 bucket changes warm latency by -3.1% to
+1.6% and lowers peak memory where the MSA stack dominates: AF3 cuEq 4.30 to
3.20 GiB and Protenix v2 reference 11.03 to 8.90 GiB. ESMFold2 samples exactly
1024 rows without bucketing and is unaffected.

**One defective ESMFold2 sample.** With `trunk_seed: 0` and `diffusion_seed: 0`
the fifth diffusion sample misplaces chain A residues 28-30 (C-N up to 22 A) in
four of the five modes; the other four samples are clean in every mode, which is
the whole ESMFold2 peptide-outlier count in the structural checks. A
reference-precision diagnostic on an A6000 (Slurm 1712460) reproduces it with the
same seeds, and it disappears when either seed is changed to 1 or when the same
seeds run with the released whole-MSA embedding. That is one defective sample in
twenty diagnostic samples; its rate is not established, so it is reported as
observed rather than attributed to the policy. ESMFold2's paper always keeps the
query row when it samples; FoldForge's shared sampler ranks all valid rows
equally and can omit it.

The earlier report, its assets, dtype audits and batching verification are
preserved in [archive/benchmark-20260916](archive/benchmark-20260916/benchmark_results.md).
