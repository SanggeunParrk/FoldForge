**Input policy versus the 2026-09-16 report.** The earlier table capped inputs at
2048 MSA rows, used no templates and a coupled seed. This one prepares up to 16384
rows, embeds 1024 sampled rows per recycle in every model, gives AF3, Protenix v2
and OpenDDE four templates per chain, and uses split trunk/diffusion seeds. On the
same A100 class the MiniWorld mode moved from 15.49 to 15.30 s (AF3), 46.94 to
47.53 s (OpenDDE), 10.81 to 11.86 s (ESMFold2) and 27.53 to 27.20 s (Protenix v2).
ESMFold2 pays the most because its MSA encoder now runs on every recurrence loop,
as its paper specifies, instead of once. The earlier report, its assets, dtype
audits and batching verification are preserved in
[archive/benchmark-20260916](archive/benchmark-20260916/benchmark_results.md).

**1024-row MSA bucket.** Measured against the same policy with the previous
smallest bucket of 2048 rows, the 1024 bucket changes warm latency by -3.1% to
+1.6% and lowers peak memory where the MSA stack dominates: AF3 cuEq 4.30 to
3.20 GiB and Protenix v2 reference 11.03 to 8.90 GiB. ESMFold2 samples exactly
1024 rows without bucketing and is unaffected.

**One defective ESMFold2 sample.** With `trunk_seed: 0` and `diffusion_seed: 0`
the fifth diffusion sample misplaces chain A residues 28-30 (C-N up to 22 A) in
four of the five modes; the other four samples are clean in every mode, which is
the whole ESMFold2 peptide-outlier count above. A reference-precision diagnostic
on an A6000 (Slurm 1712460) reproduces it with the same seeds, and it disappears
when either seed is changed to 1 or when the same seeds run with the released
whole-MSA embedding. That is one defective sample in twenty diagnostic samples;
its rate is not established, so it is reported as observed rather than attributed
to the MSA policy. ESMFold2's paper always keeps the query row when it samples;
FoldForge's shared sampler ranks all valid rows equally and can omit it.
