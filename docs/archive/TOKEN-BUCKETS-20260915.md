# Inference token buckets — 2026-09-15

All physical token dimensions now round up to multiples of 128:
`((tokens + 127) // 128) * 128`. This includes residue/input tokens,
expanded structural tokens, trunk/template/confidence Pairformer axes and
padded denoiser conditions. Examples: 769 → 896, 1025 → 1152,
1140 → 1152, 1152 → 1152, 1153 → 1280.

The old expanded-token path used the union of the token and atom autotune
ladders, sending 1140 structural tokens to 2048. That policy has been removed.
For a fixed-width pair tensor, 1152 × 1152 uses 31.64% of the elements in
2048 × 2048. This is a tensor-size comparison, not a measured model speedup
or total GPU memory reduction.

Atoms retain 1024/2048/4096/8192 and MSA rows retain
2048/4096/8192/16384. Their overflow validation is unchanged.
The common token rounding function has no 768-token ceiling. AF3's sequence-based
featurizer receives all multiples of 128 through the supported atom ceiling
8192, so it chooses the same sizes for supported inputs before building gathers.

The engine's autotune keys are unchanged: length selects a tuned configuration
with the existing floor/clamp rules; channel widths remain exact. Physical
inference allocation sizes do not define the engine's tuning ladder. Missing
channel/dtype/config cache coverage remains a separate task.

Implementation:
- `libs/team-gm/src/team_gm/modules/bucketing.py`: common token rule and atom/MSA policy.
- `libs/team-gm/src/team_gm/modules/checkpoints/padding.py`: trunk, pair stack and denoiser.
- `src/foldforge/models/bucketing.py` and `models/io/dense_atoms.py`: ESMFold2/AF3 adapters.

Applied to the local and cssb3 FoldForge worktrees and both connected team-gm
copies. Existing uncommitted work was preserved using pre-edit hash checks.

Validation: cssb3 Slurm job 43897, allocated A100 on gpu04; **90 tests passed**
in 25.04 seconds. The suite covers 1140 → 1152 data/mask preservation across
ESMFold2, pair stack, trunk and denoiser preparation; AF3 featurizer bucket
agreement; unchanged atom/MSA limits and engine key clamping; and actual
compile/CUDA-graph reuse for real token lengths 1025 and 1140. Existing backend
and shared I/O regressions also passed. No full-checkpoint end-to-end speed
benchmark was rerun. Earlier September 15 benchmark tables retain their original
2048-token results and are explicitly marked as using the previous policy.
