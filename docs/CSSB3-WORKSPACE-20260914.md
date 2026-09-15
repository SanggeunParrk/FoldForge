# cssb3 workspace migration — 2026-09-14

SSH alias: `ssh -F ~/.ssh/config cssb3` (login host: cssb-master3).
Run GPU work through Slurm partition A100, with explicit GPU, CPU and memory requests.
The separate pre-existing miniworld-engine checkout and A100 cache job 43215 are preserved.

| Workspace | Branch | Base commit |
| --- | --- | --- |
| `/home/psk6950/practice/FoldForge` | main | ebfb85a4201580fcb234b14b539fdb5783b9fdd9 |
| `/home/psk6950/practice/team-gm` | exp/miniworld | dbc204f3cf76cf4a8bff049ce6a7dd2ce9354c93 |
| FoldForge `libs/team-gm` | exp/miniworld | dbc204f3cf76cf4a8bff049ce6a7dd2ce9354c93 |

Local uncommitted changes were applied on top of these commits. `code-manifest.json`
records source SHA256 hashes; all 816 initial entries matched after transfer, including
AF3 vendor-attention batch-axis fix. No commit or push was performed.

Copied the existing FoldForge Python 3.12 environment, its interpreter and GCC 14
runtime library. Required versions: Torch 2.10.0+cu128, Triton 3.6.0,
Quack 0.5.0, CUTLASS DSL 4.5.2, FlashAttention 2.8.3.post1,
miniworld-engine d2266a035de11384c46f8cc980e6460f60925413.
This does not switch the installed engine to the other existing remote checkout.
The standalone team-gm environment is copied locally on cssb3 and reinstalled
editable against its own checkout, without changing dependency versions.

Transferred AF3, ESMFold2, Protenix v1/v2 and OpenDDE checkpoints, common MiniWorld
CCD LMDB, five qualification input directories with MSA/template LMDBs, and cached
ESMC hidden states. ESMC 6B weights were not needed for these cached-input tests.
The existing absolute template paths resolve through the team-gm validation symlink.

Validation is launched with:

```bash
cd /home/psk6950/practice/FoldForge
sbatch validation/cssb3-runtime/validate.sbatch
```

The runner requests one A100, 12 CPUs, 96 GiB RAM and four hours. It uses private
Triton, Inductor, extension and cuEquivariance caches under
`validation/cssb3-runtime/cache`, with a 75-minute timeout per model smoke test.
Runtime versions and SM80 GEMM policy checks: `validation/cssb3-runtime/runtime.json`.
Focused tests: `validation/cssb3-runtime/tests.xml`.
Real checkpoint compile/graph/bucket tests: `validation/cssb3-runtime/{af3,esmfold2}.log`.
This smoke profile uses native BF16, cuEquivariance, one recycle, two diffusion
steps and denoiser compile+CUDA graph capture. It is not full-model production
qualification or a three-backend benchmark.

Protenix/OpenDDE whole-input bucketing and complete model parity remain unfinished;
see `docs/INFERENCE-BACKENDS-BUCKETS-20260914.md` in FoldForge.

## Verified migration results

- Both environments execute BF16 FlashAttention on gpu06 (A100 80GB).
- FoldForge imports its team-gm submodule; standalone team-gm imports its own src.
- The pinned engine commit and requested Quack/CuTe versions match.
- SM80 internal mm/addmm/bmm excludes unsupported Quack calibration.
- Job 43660: 25 focused backend/bucketing tests passed in 63.79 seconds.
- AF3 real 1UBQ checkpoint inference passed in job 43660. ESMFold2 initially
  failed on team-gm LayerNorm's unsupported cuEquivariance selection; that
  LayerNorm now follows the engine policy and uses the PyTorch reference.
- Retry job 43664 on gpu04 completed with exit 0 after 2:01: 29 focused tests
  passed in 16.03 seconds and real ESMFold2 checkpoint inference passed.
- Both model smokes used native BF16, cuEquivariance, token/atom/MSA buckets
  128/1024/2048, one recycle, two diffusion steps and one sample. Each produced
  one compiled denoiser graph, one CUDA graph capture and four replay calls.
  Both checked denoiser calls had zero max-absolute error against the same
  compiled callable without capture. This does not establish whole-model
  bucketed/unbucketed or compiled/eager numerical parity.
- Git remote branches were reachable from cssb3. Existing cache job 43215 was
  still running on gpu05 at the end of migration validation.
- Source changes made during validation are mirrored locally and remotely.
  The final `code-manifest.json` includes the LayerNorm fix and updated docs.
