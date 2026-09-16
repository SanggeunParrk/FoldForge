# Consumer environment — 2026-09-13

| Component | Pinned target |
| --- | --- |
| Python | 3.12.14 (managed uv; `.python-version` selects 3.12) |
| PyTorch, cu12 extra | 2.10.0+cu128 |
| Triton | 3.6.0 (Torch dependency, recorded in `uv.lock`) |
| FlashAttention-2 | 2.8.3.post1 |
| quack-kernels | 0.5.0 |
| nvidia-cutlass-dsl | 4.5.2 |
| Transformers | Biohub fork 4.57.6, `b435f1f92dd5b4a653be57157d4f4f5ddba4f145` |
| ESM | Biohub fork 3.3.0, `fba1e42a8a0f8df2b7e08611c88807aa32a25d9c` |
| MiniWorld Engine | `d2266a035de11384c46f8cc980e6460f60925413` |

The cu12/cu13 extras include `miniworld-engine[cute]`. Root constraints preserve
Quack 0.5.0 and CuTeDSL 4.5.2 across optional environments. Installing these
packages does not override the engine's GPU-specific dispatch policy: SM86
excludes unsupported Quack GEMM candidates.

The cu12 Torch pin matches the qualified A5000 cache. Its environment fingerprint
includes **Torch itself**, even when Triton and CUDA versions match. Torch 2.11
therefore invalidated that cache; bypassing its identity check is not a repair.
The expected cache environment identity is `4cfb996f8c09`.

Transformers is pinned to the exact source commit recorded in the original
ESMFold2 environment's installed `direct_url.json`. The generic PyPI release is
not a replacement for the `ESMCModel` / ESMFold2 checkpoint API used here.
The ESM fork provides `esm.models.esmfold2.processor` and requires Python 3.12.
Its mutable Transformers `@main` dependency is overridden with the pinned source
above. Both repositories have independent `.venv` environments and lockfiles.

## Install and run

From either repository root on this cluster:

```bash
mkdir -p validation/reports/logs
sbatch scripts/setup_env.sbatch
# Once installation succeeds, for direct Python commands:
source scripts/activate_env.sh
```

This installs the locked `cu12` and `flash2` extras on an allocated A5000.
`uv` 0.12.13 is installed at `~/.local/bin/uv`; `UV_BIN` may override that path.
The script selects CUDA 12.8 and GCC 12.4 and includes the Conda toolkit's
`targets/x86_64-linux` headers and libraries. FA2's isolated build uses the same
Torch version as the runtime. A missing compatible prebuilt FA2 wheel requires
source compilation targeting SM80. The current environment validation is A5000.

On this cluster the script reuses a Python 3.12 / Torch 2.10 / CUDA 12.8 FA2 wheel
recovered from the engine environment. All 104 original package file hashes were
verified against its installed RECORD; package binaries were not rebuilt or
modified. The artifact and `provenance.json` live in
`~/.cache/miniworld-wheels/cp312-torch2.10-cu128/`. Its SHA256 is
`591add7fec7ba3b95777413d1a879a6682920ba89ee82e8928035175a66a2c72`.
Setup verifies this hash, syncs with `--no-install-package flash-attn`, then
installs the pinned FA2 wheel with `uv pip --no-config`. This explicit staging
avoids uv rebuilding FA2 solely to match its PyPI source identity.

The recovered binary requires GCC 14's C++ runtime. Only `libstdc++.so.6` and
`libgcc_s.so.1` are copied to `~/.cache/miniworld-runtime/gcc14`; the activation
helper adds that directory to `LD_LIBRARY_PATH`. The engine's entire Conda `lib`
directory must not be added: unrelated libraries can interfere with Git/TLS.

Model Slurm runners use the repository's `.venv`, clear inherited `PYTHONPATH`,
and retain offline checkpoint access. They no longer inject the shared Conda
Python or temporary stub/library folders. `TEAM_GM_ENV` (team-gm) or
`FOLDFORGE_ENV` (FoldForge) can select another complete environment.

Use the setup script to update this environment; omitting the optional extra can
remove FA2, and a direct sync may rebuild it. A CUDA-13 environment is a separate target and is not qualified by
these Ampere checks.

## Working trees

- team-gm: `exp/miniworld-integrated`.
- FoldForge: `main`; its team-gm member remains on `exp/miniworld`.
- Existing unrelated changes are preserved. Consumer changes remain local.

Raw installation and validation logs are retained in team-gm's ignored
`validation/environment-20260913/` directory. The initial Torch 2.11 attempt and
its cache mismatch are retained separately for provenance.

## Validation

Slurm job **1682646**, `gpu02` / RTX A5000, completed its batch script with exit
code 0 on 2026-09-13. Each repository ran its final setup script and tests using
its own Python 3.12.14 interpreter, without a shared `PYTHONPATH`.

| Check | team-gm | FoldForge |
| --- | --- | --- |
| Dependency consistency (`uv pip check`) | passed, 167 packages | passed, 164 packages |
| Quack 0.5.0 / CuTe 4.5.2 imports and SM86 GEMM policy | passed | passed |
| A5000 cache environment identity `4cfb996f8c09` | matched | matched |
| FA2 BF16 forward/backward versus FP32 SDPA | passed | passed |
| Native BF16 SWA forward/backward versus PyTorch | passed | passed |
| Compiled training, CUDA graph off | passed | passed |
| Compiled inference, CUDA graph capture/replay | passed | passed |
| Consumer regression tests | 77 passed | 17 passed |
| ESMFold2 CLI `--help`, including input processor import | not run | passed |

The SWA probe uses N=2, S=128, D=128; some shapes are outside the cache build
plan and fall back to autotuning. This check confirms the cache **environment**
matches, not zero cache misses for arbitrary shapes. Full checkpoint inference
and model performance were not rerun as part of this environment repair.

During environment validation, CPU SWA tests exposed CUDA backend selection for
a CPU tensor on a GPU host. Engine commit `d2266a03` fixes that device guard;
29 focused backend tests passed before the fix was pushed to engine `main`.

## Compile coverage

The environment GPU probe applies `torch.compile` with `fullgraph=False`, matching
the production default. Training disables CUDA graphs; inference explicitly
captures and replays a CUDA graph after warmup. The probe records compiled-region
and graph-break counters instead of claiming a single uninterrupted compiler graph.
The final Torch 2.10 probes each recorded 9 compiled regions, with graph breaks
including shape-key CRC32 and the FA2 backward dynamic `nonzero` path.

A separate strict `fullgraph=True` probe on the initial Torch 2.11 environment
failed at `zlib.crc32` in `autotune/shape_key.py:pack`. This environment repair does
not change that shape-key implementation; strict fullgraph support remains a
separate engine issue. Its failure is retained in the raw validation logs.

## AF-family ports

The locked environment now includes the local `foldforge-af3-data` CPU feature
package, CPU JAX 0.8.2, and the model input dependencies. Its native structure
parsers build with CMake; no JAX CUDA runtime is installed. Torch 2.10.0+cu128,
Triton 3.6.0, Quack 0.5.0, CUTLASS DSL 4.5.2 and FlashAttention 2.8.3.post1
remain pinned. The existing verified FlashAttention wheel is reused.

Rebuilding with `scripts/setup_env.sbatch` succeeded on gpu03 (job 1682660);
`uv pip check` reports compatible packages. See
`validation/logs/ports-setup-env-20260913.log`. The subsequent unification
rebuild (job 1682818) adds scoped AF3 CCD classification sets. All models now
select the same `data/ccd` database with `--ccd-db` or `FOLDFORGE_CCD_DB`;
its AF3 index lives in `data/ccd/af3`. The earlier model-specific `--assets`
paths are superseded. See [code unification](CODE-UNIFICATION-20260913.md).
