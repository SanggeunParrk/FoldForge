# scripts

Run model and measurement jobs on allocated compute nodes.

- `setup_env.sbatch`: install the pinned environment, including FA2/Quack/CuTe.
- `activate_env.sh`: activate `.venv` and the required C++ runtime.
- `fold_esmfold2.py` / `.sbatch`: compatibility entry point for the same implementation
  used by `foldforge fold esmfold2`. Live ESMC is the default; use `--lm-source cache`
  explicitly for existing embeddings.
- `esmfold2_profile.py` / `.sbatch`: stage timings with cached embeddings.
- `op_dispatch_audit.py` / `.sbatch`: inspect which backend executes.
- `build_autotune_cache.py` / `.sbatch`: bounded, resumable model-shape gap builder.
  See [cache build commands](../docs/CACHE-BUILD.md) for its private cache and replay options.

The folding command now lives in the installed model package and writes both CIF
and JSON. See [the model integration record](../docs/MODEL-INTEGRATION.md).

AF3, Protenix v1/v2 and OpenDDE now run through `foldforge fold <model>` in this
environment. See [model integration](../docs/MODEL-INTEGRATION.md) for input
schemas, checkpoint/CCD paths and complete commands. Shared CCD assets live
outside `.venv`, so environment reinstallation preserves them. All model
execution belongs on allocated GPU nodes.

The common inference and cache-building paths read `ccd_db` from the MiniWorld
input YAML. Compatibility scripts additionally accept `--ccd-db`. Prepare with
`foldforge ccd prepare` and verify with `foldforge ccd verify`.
`audit_model_boundaries.py --out docs/model-boundaries-20260913.csv` inventories
all model classes with forward methods without loading the GPU stack.

The standard entry point is now `foldforge fold <model> --spec <MiniWorld YAML>`.
All predictors use `data/ccd/preprocessed_CCD.lmdb`; see
[the format contract](../docs/MINIWORLD-FORMAT.md). The target-based scripts remain
compatibility tools for reproducing the recorded validation runs.
