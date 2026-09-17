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
  See [cache build commands](../docs/guides/CACHE-BUILD.md) for its private cache and replay options.

The folding command now lives in the installed model package and writes both CIF
and JSON. See [the model integration record](../docs/guides/MODEL-INTEGRATION.md).

AF3, Protenix v1/v2 and OpenDDE now run through `foldforge fold <model>` in this
environment. See [model integration](../docs/guides/MODEL-INTEGRATION.md) for input
schemas, checkpoint/CCD paths and complete commands. Shared CCD assets live
outside `.venv`, so environment reinstallation preserves them. All model
execution belongs on allocated GPU nodes.

The common inference and cache-building paths read `ccd_db` from the MiniWorld
input YAML. Compatibility scripts additionally accept `--ccd-db`. Prepare with
`foldforge ccd prepare` and verify with `foldforge ccd verify`.
`audit_model_boundaries.py --out docs/archive/model-boundaries-20260913.csv` inventories
all model classes with forward methods without loading the GPU stack.

The standard entry point is now `foldforge fold <model> --spec <MiniWorld YAML>`.
All predictors use `data/ccd/preprocessed_CCD.lmdb`; see
[the format contract](../docs/guides/MINIWORLD-FORMAT.md). The target-based scripts remain
compatibility tools for reproducing the recorded validation runs.

`benchmark_end_to_end.py --model <model>` defaults to five explicit modes for
AF3, OpenDDE, ESMFold2 and Protenix v2:

| Mode | Precision | Compile | Manual CUDA graph |
|---|---|---|---|
| `pytorch_eager_reference` | Model-default precision | no | no |
| `pytorch_compile_reference` | Model-default precision | yes | no |
| `pytorch_compile_bf16` | native BF16, FP32 norms | yes | no |
| `cuequiv_compile` | native BF16, FP32 norms | yes | no |
| `miniworld_graph` | native BF16, FP32 norms | yes | yes |

AF3 reference precision is `precision: af3_default`: input atom encoder and
diffusion FP32, trunk and confidence Pairformers BF16, norms and final confidence/
distogram projections FP32. Released FP32 weights are never rounded through BF16.
The AF3 `recycles` option means additional trunk passes: 10 gives 11 passes.
The benchmark retains token buckets in multiples of 128, disabled templates,
five batched diffusion samples, and denoiser-only compile/graph scope. This is
a FoldForge backend comparison, not a measurement of the official JAX runtime.

For the other models, `precision: model_default` keeps FP32 parameter storage.
OpenDDE executes in FP32 without autocast. Protenix v2 uses BF16 autocast outside
its FP32 diffusion sampler. ESMFold2 uses BF16 autocast in the input/trunk,
confidence folding trunk and diffusion pair transitions; remaining diffusion
projections stay FP32. Their reference TF32 setting is enabled. These are the
released precision scopes reproduced in FoldForge, not the original applications.
All three native BF16 comparison modes use BF16 parameters except FP32 norms,
with autocast disabled.

`--modes` selects a subset. Explicit `pytorch_eager_fp32` and
`pytorch_compile_fp32` retain whole-model FP32 diagnostics. The historical
`pytorch_eager` and `pytorch_compile` aliases remain BF16-only.
`--backends` retains the legacy compile-and-graph comparison and cannot be
combined with `--modes`. Use `--variant protenix-v2` for the v2 checkpoint.
`render_benchmark_results.py --results RUN --docs docs` validates all 20 cases
and renders the five-mode latency table and bar chart. It rejects inconsistent
inputs, precision, sample/recycle counts and execution flags.
Timing definitions and current results are in
[benchmark results](../docs/benchmark_results.md).


Precision diagnostics use `audit_af3_precision.py REPORT --spec INPUT
--config CONFIG --out RUN` inside an allocated GPU job. Add `--model esmfold2`, `--model opendde` or `--model protenix` to audit
the other adapters (AF3 is the default). Supply an eager configuration with
`precision: bf16`, `af3_default` (AF3), or `model_default` (other models),
one recycle, two diffusion steps and one sample. The report
records every parameter, observed Linear/GEMM operand dtypes, DiT module outputs
and actual CUDA autocast state. Its timings are diagnostic and must not enter
latency comparisons.

Batch diagnostics use `audit_af3_batch.py REPORT --spec INPUT --config CONFIG
--out RUN` inside an allocated GPU job. Use five samples in the input and an eager
configuration with one recycle and two diffusion steps. It compares identical
coordinates in one batched denoiser call against five independent calls, checks
one call per step, and observes MiniWorld Q/K/V `num_aug=5` with shared pair bias.
The full benchmark also rejects sequential AF3 execution through observed wrapper
call counts. Denoiser batches change random-number consumption relative to the
old sequential sampler; same-seed old/new trajectories need not match.

`audit_sample_axes.py REPORT MODEL --spec INPUT --config CONFIG --out RUN`
checks ESMFold2, Protenix v2 and OpenDDE on an allocated GPU. Use eager MiniWorld,
five samples, one recycle and two steps; supply `--lm-cache` for ESMFold2. It
records actual token kernel Q/bias shapes, excluding trunk triangle attention,
and ESMFold2 SWA construction with `num_aug=5`. SWA's FlashAttention boundary
flattens augmentation and structure batch; token attention retains `[A, B]`.
Rectangular flat-atom windows use the shared batched PyTorch path because the
engine pair-bias core currently supports square attention.

`compare_af3_precision.py --results RUN --experimental 4yx2.cif --output
quality.json --report af3_precision.md` compares all samples with the compiled
AF3-default reference (`pytorch_compile_reference`) and the experimental complex, and writes a latency chart and
quality table. Feed a merged `e2e-af3.json` containing the five measured modes.
Atom correspondence uses chain, label residue index, residue identity and atom
name; no matching-by-coordinate or best-sample substitution is performed.


Inference and benchmark commands accept `--trunk-seed` and `--diffusion-seed`
(defaults: 0/0). Input/conformer/MSA/trunk randomness and diffusion
noise/augmentation use independent streams. New result JSON records both seeds
and `seed_policy: split-v1`; historical coupled-seed results are not rewritten.
Use a new run directory when changing seed policy. The renderer refuses to mix
legacy and split-seed measurements in one comparison.
