# Build cache gaps from a real inference workload

Run on an allocated GPU using the repository's pinned environment. The driver
uses the ordinary model input, checkpoint, native BF16 and execution settings.
It preserves shipped caches and fills missing keys in a separate directory.
A successful unit merges a provenance-checked shard. On failure/timeout, only
profiles whose recorded search set exactly matches the full declared grid and
contains a finite timing are recovered. The unit remains failed; incomplete
profiles and legacy shards without search evidence are excluded. Logs and round
records remain available for another invocation with a new output directory.

```bash
source scripts/activate_env.sh
python scripts/build_autotune_cache.py \
  --model af3 --cache-dir validation/cache/a6000 \
  --out validation/cache-units/af3-1ubq --timeout 3600 -- \
  --spec configs/inference/operational-1ubq.yaml \
  --config configs/inference/graph-bf16.yaml

foldforge fold af3 --spec configs/inference/operational-1ubq.yaml \
  --config configs/inference/graph-bf16.yaml \
  --engine-cache-dir validation/cache/a6000 --out outputs/af3-1ubq
```

The example uses the cluster's operational BioMol MSA/template stores; edit paths
for another deployment. For ESMFold2 omit template conditioning from the spec.
Protenix v2 uses `--model protenix` and a config containing the v2 variant.

The driver configures the current engine Settings API and installs capture before
the first kernel so its first cache miss also receives parallel precompilation.
Round records preserve reusable measurements across units. The legacy
MINIWORLD_RUN_AUTOTUNE/MINIWORLD_AUTOTUNE_CAPTURE environment flags and the removed
capture.capturing(root=...) API are not used.

Only one writer builds into a particular cache directory at a time. Use distinct
cache directories for concurrent builds. Each unit has a whole-process-tree
wall-clock timeout, including compiler workers that create private sessions; diagnostics print elapsed time every 30 seconds. This is not
an ETA for an unknown number of missing kernel configurations.

The cache path bridge uses the pinned engine's cache root and merge interfaces.
GPU/compiler/source identity checks remain active. A directory built for A6000
is not an A5000 or A100 qualification. Changing engine versions requires rerunning
the regression and cache-coverage checks.
