# 5I28 benchmark, FoldForge side

Runs the FoldForge half of the 5I28 benchmark ([protocol](../../docs/guides/BENCHMARK-5I28.md)): every model, n = 1, 2, 4, 8, 12, 16
identical 5I28 chains (128 n tokens), the real MSA and templates of key `P0113359`, 4 trunk passes, 200 steps, 5 samples, bf16,
`compile` + CUDA graphs (`scope: trunk`), one H100.

- `run_5i28_ff.py`: one process per model, all n ascending, one `Predictor` call per n with `execution.benchmark_repeats` warm repeats
  (default 3: the 2026-10-29 run's value). The number is `model_seconds_warm_median`, the median of the warm complete model forwards
  (CUDA-synchronized, without featurization / loading / output decoding); the first forward of each size is cold and is recorded, not
  used. After every size the captured CUDA graphs and the allocator cache are released (`release_size_memory`: each size starts from
  the weights alone, so the pools of the smaller sizes cannot push the largest into an out-of-memory error). A cell is written when it
  finishes (`cells/<model>_x<n>.json`, `cells.jsonl`); an out-of-memory cell is recorded as `oom` and the run goes on. `--retry-fragmented` reruns only the 'reserved but unallocated' out-of-memory cells (the last fallback, expandable
  segments, which the sbatch script does by itself and the cell records as `retry`).
- `run_model.sbatch`: one model on one GPU (`-c 16 --hint=nomultithread`, the process pinned to the first 16 logical CPUs of the
  allocation = 16 distinct physical cores), a host-speed probe (`probe.py`) before and after, `nvidia-smi` clocks every 5 s,
  node CPU allocation / load before and after (`host.txt`, `gpu_clocks.csv` in the run directory). Resumable: submit it again.
- `gate.sbatch` + `submit_all.sh`: a CPU-only gate that waits until none of your jobs named `^b5i28` (the kit jobs) is left, then the
  nine model jobs (`--dependency=afterany:<gate>`). A timing must not share the node with another measurement.

## Commands

```bash
export OUT_ROOT=<where the run directories go>          # also holds the kernel / compile caches (CACHE_ROOT overrides)
export FOLDFORGE_5I28_FASTA=$PWD/tests/release_references/inputs/5i28/A.fasta
export FOLDFORGE_5I28_MSA_DB=<BioMolDB_20260224>/a3m_16k.lmdb          # key P0113359, 6790 rows
export FOLDFORGE_5I28_TEMPLATE_DB=<BioMolDB_20260224>/template.lmdb    # key P0113359 (20 records, 6 distinct sequences)

# one model, directly (sizes optional):
sbatch -J ff5i28-protenix1 --export=ALL,MODEL=protenix1 benchmarks/5i28_ff/run_model.sbatch
# all nine models behind the gate:
benchmarks/5i28_ff/submit_all.sh
```

Models: `protenix1 protenix2 opendde esmfold2 af3 boltz2 chai1 openfold3 rosettafold3`. Per-model settings are in `MODELS` of
`run_5i28_ff.py`: AF3 counts additional passes (`recycles` 3 = 4 passes), RoseTTAFold3 takes one template per chain (`template_n` 1)
and its family default of no input templates is switched on, as in the 2026-10-29 run. `template_dedup` stays at its default (true):
that is what gives the 4 distinct templates (`5SYD_A`, `6KOL_A`, `5YT7_A`, `6TFD_A`); FoldBench specs set it to false, a 5I28 spec must not.

The records of runs are `docs/records/5i28_foldforge_<date>.md` and `docs/records/data/`.
