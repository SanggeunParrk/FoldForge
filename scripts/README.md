# scripts

Drivers, not library code: each `*.py` runs one measurement or produces one set
of structures, and each `*.sbatch` is a thin Slurm wrapper around the `*.py` of
the same name taking the same arguments. Run on a compute node, never on the
login node.

Empty for now — the ESMFold2 drivers come over with the model. See
[../docs/PORTING-esmfold2.md](../docs/PORTING-esmfold2.md).
