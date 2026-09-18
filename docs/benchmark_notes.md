**What changed since the 4YX2 report.** The target is now 5I28 azurin (128
residues) replicated as N identical chains, so one input family covers every
length. The five-mode comparison runs at 512 tokens; the length ladder runs the
two reference modes and the MiniWorld mode from 512 tokens in 256-token steps
until every mode fails. The MSA and template policy is unchanged (`af3-msa-v1`).
The 4YX2 report is preserved in
[archive/benchmark-20260918-4yx2](archive/benchmark-20260918-4yx2/benchmark_results.md).

**Longest length that runs** (tokens; reference is the same for eager and compile):

| Model | A100 80 GB reference | A100 80 GB MiniWorld | A6000 48 GB reference | A6000 48 GB MiniWorld |
|---|---:|---:|---:|---:|
| AF3 | 1536 | 2816 | 1280 | 2048 |
| OpenDDE | 1024 | 768 | 768 | 512 |
| ESMFold2 | 768 | 1536 | 768 | 1024 |
| Protenix v2 | 2048 | 1792 | 1536 | 1536 |

**The speed ratio grows with length.** MiniWorld against reference compile goes
from 5.2x at 512 tokens to 6.9x at 1280 for AF3 on the A100, and from 4.2x to
8.2x at 1792 for Protenix v2. The A6000 shows the same trend (AF3 4.9x to 7.2x,
Protenix v2 4.6x to 7.9x). Moving from the A6000 to the A100 speeds every mode
up by 1.5x to 2.5x and leaves the ratios about where they were.

**Where MiniWorld mode runs out of memory first.** OpenDDE fails one length
earlier in MiniWorld mode than in the reference on both GPUs; at 768 tokens on
the A6000 the process log shows 27 GiB of 47 GiB held in CUDA-graph private
pools. Protenix v2 on the A100 also fails at 2048 tokens in MiniWorld mode while
the reference still runs; its peak at 1792 tokens is 60.8 GiB against 52.1 GiB.
AF3 and ESMFold2 go the other way and run 1.5x to 2x longer inputs in MiniWorld mode.
The OpenDDE A6000 run at 1280 tokens ended with a Triton illegal memory access
instead of an out-of-memory error and is shown as `exit 1`.

**Single warm forward (†).** Protenix v2 reference rows from 1280 tokens on the
A6000 and from 1536 tokens on the A100 time one warm forward instead of the
median of five, because one forward takes 20 to 50 minutes there. Those rows
ran as separate jobs, one per length and mode, and were merged with
`scripts/merge_benchmark_rows.py`.
