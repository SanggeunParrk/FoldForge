# 5I28 n >= 12 on H100: the atom DiT conditioning bug and a rerun on the fixed engine (2026-11-12)

Addendum to the [2026-11-11 record](5i28_foldforge_20261111.md) (that file is not edited). Data:
[data/5i28_atom_blocks_check_20261112.json](data/5i28_atom_blocks_check_20261112.json) (the per-chain check),
[data/5i28_foldforge_20261112_engine_5dc5396bd.csv](data/5i28_foldforge_20261112_engine_5dc5396bd.csv) (the rerun, every cell).

## The report

The engine side found that the H100 (sm_90) `cuda` backend of the atom DiT blocks (`AtomDiTOpenFold3`, `Protenix`, `Boltz2`,
`RoseTTAFold3`; not `Chai1`, whose AdaLN is the identity) computed the output gates from LN(c) instead of c, about 0.08 relative error on every
output row, whenever the conditioning had 8448..16832 rows (the launcher chose `grid.y = 2`). Fixed in the engine's local `main` at `5dc5396bd`
(not pushed, not on PyPI; the engine's earlier 2.5.3 build had the bug). The rows are those of one fold's padded atom count: FoldForge's sampler
shares the conditioning over its 5 samples (a stride-0 expand), so they are not multiplied by 5.

Cells of the 2026-11-11 record that ran the bad range on engine 2.5.1: **n = 12 (11676 atoms) and n = 16 (15568), both the diffusion-step atom
blocks and the input embedder's atom encoder, for AF3, OpenFold3, Protenix v1 / v2, OpenDDE (where it ran), Boltz-2 and RoseTTAFold3.** n <= 8
(7784 atoms, padded below 8448) and n = 2 are outside it; Chai-1 and ESMFold2 (FoldForge's own atom stack) are not affected.

## Check 1: Protenix v1 on engine 2.5.1, atom blocks on cuda against pytorch

Same inputs and seeds, `engine.ops: {atom_blocks: pytorch}` as the reference. The n chains of the assembly are identical, so a sound fold repeats
itself (chain k superposed on chain 0); comparing the whole complex is not a valid metric (the arrangement of identical chains is not fixed by
the inputs: 6-26 A even at n = 8, below the range).

| n | atoms | chain k vs chain 0, cuda (mean / max A) | chain k, cuda vs pytorch (mean / max A) | pLDDT cuda / pytorch | warm cuda / pytorch (s) |
|---|---|---|---|---|---|
| 8 | 7784 | 0.36 / 0.72 | 0.32 / 0.79 | 0.8085 / 0.8083 | 3.86 / 4.48 |
| 12 | 11676 | 0.09 / 0.60 | 0.45 / 0.85 | 0.8049 / 0.8075 | 9.06 / 10.29 |
| 16 | 15568 | 0.08 / 0.16 | 0.42 / 0.67 | 0.8159 / 0.8160 | 17.24 / 18.47 |

No jump above the range in either comparison.

## Check 2: the nine models rerun on the fixed engine

`benchmarks/5i28_ff/` at `ce23aeb` on an immutable copy of the engine's `5dc5396bd` source (`PYTHONPATH`), H100, node01, the engine's JIT cache,
n = 1..16, one process per model. The run's `meta.json` names `2.5.1` because it records the installed package, not the one imported (the runner
now also records `miniworld_engine_loaded_from`); that the fixed source ran is shown by its logs: 0 backend refusals, against 21 on 2.5.1
(`MSAPairWeightedAveraging` past 1024 tokens, `AttentionPairBias` batched), and Chai-1 runs at all.

**Timing: every cell the 2026-11-11 record has is within 4.5% of its 2.5.1 value, the affected n = 12 and 16 included** (AF3 7.97 -> 7.97 and
14.79 -> 14.82 s, Protenix v1 9.12 -> 9.02 and 17.20 -> 17.14, Boltz-2 11.92 -> 12.12 and 22.73 -> 23.09; the full table is in the CSV). The bug did
not change how long the folds take, so the n >= 12 timings of the record stand.

New cells (the 2026-11-11 record had none for these):

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| Chai-1, warm (s) | 0.551 | 1.183 | 3.706 | 14.685 | 36.191 | 71.229 (the runner's retry: a fresh process with expandable segments) |
| Chai-1, kit / FoldForge | 8.27 | 3.52 | 2.72 | 1.87 | 1.43 | 1.26 |
| OpenDDE, warm (s) | 0.668 | 1.286 | 3.551 | 13.127 | 31.275 | out of memory |
| OpenDDE, kit / FoldForge | 6.09 | 2.89 | 1.98 | 1.81 | 1.89 | (kit failed too) |

OpenDDE n = 16 runs out of memory with and without expandable segments; the allocation that fails (11.26 GiB) matches in size one structural pair
tensor, [3968, 3968, 384] in bf16 (not located further). The kit's n = 16 also fails.

## Check 3: does the fix change the outputs? Yes, by the size of any numerical change

One n = 12 fold, same inputs and seeds, engine 2.5.1 twice and the fixed engine once:

| model | 2.5.1 twice: identical coordinates, chain RMSD | 2.5.1 against fixed: identical, chain RMSD (mean / max A) | pLDDT: 2.5.1 / 2.5.1 / fixed |
|---|---|---|---|
| Protenix v1 | yes, 0.0 | no, 0.475 / 0.771 | 0.8049 / 0.8049 / 0.8067 |
| Boltz-2 | yes, 0.0 | no, 0.665 / 0.910 | 0.8358 / 0.8358 / 0.8420 |

The runs are deterministic, so the difference is the engine change. It is not isolated to the conditioning fix (`5dc5396bd` also carries the
engine's other changes since 2.5.1, e.g. the batched attention-bias and the wide MSA kernels on cuda), and its size, 0.5-0.9 A per chain and
+0.002 / +0.006 pLDDT, is that of the cuda-against-pytorch comparison above: ordinary divergence of a 200-step sampler under any numerical change.
It does not say which output is the more accurate. Structures of the 2.5.1 run at n >= 12 are not visibly damaged (Check 1); they are not the
fixed engine's.

## Not shown

- The other six affected models were not compared structure against structure (only Protenix v1 and Boltz-2 in Check 3, Protenix v1 in Check 1).
- The fixed engine is unreleased. The numbers of the 2026-11-11 record stay the ones on engine 2.5.1; a record on a released engine (the engine's
  2.5.3 is being rebuilt from `main`) replaces them.
