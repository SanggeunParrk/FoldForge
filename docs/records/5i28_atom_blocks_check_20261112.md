# 5I28 n >= 12 on H100: the atom DiT blocks on cuda against pytorch (2026-11-12)

A check made after a report from the engine side that the H100 (sm_90) `cuda` backend of the atom DiT blocks returns wrong outputs above 8192
atoms (12288 and 16384 measured, error about 0.09 against 5e-4 for pytorch; 5I28 has about 11.7k atoms at n = 12 and 15.6k at n = 16). The
[2026-11-11 record](5i28_foldforge_20261111.md) was written before that report; it is not edited, this note is its addendum. Data:
[data/5i28_atom_blocks_check_20261112.json](data/5i28_atom_blocks_check_20261112.json).

## What was run

Protenix v1, 5I28 at n = 8 (7784 atoms, below the limit: the control), 12 (11676) and 16 (15568), on one H100, engine 2.5.1 (PyPI), the same
inputs and seeds twice: default (`backend: miniworld`, the atom blocks on the engine's `cuda` backend) and `engine.ops: {atom_blocks: pytorch}`.

## Result

| n | atoms | chain k against chain 0, cuda (mean / max, A) | same, pytorch | chain k, cuda against pytorch (mean / max, A) | mean pLDDT cuda / pytorch | warm cuda / pytorch (s) |
|---|---|---|---|---|---|---|
| 8 | 7784 | 0.359 / 0.723 | 0.215 / 0.664 | 0.315 / 0.787 | 0.8085 / 0.8083 | 3.86 / 4.48 |
| 12 | 11676 | 0.085 / 0.602 | 0.079 / 0.138 | 0.445 / 0.851 | 0.8049 / 0.8075 | 9.06 / 10.29 |
| 16 | 15568 | 0.084 / 0.164 | 0.094 / 0.538 | 0.415 / 0.667 | 0.8159 / 0.8160 | 17.24 / 18.47 |

The n chains of a 5I28 assembly are identical, so a sound fold repeats itself: every chain superposed on chain 0 agrees to under 1 A, on cuda
and on pytorch alike, at all three sizes, and the same chain from the two runs agrees to 0.3-0.9 A above the limit exactly as it does below it.
pLDDT differs by at most 0.003. The cuda atom blocks were 7-14% faster, so the 2026-11-11 timings are not inflated by this.

## What this does and does not show

- Protenix v1 shows no structural damage at n = 12 and 16 in the finished structures. The engine's 200x error was measured on a block output;
  nothing here says that measurement is wrong, only that it did not reach the folds of this model.
- Not checked: the other seven models at n >= 12 (OpenFold3, Boltz-2 and RoseTTAFold3 use other atom block classes), and any size other than
  8 / 12 / 16. Until they are, their n >= 12 cells in the 2026-11-11 record are unverified in the same sense.
- A first attempt compared the whole complex between the two runs and found 6-26 A at n = 8 already: the arrangement of identical chains is
  not fixed by the inputs, so that metric cannot separate a sound structure from a broken one and is not used.
