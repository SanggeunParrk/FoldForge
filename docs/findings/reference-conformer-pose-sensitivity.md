# Reference conformer and pose leak uncontrolled information into the trunk

**Status:** recorded 2026-09-26, to be investigated further. Not a bug in any
implementation: the released OpenDDE shows the same behaviour.

## Claim

In an AF3-style model the per-residue reference conformer (`ref_pos`) is meant
as a chemistry prior. Its exact coordinates -- which RDKit conformer was drawn,
and the random rigid pose applied to it -- are arbitrary, yet they change the
prediction: the atom encoder reads `ref_pos` through a Linear layer, so the
arbitrary choice enters the single representation and is amplified by the
trunk's recycling. On a low-confidence target this moves loop conformations by
several angstroms and pLDDT by several points, with everything else fixed.

## Evidence (OpenDDE, 5I28)

Setup: 5I28, one protein chain, 127 residues, no MSA, no template, 4 trunk
passes, 200 steps, fp32. The released OpenDDE was run with every sampler draw
recorded (initial noise, per-step rotation/translation, churn noise); FoldForge,
which reproduces the release sample for sample when given the same inputs
(CA 0.00-0.16 A on 13 of 15 samples, pLDDT within 0.1), replayed the identical
draws. 3 seeds x 5 samples. Across settings only `ref_pos` differs: all 127
model input tensors were compared, the other 125 are bit-identical.

2x2 of conformer shape x per-residue pose (mean over 15 samples; pLDDT
difference to the release sample / CA RMSD to it):

| | pose = release | pose = FoldForge's draw |
|---|---|---|
| conformer = release (Protenix pickle) | -0.05 / 0.22 A | -0.23 / 2.16 A |
| conformer = FoldForge (AF3 ETKDGv3) | -1.99 / 1.27 A | -2.86 / 1.64 A |

- **Pose alone** (a random rotation plus a shift of at most 1 A per residue,
  chemically meaningless) moves the structure 2.16 A on average.
- **Conformer alone** (median 0.93 A per-residue difference, mostly side-chain
  rotamers) moves it 1.27 A and lowers pLDDT by 2 points; up to 4.
- The displacement concentrates in the flexible loop 74-92 and in 36-44 (CA up
  to 7 A in one sample); the core (46 of 128 residues) stays within 0.8 A.
- The pLDDT drop is global but largest at core aromatic/hydrophobic residues
  (Trp48, Phe15, Leu33, His35, Phe111), not in the moving loop (correlation
  between per-residue displacement and pLDDT drop: -0.38).
- The effect of the conformer depends on the pose drawn: with seed 1's pose our
  conformer reproduced the release (0.15 A, +0.4 pLDDT); with seeds 0 and 2 it
  did not (-2.9 and -3.5 pLDDT).
- The release itself: switching its own pose off moves its structures 2.3 A and
  its pLDDT from 71.2 to 73.6.

Mechanism observed: a ~4% difference in the atom encoder's per-token output
became a trunk pair representation only 0.89-correlated after four recycles (the
same amplification measured for a 2% MSA-module difference).

## Scope and caveats

- One protein-only target with low confidence (pLDDT ~71). On high-confidence
  targets (3PTB 98, 1A1K 94) every family validated within its release's spread
  with FoldForge's conformers, which suggests the effect is confined to
  uncertain regions; not yet measured directly.
- OpenDDE only (its release cannot featurise ligands or ions).

## Where the data is

Not in git (`runs/` is ignored):

- `runs/plddt_test/`: per-pair folders with the release and the four
  FoldForge variants, superposed, pLDDT in B-factors; `pairs.md`, `README.md`.
- `runs/release-compare-20260924/odde-noise/`: the recorded release draws.
- Tools: `runs/release-compare-20260924/tools/odde_noisecap.py` (record),
  `ours_noise_replay.py` (replay; conformer/pose settings), `dump_inputs.py`
  (input equality check).
