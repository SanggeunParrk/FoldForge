# Validation against the released implementations

`fast.md` and `exact.md` are `foldforge validate` verdicts for the two modes
against `outputs/`: every family's structures from its **own released code**
on 5I28 (protein), 3PTB (protein + ligand + ion) and 1A1K (DNA + protein +
ions), three seeds of five samples each, no MSA, no template.

Regenerate with `scripts/references/validate.sh fast` (or `exact`), then
`foldforge validate runs/validate-<mode> --report docs/validation/<mode>.md`.

## How a row is judged

Each check is relative to the release's own seed-to-seed spread, not to a fixed
number:

- **structure**: mean pairwise polymer CA/C1' RMSD, ours against the release's,
  passes within `rel-rel + max(0.5, 0.5 * rel-rel)`.
- **pLDDT**: the means differ by at most `max(2.0, 2 * hypot(sd_rel, sd_ours))`
  over the per-seed means.
- **ligand geometry**: sorted pairwise-distance spectrum against the CCD ideal,
  within the release's own + 0.05 A.
- **placement**: permutation-matched ligand/ion centroids after polymer
  superposition, under the structure rule.

## Current state (2026-09-26)

27 of 28 rows pass in both modes. The exception is known:

- **OpenDDE 5I28 pLDDT, 68.4 against 71.1.** Structure passes (1.81 A against
  a 2.21 A bound). The gap is the reference conformer source: OpenDDE takes its
  residues' conformers from its own RDKit CCD pickle, and FoldForge keeps AF3's
  rule for every family (decided 2026-09-25). Swapping in the release's
  conformer, with our random pose on top, reads 70.9 against 71.1. Every stage
  otherwise matches the release when fed the same input: the trunk to 4e-3
  relative through all four passes, the structural expander and refiner
  exactly, and one denoiser call to 0.02 A at every noise level.

Also worth knowing when reading the tables:

- OpenDDE's release fails to featurise any input with a ligand or ion, so it
  has 5I28 alone.
- OpenFold3's pLDDT on 5I28 sits ~1.5 above its release on every seed, inside
  the tolerance.

## Defects this comparison found

Each of these folded, looked plausible, and passed every earlier check:

- **ESMFold2's language model read a scrambled sequence.** The ESM-C pair path
  handed the tower structure-side residue indices instead of its vocabulary ids
  (alanine as `<cls>`, cysteine as leucine). The release references themselves
  had been generated with a randomly initialised ESM-C (the Hugging Face cache
  copy loads with mismatched keys and only a warning).
- **ESMFold2 read every DNA base as the one below it.** ESMFold2 orders its
  nucleic classes A G C U N DA DG DC DT; AF3 orders them A G C U DA DG DC DT N.
  Both the converter and the heads' class widening laid one onto the other in
  order. 1A1K's duplex folded 9.9 A from the release's; it now reads 0.96 A
  against the release's own 0.92.
- **ESMFold2's reference conformers** came from an older table than the one the
  release now reads, and covered no nucleotides; single-atom ions sat off the
  origin; the 5' OP3 its atom lists never carry was kept.
- **OpenDDE's MSA module saw AF3's duplicated self-row.** The release feeds the
  query once; the duplicate moved the module's output by 2%, and recycling
  grew that to a trunk pair 0.89-correlated with the release's.
- **OpenDDE's denoiser read an unposed conformer** while its trunk read the
  posed one; the release keeps a single atom array for both.
