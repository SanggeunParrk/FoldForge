# Chai-lab: with RDKit >= 2025 every ligand is dropped without an error

**Upstream:** https://github.com/chaidiscovery/chai-lab (`chai_lab` 0.6.1)

## Summary

Under RDKit 2025.x, `chai_lab.chai1.run_inference` fails to tokenise every
ligand entity and **folds the protein alone**, writing structures and scores
as usual. Pinning `rdkit==2024.9.5` restores the ligands.

## Reproduce

Trypsin + benzamidine (CCD `BEN`) + Ca2+ (PDB 3PTB), FASTA input with the
ligand given as SMILES, default settings. With RDKit 2025.x the output CIF has
no HETATM ligand records; with 2024.9.5 it has them.

## Expected

A ligand that cannot be tokenised should raise (or at least log at error
level), not disappear from the model input.

## Notes

The exact failing RDKit call was not isolated; the reference-conformer path
(ETKDGv3 with chirality) is the likely suspect, since that API changed between
the two releases. We run Chai-lab with an isolated `rdkit==2024.9.5`.
