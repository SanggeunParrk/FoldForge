# Upstream bug reports

Defects in released third-party code found while validating FoldForge against
each release (`outputs/`, `docs/validation/`). Each file is written to be filed
as-is on the upstream tracker. None has been filed yet; when one is, add its
link to the table.

| Report | Upstream | Severity | Filed |
|---|---|---|---|
| [opendde-ligand-featurization.md](opendde-ligand-featurization.md) | aurekaresearch/OpenDDE | crash on any ligand or ion | no |
| [esmfold2-hf-esmc-random-weights.md](esmfold2-hf-esmc-random-weights.md) | Biohub/esm, transformers ESMFold2 | silent wrong results | no |
| [af3-any-model-esmfold2-nucleic-order.md](af3-any-model-esmfold2-nucleic-order.md) | sokrypton/alphafold3 (`af3-any-model`) | silent wrong results on DNA/RNA | no |
| [chai-lab-rdkit-2025-ligands.md](chai-lab-rdkit-2025-ligands.md) | chaidiscovery/chai-lab | silent ligand drop | no |

"Silent" means the run finishes, writes plausible structures and confidences,
and reports no error. Those are the dangerous ones.
