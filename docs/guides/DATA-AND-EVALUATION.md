# Data and evaluation ownership

| Responsibility | Implementation |
| --- | --- |
| Chemical constants and per-frame reference geometry | `data/constants/` |
| The shared CCD database (MiniWorld BioMol LMDB) | `data/ccd/` |
| Dense featurization, the per-family input conventions, structural tokens, bond orders, chirality | `data/features/` |
| MiniWorld input specs, FASTA and LMDB resources (MSA, templates), validation | `data/inputs/` |
| JSON, pickle and structure serialization helpers | `data/io.py` |
| Judging FoldForge against the released implementations | `eval/references.py` |
| Confidence decoding (pLDDT, PAE, PDE) | `eval/dense_confidence.py` |
| Structure reading and comparison helpers | `eval/structure.py` |

Featurization is AF3's (`alphafold3` from `libs/af3-data`), applied once for
every family; `data/features/dense_conventions.py` then applies the conventions a
family's release needs (its reference conformer frame, dropped atoms, MSA
deduplication, random reference pose, ...), all selected by `DenseSpec` fields.
There is one featurization path.

Prediction output stays under repository `runs/`, enforced by
`models/io/paths.py` for the CLI and direct output writers.
