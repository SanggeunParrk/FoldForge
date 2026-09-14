# FoldForge AF3 input features

This package contains the CPU feature and native structure parsing subset of
official AlphaFold3 v3.0.1. See SOURCE.json for the exact revision and LICENSE
for its terms. JAX model inference is not included; FoldForge uses its separate
PyTorch AF3 port. CPU JAX supports feature preparation only.

Build through the FoldForge locked environment. Generate CCD indices with
`foldforge ccd prepare --components <components.cif> --rdkit <molecules.pkl> --out <database>`
and pass the same `--ccd-db <database>` to every FoldForge predictor. AF3 features
receive the explicit CCD index plus scoped glycan classification sets.
The external asset directory avoids losing generated files during venv rebuilds.
