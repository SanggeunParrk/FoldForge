"""Run the released Chai-1 on a FASTA: no MSA, 3 trunk passes, 200 steps, 5 samples.

Usage: chai_run.py FASTA OUT_DIR SEED. CHAI_DOWNLOADS_DIR must point at the
release's weights. Needs rdkit 2024.9.x: with a newer RDKit the release fails
to tokenise every ligand and silently folds without it.
"""

import sys
from pathlib import Path

import chai_lab.chai1 as chai1
import torch

fasta, out, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed)
chai1.run_inference(
    fasta_file=fasta,
    output_dir=out,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=seed,
    device="cuda:0",
    use_esm_embeddings=True,
)
