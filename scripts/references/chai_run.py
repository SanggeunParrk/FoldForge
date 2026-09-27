"""Run the released Chai-1 on a FASTA: 3 trunk passes, 200 steps, 5 samples.

No MSA unless an ``A.a3m`` sits beside the FASTA; then it is converted to the
release's own ``.aligned.pqt`` (first row the query) and handed over as its
MSA directory, so Chai-1 reads the same alignment as every other family.

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
msa_directory = None
a3m = fasta.with_name("A.a3m")
if a3m.is_file():
    from chai_lab.data.parsing.msas.aligned_pqt import (
        a3m_to_aligned_dataframe,
        expected_basename,
    )
    from chai_lab.data.parsing.msas.data_source import MSADataSource

    query = a3m.read_text().split("\n")[1].strip()
    msa_directory = out / "msas"
    msa_directory.mkdir(parents=True, exist_ok=True)
    frame = a3m_to_aligned_dataframe(
        a3m, MSADataSource.UNIREF90, insert_pairing_key=False
    )
    frame.to_parquet(msa_directory / expected_basename(query))
chai1.run_inference(
    fasta_file=fasta,
    output_dir=out / "chai",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=seed,
    device="cuda:0",
    use_esm_embeddings=True,
    msa_directory=msa_directory,
)
