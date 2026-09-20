"""Run one module of a dense-graph family on saved inputs, against a reference dump.

Porting aid. The reference outputs come from the AF3 fork that FoldForge's family
conventions mirror (refs/oracle/reference_module.py), run with the same blob and the
same inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from team_gm.modules.checkpoints.layers import skip_random_init

from foldforge.models.checkpoints.haiku import import_jax_weights_
from foldforge.modules.dense.spec import SPECS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(SPECS))
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--loaded", help="registry name: compare the fully loaded model"
    )
    args = parser.parse_args()
    from foldforge.models.architectures.af3 import AlphaFold3

    if args.loaded:
        # The full load path: precision conversion and the team-gm block wrappers.
        from foldforge.models import load

        model = load(
            args.loaded, args.blob, backend="pytorch", dtype=torch.float32, device="cpu"
        )
    else:
        with skip_random_init():
            model = AlphaFold3(spec=SPECS[args.family])
        import_jax_weights_(model, args.blob)
        model = model.float().eval()
    data = {k: torch.from_numpy(v) for k, v in np.load(args.inputs).items()}
    with torch.no_grad():
        if args.kind == "token_transformer":
            outputs = (
                model.diffusion_head.transformer(
                    act=data["act"],
                    mask=data["mask"].bool(),
                    single_cond=data["single_cond"],
                    pair_cond=data["pair_cond"],
                ),
            )
        elif args.kind == "trunk_block":
            block = model.evoformer.trunk_pairformer[int(data.get("block", 0))]
            outputs = block(
                data["pair"], data["pair_mask"], data["single"], data["seq_mask"]
            )
        elif args.kind == "msa_block":
            block = model.evoformer.msa_stack[int(data.get("block", 0))]
            outputs = block(
                msa=data["msa"],
                pair=data["pair"],
                msa_mask=data["msa_mask"],
                pair_mask=data["pair_mask"],
            )
        elif args.kind == "atom_transformer":
            from foldforge.modules.dense import atom_layout

            gather = atom_layout.GatherInfo(
                gather_idxs=data["key_idxs"],
                gather_mask=data["keys_mask"] > 0,
                input_shape=torch.tensor(data["queries_act"].shape[:2]),
            )
            outputs = (
                model.diffusion_head.atom_cross_att_encoder.atom_transformer_encoder(
                    queries_act=data["queries_act"],
                    queries_mask=data["queries_mask"].bool(),
                    queries_to_keys=gather,
                    keys_mask=data["keys_mask"].bool(),
                    queries_single_cond=data["queries_cond"],
                    keys_single_cond=data["keys_cond"],
                    pair_cond=data["pair_cond"],
                ),
            )
        else:
            message = f"unknown kind {args.kind}"
            raise SystemExit(message)
    reference = np.load(args.reference)
    for index, output in enumerate(outputs):
        ref = torch.from_numpy(reference[f"out{index}"])
        delta = (output.float() - ref).abs()
        corr = torch.corrcoef(torch.stack([output.flatten(), ref.flatten()]))[0, 1]
        scale = ref.square().mean().sqrt()
        print(  # noqa: T201 - CLI output contract
            f"out{index}: max|d| {delta.max():.3e}  rms {scale:.3e}"
            f"  max|d|/rms {delta.max() / scale:.3e}  corr {corr:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
