#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH -J ff-dispatch
#SBATCH --mem=64g
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o validation/logs/%x_%j.log
#SBATCH -e validation/logs/%x_%j.err

set -euo pipefail
cd /home/psk6950/practice/FoldForge

ENV=/home/casp17/.conda/envs/esmfold2
STUBS=$HOME/.cache/team-gm-stubs/py312
LIBS=$HOME/.cache/team-gm-libs
ENGINE=$HOME/.cache/foldforge-engine/src
export LD_LIBRARY_PATH="$LIBS/cuequivariance_ops/lib:${LD_LIBRARY_PATH:-}"
# No uv on this cluster, so the layers go on PYTHONPATH directly: FoldForge,
# the team-gm submodule, and the engine checked out at the rev pyproject pins.
export PYTHONPATH="$LIBS:$STUBS:/home/psk6950/practice/FoldForge/src:/home/psk6950/practice/FoldForge/libs/team-gm/src:$ENGINE"
export HF_HUB_OFFLINE=1

"$ENV/bin/python" scripts/op_dispatch_audit.py "$@"
