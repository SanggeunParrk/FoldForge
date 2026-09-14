#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH -J ff-autotune
#SBATCH --mem=160g
#SBATCH -c 8
#SBATCH -t 20:00:00
#SBATCH -o validation/logs/%x_%j.log
#SBATCH -e validation/logs/%x_%j.err

set -euo pipefail
cd /home/psk6950/practice/FoldForge

# Use the repository environment and its locked engine dependency.
source scripts/activate_env.sh
ENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1
# Unlock the full autotune grid so every config is benched and recorded.
export MINIWORLD_RUN_AUTOTUNE=1
export MINIWORLD_AUTOTUNE_CAPTURE=1

"$ENV/bin/python" scripts/build_autotune_cache.py "$@"
