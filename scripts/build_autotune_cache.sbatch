#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH -J ff-autotune
#SBATCH --mem=96G
#SBATCH -c 12
#SBATCH -t 04:00:00
#SBATCH -o validation/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the FoldForge repository root}"
source scripts/activate_env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/build_autotune_cache.py "$@"
