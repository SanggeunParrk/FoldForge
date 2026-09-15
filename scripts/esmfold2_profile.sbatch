#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH -J ff-profile
#SBATCH --mem=64g
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o validation/reports/logs/%x_%j.log
#SBATCH -e validation/reports/logs/%x_%j.err

set -euo pipefail
cd /home/psk6950/practice/FoldForge

# Use the repository environment and its locked engine dependency.
source scripts/activate_env.sh
ENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1

"$ENV/bin/python" scripts/esmfold2_profile.py "$@"
