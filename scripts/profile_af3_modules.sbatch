#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH -J ff-af3-profile
#SBATCH --mem=96g
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -o validation/reports/logs/%x_%j.log
#SBATCH -e validation/reports/logs/%x_%j.err

set -euo pipefail
cd /home/psk6950/practice/FoldForge

source scripts/activate_env.sh
ENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8

"$ENV/bin/python" scripts/profile_af3_modules.py "$@"
