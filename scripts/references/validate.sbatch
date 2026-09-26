#!/bin/bash
# One FoldForge fold for scripts/references/validate.sh: task i of $LIST.
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=ff-validate
#SBATCH --output=runs/references-raw/logs/validate_%A_%a.log
#SBATCH --error=runs/references-raw/logs/validate_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/activate_env.sh
read -r f t s < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
python scripts/references/fold_ours.py "$f" "$t" "$s" "validate-$MODE" --mode "$MODE" > /dev/null
