#!/bin/bash
# Regenerate outputs/: every family x target x seed with the RELEASED code.
#   sbatch scripts/references/generate.sbatch            (all 72)
#   sbatch --array=<ids> scripts/references/generate.sbatch
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=ff-refs
#SBATCH --array=0-71%4
#SBATCH --output=runs/references-raw/logs/%A_%a.log
#SBATCH --error=runs/references-raw/logs/%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
FAMILIES=(boltz2 chai1 protenix1 protenix2 openfold3 openfold3-preview2 rosettafold3 intellifold2)
TARGETS=(5i28 3ptb 1a1k)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
f=${FAMILIES[$((i / 9))]}; t=${TARGETS[$((i / 3 % 3))]}; s=${SEEDS[$((i % 3))]}
RAW=runs/references-raw/$f/$t/seed$s
rm -rf "$RAW"; mkdir -p "$RAW"
scripts/references/run_release.sh "$f" "$t" "$s" "$RAW" > "$RAW.stdout" 2>&1 || { echo "FAILED $f $t $s"; tail -20 "$RAW.stdout"; exit 1; }
source scripts/activate_env.sh
python scripts/references/collect.py "$f" "$t" "$s" "$RAW"
