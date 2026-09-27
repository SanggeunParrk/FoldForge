#!/bin/bash
# The MSA condition: every family on 5i28 with the same alignment (5i28-msa/A.a3m).
#   sbatch tools/references/generate_msa.sbatch        (all 30: 10 families x 3 seeds)
#   sbatch --array=<ids> tools/references/generate.sbatch
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=ff-refs-msa
#SBATCH --array=0-29%4
#SBATCH --output=slurm-msa_%A_%a.log
#SBATCH --error=slurm-msa_%A_%a.err
set -euo pipefail
: "${FOLDFORGE_HOME:=$HOME/.cache/foldforge}"; export FOLDFORGE_HOME
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
FAMILIES=(boltz2 chai1 protenix1 protenix2 openfold3 openfold3-preview2 rosettafold3 intellifold2 opendde esmfold2)
TARGETS=(5i28-msa)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
f=${FAMILIES[$((i / 3))]}; t=${TARGETS[0]}; s=${SEEDS[$((i % 3))]}
RAW=$FOLDFORGE_HOME/runs/references-raw/$f/$t/seed$s
rm -rf "$RAW"; mkdir -p "$RAW"
tools/references/run_release.sh "$f" "$t" "$s" "$RAW" > "$RAW.stdout" 2>&1 || { echo "FAILED $f $t $s"; tail -20 "$RAW.stdout"; exit 1; }
source tools/activate_env.sh
python tools/references/collect.py "$f" "$t" "$s" "$RAW"
