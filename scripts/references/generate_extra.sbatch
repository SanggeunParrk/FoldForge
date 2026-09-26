#!/bin/bash
# The families whose released code runs only some targets: OpenDDE's release
# fails to featurise any input with a ligand or ion (map_to_standard in its
# template featuriser, even with templates off), so it has 5i28 alone.
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=ff-refs-x
#SBATCH --array=0-11%1
#SBATCH --output=runs/references-raw/logs/x_%A_%a.log
#SBATCH --error=runs/references-raw/logs/x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
TASKS=(opendde:5i28:0 opendde:5i28:1 opendde:5i28:2
       esmfold2:5i28:0 esmfold2:5i28:1 esmfold2:5i28:2
       esmfold2:3ptb:0 esmfold2:3ptb:1 esmfold2:3ptb:2
       esmfold2:1a1k:0 esmfold2:1a1k:1 esmfold2:1a1k:2)
IFS=: read -r f t s <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
RAW=runs/references-raw/$f/$t/seed$s
rm -rf "$RAW"; mkdir -p "$RAW"
scripts/references/run_release.sh "$f" "$t" "$s" "$RAW" > "$RAW.stdout" 2>&1 || { echo "FAILED $f $t $s"; tail -20 "$RAW.stdout"; exit 1; }
source scripts/activate_env.sh
python scripts/references/collect.py "$f" "$t" "$s" "$RAW"
