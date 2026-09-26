#!/bin/bash
# Fold every family x target x seed that outputs/ holds a release reference
# for, with FoldForge, then judge:
#   scripts/references/validate.sh fast            (or exact)
#   foldforge validate runs/validate-fast --report docs/validation/fast.md
set -euo pipefail
MODE=${1:-fast}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p runs/references-raw/logs
LIST=runs/references-raw/validate-$MODE.tasks
ls -d outputs/*/*/seed* | awk -F/ '$2 != "inputs" {sub("seed","",$4); print $3" "$2" "$4}' > "$LIST"
N=$(wc -l < "$LIST")
MODE=$MODE LIST=$LIST sbatch --parsable --array=0-$((N - 1))%${THROTTLE:-3} \
  --export=ALL scripts/references/validate.sbatch
