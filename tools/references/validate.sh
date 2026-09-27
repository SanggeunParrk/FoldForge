#!/bin/bash
# Fold every family x target x seed that tests/release_references/ holds a release reference
# for, with FoldForge, then judge:
#   tools/references/validate.sh fast            (or exact)
#   foldforge validate $FOLDFORGE_HOME/runs/validate-fast --report docs/validation/fast.md
set -euo pipefail
: "${FOLDFORGE_HOME:=$HOME/.cache/foldforge}"; export FOLDFORGE_HOME
MODE=${1:-fast}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p $FOLDFORGE_HOME/runs/references-raw/logs
LIST=$FOLDFORGE_HOME/runs/references-raw/validate-$MODE.tasks
ls -d tests/release_references/*/*/seed* | awk -F/ '$2 != "inputs" {sub("seed","",$4); print $3" "$2" "$4}' > "$LIST"
N=$(wc -l < "$LIST")
MODE=$MODE LIST=$LIST sbatch --parsable --array=0-$((N - 1))%${THROTTLE:-3} \
  --export=ALL tools/references/validate.sbatch
