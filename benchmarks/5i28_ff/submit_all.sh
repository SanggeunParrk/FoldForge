#!/bin/bash
# Submit the FoldForge side of the 5I28 benchmark: one gate job, then one single-GPU job per model that starts after the gate.
#   OUT_ROOT=<dir> FOLDFORGE_5I28_FASTA=... FOLDFORGE_5I28_MSA_DB=... FOLDFORGE_5I28_TEMPLATE_DB=... benchmarks/5i28_ff/submit_all.sh
# Optional: WAIT_FOR (default '^b5i28': the kit jobs), MODELS (default: all nine), NO_GATE=1 (submit the model jobs directly).
set -euo pipefail
: "${OUT_ROOT:?}" "${FOLDFORGE_5I28_FASTA:?}" "${FOLDFORGE_5I28_MSA_DB:?}" "${FOLDFORGE_5I28_TEMPLATE_DB:?}"
HERE=$(cd "$(dirname "$0")" && pwd); export FOLDFORGE_REPO=${FOLDFORGE_REPO:-$(cd "$HERE/../.." && pwd)}
MODELS=${MODELS:-"protenix1 protenix2 opendde esmfold2 af3 boltz2 chai1 openfold3 rosettafold3"}
mkdir -p "$OUT_ROOT/logs"; cd "$OUT_ROOT/logs"
DEP=()
if [ "${NO_GATE:-0}" != 1 ]; then
  GATE=$(sbatch --parsable -J ff5i28-gate --export=ALL,WAIT_FOR="${WAIT_FOR:-^b5i28}" "$HERE/gate.sbatch")
  echo "gate job $GATE"; DEP=(--dependency=afterany:"$GATE")
fi
for m in $MODELS; do
  J=$(sbatch --parsable -J "ff5i28-$m" "${DEP[@]}" --export=ALL,MODEL="$m" "$HERE/run_model.sbatch")
  echo "model $m job $J"
done
