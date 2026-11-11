#!/bin/bash
# One model of the 5I28 benchmark (FoldForge side) on ONE GPU: every n in one process, host probe + GPU clock log around it.
#   sbatch --export=ALL,MODEL=protenix1,OUT_ROOT=<dir>,FOLDFORGE_5I28_FASTA=...,FOLDFORGE_5I28_MSA_DB=...,FOLDFORGE_5I28_TEMPLATE_DB=... \
#          -J ff5i28-protenix1 benchmarks/5i28_ff/run_model.sbatch
# Edit the partition / account / qos lines for your cluster. --hint=nomultithread: PIN physical cores (the first PIN logical CPUs of the allocation).
# Resumable: submit it again and finished cells are skipped. Optional: PIN (default 16), REPEATS (default 3), SIZES (default "1 2 4 8 12 16").
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=normal_h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH -o slurm-%x-%j.log
set -uo pipefail
: "${MODEL:?MODEL}" "${OUT_ROOT:?OUT_ROOT}" "${FOLDFORGE_5I28_FASTA:?}" "${FOLDFORGE_5I28_MSA_DB:?}" "${FOLDFORGE_5I28_TEMPLATE_DB:?}"
REPO=${FOLDFORGE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PIN=${PIN:-16}; REPEATS=${REPEATS:-3}; SIZES=${SIZES:-"1 2 4 8 12 16"}
OUT=$OUT_ROOT/$MODEL; CACHE=${CACHE_ROOT:-$OUT_ROOT/cache}
mkdir -p "$OUT" "$CACHE"
export XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export OMP_NUM_THREADS=$PIN MKL_NUM_THREADS=$PIN OPENBLAS_NUM_THREADS=$PIN PYTHONFAULTHANDLER=1

ALLOWED=$(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{ if (NF == 2) { for (i = $1; i <= $2; i++) print i } else print $1 }')
SET=$(echo "$ALLOWED" | head -n "$PIN" | paste -sd, -)
PINNED="taskset -c $SET"
RUN="pixi run --manifest-path $REPO/pyproject.toml --frozen bash -c 'export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; exec \"\$@\"' _"
nodeline() { scontrol show node "$(hostname)" | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+|CPUTot=[0-9]+" | tr '\n' ' '; }

{
  echo "JOB $SLURM_JOB_ID host=$(hostname) model=$MODEL pin=$SET sizes=$SIZES repeats=$REPEATS $(date -Is)"
  echo "NODE before $(nodeline)"
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
  eval "$PINNED $RUN python $REPO/benchmarks/5i28_ff/probe.py before"
} > "$OUT/host.txt" 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_clocks.csv" 2>&1 &
GPULOG=$!

cd "$REPO"
eval "$PINNED $RUN python benchmarks/5i28_ff/run_5i28_ff.py --model $MODEL --out $OUT --repeats $REPEATS --sizes $SIZES"
RC=$?
# the last fallback of the 2026-10-29 run: an out-of-memory error of the 'reserved but unallocated' kind is retried once with expandable segments
if grep -q '"fragmented": true' "$OUT"/cells/*.json 2>/dev/null; then
  echo "RETRY fragmented out-of-memory cells with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True eval "$PINNED $RUN python benchmarks/5i28_ff/run_5i28_ff.py --model $MODEL --out $OUT --repeats $REPEATS --sizes $SIZES --retry-fragmented"
fi
kill "$GPULOG" 2>/dev/null; wait "$GPULOG" 2>/dev/null
{ echo "NODE after $(nodeline)"; eval "$PINNED $RUN python $REPO/benchmarks/5i28_ff/probe.py after"; } >> "$OUT/host.txt" 2>&1
echo "END model=$MODEL rc=$RC $(date -Is)"
[ "$RC" = 0 ] && date -Is > "$OUT/done"
exit "$RC"
