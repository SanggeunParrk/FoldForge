#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH -J ff-benchmark
#SBATCH --mem=96g
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -o validation/reports/logs/%x_%j.log
#SBATCH -e validation/reports/logs/%x_%j.err

# One model, all five modes, on one A100 (cssb3). Pass benchmark arguments, e.g.
#   sbatch scripts/benchmark_end_to_end.sbatch --model af3 --root benchmark-x
set -euo pipefail
cd /home/psk6950/practice/FoldForge

source scripts/activate_env.sh
ENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8

"$ENV/bin/python" scripts/benchmark_end_to_end.py --targets 4yx2 \
  --benchmark-repeats 5 --steps 200 --recycles 10 --samples 5 "$@"
