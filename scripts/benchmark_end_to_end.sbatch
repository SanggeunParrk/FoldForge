#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH -J ff-benchmark
#SBATCH --mem=96g
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -o validation/reports/logs/%x_%j.log
#SBATCH -e validation/reports/logs/%x_%j.err

# One model on one GPU. Pass every benchmark argument, including --targets, e.g.
#   sbatch scripts/benchmark_end_to_end.sbatch --model af3 --root benchmark-x \
#     --targets 5i28x4
#   sbatch -p gpu --gres=gpu:A6000:1 scripts/benchmark_end_to_end.sbatch ... (local)
set -euo pipefail
cd /home/psk6950/practice/FoldForge

source scripts/activate_env.sh
ENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8

"$ENV/bin/python" scripts/benchmark_end_to_end.py \
  --benchmark-repeats 5 --steps 200 --recycles 10 --samples 5 "$@"
