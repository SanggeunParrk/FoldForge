#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -J foldforge-env
#SBATCH -o slurm-setup-env-%j.log

# Submit from this repository's root. The install needs no GPU: on the H100 cluster a CPU allocation does
# (srun -p h100 -A cssb --qos=normal_h100 --gres=none -c 8 --mem=32G pixi install).
# Ampere/Ada (A100, A5000, A6000) also want FlashAttention-2: FOLDFORGE_FLASH2=1 builds it against the
# environment's torch (about 40 minutes).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
unset PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CC="${CC:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"
export CXX="${CXX:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++}"
export MAX_JOBS=8 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=80

pixi install --locked
if [[ "${FOLDFORGE_FLASH2:-0}" == 1 ]]; then
    pixi run --locked pip install --no-build-isolation "flash-attn==2.8.3.post1"
fi
pixi run --locked python -c 'import torch, miniworld_engine, foldforge; print(torch.__version__, torch.version.cuda)'
