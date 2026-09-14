#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -J foldforge-env
#SBATCH -o validation/logs/setup-env-%j.log

# Submit from this repository's root. Ampere: A5000/A6000/A100.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
unset PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-/home/psk6950/practice/miniworld-engine/.pixi/envs/default}"
export CC="${CC:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}"
export CXX="${CXX:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++}"
# Conda's split CUDA toolkit stores headers/libraries below targets/.
if [[ -d "$CUDA_HOME/targets/x86_64-linux/include" ]]; then
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export MAX_JOBS=8 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=80
UV_BIN=${UV_BIN:-$HOME/.local/bin/uv}
# Reuse the byte-verified engine wheel for this exact Python/Torch/CUDA ABI.
# uv's PyPI source identity otherwise forces a redundant FA2 source build.
FA2_WHEEL_DIR="$HOME/.cache/miniworld-wheels/cp312-torch2.10-cu128"
FA2_WHEEL="$FA2_WHEEL_DIR/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl"
if [[ -f "$FA2_WHEEL" ]]; then
    EXPECTED_SHA=591add7fec7ba3b95777413d1a879a6682920ba89ee82e8928035175a66a2c72
    ACTUAL_SHA=$(sha256sum "$FA2_WHEEL")
    [[ ${ACTUAL_SHA%% *} == "$EXPECTED_SHA" ]] || { echo 'FA2 wheel checksum mismatch' >&2; exit 1; }
    MINIWORLD_CXX_RUNTIME=${MINIWORLD_CXX_RUNTIME:-$HOME/.cache/miniworld-runtime/gcc14}
    if [[ ! -f "$MINIWORLD_CXX_RUNTIME/libstdc++.so.6" || ! -f "$MINIWORLD_CXX_RUNTIME/libgcc_s.so.1" ]]; then
        mkdir -p "$MINIWORLD_CXX_RUNTIME"
        cp -L "$CUDA_HOME/lib/libstdc++.so.6" "$CUDA_HOME/lib/libgcc_s.so.1" "$MINIWORLD_CXX_RUNTIME/"
    fi
    export LD_LIBRARY_PATH="$MINIWORLD_CXX_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$UV_BIN" sync --locked --extra cu12 --extra flash2 --no-install-package flash-attn
    "$UV_BIN" pip install --no-config --python .venv/bin/python --no-deps \
        --no-index --find-links "$FA2_WHEEL_DIR" "flash-attn==2.8.3.post1"
else
    # A fresh machine can build the same pinned release against runtime Torch.
    "$UV_BIN" sync --locked --extra cu12 --extra flash2
fi
"$UV_BIN" pip check --python .venv/bin/python
.venv/bin/python -c 'import torch, flash_attn, quack, cutlass; print(torch.__version__, flash_attn.__version__)'
