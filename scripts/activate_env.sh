#!/usr/bin/env bash
# Source from Bash before running Python directly; Slurm runners source this too.
MINIWORLD_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MINIWORLD_ENV_DIR=${FOLDFORGE_ENV:-$MINIWORLD_REPO_ROOT/.venv}
source "$MINIWORLD_ENV_DIR/bin/activate"
# The verified engine FA2 wheel was built with GCC 14's C++ runtime.
MINIWORLD_CXX_RUNTIME=${MINIWORLD_CXX_RUNTIME:-/home/psk6950/.cache/miniworld-runtime/gcc14}
if [[ -f "$MINIWORLD_CXX_RUNTIME/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="$MINIWORLD_CXX_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# JIT compilation uses a CUDA-12.8-compatible host compiler on this cluster.
if [[ -x /opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ ]]; then
    export CC=${CC:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}
    export CXX=${CXX:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++}
fi
unset PYTHONPATH
