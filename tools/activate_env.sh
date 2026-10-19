#!/usr/bin/env bash
# Source from Bash before running Python directly; Slurm runners source this too.
# Weights, databases, release envs and runs live outside the checkout.
: "${FOLDFORGE_HOME:=$HOME/.cache/foldforge}"; export FOLDFORGE_HOME
FOLDFORGE_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# The pixi environment (`pixi install` at the repository root): ${FOLDFORGE_ENV:-default}.
FOLDFORGE_ENV_DIR=${FOLDFORGE_ENV_DIR:-$FOLDFORGE_REPO_ROOT/.pixi/envs/${FOLDFORGE_ENV:-default}}
source "$FOLDFORGE_ENV_DIR/bin/activate" 2>/dev/null || export PATH="$FOLDFORGE_ENV_DIR/bin:$PATH"
export CONDA_PREFIX=$FOLDFORGE_ENV_DIR
# conda-forge's libstdc++ must come first for the engine's JIT-built extensions.
export LD_LIBRARY_PATH="$FOLDFORGE_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# JIT compilation uses a CUDA-compatible host compiler on this cluster.
if [[ -x /opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ ]]; then
    export CC=${CC:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/gcc}
    export CXX=${CXX:-/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++}
fi
unset PYTHONPATH
