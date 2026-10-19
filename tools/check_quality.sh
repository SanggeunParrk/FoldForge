#!/usr/bin/env bash
# Run from any directory, using the existing project environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
pixi run --frozen ruff check src tests scripts
pixi run --frozen ruff format --check src tests scripts
pixi run --frozen pyright
