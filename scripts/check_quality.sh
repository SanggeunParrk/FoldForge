#!/usr/bin/env bash
# Run from any directory, using the existing project environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pyright
