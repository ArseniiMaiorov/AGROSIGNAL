#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $PY_BIN -m pip install -r $ROOT_DIR/backend/training/requirements.cpu.txt"
else
  "$PY_BIN" -m pip install -r "$ROOT_DIR/backend/training/requirements.cpu.txt"
  echo "Training dependencies installed."
fi
