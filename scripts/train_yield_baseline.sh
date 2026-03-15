#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/backend/training/config/yield_xgboost_v2.yaml}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONPATH="$ROOT_DIR/backend:${PYTHONPATH:-}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR"
ARGS=(--config "$CONFIG_PATH" "$@")
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
"$PY_BIN" "$ROOT_DIR/backend/training/train_yield_baseline_v2.py" "${ARGS[@]}"
