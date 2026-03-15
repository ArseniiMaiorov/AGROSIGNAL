#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONPATH="$ROOT_DIR/backend:${PYTHONPATH:-}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR"
ARGS=("$@")
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
"$PY_BIN" "$ROOT_DIR/backend/training/prepare_yield_training_corpus.py" "${ARGS[@]}"
