#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
MATRIX_PATH="${MATRIX_PATH:-$ROOT_DIR/backend/training/release_russia_qa_matrix.json}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR"

if [[ ! -f "$MATRIX_PATH" ]]; then
  echo "Matrix file not found: $MATRIX_PATH" >&2
  exit 1
fi

"$PY_BIN" "$ROOT_DIR/backend/training/scripts/validate_release_qa_matrix.py" "$MATRIX_PATH"
"$PY_BIN" "$ROOT_DIR/backend/training/scripts/run_release_qa_matrix.py" "$MATRIX_PATH" "$@"
if [[ -f "$ROOT_DIR/backend/training/release_russia_qa_results.jsonl" ]]; then
  "$PY_BIN" "$ROOT_DIR/backend/training/scripts/summarize_release_qa_by_band.py" \
    "$ROOT_DIR/backend/training/release_russia_qa_results.jsonl"
fi
