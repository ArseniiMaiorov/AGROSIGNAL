#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"

cd "$ROOT_DIR"
"$PY_BIN" "$ROOT_DIR/backend/training/scripts/summarize_release_qa_by_band.py" "$@"
