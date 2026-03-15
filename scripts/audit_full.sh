#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR"

echo "[1/7] Backend regression + coverage"
PYTHONPATH="$ROOT_DIR/backend:${PYTHONPATH:-}" "$PY_BIN" -m pytest -q --cov=backend --cov-report=term backend/tests

echo "[2/7] Frontend unit tests + coverage"
npm --prefix frontend run test:coverage

echo "[3/7] Frontend production build"
npm --prefix frontend run build

if curl -fsS "http://localhost:8000/health" >/dev/null 2>&1; then
  echo "[4/7] Live release smoke"
  "$PY_BIN" scripts/release_smoke.py
  echo "[5/7] Live crop suitability audit"
  "$PY_BIN" scripts/crop_suitability_audit.py
else
  echo "[4/7] Live release smoke skipped: backend is not reachable on http://localhost:8000"
  echo "[5/7] Live crop suitability audit skipped: backend is not reachable on http://localhost:8000"
fi

if [[ -f "$ROOT_DIR/backend/training/release_russia_qa_matrix.json" ]]; then
  echo "[6/7] Release QA matrix validation"
  "$PY_BIN" "$ROOT_DIR/backend/training/scripts/validate_release_qa_matrix.py" \
    "$ROOT_DIR/backend/training/release_russia_qa_matrix.json"
else
  echo "[6/7] Release QA matrix validation skipped: manifest not found"
fi

if [[ -f "$ROOT_DIR/backend/training/release_russia_qa_results.jsonl" ]]; then
  echo "[7/8] QA band summary"
  "$PY_BIN" "$ROOT_DIR/backend/training/scripts/summarize_release_qa_by_band.py" \
    "$ROOT_DIR/backend/training/release_russia_qa_results.jsonl"
else
  echo "[7/8] QA band summary skipped: results not found"
fi

echo "[8/8] Playwright smoke"
npm --prefix frontend run test:e2e

echo "Audit complete."
