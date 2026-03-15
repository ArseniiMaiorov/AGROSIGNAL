#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
PROMOTION_MARKER="$ROOT_DIR/backend/debug/runs/promoted_release_candidate.json"
ALLOW_MODEL_PROMOTE="${ALLOW_MODEL_PROMOTE:-0}"

cd "$ROOT_DIR"

"$ROOT_DIR/scripts/train_orchestrated.sh"

echo "[promote] Validating release QA matrix manifest"
if [[ -f "$ROOT_DIR/backend/training/release_russia_qa_matrix.json" ]]; then
  "$PY_BIN" "$ROOT_DIR/backend/training/scripts/validate_release_qa_matrix.py" \
    "$ROOT_DIR/backend/training/release_russia_qa_matrix.json"
fi

if [[ -f "$ROOT_DIR/backend/training/release_russia_qa_results.jsonl" ]]; then
  echo "[promote] Summarizing QA by region band"
  "$PY_BIN" "$ROOT_DIR/backend/training/scripts/summarize_release_qa_by_band.py" \
    "$ROOT_DIR/backend/training/release_russia_qa_results.jsonl"
fi

if [[ "$ALLOW_MODEL_PROMOTE" != "1" ]]; then
  echo "[promote] Training finished. Automatic promotion is disabled."
  echo "[promote] Re-run with ALLOW_MODEL_PROMOTE=1 only after manual holdout review."
  exit 0
fi

echo "[promote] Writing promotion marker"
"$PY_BIN" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
marker = root / "backend" / "debug" / "runs" / "promoted_release_candidate.json"
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(
    json.dumps(
        {
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "mode": "operator-confirmed",
            "note": "Weights are already trained in-place; this marker records operator promotion.",
        },
        ensure_ascii=True,
        indent=2,
    ),
    encoding="utf-8",
)
print(marker)
PY

echo "[promote] Promotion marker written: $PROMOTION_MARKER"
