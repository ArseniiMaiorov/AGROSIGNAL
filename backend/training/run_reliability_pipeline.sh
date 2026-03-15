#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
HOLDOUT_JSON="${HOLDOUT_JSON:-$BACKEND_DIR/training/holdout_aoi_template.json}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Python executable not found: $PY_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/8] Validate holdout config"
"$PY_BIN" - "$HOLDOUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

holdout_path = Path(sys.argv[1]).resolve()
if not holdout_path.exists():
    raise SystemExit(f"ERROR: holdout file not found: {holdout_path}")
if holdout_path.name == "holdout_aoi_template.json":
    raise SystemExit(
        "ERROR: HOLDOUT_JSON points to template file. Provide a real holdout config."
    )

payload = json.loads(holdout_path.read_text(encoding="utf-8"))
items = payload.get("items", []) if isinstance(payload, dict) else payload
if not isinstance(items, list) or not items:
    raise SystemExit("ERROR: holdout must contain a non-empty list of AOIs")

errors = []
for idx, item in enumerate(items):
    item_id = str(item.get("id") or f"index_{idx}")
    req = item.get("request")
    gt = item.get("ground_truth_geojson")
    if not isinstance(req, dict):
        errors.append(f"{item_id}: missing/invalid request")
        continue
    if not isinstance(gt, str) or not gt.strip():
        errors.append(f"{item_id}: missing ground_truth_geojson")
        continue
    if "/absolute/path/" in gt or "your_holdout" in gt:
        errors.append(f"{item_id}: placeholder GT path: {gt}")
        continue
    gt_path = Path(gt)
    if not gt_path.exists():
        errors.append(f"{item_id}: GT file not found: {gt_path}")

if errors:
    raise SystemExit(
        "ERROR: holdout validation failed:\n- " + "\n- ".join(errors)
    )
print(f"OK: holdout validated ({holdout_path}) with {len(items)} AOI entries")
PY

echo "[2/8] Check training environment"
"$PY_BIN" "$BACKEND_DIR/training/check_training_env.py" --skip-pickle

echo "[3/8] Fetch real Sentinel-2 tiles"
"$PY_BIN" "$BACKEND_DIR/training/fetch_real_tiles.py"

echo "[4/8] Generate weak labels"
"$PY_BIN" "$BACKEND_DIR/training/generate_weak_labels_real_tiles.py" --full-rebuild

echo "[5/8] Train BoundaryUNet and export ONNX"
"$PY_BIN" "$BACKEND_DIR/training/gen_data.py" --ml-feature-profile v2_16ch

echo "[6/8] Train object classifier"
"$PY_BIN" "$BACKEND_DIR/training/train_object_classifier.py"

echo "[7/8] Torch vs ONNX parity check"
"$PY_BIN" "$BACKEND_DIR/training/check_torch_onnx_parity.py" \
  --tiles-dir "$BACKEND_DIR/debug/runs/real_tiles" \
  --labels-dir "$BACKEND_DIR/debug/runs/real_tiles_labels_weak" \
  --torch-model "$BACKEND_DIR/models/boundary_unet_v2.pth" \
  --onnx-model "$BACKEND_DIR/models/boundary_unet_v2.onnx" \
  --norm "$BACKEND_DIR/models/boundary_unet_v2.norm.json"

echo "[8/8] Holdout A/B evaluation (rule-based vs ML-primary)"
"$PY_BIN" "$BACKEND_DIR/training/run_holdout_ab.py" --holdout "$HOLDOUT_JSON"

echo "Done."
