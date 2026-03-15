#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
REGIONAL_HOLDOUT_JSON="${REGIONAL_HOLDOUT_JSON:-$BACKEND_DIR/training/holdout_boundary_gt.json}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Python executable not found: $PY_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/7] Validate regional holdout manifest"
"$PY_BIN" - "$REGIONAL_HOLDOUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

required = {
    "id",
    "ground_truth_geojson",
    "region_band",
    "region_boundary_profile_target",
    "error_mode_tag",
    "parcel_shape_class",
    "adjacency_tag",
}

manifest_path = Path(sys.argv[1]).resolve()
if not manifest_path.exists():
    raise SystemExit(f"ERROR: manifest not found: {manifest_path}")

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
items = payload.get("items", []) if isinstance(payload, dict) else payload
if not isinstance(items, list) or not items:
    raise SystemExit("ERROR: manifest must contain a non-empty list of items")

errors = []
for idx, item in enumerate(items):
    item_id = str(item.get("id") or f"index_{idx}")
    missing = sorted(k for k in required if k not in item or item.get(k) in (None, ""))
    if missing:
        errors.append(f"{item_id}: missing fields: {', '.join(missing)}")
        continue
    gt_path = Path(str(item["ground_truth_geojson"]))
    if "/absolute/path/" in str(gt_path):
        errors.append(f"{item_id}: placeholder ground_truth_geojson: {gt_path}")
        continue
    if not gt_path.exists():
        errors.append(f"{item_id}: GT file not found: {gt_path}")

if errors:
    raise SystemExit("ERROR: regional manifest validation failed:\n- " + "\n- ".join(errors))

print(f"OK: regional manifest validated ({manifest_path}) with {len(items)} items")
PY

echo "[2/7] Check training environment"
"$PY_BIN" "$BACKEND_DIR/training/check_training_env.py" --skip-pickle

echo "[3/7] Fetch real Sentinel-2 tiles"
"$PY_BIN" "$BACKEND_DIR/training/fetch_real_tiles.py"

echo "[4/7] Generate weak labels"
"$PY_BIN" "$BACKEND_DIR/training/generate_weak_labels_real_tiles.py" --full-rebuild

echo "[5/7] Train BoundaryUNet with region-aware metadata"
"$PY_BIN" "$BACKEND_DIR/training/gen_data.py" --ml-feature-profile v2_16ch

echo "[6/7] Torch vs ONNX parity"
"$PY_BIN" "$BACKEND_DIR/training/check_torch_onnx_parity.py" \
  --tiles-dir "$BACKEND_DIR/debug/runs/real_tiles" \
  --labels-dir "$BACKEND_DIR/debug/runs/real_tiles_labels_weak" \
  --torch-model "$BACKEND_DIR/models/boundary_unet_v2.pth" \
  --onnx-model "$BACKEND_DIR/models/boundary_unet_v2.onnx" \
  --norm "$BACKEND_DIR/models/boundary_unet_v2.norm.json"

echo "[7/7] Holdout A/B for regional profiles"
"$PY_BIN" "$BACKEND_DIR/training/run_holdout_ab.py" --holdout "$REGIONAL_HOLDOUT_JSON"

echo "Done."
