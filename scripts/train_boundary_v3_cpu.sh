#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/backend/training/config/boundary_unet_v3_cpu.yaml}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONPATH="$ROOT_DIR/backend:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR"

TRAIN_ARGS=(--config "$CONFIG_PATH")
if [[ "$DRY_RUN" == "1" ]]; then
  TRAIN_ARGS+=(--dry-run)
fi
"$PY_BIN" "$ROOT_DIR/backend/training/train_boundary_v3_cpu.py" "${TRAIN_ARGS[@]}"

echo "[parity] Running ONNX parity check for boundary_unet_v3_cpu"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $PY_BIN $ROOT_DIR/backend/training/check_torch_onnx_parity.py --tiles-dir $ROOT_DIR/backend/debug/runs/real_tiles --labels-dir $ROOT_DIR/backend/debug/runs/real_tiles_labels_weak --torch-model $ROOT_DIR/backend/models/boundary_unet_v3_cpu.pth --onnx-model $ROOT_DIR/backend/models/boundary_unet_v3_cpu.onnx --norm $ROOT_DIR/backend/models/boundary_unet_v3_cpu.norm.json"
else
  "$PY_BIN" "$ROOT_DIR/backend/training/check_torch_onnx_parity.py" \
    --tiles-dir "$ROOT_DIR/backend/debug/runs/real_tiles" \
    --labels-dir "$ROOT_DIR/backend/debug/runs/real_tiles_labels_weak" \
    --torch-model "$ROOT_DIR/backend/models/boundary_unet_v3_cpu.pth" \
    --onnx-model "$ROOT_DIR/backend/models/boundary_unet_v3_cpu.onnx" \
    --norm "$ROOT_DIR/backend/models/boundary_unet_v3_cpu.norm.json"
fi
