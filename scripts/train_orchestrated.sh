#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
BACKEND_DIR="$ROOT_DIR/backend"
export PYTHONPATH="$BACKEND_DIR:${PYTHONPATH:-}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

export SH_FAILOVER_ENABLED="${SH_FAILOVER_ENABLED:-true}"
export BATCHED="${BATCHED:-1}"
export LOW_MEM="${LOW_MEM:-1}"
export EXPERIMENTAL_LSTM="${EXPERIMENTAL_LSTM:-0}"
export BACKBONE="${BACKBONE:-efficientnet_b0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DEFAULT_SCENE_CACHE_DIR="$ROOT_DIR/backend/cache/sentinel_scenes"
FALLBACK_SCENE_CACHE_DIR="$ROOT_DIR/backend/debug/cache/sentinel_scenes"
if [[ -z "${SCENE_CACHE_DIR:-}" ]]; then
  if [[ -d "$DEFAULT_SCENE_CACHE_DIR" && ! -w "$DEFAULT_SCENE_CACHE_DIR" ]]; then
    mkdir -p "$FALLBACK_SCENE_CACHE_DIR"
    export SCENE_CACHE_DIR="$FALLBACK_SCENE_CACHE_DIR"
  else
    mkdir -p "$DEFAULT_SCENE_CACHE_DIR"
    export SCENE_CACHE_DIR="$DEFAULT_SCENE_CACHE_DIR"
  fi
fi

cd "$ROOT_DIR"

STAGES_RAW="${STAGES:-download,prepare,train-boundary,benchmark,export,register}"
export STAGES="$STAGES_RAW"
IFS=',' read -r -a STAGES <<< "$STAGES_RAW"

function has_stage() {
  local needle="$1"
  for stage in "${STAGES[@]}"; do
    if [[ "$stage" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

function corpus_ready() {
  local tiles_dir="$ROOT_DIR/backend/debug/runs/real_tiles"
  local labels_dir="$ROOT_DIR/backend/debug/runs/real_tiles_labels_weak"
  [[ -d "$tiles_dir" && -d "$labels_dir" ]] || return 1
  shopt -s nullglob
  local tile_paths=("$tiles_dir"/*.npz)
  shopt -u nullglob
  [[ ${#tile_paths[@]} -gt 0 ]] || return 1
  local tile_path tile_name
  for tile_path in "${tile_paths[@]}"; do
    tile_name="$(basename "$tile_path" .npz)"
    if [[ -f "$labels_dir/${tile_name}_label.tif" ]]; then
      return 0
    fi
  done
  return 1
}

echo "[0/6] Runtime and training preflight"
"$PY_BIN" scripts/preflight_check.py
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] Skipping training dependency validation"
else
  "$PY_BIN" "$BACKEND_DIR/training/check_training_env.py" --skip-pickle
fi

PIPELINE_MODE="${PIPELINE_MODE:-regional}"
if [[ "${HOLDOUT_JSON:-}" != "" && -f "${HOLDOUT_JSON}" && "$(basename "${HOLDOUT_JSON}")" != "holdout_aoi_template.json" ]]; then
  PIPELINE_MODE="reliability"
fi

echo "[1/6] Controlled Sentinel Hub failover is enabled"
echo "        account order: primary -> reserv -> second_reserv"
echo "        scene cache: $SCENE_CACHE_DIR"

if has_stage download || has_stage prepare; then
  echo "[2/6] Open/public corpus download + weak label prep"
  "$ROOT_DIR/scripts/train_open_data_download.sh"
elif has_stage train-boundary && ! corpus_ready; then
  echo "[2/6] Boundary corpus is missing; auto-running download + prepare"
  "$ROOT_DIR/scripts/train_open_data_download.sh"
else
  echo "[2/6] Download/prepare stages skipped"
fi

if has_stage train-boundary; then
  if [[ "$PIPELINE_MODE" == "reliability" ]]; then
    echo "[3/6] Running reliability pipeline"
    "$BACKEND_DIR/training/run_reliability_pipeline.sh"
  else
    echo "[3/6] Training boundary_unet_v3_cpu candidate"
    "$ROOT_DIR/scripts/train_boundary_v3_cpu.sh"
  fi
else
  echo "[3/6] Boundary training skipped"
fi

if has_stage train-yield-corpus; then
  echo "[4/6] Preparing yield corpus"
  "$ROOT_DIR/scripts/train_yield_corpus.sh"
fi
if has_stage train-yield-baseline; then
  echo "[4/6] Training yield baseline"
  "$ROOT_DIR/scripts/train_yield_baseline.sh"
fi
if has_stage train-yield-ensemble; then
  echo "[4/6] Training yield ensemble"
  "$ROOT_DIR/scripts/train_yield_ensemble.sh"
fi
if ! has_stage train-yield-corpus && ! has_stage train-yield-baseline && ! has_stage train-yield-ensemble; then
  echo "[4/6] Yield stages skipped"
fi

if has_stage benchmark; then
  echo "[5/6] Summarizing QA by region band"
  "$ROOT_DIR/scripts/benchmark_north_south.sh"
else
  echo "[5/6] Benchmark stage skipped"
fi

echo "[6/6] Writing candidate manifest"
"$PY_BIN" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import os

root = Path.cwd()
backend = root / "backend"
out_dir = backend / "debug" / "runs"
out_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_path": str((backend / "models" / "boundary_unet_v3_cpu.pth").resolve()),
    "onnx_path": str((backend / "models" / "boundary_unet_v3_cpu.onnx").resolve()),
    "norm_path": str((backend / "models" / "boundary_unet_v3_cpu.norm.json").resolve()),
    "object_classifier_path": str((backend / "models" / "object_classifier.pkl").resolve()),
    "qa_matrix_path": str((backend / "training" / "release_russia_qa_matrix.json").resolve()),
    "qa_band_summary_path": str((backend / "debug" / "runs" / "release_qa_band_summary.json").resolve()),
    "failover_mode": "primary->reserv->second_reserv",
    "stages": [token.strip() for token in os.environ.get("STAGES", "").split(",") if token.strip()],
    "batched": os.environ.get("BATCHED", "1") == "1",
    "low_mem": os.environ.get("LOW_MEM", "1") == "1",
    "backbone": os.environ.get("BACKBONE", "efficientnet_b0"),
}
(out_dir / "train_candidate_manifest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
print(out_dir / "train_candidate_manifest.json")
PY

echo "Candidate ready. Review holdout/QA metrics before promotion."
