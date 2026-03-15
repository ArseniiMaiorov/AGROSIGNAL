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

export SH_FAILOVER_ENABLED="${SH_FAILOVER_ENABLED:-true}"

DEFAULT_SCENE_CACHE_DIR="$ROOT_DIR/backend/cache/sentinel_scenes"
FALLBACK_SCENE_CACHE_DIR="$ROOT_DIR/backend/debug/cache/sentinel_scenes"
if [[ -z "${SCENE_CACHE_DIR:-}" ]]; then
  if [[ -d "$DEFAULT_SCENE_CACHE_DIR" && ! -w "$DEFAULT_SCENE_CACHE_DIR" ]]; then
    mkdir -p "$FALLBACK_SCENE_CACHE_DIR"
    export SCENE_CACHE_DIR="$FALLBACK_SCENE_CACHE_DIR"
    echo "[cache] default scene cache is not writable; using fallback: $SCENE_CACHE_DIR"
  else
    mkdir -p "$DEFAULT_SCENE_CACHE_DIR"
    export SCENE_CACHE_DIR="$DEFAULT_SCENE_CACHE_DIR"
  fi
fi

cd "$ROOT_DIR"

echo "[1/3] Building open/public corpus manifest"
"$PY_BIN" "$ROOT_DIR/backend/training/prepare_open_boundary_corpus.py"

echo "[2/3] Fetching Sentinel training tiles with failover"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $PY_BIN $ROOT_DIR/backend/training/fetch_real_tiles.py $*"
else
  "$PY_BIN" "$ROOT_DIR/backend/training/fetch_real_tiles.py" "$@"
fi

echo "[3/3] Generating weak labels"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] $PY_BIN $ROOT_DIR/backend/training/generate_weak_labels_real_tiles.py --full-rebuild"
else
  "$PY_BIN" "$ROOT_DIR/backend/training/generate_weak_labels_real_tiles.py" --full-rebuild
fi
