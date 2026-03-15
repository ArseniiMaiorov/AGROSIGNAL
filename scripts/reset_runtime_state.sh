#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[reset] stopping docker compose stack and removing compose volumes"
docker compose down -v --remove-orphans

remove_path_via_docker() {
  local target="$1"
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  echo "[reset] retrying removal via docker for $target"
  docker run --rm -v "$ROOT_DIR:/workspace" alpine:3.20 sh -c "rm -rf -- \"/workspace/$target\""
}

remove_path() {
  local target="$1"
  if [ -e "$target" ]; then
    echo "[reset] removing $target"
    if ! rm -rf "$target" 2>/dev/null; then
      if ! remove_path_via_docker "$target"; then
        echo "[reset] failed to remove $target" >&2
        return 1
      fi
    fi
  fi
}

remove_path "backend/cache"
remove_path "backend/debug"
remove_path "backend/celerybeat-schedule"
remove_path "frontend/dist"
remove_path "frontend/coverage"
remove_path "frontend/test-results"
remove_path ".coverage"
remove_path "backend/training/release_russia_qa_results_sample.jsonl"
remove_path "backend/training/release_russia_qa_results_live_sample.jsonl"
remove_path "backend/training/release_russia_qa_matrix_live_sample.json"

mkdir -p backend/cache backend/debug

echo "[reset] done"
echo "[reset] kept untouched: .env, model weights, source code"
echo "[reset] next:"
echo "  docker compose up -d --build"
