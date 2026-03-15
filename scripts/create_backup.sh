#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$ROOT_DIR")"
BACKUP_DIR="$PARENT_DIR/autodetect-backup-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$BACKUP_DIR"
rsync -av \
  --exclude='training_data/' \
  --exclude='*.tif' \
  --exclude='*.npz' \
  --exclude='*.pkl' \
  --exclude='*.h5' \
  --exclude='*.pth' \
  --exclude='__pycache__/' \
  --exclude='.venv/' \
  --exclude='node_modules/' \
  --exclude='.git/' \
  "$ROOT_DIR/" "$BACKUP_DIR/"

cat > "$BACKUP_DIR/BACKUP_INFO.txt" <<INFO
Дата бэкапа: $(date --iso-8601=seconds)
Исходный путь: $ROOT_DIR
Содержимое: код проекта без тяжёлых артефактов обучения
Размер: $(du -sh "$BACKUP_DIR" | cut -f1)
INFO

echo "Бэкап создан: $BACKUP_DIR"
