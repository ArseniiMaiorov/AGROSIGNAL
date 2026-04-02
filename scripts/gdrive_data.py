#!/usr/bin/env python3
"""
gdrive_data.py — управление большими данными проекта через Google Drive (rclone).

Команды:
  status   — показать что есть локально / на GDrive
  push     — загрузить локальные данные на GDrive
  pull     — скачать данные с GDrive локально
  clean    — удалить локальные копии (данные остаются на GDrive)
  mount    — примонтировать GDrive как папку (требует FUSE)

Использование:
  python scripts/gdrive_data.py status
  python scripts/gdrive_data.py pull real_tiles
  python scripts/gdrive_data.py push sentinel_cache
  python scripts/gdrive_data.py clean sentinel_cache
  python scripts/gdrive_data.py pull             # скачать всё

Первоначальная настройка rclone (один раз):
  sudo apt install rclone             # или: curl https://rclone.org/install.sh | sudo bash
  rclone config                       # выбрать Google Drive, пройти OAuth
  # Имя remote по умолчанию: "gdrive" (можно изменить через RCLONE_REMOTE в .env)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Имя rclone remote (настраивается один раз через rclone config)
RCLONE_BIN = os.getenv("RCLONE_BIN", str(Path.home() / ".local/bin/rclone"))
RCLONE_REMOTE = os.getenv("GDRIVE_RCLONE_REMOTE", "gdrive")
# root_folder_id задан в rclone.conf → base = корень папки GDrive
_BASE = os.getenv("GDRIVE_BASE_PATH", ".").strip("/.")
GDRIVE_BASE = _BASE if (_BASE and _BASE != ".") else ""

# ──────────────────────────────────────────────────────────────────────────────
# МАНИФЕСТ: что хранится на GDrive
# key → { local, remote_subpath, type, size_hint, description }
# ──────────────────────────────────────────────────────────────────────────────
DATA_MANIFEST: dict[str, dict[str, Any]] = {

    "real_tiles": {
        "local": "backend/debug/runs/real_tiles",
        "remote": "training/real_tiles",
        "type": "directory",
        "size_hint": "~2.7 GB",
        "description": "64 обработанных NPZ тайла Sentinel-2 (глобальный датасет)",
        "required_for": ["training", "labels", "finetune", "classifier"],
    },

    "real_tiles_labels_weak": {
        "local": "backend/debug/runs/real_tiles_labels_weak",
        "remote": "training/real_tiles_labels_weak",
        "type": "directory",
        "size_hint": "~800 MB",
        "description": "Слабые метки полей (TIF) + сводный CSV",
        "required_for": ["finetune", "classifier"],
    },

    "sentinel_cache": {
        "local": "backend/debug/cache/sentinel_scenes",
        "remote": "cache/sentinel_scenes",
        "type": "directory",
        "size_hint": "~26 GB",
        "description": "Сырые сцены Sentinel Hub (кэш API — не обязателен)",
        "required_for": [],
    },

    "models": {
        "local": "backend/models",
        "remote": "models",
        "type": "directory",
        "size_hint": "~3 GB",
        "description": "Модели: UNet .pth/.onnx, классификатор .pkl, SAM .pt",
        "required_for": ["inference"],
    },

    "dataset_tar": {
        "local": "artifacts/agrosignal_dataset_20260316.tar",
        "remote": "artifacts/agrosignal_dataset_20260316.tar",
        "type": "file",
        "size_hint": "~2.7 GB",
        "description": "Архив обучающего датасета (резервная копия)",
        "required_for": [],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────────────────────────────────────

def _check_rclone() -> bool:
    return Path(RCLONE_BIN).exists() or shutil.which("rclone") is not None


def _rclone(*args: str, **kwargs) -> subprocess.CompletedProcess:
    bin_ = RCLONE_BIN if Path(RCLONE_BIN).exists() else (shutil.which("rclone") or "rclone")
    return subprocess.run([bin_, *args], **kwargs)


def _remote_path(remote_subpath: str) -> str:
    if GDRIVE_BASE:
        return f"{RCLONE_REMOTE}:{GDRIVE_BASE}/{remote_subpath}"
    return f"{RCLONE_REMOTE}:{remote_subpath}"


def _local_path(entry: dict[str, Any]) -> Path:
    return PROJECT_ROOT / entry["local"]


def _local_exists(entry: dict[str, Any]) -> bool:
    p = _local_path(entry)
    if entry["type"] == "directory":
        return p.is_dir() and any(p.iterdir())
    return p.is_file()


def _remote_exists(entry: dict[str, Any]) -> bool:
    if not _check_rclone():
        return False
    remote = _remote_path(entry["remote"])
    result = _rclone("lsf", remote, "--max-depth", "1", capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def _du(path: Path) -> str:
    """Return human-readable size of a local path."""
    if not path.exists():
        return "—"
    result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.split()[0]
    return "?"


# ──────────────────────────────────────────────────────────────────────────────
# КОМАНДЫ
# ──────────────────────────────────────────────────────────────────────────────

def cmd_status(keys: list[str]) -> None:
    targets = {k: DATA_MANIFEST[k] for k in keys}
    has_rclone = _check_rclone()

    col = "{:<26} {:<10} {:<10} {:<12} {}"
    print(col.format("Name", "Local", "GDrive", "Size", "Description"))
    print("─" * 80)

    for key, entry in targets.items():
        local_ok = _local_exists(entry)
        local_size = _du(_local_path(entry)) if local_ok else "—"

        if has_rclone:
            remote_ok = _remote_exists(entry)
            gdrive_str = "✓" if remote_ok else "✗"
        else:
            gdrive_str = "no rclone"

        local_str = f"✓ {local_size}" if local_ok else "✗"
        print(col.format(key, local_str, gdrive_str, entry["size_hint"], entry["description"]))

    if not has_rclone:
        print("\n  rclone не установлен. Установка:")
        print("    curl https://rclone.org/install.sh | sudo bash")
        print("    rclone config   # настроить Google Drive один раз")


def cmd_push(keys: list[str], dry_run: bool = False) -> None:
    if not _check_rclone():
        print("Ошибка: rclone не установлен.")
        sys.exit(1)

    for key in keys:
        entry = DATA_MANIFEST[key]
        local = _local_path(entry)

        if not _local_exists(entry):
            print(f"  {key}: нет локально — пропуск")
            continue

        remote = _remote_path(entry["remote"])
        print(f"  push {key}  {local} → {remote}  ({entry['size_hint']})")

        args = ["copy", "--progress", "--transfers", "4"]
        if dry_run:
            args.insert(0, "--dry-run")
        if entry["type"] == "directory":
            args += [str(local), remote]
        else:
            args += [str(local), remote.rsplit("/", 1)[0]]

        result = _rclone(*args)
        if result.returncode != 0:
            print(f"  Ошибка при загрузке {key}")
        else:
            print(f"  {key}: загружено ✓")


def cmd_pull(keys: list[str], dry_run: bool = False) -> None:
    if not _check_rclone():
        print("Ошибка: rclone не установлен.")
        print("  curl https://rclone.org/install.sh | sudo bash")
        print("  rclone config")
        sys.exit(1)

    for key in keys:
        entry = DATA_MANIFEST[key]
        local = _local_path(entry)

        if _local_exists(entry):
            print(f"  {key}: уже есть локально ({_du(local)}) — пропуск")
            continue

        remote = _remote_path(entry["remote"])
        print(f"  pull {key}  {remote} → {local}  ({entry['size_hint']})")

        local.parent.mkdir(parents=True, exist_ok=True)

        args = ["copy", "--progress", "--transfers", "4"]
        if dry_run:
            args.insert(0, "--dry-run")
        if entry["type"] == "directory":
            local.mkdir(parents=True, exist_ok=True)
            args += [remote, str(local)]
        else:
            args += [remote, str(local.parent)]

        result = _rclone(*args)
        if result.returncode != 0:
            print(f"  Ошибка при скачивании {key}")
        else:
            print(f"  {key}: скачано ✓")


def cmd_clean(keys: list[str], force: bool = False) -> None:
    for key in keys:
        entry = DATA_MANIFEST[key]
        local = _local_path(entry)

        if not _local_exists(entry):
            print(f"  {key}: не найдено локально")
            continue

        size = _du(local)
        if not force:
            answer = input(f"  Удалить локально {key} ({size})? [y/N] ").strip().lower()
            if answer != "y":
                print(f"  {key}: пропуск")
                continue

        if entry["type"] == "directory":
            shutil.rmtree(local)
        else:
            local.unlink()
        print(f"  {key}: удалено ({size} освобождено)")


def cmd_mount(mountpoint: str) -> None:
    if not _check_rclone():
        print("Ошибка: rclone не установлен.")
        sys.exit(1)

    mnt = Path(mountpoint)
    mnt.mkdir(parents=True, exist_ok=True)
    remote = f"{RCLONE_REMOTE}:{GDRIVE_BASE}"

    print(f"Монтирование {remote} → {mnt}")
    print("Остановить: Ctrl+C")
    _rclone(
        "mount", remote, str(mnt),
        "--vfs-cache-mode", "writes",
        "--vfs-read-chunk-size", "32M",
        "--buffer-size", "64M",
        "--dir-cache-time", "10m",
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_keys(raw: list[str]) -> list[str]:
    if not raw or raw == ["all"]:
        return list(DATA_MANIFEST.keys())
    invalid = [k for k in raw if k not in DATA_MANIFEST]
    if invalid:
        print(f"Неизвестные ключи: {invalid}")
        print(f"Доступные: {list(DATA_MANIFEST.keys())}")
        sys.exit(1)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Управление данными проекта через Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Ключи данных:",
            *[f"  {k:<30} {v['size_hint']:<10} {v['description']}"
              for k, v in DATA_MANIFEST.items()],
        ]),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Показать состояние данных")
    p_status.add_argument("keys", nargs="*", default=["all"])

    p_push = sub.add_parser("push", help="Загрузить данные на GDrive")
    p_push.add_argument("keys", nargs="*", default=["all"])
    p_push.add_argument("--dry-run", action="store_true")

    p_pull = sub.add_parser("pull", help="Скачать данные с GDrive")
    p_pull.add_argument("keys", nargs="*", default=["all"])
    p_pull.add_argument("--dry-run", action="store_true")

    p_clean = sub.add_parser("clean", help="Удалить локальные копии")
    p_clean.add_argument("keys", nargs="*", default=["all"])
    p_clean.add_argument("--force", action="store_true", help="Без подтверждения")

    p_mount = sub.add_parser("mount", help="Примонтировать GDrive как папку")
    p_mount.add_argument("--mountpoint", default=str(PROJECT_ROOT / "mnt/gdrive"))

    args = parser.parse_args()

    if args.cmd == "status":
        cmd_status(_resolve_keys(args.keys))
    elif args.cmd == "push":
        cmd_push(_resolve_keys(args.keys), dry_run=args.dry_run)
    elif args.cmd == "pull":
        cmd_pull(_resolve_keys(args.keys), dry_run=args.dry_run)
    elif args.cmd == "clean":
        cmd_clean(_resolve_keys(args.keys), force=args.force)
    elif args.cmd == "mount":
        cmd_mount(args.mountpoint)


if __name__ == "__main__":
    main()
