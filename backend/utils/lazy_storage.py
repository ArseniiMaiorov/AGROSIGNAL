"""
lazy_storage.py — прозрачная работа с Google Drive.

Все большие данные хранятся на GDrive, локально только рабочая копия.
Скрипты обращаются через ensure() — данные скачаются автоматически если нужно.
После обучения результаты автоматически пушатся обратно через push().

GDrive папка: https://drive.google.com/drive/folders/1jtUAN8KRcJb73dxTME-Gi1d4TvGF3P2z
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RCLONE_BIN = os.getenv(
    "RCLONE_BIN",
    str(Path.home() / ".local/bin/rclone"),
)
RCLONE_REMOTE = os.getenv("GDRIVE_RCLONE_REMOTE", "gdrive")

# root_folder_id задан в rclone.conf → remote "gdrive:" = корень папки GDrive
# Поддиректории внутри этой папки:
_MANIFEST: dict[str, dict[str, str]] = {
    "real_tiles": {
        "local": "backend/debug/runs/real_tiles",
        "remote": "training/real_tiles",
        "type": "directory",
        "size_hint": "~2.7 GB",
    },
    "real_tiles_labels_weak": {
        "local": "backend/debug/runs/real_tiles_labels_weak",
        "remote": "training/real_tiles_labels_weak",
        "type": "directory",
        "size_hint": "~800 MB",
    },
    "sentinel_cache": {
        "local": "backend/debug/cache/sentinel_scenes",
        "remote": "cache/sentinel_scenes",
        "type": "directory",
        "size_hint": "~26 GB",
    },
    "models": {
        "local": "backend/models",
        "remote": "models",
        "type": "directory",
        "size_hint": "~3 GB",
    },
    "dataset_tar": {
        "local": "artifacts/agrosignal_dataset_20260316.tar",
        "remote": "artifacts/agrosignal_dataset_20260316.tar",
        "type": "file",
        "size_hint": "~2.7 GB",
    },
}


def _rclone(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    rclone = RCLONE_BIN if Path(RCLONE_BIN).exists() else shutil.which("rclone")
    if not rclone:
        raise FileNotFoundError(
            "rclone не найден. Установлен автоматически в ~/.local/bin/rclone — "
            "добавь его в PATH: export PATH=$HOME/.local/bin:$PATH"
        )
    return subprocess.run([rclone, *args], check=check)


def _remote(subpath: str) -> str:
    base = os.getenv("GDRIVE_BASE_PATH", ".").strip("/.")
    if base and base != ".":
        return f"{RCLONE_REMOTE}:{base}/{subpath}"
    return f"{RCLONE_REMOTE}:{subpath}"


def _local(key: str) -> Path:
    return PROJECT_ROOT / _MANIFEST[key]["local"]


def _exists_locally(key: str) -> bool:
    p = _local(key)
    if _MANIFEST[key]["type"] == "directory":
        return p.is_dir() and any(p.iterdir())
    return p.is_file()


# ──────────────────────────────────────────────────────────────────────────────

def ensure(*keys: str, fatal: bool = True) -> dict[str, bool]:
    """
    Убедиться что данные доступны локально.
    Если нет — скачать с GDrive автоматически.

    Использование:
        from utils.lazy_storage import ensure
        ensure("real_tiles")
        ensure("real_tiles", "models")
    """
    bad = [k for k in keys if k not in _MANIFEST]
    if bad:
        raise KeyError(f"Неизвестные ключи: {bad}. Доступные: {list(_MANIFEST)}")

    results: dict[str, bool] = {}
    for key in keys:
        if _exists_locally(key):
            results[key] = True
            continue

        entry = _MANIFEST[key]
        remote = _remote(entry["remote"])
        local = _local(key)
        print(f"[gdrive] ↓ {key}  {remote} → {local}  ({entry['size_hint']})")

        local.parent.mkdir(parents=True, exist_ok=True)

        if entry["type"] == "directory":
            local.mkdir(parents=True, exist_ok=True)
            cmd = ["copy", remote, str(local), "--progress", "--transfers", "4"]
        else:
            cmd = ["copy", remote, str(local.parent), "--progress"]

        try:
            _rclone(*cmd)
            results[key] = True
            print(f"[gdrive] ✓ {key} готов")
        except subprocess.CalledProcessError as exc:
            results[key] = False
            msg = (
                f"[gdrive] ✗ Не удалось скачать '{key}' (код {exc.returncode})\n"
                f"  Попробуй вручную: python scripts/gdrive_data.py pull {key}\n"
                f"  Или сначала: python scripts/setup_gdrive.py  (авторизация)"
            )
            if fatal:
                print(msg)
                sys.exit(1)
            else:
                print(msg)

    return results


def push(*keys: str, silent: bool = False) -> dict[str, bool]:
    """
    Загрузить локальные данные на GDrive.

    Использование:
        from utils.lazy_storage import push
        push("models")            # после обучения
        push("real_tiles_labels_weak", "models")
    """
    bad = [k for k in keys if k not in _MANIFEST]
    if bad:
        raise KeyError(f"Неизвестные ключи: {bad}")

    results: dict[str, bool] = {}
    for key in keys:
        if not _exists_locally(key):
            if not silent:
                print(f"[gdrive] {key}: нет локально — пропуск")
            results[key] = False
            continue

        entry = _MANIFEST[key]
        local = _local(key)
        remote = _remote(entry["remote"])

        if not silent:
            print(f"[gdrive] ↑ {key}  {local} → {remote}")

        if entry["type"] == "directory":
            cmd = ["copy", str(local), remote, "--progress", "--transfers", "4"]
        else:
            cmd = ["copy", str(local), remote.rsplit(":", 1)[0] + ":" +
                   "/".join(remote.split(":")[1].rsplit("/", 1)[:-1])]

        try:
            _rclone(*cmd)
            results[key] = True
            if not silent:
                print(f"[gdrive] ✓ {key} загружен")
        except subprocess.CalledProcessError as exc:
            results[key] = False
            if not silent:
                print(f"[gdrive] ✗ {key}: ошибка загрузки (код {exc.returncode})")

    return results
