#!/usr/bin/env python3
"""
setup_dropbox.py — одноразовая настройка Dropbox через rclone.

Использование:
    python scripts/setup_dropbox.py
"""
from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

RCLONE_BIN = Path.home() / ".local/bin/rclone"
CONFIG_PATH = Path.home() / ".config/rclone/rclone.conf"
REMOTE_NAME = "dropbox"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    bin_ = str(RCLONE_BIN) if RCLONE_BIN.exists() else "rclone"
    return subprocess.run([bin_, *cmd], **kwargs)


def ensure_remote() -> None:
    config = configparser.RawConfigParser()
    config.optionxform = str
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH, encoding="utf-8")
    if not config.has_section(REMOTE_NAME):
        config.add_section(REMOTE_NAME)
    config[REMOTE_NAME]["type"] = "dropbox"
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        config.write(handle)


def main() -> None:
    print("=" * 55)
    print("  Настройка Dropbox для проекта ТерраINFO")
    print("=" * 55)

    if not RCLONE_BIN.exists():
        print("\n❌  rclone не найден в ~/.local/bin/rclone")
        print("Установи его и запусти скрипт снова.")
        sys.exit(1)

    ensure_remote()
    print("\nСейчас откроется браузер Dropbox.")
    print("1. Войди в нужный Dropbox-аккаунт")
    print("2. Разреши rclone доступ")
    print("3. Вернись в приложение и нажми «Проверить подключение»")
    input("\nНажми Enter чтобы продолжить … ")

    result = run(["config", "reconnect", f"{REMOTE_NAME}:"])
    if result.returncode != 0:
        print("\n❌  Авторизация Dropbox не завершена")
        sys.exit(result.returncode)

    print("\n✓  Dropbox авторизован")


if __name__ == "__main__":
    main()
