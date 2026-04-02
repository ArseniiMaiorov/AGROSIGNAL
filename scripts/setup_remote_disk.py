#!/usr/bin/env python3
"""
setup_remote_disk.py — одноразовая настройка удалённого WebDAV-диска через rclone.

Использование:
    python scripts/setup_remote_disk.py "https://cloud.example.com/remote.php/dav/files/user/projects"
"""
from __future__ import annotations

import argparse
import configparser
import getpass
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

RCLONE_BIN = Path.home() / ".local/bin/rclone"
CONFIG_PATH = Path.home() / ".config/rclone/rclone.conf"
DEFAULT_REMOTE_NAME = "webdav"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    bin_ = str(RCLONE_BIN) if RCLONE_BIN.exists() else "rclone"
    return subprocess.run([bin_, *cmd], **kwargs)


def detect_vendor(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if "nextcloud" in host or "remote.php" in path or "public.php/webdav" in path:
        return "nextcloud"
    if "owncloud" in host:
        return "owncloud"
    if "sharepoint" in host:
        return "sharepoint"
    return "other"


def obscure_password(password: str) -> str:
    result = run(["obscure", password], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "rclone obscure failed")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Настроить WebDAV remote для проекта ТерраINFO")
    parser.add_argument("url", nargs="?", help="Полный WebDAV URL или URL каталога на удалённом диске")
    parser.add_argument("--remote-name", default=DEFAULT_REMOTE_NAME)
    parser.add_argument("--vendor", default="")
    args = parser.parse_args()

    print("=" * 55)
    print("  Настройка удалённого диска (WebDAV) для ТерраINFO")
    print("=" * 55)

    if not RCLONE_BIN.exists():
        print("\n❌  rclone не найден в ~/.local/bin/rclone")
        print("Установи его и запусти скрипт снова.")
        sys.exit(1)

    raw_url = (args.url or input("\nWebDAV URL: ").strip()).strip()
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        print("\n❌  Нужен корректный URL вида https://host/path")
        sys.exit(1)

    vendor = (args.vendor or detect_vendor(raw_url)).strip() or "other"
    username = input("Логин WebDAV: ").strip()
    password = getpass.getpass("Пароль WebDAV: ")
    if not username or not password:
        print("\n❌  Логин и пароль обязательны")
        sys.exit(1)

    try:
        obscured_password = obscure_password(password)
    except Exception as exc:
        print(f"\n❌  Не удалось подготовить пароль для rclone: {exc}")
        sys.exit(1)

    config = configparser.RawConfigParser()
    config.optionxform = str
    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH, encoding="utf-8")
    if config.has_section(args.remote_name):
        config.remove_section(args.remote_name)
    config.add_section(args.remote_name)
    config[args.remote_name]["type"] = "webdav"
    config[args.remote_name]["url"] = f"{parsed.scheme}://{parsed.netloc}"
    config[args.remote_name]["vendor"] = vendor
    config[args.remote_name]["user"] = username
    config[args.remote_name]["pass"] = obscured_password
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        config.write(handle)

    print("\n✓  Remote сохранён в rclone.conf")
    print("  Вернись в приложение и нажми «Проверить подключение».")


if __name__ == "__main__":
    main()
