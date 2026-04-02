#!/usr/bin/env python3
"""
setup_gdrive.py — одноразовая настройка Google Drive через rclone.

Запусти один раз, откроется браузер, войди в Google — готово.
После этого все скрипты работают с GDrive автоматически.

Использование:
    python scripts/setup_gdrive.py
"""
import os
import subprocess
import sys
from pathlib import Path

RCLONE_BIN = Path.home() / ".local/bin/rclone"
CONFIG_PATH = Path.home() / ".config/rclone/rclone.conf"
FOLDER_ID = "1jtUAN8KRcJb73dxTME-Gi1d4TvGF3P2z"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    bin_ = str(RCLONE_BIN) if RCLONE_BIN.exists() else "rclone"
    return subprocess.run([bin_, *cmd], **kwargs)


def main() -> None:
    print("=" * 55)
    print("  Настройка Google Drive для проекта AGROSIGNAL")
    print("=" * 55)

    # Проверить rclone
    if not RCLONE_BIN.exists():
        print("\n❌  rclone не найден в ~/.local/bin/rclone")
        print("Установи его:")
        print("  curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip")
        print("  cd /tmp && unzip -q rclone.zip && cp rclone-v*/rclone ~/.local/bin/rclone")
        print("  chmod +x ~/.local/bin/rclone")
        sys.exit(1)
    else:
        result = run(["version"], capture_output=True, text=True)
        version = result.stdout.split("\n")[0] if result.returncode == 0 else "?"
        print(f"\n✓  rclone найден: {version}")

    # Проверить конфиг
    if CONFIG_PATH.exists():
        content = CONFIG_PATH.read_text()
        if "token" in content:
            print("✓  rclone конфиг с токеном уже есть")
            # Проверить доступ к GDrive
            print("\nПроверяю доступ к Google Drive …")
            result = run(["lsf", "gdrive:", "--max-depth", "1"], capture_output=True, text=True)
            if result.returncode == 0:
                files = result.stdout.strip().split("\n") if result.stdout.strip() else []
                print(f"✓  Папка GDrive доступна ({len(files)} объектов)")
                if files:
                    print("  Содержимое:", ", ".join(files[:5]))
            else:
                print("⚠  Не удалось подключиться. Переавторизуюсь …")
                _do_auth()
            return
        else:
            print("✓  Конфиг найден, но без токена — нужна авторизация")
    else:
        print("  Конфиг не найден — создаю …")
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            f"[gdrive]\n"
            f"type = drive\n"
            f"scope = drive\n"
            f"root_folder_id = {FOLDER_ID}\n"
        )
        print("✓  Конфиг создан")

    _do_auth()


def _do_auth() -> None:
    print()
    print("─" * 55)
    print("  Авторизация Google Drive")
    print("─" * 55)
    print("  Сейчас откроется браузер.")
    print("  1. Войди в свой Google аккаунт")
    print("  2. Разреши rclone доступ к Google Drive")
    print("  3. Вернись в терминал")
    print()
    input("  Нажми Enter чтобы открыть браузер …")

    result = run(["config", "reconnect", "gdrive:"])

    if result.returncode == 0:
        print()
        print("✓  Авторизация успешна!")
        print()
        print("─" * 55)
        print("  Проверяю подключение …")
        check = run(["lsf", "gdrive:", "--max-depth", "1"], capture_output=True, text=True)
        if check.returncode == 0:
            files = check.stdout.strip().split("\n") if check.stdout.strip() else []
            print(f"✓  GDrive папка доступна ({len(files)} объектов)")
        else:
            print("⚠  Список файлов не получен (папка может быть пустой)")
        print()
        print("=" * 55)
        print("  Готово! Теперь запускай:")
        print()
        print("  # Загрузить текущие данные на GDrive:")
        print("  python scripts/gdrive_data.py push")
        print()
        print("  # Статус:")
        print("  python scripts/gdrive_data.py status")
        print()
        print("  # Обучение (данные скачаются автоматически):")
        print("  .venv/bin/python3 backend/training/train_regional_supplement.py --epochs 50")
        print("=" * 55)
    else:
        print()
        print("❌  Авторизация не завершена (код %d)" % result.returncode)
        sys.exit(1)


if __name__ == "__main__":
    main()
