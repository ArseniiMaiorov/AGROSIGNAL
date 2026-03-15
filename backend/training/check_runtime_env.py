#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _read_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _bool_literal_check(env_file: Path) -> list[str]:
    try:
        from core.config import (
            BOOL_ALLOWED_LITERALS,
            get_bool_env_alias_map,
            parse_env_bool,
        )
    except ModuleNotFoundError as exc:
        return [
            (
                "Runtime preflight requires project dependencies "
                f"(missing module: {exc.name}). Run with project venv."
            )
        ]

    values = _read_env_file(env_file)
    errors: list[str] = []
    for _field_name, aliases in get_bool_env_alias_map().items():
        for alias in aliases:
            if alias not in values:
                continue
            raw = values.get(alias)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                parse_env_bool(raw, field_name=alias)
            except ValueError as exc:
                errors.append(str(exc))

    if errors:
        allowed = ", ".join(BOOL_ALLOWED_LITERALS)
        errors.append(f"Allowed boolean literals: {allowed}")
    return errors


def _settings_check(env_file: Path) -> list[str]:
    try:
        from core.config import get_settings
    except ModuleNotFoundError as exc:
        return [
            (
                "Runtime preflight requires project dependencies "
                f"(missing module: {exc.name}). Run with project venv."
            )
        ]

    values = _read_env_file(env_file)
    # Ensure explicit env-file values are visible even if shell env differs.
    # Restore original environment after check to avoid side effects.
    previous_env: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        get_settings.cache_clear()
        _ = get_settings()
    except Exception as exc:
        return [f"Settings validation failed: {exc}"]
    finally:
        for key, old_value in previous_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate runtime .env before starting backend/celery workers"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to env file (default: .env in project root)",
    )
    args = parser.parse_args()

    env_file = args.env_file.resolve()
    if not env_file.exists():
        print(f"ERROR: env file not found: {env_file}")
        return 1

    errors: list[str] = []
    errors.extend(_bool_literal_check(env_file))
    errors.extend(_settings_check(env_file))
    errors = list(dict.fromkeys(errors))

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print(f"OK: runtime env validation passed ({env_file})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
