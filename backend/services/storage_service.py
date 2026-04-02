"""Настройки локального/облачного хранения для UI."""
from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import UUID

from core.logging import get_logger

logger = get_logger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent if (BACKEND_ROOT.parent / "frontend").exists() else BACKEND_ROOT
PROFILE_PATH = BACKEND_ROOT / "debug" / "storage_profiles.json"
RCLONE_CONFIG_PATH = Path(os.getenv("RCLONE_CONFIG", str(Path.home() / ".config" / "rclone" / "rclone.conf")))
LOCAL_PROJECT_WORKSPACE_DIRS: tuple[str, ...] = (
    "backend/models",
    "backend/debug/runs/real_tiles",
    "backend/debug/runs/real_tiles_labels_weak",
    "backend/debug/cache/sentinel_scenes",
    "artifacts",
)

LOCAL_BACKEND_WORKSPACE_DIRS: tuple[str, ...] = (
    "models",
    "debug/runs/real_tiles",
    "debug/runs/real_tiles_labels_weak",
    "debug/cache/sentinel_scenes",
    "artifacts",
)

CLOUD_WORKSPACE_DIRS: tuple[str, ...] = (
    "models",
    "training",
    "training/real_tiles",
    "training/real_tiles_labels_weak",
    "cache",
    "cache/sentinel_scenes",
    "artifacts",
)

PROVIDER_LABELS = {
    "google_drive": "Google Drive",
    "dropbox": "Dropbox",
    "yandex_disk": "Яндекс Диск",
    "webdav": "Удалённый диск",
    "unknown": "Неизвестное хранилище",
}

PROVIDER_LOGIN_URLS = {
    "google_drive": "https://accounts.google.com/",
    "dropbox": "https://www.dropbox.com/login",
    "yandex_disk": "https://passport.yandex.ru/auth",
    "webdav": None,
}

PROVIDER_RCLONE_TYPES = {
    "google_drive": "drive",
    "dropbox": "dropbox",
    "yandex_disk": "yandex",
    "webdav": "webdav",
}

PROVIDER_RCLONE_HINT_ENVS = {
    "google_drive": "GDRIVE_RCLONE_REMOTE",
    "dropbox": "DROPBOX_RCLONE_REMOTE",
    "yandex_disk": "YANDEX_RCLONE_REMOTE",
    "webdav": "WEBDAV_RCLONE_REMOTE",
}

PROVIDER_RCLONE_HINT_DEFAULTS = {
    "google_drive": "gdrive",
    "dropbox": "dropbox",
    "yandex_disk": "yandex",
    "webdav": "webdav",
}

PROVIDER_SETUP_COMMANDS = {
    "google_drive": "python scripts/setup_gdrive.py",
    "dropbox": "python scripts/setup_dropbox.py",
    "yandex_disk": "python scripts/setup_yandex_disk.py",
}

RCLONE_ERROR_PREFIX_RE = re.compile(r"^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+(?:ERROR|NOTICE)\s*:\s*", re.IGNORECASE)


class StorageService:
    """Профиль хранения данных для организации."""

    def get_config(self, organization_id: UUID) -> dict[str, Any]:
        record = self._load_profiles().get(str(organization_id)) or self._default_record()
        if record.get("storage_mode") == "cloud":
            runtime = self._inspect_cloud_record(record)
        else:
            runtime = self._prepare_local_runtime()
        record = self._merge_runtime(record, runtime)
        return self._serialize(record)

    def update_config(self, organization_id: UUID, payload: dict[str, Any]) -> dict[str, Any]:
        profiles = self._load_profiles()
        now = self._utcnow()
        storage_mode = str(payload.get("storage_mode") or "local").strip().lower()
        record = profiles.get(str(organization_id)) or self._default_record()
        record["storage_mode"] = storage_mode
        record["updated_at"] = now

        if storage_mode == "local":
            workspace_root, _ = self._local_workspace_specs()
            record.update(
                {
                    "cloud_url": None,
                    "provider": None,
                    "provider_label": None,
                    "remote_name": None,
                    "cloud_root_id": None,
                    "cloud_root_path": None,
                    "cloud_base_url": None,
                    "cloud_vendor": None,
                    "cloud_link_kind": None,
                    "workspace_root": str(workspace_root),
                }
            )
            runtime = self._prepare_local_runtime()
        else:
            cloud_url = str(payload.get("cloud_url") or "").strip()
            detection = self._detect_provider(cloud_url)
            record.update(
                {
                    "cloud_url": cloud_url,
                    "provider": detection["provider"],
                    "provider_label": detection["provider_label"],
                    "workspace_root": detection.get("workspace_root") or cloud_url,
                    "remote_name": self._build_remote_name(organization_id, detection["provider"]),
                    "cloud_root_id": detection.get("cloud_root_id"),
                    "cloud_root_path": detection.get("cloud_root_path"),
                    "cloud_base_url": detection.get("cloud_base_url"),
                    "cloud_vendor": detection.get("cloud_vendor"),
                    "cloud_link_kind": detection.get("cloud_link_kind"),
                }
            )
            runtime = self._connect_cloud_record(record, create_hierarchy=True)

        merged = self._merge_runtime(record, runtime)
        profiles[str(organization_id)] = merged
        self._save_profiles(profiles)
        return self._serialize(merged)

    def connect_cloud(self, organization_id: UUID) -> dict[str, Any]:
        profiles = self._load_profiles()
        record = profiles.get(str(organization_id)) or self._default_record()
        if record.get("storage_mode") != "cloud":
            runtime = self._prepare_local_runtime()
            merged = self._merge_runtime(record, runtime)
            profiles[str(organization_id)] = merged
            self._save_profiles(profiles)
            return self._serialize(merged)

        runtime = self._connect_cloud_record(record, create_hierarchy=True)
        merged = self._merge_runtime(record, runtime)
        merged["updated_at"] = self._utcnow()
        profiles[str(organization_id)] = merged
        self._save_profiles(profiles)
        return self._serialize(merged)

    def _default_record(self) -> dict[str, Any]:
        workspace_root, _ = self._local_workspace_specs()
        return {
            "storage_mode": "local",
            "cloud_url": None,
            "provider": None,
            "provider_label": None,
            "remote_name": None,
            "cloud_root_id": None,
            "cloud_root_path": None,
            "cloud_base_url": None,
            "cloud_vendor": None,
            "cloud_link_kind": None,
            "workspace_root": str(workspace_root),
            "updated_at": self._utcnow(),
        }

    def _local_workspace_specs(self) -> tuple[Path, list[tuple[str, Path]]]:
        if PROJECT_ROOT != BACKEND_ROOT:
            workspace_root = PROJECT_ROOT
            relative_paths = LOCAL_PROJECT_WORKSPACE_DIRS
        else:
            workspace_root = BACKEND_ROOT
            relative_paths = LOCAL_BACKEND_WORKSPACE_DIRS
        return workspace_root, [(relative_path, workspace_root / relative_path) for relative_path in relative_paths]

    def _prepare_local_runtime(self) -> dict[str, Any]:
        workspace_root, specs = self._local_workspace_specs()
        folders: list[str] = []
        for relative_path, path in specs:
            path.mkdir(parents=True, exist_ok=True)
            folders.append(relative_path)
        return {
            "status": "local_ready",
            "message": "Локальное хранилище активно.",
            "auth_state": "not_required",
            "auth_required": False,
            "rclone_available": bool(self._resolve_rclone_bin()),
            "hierarchy_ready": True,
            "workspace_root": str(workspace_root),
            "workspace_folders": folders,
            "auth_prompt": None,
        }

    def _inspect_cloud_record(self, record: dict[str, Any]) -> dict[str, Any]:
        provider = str(record.get("provider") or "unknown")
        if provider == "unknown":
            return {
                "status": "unsupported",
                "message": "Не удалось определить тип облака по ссылке.",
                "auth_state": "unsupported",
                "auth_required": False,
                "rclone_available": bool(self._resolve_rclone_bin()),
                "hierarchy_ready": False,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": None,
            }

        explicit_root_missing = self._missing_explicit_root(record)
        if explicit_root_missing:
            return self._build_link_issue_runtime(record)

        rclone_bin = self._resolve_rclone_bin()
        if not rclone_bin:
            return {
                "status": "unavailable",
                "message": "rclone недоступен в runtime backend.",
                "auth_state": "unavailable",
                "auth_required": False,
                "rclone_available": False,
                "hierarchy_ready": False,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": self._build_auth_prompt(provider, missing_rclone=True),
            }

        source_remote = self._find_source_remote(provider, record)
        if not source_remote:
            return {
                "status": "pending_auth",
                "message": f"Нужна авторизация {PROVIDER_LABELS[provider]} перед инициализацией облачной папки.",
                "auth_state": "needs_auth",
                "auth_required": True,
                "rclone_available": True,
                "hierarchy_ready": False,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": self._build_auth_prompt(provider, cloud_url=str(record.get("cloud_url") or "")),
            }

        remote_name = str(record.get("remote_name") or "")
        remote_root = self._remote_root_for_record(record)
        if remote_name and self._rclone_path_exists(remote_root):
            hierarchy_ready = self._cloud_hierarchy_ready(remote_root)
            if not hierarchy_ready:
                previous_status = str(record.get("status") or "")
                previous_message = str(record.get("message") or "").strip()
                if previous_status.endswith("error") or previous_status == "connected_with_warnings":
                    if previous_status.endswith("error") and previous_message and "googleapi:" not in previous_message.lower():
                        return {
                            "status": previous_status,
                            "message": previous_message,
                            "auth_state": record.get("auth_state", "folder_access_error"),
                            "auth_required": bool(record.get("auth_required", True)),
                            "rclone_available": True,
                            "hierarchy_ready": False,
                            "workspace_root": record.get("workspace_root"),
                            "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                            "auth_prompt": record.get("auth_prompt") or self._build_auth_prompt(provider, access_issue="folder_access"),
                        }
                    diagnostic = self._build_cloud_access_diagnostic(
                        provider=provider,
                        detail=previous_message,
                        cloud_root_id=str(record.get("cloud_root_id") or ""),
                    )
                    return {
                        "status": diagnostic["status"] if previous_message else (previous_status or "folder_access_error"),
                        "message": diagnostic["message"] if previous_message else "Облачная папка найдена, но рабочую иерархию внутри неё пока создать не удалось.",
                        "auth_state": diagnostic["auth_state"] if previous_message else record.get("auth_state", "folder_access_error"),
                        "auth_required": diagnostic["auth_required"] if previous_message else bool(record.get("auth_required", True)),
                        "rclone_available": True,
                        "hierarchy_ready": False,
                        "workspace_root": record.get("workspace_root"),
                        "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                        "auth_prompt": diagnostic["auth_prompt"] if previous_message else (record.get("auth_prompt") or self._build_auth_prompt(provider, access_issue="folder_access")),
                    }
                return {
                    "status": "pending_workspace",
                    "message": f"{PROVIDER_LABELS[provider]} доступен, но рабочая иерархия в выбранной папке ещё не создана.",
                    "auth_state": "connected",
                    "auth_required": False,
                    "rclone_available": True,
                    "hierarchy_ready": False,
                    "workspace_root": record.get("workspace_root"),
                    "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                    "auth_prompt": None,
                }
            return {
                "status": "connected",
                "message": f"{PROVIDER_LABELS[provider]} подключён. Облачная папка доступна.",
                "auth_state": "connected",
                "auth_required": False,
                "rclone_available": True,
                "hierarchy_ready": True,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": None,
            }

        return {
            "status": "pending_connection",
            "message": f"{PROVIDER_LABELS[provider]} определён. Нажмите «Проверить подключение», чтобы подготовить папку.",
            "auth_state": "source_ready",
            "auth_required": False,
            "rclone_available": True,
            "hierarchy_ready": False,
            "workspace_root": record.get("workspace_root"),
            "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
            "auth_prompt": None,
        }

    def _connect_cloud_record(self, record: dict[str, Any], *, create_hierarchy: bool) -> dict[str, Any]:
        provider = str(record.get("provider") or "unknown")
        if provider not in PROVIDER_RCLONE_TYPES:
            return self._inspect_cloud_record(record)

        if self._missing_explicit_root(record):
            return self._build_link_issue_runtime(record)

        cloud_root_id = str(record.get("cloud_root_id") or "").strip()
        if provider == "google_drive" and not cloud_root_id:
            return {
                "status": "link_format_error",
                "message": "В ссылке Google Drive не найден идентификатор папки.",
                "auth_state": "invalid_link",
                "auth_required": False,
                "rclone_available": bool(self._resolve_rclone_bin()),
                "hierarchy_ready": False,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": None,
            }

        rclone_bin = self._resolve_rclone_bin()
        if not rclone_bin:
            return self._inspect_cloud_record(record)

        source_remote = self._find_source_remote(provider, record)
        if not source_remote:
            return self._inspect_cloud_record(record)

        remote_name = str(record.get("remote_name") or self._build_remote_name(UUID(int=0), provider))
        try:
            self._clone_remote_config(source_remote, remote_name, record=record)
        except Exception as exc:
            logger.error("storage_remote_clone_failed", error=str(exc), exc_info=True)
            return {
                "status": "error",
                "message": f"Не удалось подготовить rclone remote: {exc}",
                "auth_state": "error",
                "auth_required": False,
                "rclone_available": True,
                "hierarchy_ready": False,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": None,
            }

        remote_root = self._remote_root_for_record(record, remote_name=remote_name)
        if not self._rclone_path_exists(remote_root):
            diagnostic = self._build_cloud_access_diagnostic(
                provider=provider,
                detail=f"not found: {record.get('cloud_root_path') or record.get('cloud_root_id') or record.get('cloud_url') or remote_root}",
                cloud_root_id=cloud_root_id,
            )
            return {
                "status": diagnostic["status"],
                "message": diagnostic["message"],
                "auth_state": diagnostic["auth_state"],
                "auth_required": diagnostic["auth_required"],
                "rclone_available": True,
                "hierarchy_ready": False,
                "workspace_root": record.get("workspace_root"),
                "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                "auth_prompt": diagnostic["auth_prompt"],
            }

        hierarchy_ready = True
        message = f"{PROVIDER_LABELS[provider]} подключён. Облачная структура готова."
        if create_hierarchy:
            errors = self._ensure_cloud_workspace(remote_root)
            hierarchy_ready = len(errors) == 0
            if errors:
                diagnostic = self._build_cloud_access_diagnostic(
                    provider=provider,
                    detail=errors[0],
                    cloud_root_id=cloud_root_id,
                )
                return {
                    "status": diagnostic["status"],
                    "message": diagnostic["message"],
                    "auth_state": diagnostic["auth_state"],
                    "auth_required": diagnostic["auth_required"],
                    "rclone_available": True,
                    "hierarchy_ready": False,
                    "workspace_root": record.get("workspace_root"),
                    "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
                    "auth_prompt": diagnostic["auth_prompt"],
                }

        return {
            "status": "connected" if hierarchy_ready else "connected_with_warnings",
            "message": message,
            "auth_state": "connected",
            "auth_required": False,
            "rclone_available": True,
            "hierarchy_ready": hierarchy_ready,
            "workspace_root": record.get("workspace_root"),
            "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
            "auth_prompt": None,
        }

    def _merge_runtime(self, record: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
        merged = dict(record)
        merged.update(runtime)
        if not merged.get("updated_at"):
            merged["updated_at"] = self._utcnow()
        return merged

    def _serialize(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "storage_mode": record.get("storage_mode", "local"),
            "cloud_url": record.get("cloud_url"),
            "provider": record.get("provider"),
            "provider_label": record.get("provider_label"),
            "status": record.get("status", "local_ready"),
            "message": record.get("message"),
            "auth_state": record.get("auth_state", "not_required"),
            "auth_required": bool(record.get("auth_required", False)),
            "rclone_available": bool(record.get("rclone_available", False)),
            "remote_name": record.get("remote_name"),
            "hierarchy_ready": bool(record.get("hierarchy_ready", False)),
            "workspace_root": record.get("workspace_root"),
            "workspace_folders": list(record.get("workspace_folders") or []),
            "auth_prompt": record.get("auth_prompt"),
            "updated_at": record.get("updated_at"),
        }

    def _detect_provider(self, raw_url: str) -> dict[str, Any]:
        parsed = urlparse(raw_url)
        host = parsed.netloc.lower()
        if "drive.google.com" in host or "docs.google.com" in host:
            folder_id = self._extract_google_drive_folder_id(raw_url)
            return {
                "provider": "google_drive",
                "provider_label": PROVIDER_LABELS["google_drive"],
                "cloud_root_id": folder_id,
                "cloud_root_path": "",
                "cloud_link_kind": "folder_link",
                "workspace_root": raw_url,
            }
        if "dropbox.com" in host:
            dropbox_path, link_kind = self._extract_dropbox_folder_path(raw_url)
            return {
                "provider": "dropbox",
                "provider_label": PROVIDER_LABELS["dropbox"],
                "cloud_root_id": None,
                "cloud_root_path": dropbox_path,
                "cloud_link_kind": link_kind,
                "workspace_root": raw_url,
            }
        if "disk.yandex." in host or "yadi.sk" in host:
            yandex_path, link_kind = self._extract_yandex_disk_path(raw_url)
            return {
                "provider": "yandex_disk",
                "provider_label": PROVIDER_LABELS["yandex_disk"],
                "cloud_root_id": None,
                "cloud_root_path": yandex_path,
                "cloud_link_kind": link_kind,
                "workspace_root": raw_url,
            }
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            webdav = self._detect_webdav_target(raw_url)
            return {
                "provider": "webdav",
                "provider_label": PROVIDER_LABELS["webdav"],
                "cloud_root_id": None,
                "cloud_root_path": webdav["cloud_root_path"],
                "cloud_base_url": webdav["cloud_base_url"],
                "cloud_vendor": webdav["cloud_vendor"],
                "cloud_link_kind": webdav["cloud_link_kind"],
                "workspace_root": webdav["workspace_root"],
            }
        return {
            "provider": "unknown",
            "provider_label": PROVIDER_LABELS["unknown"],
            "cloud_root_id": None,
            "cloud_root_path": None,
            "cloud_link_kind": "unknown",
            "workspace_root": raw_url,
        }

    def _extract_google_drive_folder_id(self, raw_url: str) -> str | None:
        patterns = (
            r"/folders/([a-zA-Z0-9_-]+)",
            r"[?&]id=([a-zA-Z0-9_-]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, raw_url)
            if match:
                return match.group(1)
        query = parse_qs(urlparse(raw_url).query)
        if query.get("id"):
            return query["id"][0]
        return None

    def _extract_dropbox_folder_path(self, raw_url: str) -> tuple[str | None, str]:
        parsed = urlparse(raw_url)
        path = unquote(parsed.path or "").strip()
        if path.startswith("/home"):
            relative = path[len("/home"):].strip("/")
            return relative, "folder_link"
        return None, "shared_link"

    def _extract_yandex_disk_path(self, raw_url: str) -> tuple[str | None, str]:
        parsed = urlparse(raw_url)
        query = parse_qs(parsed.query or "")
        if query.get("path"):
            return unquote(query["path"][0]).strip("/"), "folder_link"
        path = unquote(parsed.path or "").strip()
        if path.startswith("/client/disk"):
            relative = path[len("/client/disk"):].strip("/")
            return relative, "folder_link"
        return None, "shared_link"

    def _detect_webdav_target(self, raw_url: str) -> dict[str, str]:
        parsed = urlparse(raw_url)
        path = unquote(parsed.path or "").strip("/")
        host = parsed.netloc.lower()
        cloud_vendor = self._guess_webdav_vendor(host, path)
        cloud_link_kind = "webdav_endpoint" if self._looks_like_webdav_endpoint(host, path) else "server_link"
        sanitized = parsed._replace(params="", query="", fragment="")
        workspace_root = sanitized.geturl()
        return {
            "cloud_base_url": f"{parsed.scheme}://{parsed.netloc}",
            "cloud_root_path": path,
            "cloud_vendor": cloud_vendor,
            "cloud_link_kind": cloud_link_kind,
            "workspace_root": workspace_root,
        }

    def _guess_webdav_vendor(self, host: str, path: str) -> str:
        lower_host = host.lower()
        lower_path = path.lower()
        if "nextcloud" in lower_host or "remote.php" in lower_path or "public.php/webdav" in lower_path:
            return "nextcloud"
        if "owncloud" in lower_host:
            return "owncloud"
        if "sharepoint" in lower_host:
            return "sharepoint"
        return "other"

    def _looks_like_webdav_endpoint(self, host: str, path: str) -> bool:
        lower_host = host.lower()
        lower_path = path.lower()
        hints = (
            "/remote.php/dav",
            "/remote.php/webdav",
            "/public.php/webdav",
            "/webdav",
            "/dav",
        )
        return any(hint in lower_path for hint in hints) or "nextcloud" in lower_host or "owncloud" in lower_host or "sharepoint" in lower_host

    def _load_profiles(self) -> dict[str, Any]:
        if not PROFILE_PATH.exists():
            return {}
        try:
            return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("storage_profiles_read_failed", error=str(exc), exc_info=True)
            return {}

    def _save_profiles(self, payload: dict[str, Any]) -> None:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_rclone_bin(self) -> str | None:
        configured = os.getenv("RCLONE_BIN", "").strip()
        candidates = [configured] if configured else []
        candidates.extend([shutil.which("rclone") or ""])
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

    def _load_rclone_config(self) -> configparser.RawConfigParser:
        config = configparser.RawConfigParser()
        config.optionxform = str
        if RCLONE_CONFIG_PATH.exists():
            config.read(RCLONE_CONFIG_PATH, encoding="utf-8")
        return config

    def _find_source_remote(self, provider: str, record: dict[str, Any] | None = None) -> str | None:
        rclone_type = PROVIDER_RCLONE_TYPES.get(provider)
        if not rclone_type or not RCLONE_CONFIG_PATH.exists():
            return None
        config = self._load_rclone_config()
        preferred = os.getenv(
            PROVIDER_RCLONE_HINT_ENVS.get(provider, ""),
            PROVIDER_RCLONE_HINT_DEFAULTS.get(provider, ""),
        ).strip()
        if preferred and config.has_section(preferred):
            section = config[preferred]
            if section.get("type") == rclone_type and self._section_has_auth(provider, section, record):
                return preferred
        for section_name in config.sections():
            section = config[section_name]
            if section.get("type") == rclone_type and self._section_has_auth(provider, section, record):
                return section_name
        return None

    def _section_has_auth(
        self,
        provider: str,
        section: configparser.SectionProxy,
        record: dict[str, Any] | None = None,
    ) -> bool:
        if provider in {"google_drive", "dropbox", "yandex_disk"}:
            return bool(section.get("token"))
        if provider == "webdav":
            if not (section.get("user") or section.get("pass") or section.get("bearer_token") or section.get("bearer_token_command")):
                return False
            if not record:
                return True
            target_base_url = str(record.get("cloud_base_url") or "").strip()
            if not target_base_url:
                return True
            section_url = str(section.get("url") or "").strip()
            return self._urls_share_host(section_url, target_base_url)
        return False

    def _urls_share_host(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_parsed = urlparse(left)
        right_parsed = urlparse(right)
        return bool(left_parsed.netloc and left_parsed.netloc == right_parsed.netloc)

    def _clone_remote_config(self, source_remote: str, target_remote: str, *, record: dict[str, Any]) -> None:
        provider = str(record.get("provider") or "unknown")
        config = self._load_rclone_config()
        if not config.has_section(source_remote):
            raise FileNotFoundError(f"Исходный remote '{source_remote}' не найден")
        if config.has_section(target_remote):
            config.remove_section(target_remote)
        config.add_section(target_remote)
        source_section = config[source_remote]
        target_section = config[target_remote]
        for key, value in source_section.items():
            target_section[key] = value
        target_section["type"] = PROVIDER_RCLONE_TYPES[provider]
        if provider == "google_drive":
            target_section["root_folder_id"] = str(record.get("cloud_root_id") or "").strip()
            target_section["scope"] = target_section.get("scope", "drive")
        elif provider == "webdav":
            target_section["url"] = str(record.get("cloud_base_url") or target_section.get("url") or "").strip()
            target_section["vendor"] = str(record.get("cloud_vendor") or target_section.get("vendor") or "other").strip() or "other"
        RCLONE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RCLONE_CONFIG_PATH.open("w", encoding="utf-8") as handle:
            config.write(handle)

    def _rclone_path_exists(self, remote_path: str) -> bool:
        result = self._run_rclone("lsf", remote_path, "--max-depth", "1")
        return result.returncode == 0

    def _remote_root_for_record(self, record: dict[str, Any], *, remote_name: str | None = None) -> str:
        resolved_remote = str(remote_name or record.get("remote_name") or "").strip()
        root_path = str(record.get("cloud_root_path") or "").strip("/")
        if not resolved_remote:
            return ":"
        if root_path:
            return f"{resolved_remote}:{root_path}"
        return f"{resolved_remote}:"

    def _join_remote_path(self, remote_root: str, relative_path: str) -> str:
        normalized = relative_path.strip("/")
        if remote_root.endswith(":"):
            return f"{remote_root}{normalized}"
        return f"{remote_root.rstrip('/')}/{normalized}"

    def _ensure_cloud_workspace(self, remote_root: str) -> list[str]:
        errors: list[str] = []
        for folder in sorted(CLOUD_WORKSPACE_DIRS, key=lambda item: (item.count("/"), item)):
            result = self._run_rclone("mkdir", self._join_remote_path(remote_root, folder))
            if result.returncode != 0:
                detail = self._normalize_rclone_error(result.stderr or result.stdout or "") or f"mkdir {folder} failed"
                errors.append(detail)
        return errors

    def _cloud_hierarchy_ready(self, remote_root: str) -> bool:
        for folder in CLOUD_WORKSPACE_DIRS:
            result = self._run_rclone("lsf", self._join_remote_path(remote_root, folder), "--max-depth", "0")
            if result.returncode != 0:
                return False
        return True

    def _missing_explicit_root(self, record: dict[str, Any]) -> bool:
        provider = str(record.get("provider") or "unknown")
        if provider == "google_drive":
            return not str(record.get("cloud_root_id") or "").strip()
        if provider in {"dropbox", "yandex_disk"}:
            return str(record.get("cloud_link_kind") or "") != "folder_link" or record.get("cloud_root_path") is None
        if provider == "webdav":
            return not str(record.get("cloud_base_url") or "").strip()
        return True

    def _build_link_issue_runtime(self, record: dict[str, Any]) -> dict[str, Any]:
        provider = str(record.get("provider") or "unknown")
        provider_label = PROVIDER_LABELS.get(provider, provider)
        link_kind = str(record.get("cloud_link_kind") or "")
        auth_prompt = self._build_auth_prompt(provider, access_issue="folder_link")
        message = f"Не удалось определить рабочую папку для {provider_label} по этой ссылке."
        if provider == "dropbox":
            message = (
                "Для Dropbox сейчас нужна прямая ссылка на папку из интерфейса Dropbox (`/home/...`). "
                "Публичные shared-ссылки не дают безопасно определить путь для записи."
            )
        elif provider == "yandex_disk":
            message = (
                "Для Яндекс Диска сейчас нужна прямая ссылка на папку из интерфейса Диска (`/client/disk/...`). "
                "Публичные ссылки `yadi.sk` подходят только для чтения и не годятся как рабочий каталог проекта."
            )
        elif provider == "webdav" and link_kind != "webdav_endpoint":
            message = (
                "Ссылка распознана как удалённый диск, но для автонастройки нужна прямая WebDAV-ссылка на каталог "
                "или заранее авторизованный remote для этого сервера."
            )
            auth_prompt = self._build_auth_prompt(provider, access_issue="webdav_link", cloud_url=str(record.get("cloud_url") or ""))
        return {
            "status": "link_format_error",
            "message": message,
            "auth_state": "invalid_link",
            "auth_required": provider != "unknown",
            "rclone_available": bool(self._resolve_rclone_bin()),
            "hierarchy_ready": False,
            "workspace_root": record.get("workspace_root"),
            "workspace_folders": list(CLOUD_WORKSPACE_DIRS),
            "auth_prompt": auth_prompt,
        }

    def _run_rclone(self, *args: str) -> subprocess.CompletedProcess[str]:
        rclone_bin = self._resolve_rclone_bin()
        if not rclone_bin:
            return subprocess.CompletedProcess(args=["rclone", *args], returncode=127, stdout="", stderr="rclone missing")
        try:
            return subprocess.run(
                [rclone_bin, *args],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
        except Exception as exc:
            logger.error("storage_rclone_exec_failed", args=args, error=str(exc), exc_info=True)
            return subprocess.CompletedProcess(args=[rclone_bin, *args], returncode=1, stdout="", stderr=str(exc))

    def _build_auth_prompt(
        self,
        provider: str,
        *,
        missing_rclone: bool = False,
        requires_command: bool = True,
        access_issue: str | None = None,
        cloud_url: str = "",
    ) -> dict[str, Any]:
        provider_label = PROVIDER_LABELS.get(provider, provider)
        if access_issue == "folder_access":
            return {
                "provider_label": provider_label,
                "title": f"Проверьте доступ к папке {provider_label}",
                "description": "Backend видит ссылку, но не может создать рабочие каталоги внутри указанной папки.",
                "steps": [
                    f"Откройте ссылку на папку под тем же аккаунтом {provider_label}, который авторизован в rclone.",
                    "Проверьте, что ссылка ведёт на обычную папку, а не на shortcut.",
                    "Убедитесь, что у этого аккаунта есть права «Редактор» на папку.",
                    "После этого снова нажмите «Проверить подключение».",
                ],
                "login_url": PROVIDER_LOGIN_URLS.get(provider),
                "suggested_command": self._setup_command_for_provider(provider, cloud_url=cloud_url),
            }
        if access_issue == "folder_link":
            steps = [
                "Откройте нужную папку в веб-интерфейсе этого облака.",
                "Скопируйте прямую ссылку именно на папку из своего аккаунта.",
                "Вставьте её в меню и снова нажмите «Сохранить и подключить».",
            ]
            if provider == "dropbox":
                steps[1] = "Для Dropbox используйте ссылку вида `https://www.dropbox.com/home/...`, а не shared link `scl/fo`."
            elif provider == "yandex_disk":
                steps[1] = "Для Яндекс Диска используйте ссылку вида `https://disk.yandex.ru/client/disk/...`, а не публичную `yadi.sk`."
            elif provider == "webdav":
                steps = [
                    "Убедитесь, что ссылка ведёт на каталог WebDAV/Nextcloud/ownCloud/SharePoint, а не на обычную HTML-страницу.",
                    "Если у вас только веб-ссылка, получите у администратора прямой WebDAV URL.",
                    "После этого вставьте ссылку снова и повторите подключение.",
                ]
            return {
                "provider_label": provider_label,
                "title": f"Нужна корректная ссылка на папку {provider_label}",
                "description": "По текущей ссылке нельзя безопасно определить рабочий каталог, в который проект сможет создавать служебную иерархию.",
                "steps": steps,
                "login_url": PROVIDER_LOGIN_URLS.get(provider),
                "suggested_command": self._setup_command_for_provider(provider, cloud_url=cloud_url),
            }
        if access_issue == "webdav_link":
            return {
                "provider_label": provider_label,
                "title": "Требуется вход на удалённый диск",
                "description": "Ссылка распознана как удалённый диск/WebDAV. Для записи нужен одноразовый вход и сохранение remote для этого сервера.",
                "steps": [
                    "Запустите helper-команду ниже и введите логин/пароль от удалённого диска.",
                    "Если это Nextcloud/ownCloud, используйте WebDAV-логин и пароль приложения или обычный пароль сервера.",
                    "После сохранения remote вернитесь в приложение и нажмите «Проверить подключение».",
                ],
                "login_url": None,
                "suggested_command": self._setup_command_for_provider(provider, cloud_url=cloud_url),
            }
        if missing_rclone:
            return {
                "provider_label": provider_label,
                "title": "Не найден rclone",
                "description": "Backend не видит бинарник rclone, поэтому не может подключить облачное хранилище.",
                "steps": [
                    "Установите rclone на хост-машине.",
                    "Перезапустите backend после установки.",
                    "Вернитесь в меню и снова проверьте подключение.",
                ],
                "login_url": PROVIDER_LOGIN_URLS.get(provider),
                "suggested_command": "curl https://rclone.org/install.sh | sudo bash",
            }
        if provider == "webdav":
            return self._build_auth_prompt(provider, access_issue="webdav_link", cloud_url=cloud_url)
        steps = [
            f"Войдите в {provider_label} в браузере.",
            "Авторизуйте системный remote rclone один раз.",
            "После авторизации вернитесь в приложение и нажмите «Проверить подключение».",
        ]
        return {
            "provider_label": provider_label,
            "title": f"Требуется вход в {provider_label}",
            "description": "Ссылка на облачную папку распознана, но действующий токен доступа ещё не найден.",
            "steps": steps,
            "login_url": PROVIDER_LOGIN_URLS.get(provider),
            "suggested_command": self._setup_command_for_provider(provider, cloud_url=cloud_url) if requires_command else None,
        }

    def _normalize_rclone_error(self, detail: str) -> str:
        text = str(detail or "").strip()
        if not text:
            return ""
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = RCLONE_ERROR_PREFIX_RE.sub("", raw_line.strip())
            if not line:
                continue
            lines.append(line)
        if not lines:
            return ""
        for prefix in ("Failed to mkdir:", "failed to make directory:"):
            for line in reversed(lines):
                if prefix.lower() in line.lower():
                    return line
        return lines[-1]

    def _build_cloud_access_diagnostic(
        self,
        *,
        provider: str,
        detail: str,
        cloud_root_id: str,
    ) -> dict[str, Any]:
        normalized = self._normalize_rclone_error(detail)
        lower = normalized.lower()
        provider_label = PROVIDER_LABELS.get(provider, provider)
        if provider == "google_drive" and (
            "file not found" in lower
            or "notfound" in lower
            or cloud_root_id.lower() in lower
        ):
            return {
                "status": "folder_access_error",
                "message": (
                    f"Не удалось создать рабочую иерархию в {provider_label}. "
                    "Обычно это значит, что у аккаунта rclone нет прав редактора на эту папку, "
                    "ссылка ведёт на shortcut вместо обычной папки, либо папка недоступна этому аккаунту."
                ),
                "auth_state": "folder_access_error",
                "auth_required": True,
                "auth_prompt": self._build_auth_prompt(provider, access_issue="folder_access"),
            }
        if "not found" in lower or "directory not found" in lower or "object not found" in lower:
            if provider == "dropbox":
                return {
                    "status": "folder_access_error",
                    "message": "Не удалось найти папку Dropbox по этой ссылке. Обычно это значит, что вставлена shared-ссылка вместо прямого URL `/home/...` или папка не добавлена в тот Dropbox-аккаунт, который авторизован в rclone.",
                    "auth_state": "folder_access_error",
                    "auth_required": True,
                    "auth_prompt": self._build_auth_prompt(provider, access_issue="folder_link"),
                }
            if provider == "yandex_disk":
                return {
                    "status": "folder_access_error",
                    "message": "Не удалось открыть выбранную папку Яндекс Диска. Обычно это значит, что вставлена публичная ссылка вместо прямого URL `/client/disk/...` или у текущего аккаунта нет прав записи в эту папку.",
                    "auth_state": "folder_access_error",
                    "auth_required": True,
                    "auth_prompt": self._build_auth_prompt(provider, access_issue="folder_link"),
                }
            if provider == "webdav":
                return {
                    "status": "folder_access_error",
                    "message": "Удалённый диск найден, но указанный каталог недоступен. Проверьте WebDAV URL, права на папку и учётные данные для этого сервера.",
                    "auth_state": "folder_access_error",
                    "auth_required": True,
                    "auth_prompt": self._build_auth_prompt(provider, access_issue="webdav_link"),
                }
        if "permission" in lower or "access denied" in lower or "insufficient" in lower:
            return {
                "status": "folder_permission_error",
                "message": f"{provider_label} подключён, но текущему аккаунту не хватает прав для создания рабочих папок в выбранном каталоге.",
                "auth_state": "folder_access_error",
                "auth_required": True,
                "auth_prompt": self._build_auth_prompt(provider, access_issue="folder_access"),
            }
        return {
            "status": "connected_with_warnings",
            "message": f"Облако подключено, но рабочую иерархию пока создать не удалось: {normalized or 'неизвестная ошибка rclone'}",
            "auth_state": "warning",
            "auth_required": False,
            "auth_prompt": None,
        }

    def _build_remote_name(self, organization_id: UUID, provider: str) -> str:
        suffix = organization_id.hex[:8] if organization_id.int else "default"
        provider_suffix = {
            "google_drive": "gdrive",
            "dropbox": "dropbox",
            "yandex_disk": "yadisk",
            "webdav": "webdav",
        }.get(provider, "cloud")
        return f"agrosignal_{provider_suffix}_{suffix}"

    def _setup_command_for_provider(self, provider: str, *, cloud_url: str = "") -> str | None:
        if provider == "webdav":
            return f"python scripts/setup_remote_disk.py \"{cloud_url}\"" if cloud_url else "python scripts/setup_remote_disk.py"
        return PROVIDER_SETUP_COMMANDS.get(provider)

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()
