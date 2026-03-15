"""Сервис статуса системы."""
from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import desc
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from services.payload_meta import build_freshness
from services.temporal_analytics_service import GEOMETRY_FOUNDATION
from storage.db import AoiRun, Layer, Organization, User, _slugify

try:
    from processing.fields.object_classifier import FEATURE_COLUMNS
    from utils.classifier_schema import validate_classifier_file
    _CLASSIFIER_AVAILABLE = True
except ImportError:
    _CLASSIFIER_AVAILABLE = False
    FEATURE_COLUMNS = []

logger = get_logger(__name__)
APP_VERSION = "1.0.0"


class StatusService:
    """Сбор сводного статуса backend-компонентов."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def get_system_status(self, *, organization_id: UUID) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        components = {
            "database": await self._check_database(),
            "redis": await self._check_redis(),
            "auth_bootstrap": await self._check_auth_bootstrap(),
            "sentinel_model": self._check_model_file(self.settings.ML_MODEL_PATH),
            "edge_refiner": self._check_edge_refiner(),
            "sentinel_provider": await self._check_sentinel_provider(organization_id=organization_id),
            "classifier": self._check_classifier(),
            "weather": await self._check_weather_cache(organization_id=organization_id),
            "layer_catalog": await self._check_layer_catalog(),
        }
        critical = {"database"}
        statuses = {name: data["status"] for name, data in components.items()}
        critical_offline = any(
            status == "offline" for name, status in statuses.items() if name in critical
        )
        any_offline = any(status == "offline" for status in statuses.values())
        overall = "online"
        if any_offline:
            overall = "degraded"
        if critical_offline:
            overall = "offline"
        running_result = await self.db.execute(
            text("SELECT count(*) FROM aoi_runs WHERE organization_id = :organization_id AND status = 'running'"),
            {"organization_id": str(organization_id)},
        )
        runs_result = await self.db.execute(
            text("SELECT count(*) FROM aoi_runs WHERE organization_id = :organization_id"),
            {"organization_id": str(organization_id)},
        )
        return {
            "status": overall,
            "timestamp": now.isoformat(),
            "components": components,
            "runs": {
                "running": int(running_result.scalar_one() or 0),
                "total": int(runs_result.scalar_one() or 0),
            },
            "build": self._build_info(),
            "model_truth": self._model_truth_info(),
            "freshness": build_freshness(
                provider="backend",
                fetched_at=now,
                cache_written_at=now,
                model_version=self.settings.MODEL_VERSION,
                dataset_version=self.settings.TRAIN_DATA_VERSION,
            ),
        }

    async def get_bootstrap_status(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        components = {
            "database": await self._check_database(),
            "redis": await self._check_redis(),
            "auth_bootstrap": await self._check_auth_bootstrap(),
            "sentinel_model": self._check_model_file(self.settings.ML_MODEL_PATH),
            "edge_refiner": self._check_edge_refiner(),
            "classifier": self._check_classifier(),
            "weather_provider": await self._check_weather_provider(),
            "layer_catalog": await self._check_layer_catalog(),
            "scene_cache": self._check_scene_cache(),
            "satellite_browse": self._check_satellite_browse(),
            "prediction_engine": self._check_runtime_module("services.yield_service", detail=self.settings.YIELD_MODEL_VERSION),
            "scenario_engine": self._check_runtime_module("services.modeling_service", detail=self.settings.YIELD_MODEL_VERSION),
            "archive_service": self._check_archive_runtime(),
            "async_jobs": self._check_runtime_module("tasks.analytics", detail="prediction+scenario"),
            "release_smoke": self._check_project_artifact("scripts/release_smoke.py", optional=True),
            "training_pipeline": self._check_project_artifact("backend/training/run_regional_retrain_pipeline.sh"),
            "training_commands": self._check_project_artifact("scripts/train_orchestrated.sh"),
            "cpu_training_deps": self._check_project_artifact("backend/training/requirements.cpu.txt"),
            "qa_matrix": self._check_project_artifact("backend/training/release_russia_qa_matrix.json"),
            "qa_band_summary": self._check_project_artifact("backend/debug/runs/release_qa_band_summary.json", optional=True),
            "docs": self._check_project_artifact("AUTODETECT.md", optional=True),
        }
        critical = {"database", "auth_bootstrap"}
        statuses = {name: data["status"] for name, data in components.items()}
        critical_offline = any(
            status == "offline" for name, status in statuses.items() if name in critical
        )
        any_offline = any(status == "offline" for status in statuses.values())
        overall = "online"
        if any_offline:
            overall = "degraded"
        if critical_offline:
            overall = "offline"
        return {
            "status": overall,
            "timestamp": now.isoformat(),
            "components": components,
            "build": self._build_info(),
            "model_truth": self._model_truth_info(),
            "auth": {
                "enabled": bool(self.settings.AUTH_BOOTSTRAP_ENABLED),
                "bootstrap_admin_email": self.settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower(),
                "bootstrap_org_slug": _slugify(self.settings.AUTH_BOOTSTRAP_ORG_NAME),
                "bootstrap_org_name": self.settings.AUTH_BOOTSTRAP_ORG_NAME,
            },
            "freshness": build_freshness(
                provider="backend",
                fetched_at=now,
                cache_written_at=now,
                model_version=self.settings.MODEL_VERSION,
                dataset_version=self.settings.TRAIN_DATA_VERSION,
            ),
        }

    async def _check_database(self) -> dict[str, Any]:
        try:
            result = await self.db.execute(text("SELECT 1"))
            return {"status": "online", "detail": int(result.scalar_one())}
        except Exception as exc:
            logger.error("status_database_failed", error=str(exc), exc_info=True)
            return {"status": "offline", "detail": str(exc)}

    async def _check_redis(self) -> dict[str, Any]:
        try:
            import redis
            client = redis.from_url(self.settings.REDIS_URL, socket_timeout=2.0)
            pong = client.ping()
            return {"status": "online" if pong else "offline", "detail": bool(pong)}
        except Exception as exc:
            logger.error("status_redis_failed", error=str(exc), exc_info=True)
            return {"status": "offline", "detail": str(exc)}

    def _check_model_file(self, relative_path: str) -> dict[str, Any]:
        path = Path("/app") / relative_path if relative_path.startswith("models/") else Path(relative_path)
        if path.exists():
            return {"status": "online", "detail": str(path)}
        local_path = Path(__file__).resolve().parents[1] / relative_path
        if local_path.exists():
            return {"status": "online", "detail": str(local_path)}
        return {"status": "offline", "detail": relative_path}

    def _check_optional_model_file(self, relative_path: str, *, disabled: bool, fallback_detail: str) -> dict[str, Any]:
        if disabled:
            return {"status": "online", "detail": "disabled by config"}
        path = Path("/app") / relative_path if relative_path.startswith("models/") else Path(relative_path)
        if path.exists():
            return {"status": "online", "detail": str(path)}
        local_path = Path(__file__).resolve().parents[1] / relative_path
        if local_path.exists():
            return {"status": "online", "detail": str(local_path)}
        return {"status": "degraded", "detail": fallback_detail}

    def _check_edge_refiner(self) -> dict[str, Any]:
        return self._check_optional_model_file(
            self.settings.UNET_EDGE_MODEL,
            disabled=not bool(self.settings.FEATURE_UNET_EDGE),
            fallback_detail="optional heuristic fallback active",
        )

    def _check_classifier(self) -> dict[str, Any]:
        if not _CLASSIFIER_AVAILABLE:
            return {"status": "offline", "detail": "Модуль классификатора недоступен"}
        configured_path = Path(self.settings.OBJECT_CLASSIFIER_PATH)
        candidates: list[Path] = []
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            candidates.extend(
                [
                    Path("/app") / configured_path,
                    Path(__file__).resolve().parents[1] / configured_path,
                    configured_path,
                ]
            )
        candidates.extend(
            [
                Path(__file__).resolve().parents[1] / "models" / "object_classifier.pkl",
                Path("/app/models/object_classifier.pkl"),
                Path(__file__).resolve().parents[1] / "object_classifier.pkl",
                Path("/app/object_classifier.pkl"),
            ]
        )
        unique_candidates: list[Path] = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        checked_errors: list[str] = []
        for classifier_path in unique_candidates:
            if not classifier_path.exists():
                continue
            try:
                meta = validate_classifier_file(classifier_path, tuple(FEATURE_COLUMNS))
                return {
                    "status": "online",
                    "detail": f"{classifier_path.name}: {meta['feature_count']} признаков",
                }
            except Exception as exc:
                checked_errors.append(str(exc))
                logger.error("status_classifier_failed", path=str(classifier_path), error=str(exc), exc_info=True)
        detail = checked_errors[0] if checked_errors else "Файл object_classifier.pkl не найден"
        return {"status": "offline", "detail": detail}

    def _check_scene_cache(self) -> dict[str, Any]:
        scene_dir = Path(self.settings.SCENE_CACHE_DIR)
        if not scene_dir.is_absolute():
            scene_dir = Path(__file__).resolve().parents[1] / scene_dir
        if not scene_dir.exists():
            return {"status": "degraded", "detail": str(scene_dir)}
        items = list(scene_dir.glob("*.npz"))
        return {"status": "online", "detail": f"{scene_dir} ({len(items)} scenes)"}

    def _check_runtime_module(self, module_path: str, *, detail: str | None = None) -> dict[str, Any]:
        try:
            importlib.import_module(module_path)
            return {"status": "online", "detail": detail or module_path}
        except Exception as exc:
            logger.error("status_runtime_module_failed", module=module_path, error=str(exc), exc_info=True)
            return {"status": "degraded", "detail": f"{module_path}: {exc}"}

    def _check_archive_runtime(self) -> dict[str, Any]:
        archive_dir = Path(self.settings.ARCHIVE_DIR)
        if not archive_dir.is_absolute():
            archive_dir = Path(__file__).resolve().parents[1] / archive_dir
        if not archive_dir.exists():
            return {"status": "degraded", "detail": str(archive_dir)}
        return {"status": "online", "detail": str(archive_dir)}

    def _check_satellite_browse(self) -> dict[str, Any]:
        primary = bool(self.settings.SH_CLIENT_ID and self.settings.SH_CLIENT_SECRET)
        reserve = bool(self.settings.SH_CLIENT_ID_reserv and self.settings.SH_CLIENT_SECRET_reserv)
        second_reserve = bool(
            self.settings.SH_CLIENT_ID_second_reserv and self.settings.SH_CLIENT_SECRET_second_reserv
        )
        if primary:
            return {
                "status": "online",
                "detail": f"primary=yes, reserve={'yes' if reserve else 'no'}, second_reserve={'yes' if second_reserve else 'no'}",
            }
        if reserve or second_reserve:
            return {
                "status": "degraded",
                "detail": "primary credentials missing, reserve failover only",
            }
        return {"status": "offline", "detail": "Sentinel Hub credentials are not configured"}

    def _check_project_artifact(self, relative_path: str, *, optional: bool = False) -> dict[str, Any]:
        candidate_roots = [
            Path(__file__).resolve().parents[2],  # local repo root
            Path(__file__).resolve().parents[1],  # container /app root
        ]
        relative_variants = [relative_path]
        if relative_path.startswith("backend/"):
            relative_variants.append(relative_path.removeprefix("backend/"))
        for root in candidate_roots:
            for variant in relative_variants:
                path = root / variant
                if path.exists():
                    return {"status": "online", "detail": str(path)}
        runtime_root = Path(__file__).resolve().parents[1]
        if optional and not (runtime_root / "scripts").exists():
            return {"status": "unknown", "detail": "not bundled into runtime image"}
        return {"status": "degraded", "detail": f"missing: {relative_path}"}

    async def _check_auth_bootstrap(self) -> dict[str, Any]:
        if not bool(self.settings.AUTH_REQUIRED):
            return {"status": "online", "detail": "Auth disabled"}
        if not bool(self.settings.AUTH_BOOTSTRAP_ENABLED):
            return {"status": "online", "detail": "Bootstrap admin disabled"}
        try:
            org_slug = _slugify(self.settings.AUTH_BOOTSTRAP_ORG_NAME)
            org = (
                await self.db.execute(select(Organization).where(Organization.slug == org_slug))
            ).scalar_one_or_none()
            user = (
                await self.db.execute(
                    select(User).where(User.email == self.settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower())
                )
            ).scalar_one_or_none()
            if org is None or user is None:
                missing = []
                if org is None:
                    missing.append("organization")
                if user is None:
                    missing.append("admin_user")
                return {"status": "offline", "detail": f"Missing bootstrap entities: {', '.join(missing)}"}
            return {
                "status": "online",
                "detail": {
                    "organization_slug": org_slug,
                    "admin_email": self.settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower(),
                },
            }
        except Exception as exc:
            logger.error("status_auth_bootstrap_failed", error=str(exc), exc_info=True)
            return {"status": "offline", "detail": str(exc)}

    async def _check_layer_catalog(self) -> dict[str, Any]:
        try:
            count_result = await self.db.execute(select(Layer))
            count = len(list(count_result.scalars().all()))
            if count <= 0:
                return {"status": "offline", "detail": "Layer catalog is empty"}
            return {"status": "online", "detail": f"{count} layers"}
        except Exception as exc:
            logger.error("status_layer_catalog_failed", error=str(exc), exc_info=True)
            return {"status": "offline", "detail": str(exc)}

    async def _check_weather_provider(self) -> dict[str, Any]:
        provider = str(self.settings.WEATHER_PROVIDER).strip().lower()
        if provider == "openmeteo":
            return {"status": "online", "detail": self.settings.OPENMETEO_BASE_URL}
        if provider == "openweather":
            if self.settings.OPENWEATHER_API_KEY:
                return {"status": "online", "detail": self.settings.OPENWEATHER_BASE_URL}
            return {"status": "degraded", "detail": "OPENWEATHER_API_KEY is not configured"}
        return {"status": "degraded", "detail": f"Unsupported weather provider '{provider}'"}

    async def _check_weather_cache(self, *, organization_id: UUID) -> dict[str, Any]:
        try:
            result = await self.db.execute(
                text("SELECT max(observed_at) FROM weather_data WHERE organization_id = :organization_id"),
                {"organization_id": str(organization_id)},
            )
            last = result.scalar_one_or_none()
            if last is None:
                return {"status": "degraded", "detail": "Кэш погоды пуст"}
            return {"status": "online", "detail": last.isoformat()}
        except Exception as exc:
            logger.error("status_weather_failed", error=str(exc), exc_info=True)
            return {"status": "degraded", "detail": str(exc)}

    async def _check_sentinel_provider(self, *, organization_id: UUID) -> dict[str, Any]:
        try:
            result = await self.db.execute(
                select(AoiRun)
                .where(AoiRun.organization_id == organization_id)
                .order_by(desc(AoiRun.created_at))
                .limit(1)
            )
            run = result.scalar_one_or_none()
            runtime = dict((getattr(run, "params", None) or {}).get("runtime") or {}) if run is not None else {}
            account = str(runtime.get("sentinel_account_used") or "primary")
            failover_level = int(runtime.get("sentinel_failover_level") or 0)
            tta_mode = str(runtime.get("tta_mode") or "none")
            s1_planned = bool(runtime.get("s1_planned"))
            detail = f"account={account}, failover={failover_level}, tta={tta_mode}, s1={'on' if s1_planned else 'off'}"
            return {
                "status": "online",
                "detail": detail,
            }
        except Exception as exc:
            logger.error("status_sentinel_provider_failed", error=str(exc), exc_info=True)
            return {"status": "degraded", "detail": str(exc)}

    def _build_info(self) -> dict[str, Any]:
        return {
            "app_version": APP_VERSION,
            "model_version": self.settings.MODEL_VERSION,
            "train_data_version": self.settings.TRAIN_DATA_VERSION,
            "yield_model_version": self.settings.YIELD_MODEL_VERSION,
            "feature_stack_version": self.settings.FEATURE_STACK_VERSION,
        }

    def _model_truth_info(self) -> dict[str, Any]:
        return {
            "head_count": int(GEOMETRY_FOUNDATION.get("head_count") or 3),
            "heads": list(GEOMETRY_FOUNDATION.get("heads") or []),
            "tta_standard": str(GEOMETRY_FOUNDATION.get("tta_standard") or "flip2"),
            "tta_quality": str(GEOMETRY_FOUNDATION.get("tta_quality") or "rotate4"),
            "retrain_description": str(GEOMETRY_FOUNDATION.get("retrain_description") or ""),
        }
