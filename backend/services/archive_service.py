"""Сервис создания и выдачи архивов по полям."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from services.field_analytics_service import FieldAnalyticsService
from services.modeling_service import ModelingService
from services.payload_meta import build_freshness
from services.weather_service import WeatherService
from storage.db import ArchiveEntry, Field

logger = get_logger(__name__)


class ArchiveService:
    """Создание ZIP-архивов с геометрией, метаданными и погодой."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self.weather_service = WeatherService(db)
        self.field_analytics_service = FieldAnalyticsService(db)
        self.modeling_service = ModelingService(db)
        archive_dir = Path(self.settings.ARCHIVE_DIR)
        if not archive_dir.is_absolute():
            archive_dir = Path(__file__).resolve().parents[1] / archive_dir
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    async def list_archives(self, *, organization_id: UUID, field_id: UUID | None = None) -> list[dict[str, Any]]:
        stmt = (
            select(ArchiveEntry)
            .where(ArchiveEntry.organization_id == organization_id)
            .order_by(ArchiveEntry.created_at.desc())
        )
        if field_id is not None:
            stmt = stmt.where(ArchiveEntry.field_id == field_id)
        result = await self.db.execute(stmt)
        entries = result.scalars().all()
        return [self._entry_to_dict(entry) for entry in entries]

    async def create_archive(
        self,
        *,
        organization_id: UUID,
        field_id: UUID,
        date_from: datetime,
        date_to: datetime,
        layers: list[str],
    ) -> dict[str, Any]:
        field = None
        if hasattr(self.db, "get"):
            field = await self.db.get(Field, field_id)
            org_id = getattr(field, "organization_id", None) if field is not None else None
            if field is not None and org_id is not None and org_id != organization_id:
                field = None
        if field is None:
            field_result = await self.db.execute(
                select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
            )
            field = self._scalar_one_or_none(field_result)
        if field is None:
            raise ValueError("Поле не найдено")
        if str(getattr(field, "source", "") or "").strip().lower() == "autodetect_preview":
            raise ValueError("Preview-контур из быстрого режима нельзя архивировать. Запустите Standard или Quality.")

        safe_field_id = str(field_id).replace("/", "").replace("..", "").replace("\\", "")
        file_name = f"field_{safe_field_id}_{date_from:%Y%m%d}_{date_to:%Y%m%d}.zip"
        file_path = (self.archive_dir / file_name).resolve()
        if not str(file_path).startswith(str(self.archive_dir.resolve())):
            raise ValueError("Недопустимое имя файла архива")
        geom = to_shape(field.geom)
        centroid = geom.centroid
        current_weather = await self.weather_service.get_current_weather(
            centroid.y,
            centroid.x,
            organization_id=organization_id,
        )
        archive_snapshots = await self.field_analytics_service.build_archive_snapshots(
            field.id,
            organization_id=organization_id,
        )
        archive_snapshots["weather_snapshot"] = current_weather
        archive_snapshots["model_meta"] = {
            **dict(archive_snapshots.get("model_meta") or {}),
            "model_version": (archive_snapshots.get("model_meta") or {}).get("model_version") or self.settings.MODEL_VERSION,
            "dataset_version": (archive_snapshots.get("model_meta") or {}).get("dataset_version") or self.settings.TRAIN_DATA_VERSION,
            "yield_model_version": self.settings.YIELD_MODEL_VERSION,
        }
        archive_snapshots["scenario_snapshot"] = {
            "items": await self.modeling_service.list_scenarios(field.id, organization_id=organization_id),
        }

        with zipfile.ZipFile(file_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "field.geojson",
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": json.loads(json.dumps(to_shape(field.geom).__geo_interface__)),
                                "properties": {
                                    "field_id": str(field.id),
                                    "source": field.source,
                                    "area_m2": field.area_m2,
                                    "perimeter_m": field.perimeter_m,
                                    "quality_score": field.quality_score,
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            archive.writestr(
                "meta.json",
                json.dumps(
                    {
                        "field_id": str(field.id),
                        "date_from": date_from.isoformat(),
                        "date_to": date_to.isoformat(),
                        "layers": layers,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "weather_snapshot": current_weather,
                        "field_snapshot": archive_snapshots["field_snapshot"],
                        "prediction_snapshot": archive_snapshots["prediction_snapshot"],
                        "metrics_snapshot": archive_snapshots["metrics_snapshot"],
                        "scenario_snapshot": archive_snapshots["scenario_snapshot"],
                        "model_meta": archive_snapshots["model_meta"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            archive.writestr("weather.csv", self._weather_csv(current_weather))

        entry = ArchiveEntry(
            organization_id=organization_id,
            field_id=field.id,
            date_from=date_from,
            date_to=date_to,
            layers=layers,
            file_path=str(file_path),
            status="ready",
            expires_at=datetime.now(timezone.utc) + timedelta(days=int(self.settings.ARCHIVE_TTL_DAYS)),
            meta={
                "prediction_available": archive_snapshots["prediction_snapshot"] is not None,
                "snapshot_ready": True,
            },
            field_snapshot=archive_snapshots["field_snapshot"],
            prediction_snapshot=archive_snapshots["prediction_snapshot"],
            metrics_snapshot=archive_snapshots["metrics_snapshot"],
            weather_snapshot=archive_snapshots["weather_snapshot"],
            scenario_snapshot=archive_snapshots["scenario_snapshot"],
            model_meta=archive_snapshots["model_meta"],
        )
        self.db.add(entry)
        await self.db.flush()
        await self.field_analytics_service.attach_archive_series(
            field.id,
            entry.id,
            entry.created_at or datetime.now(timezone.utc),
            organization_id=organization_id,
        )
        logger.info("archive_created", field_id=str(field_id), archive_id=entry.id, file_path=str(file_path))
        return self._entry_to_dict(entry)

    async def get_archive_path(self, archive_id: int, *, organization_id: UUID) -> str:
        entry = None
        if hasattr(self.db, "get"):
            entry = await self.db.get(ArchiveEntry, archive_id)
            org_id = getattr(entry, "organization_id", None) if entry is not None else None
            if entry is not None and org_id is not None and org_id != organization_id:
                entry = None
        if entry is None:
            entry_result = await self.db.execute(
                select(ArchiveEntry).where(ArchiveEntry.id == archive_id).where(ArchiveEntry.organization_id == organization_id)
            )
            entry = self._scalar_one_or_none(entry_result)
        if entry is None:
            raise ValueError("Архив не найден")
        return entry.file_path

    async def get_archive_view(self, archive_id: int, *, organization_id: UUID) -> dict[str, Any]:
        entry = None
        if hasattr(self.db, "get"):
            entry = await self.db.get(ArchiveEntry, archive_id)
            org_id = getattr(entry, "organization_id", None) if entry is not None else None
            if entry is not None and org_id is not None and org_id != organization_id:
                entry = None
        if entry is None:
            entry_result = await self.db.execute(
                select(ArchiveEntry).where(ArchiveEntry.id == archive_id).where(ArchiveEntry.organization_id == organization_id)
            )
            entry = self._scalar_one_or_none(entry_result)
        if entry is None:
            raise ValueError("Архив не найден")
        return {
            "archive": self._entry_to_dict(entry),
            "snapshot": {
                "field_snapshot": dict(entry.field_snapshot or {}),
                "prediction_snapshot": dict(entry.prediction_snapshot or {}),
                "metrics_snapshot": dict(entry.metrics_snapshot or {}),
                "weather_snapshot": dict(entry.weather_snapshot or {}),
                "scenario_snapshot": dict(entry.scenario_snapshot or {}),
                "model_meta": dict(entry.model_meta or {}),
            },
        }

    async def cleanup_expired(self) -> dict[str, int]:
        now = datetime.now(timezone.utc)
        stmt = select(ArchiveEntry).where(ArchiveEntry.expires_at < now)
        result = await self.db.execute(stmt)
        entries = list(result.scalars().all())
        removed_files = 0
        removed_rows = 0
        for entry in entries:
            path = Path(entry.file_path)
            if path.exists():
                path.unlink()
                removed_files += 1
            await self.db.delete(entry)
            removed_rows += 1
        await self.db.flush()
        logger.info("archive_cleanup_finished", removed_rows=removed_rows, removed_files=removed_files)
        return {"removed_rows": removed_rows, "removed_files": removed_files}

    @staticmethod
    def _weather_csv(weather: dict[str, Any]) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["параметр", "значение"])
        for key, value in weather.items():
            writer.writerow([key, value])
        return buffer.getvalue()

    @staticmethod
    def _scalar_one_or_none(result: Any) -> Any:
        if hasattr(result, "scalar_one_or_none"):
            return result.scalar_one_or_none()
        scalars = result.scalars() if hasattr(result, "scalars") else None
        if scalars is None:
            return None
        if hasattr(scalars, "first"):
            return scalars.first()
        rows = scalars.all() if hasattr(scalars, "all") else list(scalars)
        return rows[0] if rows else None

    @staticmethod
    def _entry_to_dict(entry: ArchiveEntry) -> dict[str, Any]:
        model_meta = dict(entry.model_meta or {})
        return {
            "id": entry.id,
            "field_id": str(entry.field_id),
            "date_from": entry.date_from.isoformat(),
            "date_to": entry.date_to.isoformat(),
            "layers": list(entry.layers or []),
            "file_path": entry.file_path,
            "status": entry.status,
            "expires_at": entry.expires_at.isoformat(),
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "meta": dict(entry.meta or {}),
            "field_snapshot": dict(entry.field_snapshot or {}),
            "prediction_snapshot": dict(entry.prediction_snapshot or {}),
            "metrics_snapshot": dict(entry.metrics_snapshot or {}),
            "weather_snapshot": dict(entry.weather_snapshot or {}),
            "scenario_snapshot": dict(entry.scenario_snapshot or {}),
            "model_meta": model_meta,
            "freshness": build_freshness(
                provider="archive",
                fetched_at=entry.created_at,
                cache_written_at=entry.created_at,
                source_published_at=entry.date_to,
                stale=bool(entry.expires_at and entry.expires_at < datetime.now(timezone.utc)),
                model_version=model_meta.get("model_version"),
                dataset_version=model_meta.get("dataset_version"),
            ),
        }
