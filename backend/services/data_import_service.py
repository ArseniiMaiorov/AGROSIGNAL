from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import geopandas as gpd
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Polygon, box, shape
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from services.audit_service import record_audit_event
from storage.db import (
    AoiRun,
    Crop,
    CropAssignment,
    DataImportError,
    DataImportJob,
    Field,
    FieldSeason,
    ManagementEvent,
    SoilProfile,
    WeatherDaily,
    YieldObservation,
    utcnow,
)

logger = get_logger(__name__)

CSV_IMPORT_TYPES = {"yield_history", "crop_plan", "soil_samples", "management_events", "weather_daily"}
BOUNDARY_IMPORT_TYPES = {"field_boundaries"}
SUPPORTED_IMPORT_TYPES = CSV_IMPORT_TYPES | BOUNDARY_IMPORT_TYPES


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_datetime_like(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        return _as_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO date/datetime") from exc


def _parse_date_like(value: Any, *, field_name: str) -> date:
    return _parse_datetime_like(value, field_name=field_name).date()


def _parse_int(value: Any, *, field_name: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required")
    try:
        return int(float(str(value).strip()))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _parse_float(value: Any, *, field_name: str, required: bool = False) -> float | None:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


class DataImportService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        import_dir = Path(self.settings.DATA_IMPORT_DIR)
        if not import_dir.is_absolute():
            import_dir = Path(__file__).resolve().parents[1] / import_dir
        self.import_dir = import_dir
        self.import_dir.mkdir(parents=True, exist_ok=True)

    async def list_jobs(self, *, organization_id: UUID) -> list[dict[str, Any]]:
        stmt = (
            select(DataImportJob)
            .where(DataImportJob.organization_id == organization_id)
            .order_by(DataImportJob.created_at.desc(), DataImportJob.id.desc())
        )
        jobs = (await self.db.execute(stmt)).scalars().all()
        return [self._job_to_dict(job) for job in jobs]

    async def get_job(self, job_id: int, *, organization_id: UUID) -> dict[str, Any]:
        job = await self._job(job_id, organization_id=organization_id)
        return self._job_to_dict(job)

    async def create_import(
        self,
        *,
        organization_id: UUID,
        created_by_user_id: UUID,
        import_type: str,
        source_filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        normalized_type = import_type.strip().lower()
        if normalized_type not in SUPPORTED_IMPORT_TYPES:
            raise ValueError(f"Unsupported import_type '{import_type}'")
        if not content:
            raise ValueError("Empty import payload")

        file_token = uuid.uuid4().hex
        target_dir = self.import_dir / str(organization_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{file_token}_{Path(source_filename).name}"
        target_path.write_bytes(content)

        job = DataImportJob(
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            import_type=normalized_type,
            status="uploaded",
            source_filename=Path(source_filename).name,
            source_path=str(target_path),
        )
        self.db.add(job)
        await self.db.flush()
        await self.preview_job(job.id, organization_id=organization_id)
        await record_audit_event(
            self.db,
            action="imports.preview",
            resource_type="data_import_job",
            resource_id=str(job.id),
            organization_id=organization_id,
            actor_user_id=created_by_user_id,
            payload={"import_type": normalized_type, "source_filename": job.source_filename},
        )
        return self._job_to_dict(job)

    async def preview_job(self, job_id: int, *, organization_id: UUID) -> dict[str, Any]:
        job = await self._job(job_id, organization_id=organization_id)
        await self._replace_errors(job)

        preview_summary: dict[str, Any]
        if job.import_type in CSV_IMPORT_TYPES:
            rows = self._read_csv_rows(Path(job.source_path))
            preview_summary = await self._preview_csv(job, rows)
        elif job.import_type in BOUNDARY_IMPORT_TYPES:
            gdf = self._read_boundaries(Path(job.source_path))
            preview_summary = await self._preview_boundaries(job, gdf)
        else:
            raise ValueError(f"Unsupported import type {job.import_type}")

        job.preview_summary = preview_summary
        job.status = "previewed"
        await self.db.flush()
        return self._job_to_dict(job)

    async def commit_job(
        self,
        job_id: int,
        *,
        organization_id: UUID,
        actor_user_id: UUID,
    ) -> dict[str, Any]:
        job = await self._job(job_id, organization_id=organization_id)
        await self._replace_errors(job)

        if job.import_type in CSV_IMPORT_TYPES:
            rows = self._read_csv_rows(Path(job.source_path))
            commit_summary = await self._commit_csv(job, rows)
        elif job.import_type in BOUNDARY_IMPORT_TYPES:
            gdf = self._read_boundaries(Path(job.source_path))
            commit_summary = await self._commit_boundaries(job, gdf)
        else:
            raise ValueError(f"Unsupported import type {job.import_type}")

        job.commit_summary = commit_summary
        job.status = "committed"
        job.committed_at = utcnow()
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="imports.commit",
            resource_type="data_import_job",
            resource_id=str(job.id),
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            payload=commit_summary,
        )
        return self._job_to_dict(job)

    async def list_errors(self, job_id: int, *, organization_id: UUID) -> list[dict[str, Any]]:
        await self._job(job_id, organization_id=organization_id)
        stmt = (
            select(DataImportError)
            .where(DataImportError.organization_id == organization_id)
            .where(DataImportError.import_job_id == job_id)
            .order_by(DataImportError.id.asc())
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return [self._error_to_dict(row) for row in rows]

    def _read_csv_rows(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    def _read_boundaries(self, path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        return gdf

    async def _preview_csv(self, job: DataImportJob, rows: list[dict[str, Any]]) -> dict[str, Any]:
        required = self._required_columns(job.import_type)
        errors: list[dict[str, Any]] = []
        seasons: set[int] = set()
        fields: set[str] = set()

        for idx, row in enumerate(rows, start=2):
            try:
                self._validate_csv_row(job.import_type, row)
                field_external_id = str(row["field_external_id"]).strip()
                season_year = row.get("season_year")
                fields.add(field_external_id)
                if season_year not in (None, ""):
                    seasons.add(_parse_int(season_year, field_name="season_year"))
            except ValueError as exc:
                errors.append(
                    {
                        "row_number": idx,
                        "error_code": "validation_error",
                        "error_message": str(exc),
                        "raw_record": row,
                    }
                )

        await self._replace_errors(job, errors=errors)
        return {
            "rows_total": len(rows),
            "required_columns": sorted(required),
            "columns": sorted(rows[0].keys()) if rows else [],
            "valid_rows": max(0, len(rows) - len(errors)),
            "invalid_rows": len(errors),
            "unique_field_external_ids": len(fields),
            "season_years": sorted(seasons),
        }

    async def _preview_boundaries(self, job: DataImportJob, gdf: gpd.GeoDataFrame) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        field_ids: set[str] = set()
        geometry_types: set[str] = set()
        if "field_external_id" not in gdf.columns:
            errors.append(
                {
                    "row_number": 1,
                    "error_code": "missing_column",
                    "error_message": "field_external_id column is required for field_boundaries",
                    "raw_record": {"columns": list(gdf.columns)},
                }
            )
        else:
            for idx, row in enumerate(gdf.itertuples(index=False), start=2):
                payload = row._asdict()
                geom = payload.get("geometry")
                field_external_id = payload.get("field_external_id")
                if not field_external_id:
                    errors.append(
                        {
                            "row_number": idx,
                            "error_code": "missing_field_external_id",
                            "error_message": "field_external_id is required",
                            "raw_record": {k: v for k, v in payload.items() if k != "geometry"},
                        }
                    )
                    continue
                if geom is None or geom.is_empty:
                    errors.append(
                        {
                            "row_number": idx,
                            "error_code": "invalid_geometry",
                            "error_message": "geometry is empty",
                            "raw_record": {"field_external_id": field_external_id},
                        }
                    )
                    continue
                if geom.geom_type not in {"Polygon", "MultiPolygon"}:
                    errors.append(
                        {
                            "row_number": idx,
                            "error_code": "invalid_geometry_type",
                            "error_message": f"Unsupported geometry type {geom.geom_type}",
                            "raw_record": {"field_external_id": field_external_id},
                        }
                    )
                    continue
                field_ids.add(str(field_external_id))
                geometry_types.add(geom.geom_type)

        await self._replace_errors(job, errors=errors)
        bounds = list(map(float, gdf.total_bounds)) if not gdf.empty else []
        return {
            "rows_total": int(len(gdf.index)),
            "valid_rows": int(max(0, len(gdf.index) - len(errors))),
            "invalid_rows": int(len(errors)),
            "columns": [str(col) for col in gdf.columns],
            "geometry_types": sorted(geometry_types),
            "unique_field_external_ids": len(field_ids),
            "bbox_4326": bounds,
        }

    async def _commit_csv(self, job: DataImportJob, rows: list[dict[str, Any]]) -> dict[str, Any]:
        inserted = 0
        updated = 0
        skipped = 0
        errors: list[dict[str, Any]] = []

        crop_map = await self._crop_map()
        for idx, row in enumerate(rows, start=2):
            try:
                self._validate_csv_row(job.import_type, row)
                field = await self._resolve_field(
                    organization_id=job.organization_id,
                    field_external_id=str(row["field_external_id"]).strip(),
                )

                if job.import_type == "yield_history":
                    season_year = _parse_int(row["season_year"], field_name="season_year")
                    crop_code = str(row["crop_code"]).strip().lower()
                    field_season, was_created = await self._get_or_create_field_season(
                        organization_id=job.organization_id,
                        field=field,
                        season_year=season_year,
                    )
                    updated += 0 if was_created else 1
                    inserted += 1 if was_created else 0
                    await self._upsert_crop_assignment(
                        organization_id=job.organization_id,
                        field_season=field_season,
                        crop_code=crop_code,
                        payload={"import_job_id": job.id, "source_filename": job.source_filename},
                    )
                    crop_id = crop_map.get(crop_code)
                    observation = YieldObservation(
                        organization_id=job.organization_id,
                        field_season_id=field_season.id,
                        yield_kg_ha=_parse_float(row["yield_kg_ha"], field_name="yield_kg_ha", required=True) or 0.0,
                        observed_at=_parse_datetime_like(row.get("observed_at") or f"{season_year}-09-01", field_name="observed_at"),
                        source="customer_import",
                        payload={
                            "import_job_id": job.id,
                            "source_filename": job.source_filename,
                            "crop_code": crop_code,
                            "crop_id": crop_id,
                            "raw_row": row,
                        },
                    )
                    self.db.add(observation)
                    inserted += 1
                elif job.import_type == "crop_plan":
                    season_year = _parse_int(row["season_year"], field_name="season_year")
                    crop_code = str(row["crop_code"]).strip().lower()
                    field_season, was_created = await self._get_or_create_field_season(
                        organization_id=job.organization_id,
                        field=field,
                        season_year=season_year,
                    )
                    updated += 0 if was_created else 1
                    inserted += 1 if was_created else 0
                    await self._upsert_crop_assignment(
                        organization_id=job.organization_id,
                        field_season=field_season,
                        crop_code=crop_code,
                        payload={"import_job_id": job.id, "source_filename": job.source_filename, "raw_row": row},
                    )
                    updated += 1
                elif job.import_type == "soil_samples":
                    soil = SoilProfile(
                        organization_id=job.organization_id,
                        field_id=field.id,
                        sampled_at=_parse_datetime_like(row["sampled_at"], field_name="sampled_at"),
                        source="customer_import",
                        texture_class=(str(row.get("texture_class")).strip() or None) if row.get("texture_class") is not None else None,
                        organic_matter_pct=_parse_float(row.get("organic_matter_pct"), field_name="organic_matter_pct"),
                        ph=_parse_float(row.get("ph"), field_name="ph"),
                        n_ppm=_parse_float(row.get("n_ppm"), field_name="n_ppm"),
                        p_ppm=_parse_float(row.get("p_ppm"), field_name="p_ppm"),
                        k_ppm=_parse_float(row.get("k_ppm"), field_name="k_ppm"),
                        payload={"import_job_id": job.id, "source_filename": job.source_filename, "raw_row": row},
                    )
                    self.db.add(soil)
                    inserted += 1
                elif job.import_type == "management_events":
                    season_year = _parse_int(row["season_year"], field_name="season_year")
                    field_season, was_created = await self._get_or_create_field_season(
                        organization_id=job.organization_id,
                        field=field,
                        season_year=season_year,
                    )
                    inserted += 1 if was_created else 0
                    event = ManagementEvent(
                        organization_id=job.organization_id,
                        field_season_id=field_season.id,
                        event_date=_parse_datetime_like(row["event_date"], field_name="event_date"),
                        event_type=str(row["event_type"]).strip(),
                        amount=_parse_float(row.get("amount"), field_name="amount"),
                        unit=(str(row.get("unit")).strip() or None) if row.get("unit") is not None else None,
                        source="customer_import",
                        payload={"import_job_id": job.id, "source_filename": job.source_filename, "raw_row": row},
                    )
                    self.db.add(event)
                    inserted += 1
                elif job.import_type == "weather_daily":
                    season_year = _parse_int(row["season_year"], field_name="season_year")
                    field_season, was_created = await self._get_or_create_field_season(
                        organization_id=job.organization_id,
                        field=field,
                        season_year=season_year,
                    )
                    inserted += 1 if was_created else 0
                    observed_on = _parse_date_like(row["observed_on"], field_name="observed_on")
                    existing = (
                        await self.db.execute(
                            select(WeatherDaily)
                            .where(WeatherDaily.organization_id == job.organization_id)
                            .where(WeatherDaily.field_season_id == field_season.id)
                            .where(WeatherDaily.observed_on == observed_on)
                        )
                    ).scalar_one_or_none()
                    payload = {"import_job_id": job.id, "source_filename": job.source_filename, "raw_row": row}
                    if existing is None:
                        self.db.add(
                            WeatherDaily(
                                organization_id=job.organization_id,
                                field_season_id=field_season.id,
                                observed_on=observed_on,
                                temperature_mean_c=_parse_float(row.get("temperature_mean_c"), field_name="temperature_mean_c"),
                                precipitation_mm=_parse_float(row.get("precipitation_mm"), field_name="precipitation_mm"),
                                gdd=_parse_float(row.get("gdd"), field_name="gdd"),
                                vpd=_parse_float(row.get("vpd"), field_name="vpd"),
                                soil_moisture=_parse_float(row.get("soil_moisture"), field_name="soil_moisture"),
                                source="customer_import",
                                payload=payload,
                            )
                        )
                        inserted += 1
                    else:
                        existing.temperature_mean_c = _parse_float(row.get("temperature_mean_c"), field_name="temperature_mean_c")
                        existing.precipitation_mm = _parse_float(row.get("precipitation_mm"), field_name="precipitation_mm")
                        existing.gdd = _parse_float(row.get("gdd"), field_name="gdd")
                        existing.vpd = _parse_float(row.get("vpd"), field_name="vpd")
                        existing.soil_moisture = _parse_float(row.get("soil_moisture"), field_name="soil_moisture")
                        existing.payload = payload
                        updated += 1
            except ValueError as exc:
                skipped += 1
                errors.append(
                    {
                        "row_number": idx,
                        "error_code": "commit_validation_error",
                        "error_message": str(exc),
                        "raw_record": row,
                    }
                )

        await self._replace_errors(job, errors=errors)
        return {
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "error_count": len(errors),
        }

    async def _commit_boundaries(self, job: DataImportJob, gdf: gpd.GeoDataFrame) -> dict[str, Any]:
        if "field_external_id" not in gdf.columns:
            raise ValueError("field_external_id column is required for field_boundaries")

        if gdf.empty:
            return {"inserted": 0, "updated": 0, "skipped": 0, "error_count": 0}

        aoi_geom = box(*gdf.total_bounds)
        run = AoiRun(
            organization_id=job.organization_id,
            created_by_user_id=job.created_by_user_id,
            aoi=from_shape(aoi_geom, srid=4326),
            time_start=utcnow(),
            time_end=utcnow(),
            params={"source": "field_boundaries_import", "import_job_id": job.id},
            status="completed",
            progress=100,
        )
        self.db.add(run)
        await self.db.flush()

        inserted = 0
        updated = 0
        skipped = 0
        errors: list[dict[str, Any]] = []

        for idx, row in enumerate(gdf.itertuples(index=False), start=2):
            payload = row._asdict()
            raw_geom = payload.pop("geometry")
            field_external_id = str(payload.get("field_external_id") or "").strip()
            try:
                if not field_external_id:
                    raise ValueError("field_external_id is required")
                if raw_geom is None or raw_geom.is_empty:
                    raise ValueError("geometry is empty")
                if raw_geom.geom_type == "Polygon":
                    geom = MultiPolygon([raw_geom])
                elif raw_geom.geom_type == "MultiPolygon":
                    geom = raw_geom
                else:
                    raise ValueError(f"unsupported geometry type {raw_geom.geom_type}")

                metric_series = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(3857)
                metric_geom = metric_series.iloc[0]
                area_m2 = float(metric_geom.area)
                perimeter_m = float(metric_geom.length)

                existing = (
                    await self.db.execute(
                        select(Field)
                        .where(Field.organization_id == job.organization_id)
                        .where(Field.external_field_id == field_external_id)
                    )
                ).scalar_one_or_none()

                if existing is None:
                    self.db.add(
                        Field(
                            organization_id=job.organization_id,
                            aoi_run_id=run.id,
                            geom=from_shape(geom, srid=4326),
                            area_m2=area_m2,
                            perimeter_m=perimeter_m,
                            quality_score=1.0,
                            source="customer_import",
                            external_field_id=field_external_id,
                        )
                    )
                    inserted += 1
                else:
                    existing.aoi_run_id = run.id
                    existing.geom = from_shape(geom, srid=4326)
                    existing.area_m2 = area_m2
                    existing.perimeter_m = perimeter_m
                    existing.quality_score = 1.0
                    existing.source = "customer_import"
                    updated += 1
            except ValueError as exc:
                skipped += 1
                errors.append(
                    {
                        "row_number": idx,
                        "error_code": "boundary_commit_error",
                        "error_message": str(exc),
                        "raw_record": {"field_external_id": field_external_id, **payload},
                    }
                )

        await self._replace_errors(job, errors=errors)
        return {
            "aoi_run_id": str(run.id),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "error_count": len(errors),
        }

    def _required_columns(self, import_type: str) -> set[str]:
        if import_type == "yield_history":
            return {"field_external_id", "season_year", "crop_code", "yield_kg_ha"}
        if import_type == "crop_plan":
            return {"field_external_id", "season_year", "crop_code"}
        if import_type == "soil_samples":
            return {"field_external_id", "sampled_at"}
        if import_type == "management_events":
            return {"field_external_id", "season_year", "event_date", "event_type"}
        if import_type == "weather_daily":
            return {"field_external_id", "season_year", "observed_on"}
        if import_type == "field_boundaries":
            return {"field_external_id", "geometry"}
        raise ValueError(f"Unsupported import type {import_type}")

    def _validate_csv_row(self, import_type: str, row: dict[str, Any]) -> None:
        required = self._required_columns(import_type)
        missing = [column for column in required if row.get(column) in (None, "")]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        str(row["field_external_id"]).strip() or (_ for _ in ()).throw(ValueError("field_external_id is required"))

        if import_type in {"yield_history", "crop_plan", "management_events", "weather_daily"}:
            _parse_int(row["season_year"], field_name="season_year")
        if import_type == "yield_history":
            _parse_float(row["yield_kg_ha"], field_name="yield_kg_ha", required=True)
            crop_code = str(row["crop_code"]).strip().lower()
            if not crop_code:
                raise ValueError("crop_code is required")
        if import_type == "crop_plan":
            crop_code = str(row["crop_code"]).strip().lower()
            if not crop_code:
                raise ValueError("crop_code is required")
        if import_type == "soil_samples":
            _parse_datetime_like(row["sampled_at"], field_name="sampled_at")
        if import_type == "management_events":
            _parse_datetime_like(row["event_date"], field_name="event_date")
            event_type = str(row["event_type"]).strip()
            if not event_type:
                raise ValueError("event_type is required")
        if import_type == "weather_daily":
            _parse_date_like(row["observed_on"], field_name="observed_on")

    async def _crop_map(self) -> dict[str, int]:
        rows = (await self.db.execute(select(Crop.code, Crop.id))).all()
        return {str(code): int(crop_id) for code, crop_id in rows}

    async def _resolve_field(self, *, organization_id: UUID, field_external_id: str) -> Field:
        stmt = (
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.external_field_id == field_external_id)
            .order_by(Field.created_at.desc())
            .limit(1)
        )
        field = (await self.db.execute(stmt)).scalar_one_or_none()
        if field is None:
            raise ValueError(f"Field with external_field_id '{field_external_id}' not found")
        return field

    async def _get_or_create_field_season(
        self,
        *,
        organization_id: UUID,
        field: Field,
        season_year: int,
    ) -> tuple[FieldSeason, bool]:
        stmt = (
            select(FieldSeason)
            .where(FieldSeason.organization_id == organization_id)
            .where(FieldSeason.field_id == field.id)
            .where(FieldSeason.season_year == season_year)
        )
        season = (await self.db.execute(stmt)).scalar_one_or_none()
        if season is not None:
            return season, False
        season = FieldSeason(
            organization_id=organization_id,
            field_id=field.id,
            season_year=season_year,
            label=f"{season_year}",
            external_field_id=field.external_field_id,
        )
        self.db.add(season)
        await self.db.flush()
        return season, True

    async def _upsert_crop_assignment(
        self,
        *,
        organization_id: UUID,
        field_season: FieldSeason,
        crop_code: str,
        payload: dict[str, Any],
    ) -> CropAssignment:
        crop_code = crop_code.strip().lower()
        crop_map = await self._crop_map()
        crop_id = crop_map.get(crop_code)
        stmt = (
            select(CropAssignment)
            .where(CropAssignment.organization_id == organization_id)
            .where(CropAssignment.field_season_id == field_season.id)
            .order_by(CropAssignment.assigned_at.desc())
            .limit(1)
        )
        existing = (await self.db.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = CropAssignment(
                organization_id=organization_id,
                field_season_id=field_season.id,
                crop_id=crop_id,
                crop_code=crop_code,
                source="customer_import",
                payload=payload,
            )
            self.db.add(existing)
            await self.db.flush()
            return existing
        existing.crop_id = crop_id
        existing.crop_code = crop_code
        existing.payload = payload
        return existing

    async def _replace_errors(self, job: DataImportJob, *, errors: list[dict[str, Any]] | None = None) -> None:
        await self.db.execute(delete(DataImportError).where(DataImportError.import_job_id == job.id))
        errors = errors or []
        max_errors = max(1, int(self.settings.DATA_IMPORT_MAX_ERRORS))
        for item in errors[:max_errors]:
            self.db.add(
                DataImportError(
                    organization_id=job.organization_id,
                    import_job_id=job.id,
                    row_number=item.get("row_number"),
                    error_code=str(item.get("error_code") or "validation_error"),
                    error_message=str(item.get("error_message") or "Validation error"),
                    raw_record=item.get("raw_record") or {},
                )
            )
        job.error_count = min(len(errors), max_errors)
        await self.db.flush()

    async def _job(self, job_id: int, *, organization_id: UUID) -> DataImportJob:
        stmt = (
            select(DataImportJob)
            .where(DataImportJob.organization_id == organization_id)
            .where(DataImportJob.id == job_id)
        )
        job = (await self.db.execute(stmt)).scalar_one_or_none()
        if job is None:
            raise ValueError("Import job not found")
        return job

    @staticmethod
    def _job_to_dict(job: DataImportJob) -> dict[str, Any]:
        return {
            "id": int(job.id),
            "import_type": job.import_type,
            "status": job.status,
            "source_filename": job.source_filename,
            "preview_summary": dict(job.preview_summary or {}),
            "commit_summary": dict(job.commit_summary or {}),
            "error_count": int(job.error_count or 0),
            "created_at": job.created_at.isoformat() if job.created_at else None,
        }

    @staticmethod
    def _error_to_dict(row: DataImportError) -> dict[str, Any]:
        return {
            "id": int(row.id),
            "row_number": row.row_number,
            "error_code": row.error_code,
            "error_message": row.error_message,
            "raw_record": dict(row.raw_record or {}),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
