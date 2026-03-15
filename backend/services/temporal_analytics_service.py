"""Seasonal analytics, management zones, and explainability helpers."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import mean
from typing import Any
from uuid import UUID

# Module-level TTL cache for get_temporal_analytics — avoids repeated heavy DB+CPU on same request
_TEMPORAL_ANALYTICS_CACHE: dict[str, tuple[float, dict]] = {}
_TEMPORAL_CACHE_TTL = 180  # seconds (3 minutes)

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from providers.sentinelhub.client import SentinelHubClient
from services.payload_meta import build_freshness
from services.weekly_profile_service import (
    FEATURE_SCHEMA_VERSION,
    current_season_year,
    ensure_weekly_profile,
    load_weekly_feature_rows,
)
from storage.db import CropAssignment, Field, FieldFeatureWeekly, FieldMetricSeries, FieldSeason, GridCell, SoilProfile, WeatherDaily, YieldObservation

METRIC_META: dict[str, dict[str, str]] = {
    "ndvi": {"label": "NDVI", "unit": ""},
    "ndmi": {"label": "NDMI", "unit": ""},
    "ndwi": {"label": "NDWI", "unit": ""},
    "bsi": {"label": "BSI", "unit": ""},
    "gdd": {"label": "ГСТ (нед.)", "unit": "°C·д"},
    "gdd_cumulative": {"label": "ГСТ (накоп.)", "unit": "°C·д"},
    "vpd": {"label": "VPD", "unit": "kPa"},
    "soil_moisture": {"label": "Влага почвы", "unit": "%"},
    "precipitation": {"label": "Осадки", "unit": "мм"},
    "wind": {"label": "Ветер", "unit": "м/с"},
}

ZONE_LABELS = {
    "low": "Низкий потенциал",
    "medium": "Средний потенциал",
    "high": "Высокий потенциал",
}

GEOMETRY_FOUNDATION = {
    "head_count": 3,
    "heads": ["extent", "boundary", "distance"],
    "tta_standard": "flip2",
    "tta_quality": "rotate4",
    "retrain_description": (
        "Текущая переобучаемая модель сохраняет 3-head topology "
        "(extent/boundary/distance) и дообучается для качества границ, полноты, "
        "устойчивости по России и северным регионам."
    ),
}

logger = get_logger(__name__)

TEMPORAL_BACKFILL_MAX_DAYS = 366
TEMPORAL_BACKFILL_WINDOW_DAYS = 14
TEMPORAL_SATELLITE_METRICS: tuple[str, ...] = ("ndvi", "ndmi", "ndwi", "bsi")
TEMPORAL_RANGE_READY_MIN_POINTS = 2
TEMPORAL_VALID_SCL_CLASSES = {4, 5, 6}
TEMPORAL_BACKFILL_SOURCE = "historical_backfill"
TEMPORAL_DEFAULT_CLOUD_PCT = 40
TEMPORAL_RASTER_SIZE = 96


@dataclass(slots=True)
class WeeklyPoint:
    observed_at: date
    value: float
    source: str
    coverage: float | None = None


class TemporalBackfillError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_driver_breakdown(
    drivers: list[dict[str, Any]] | None,
    *,
    baseline_yield_kg_ha: float | None,
    scenario_yield_kg_ha: float | None = None,
    baseline_inputs: dict[str, Any] | None = None,
    scenario_inputs: dict[str, Any] | None = None,
    source: str = "model",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    baseline_inputs = baseline_inputs or {}
    scenario_inputs = scenario_inputs or {}
    active_yield = float(scenario_yield_kg_ha or baseline_yield_kg_ha or 0.0)
    for index, raw in enumerate(drivers or []):
        label = str(raw.get("label") or raw.get("factor") or f"driver_{index + 1}")
        input_key = str(raw.get("input_key") or raw.get("input") or label)
        raw_effect = raw.get("effect_pct")
        if raw_effect is None:
            raw_effect = raw.get("effect")
        try:
            effect = float(raw_effect or 0.0)
        except Exception:
            effect = 0.0
        effect_pct = effect * 100.0 if abs(effect) <= 1.5 else effect
        effect_kg = float(raw.get("effect_kg_ha") or 0.0)
        if abs(effect_kg) < 1e-9 and active_yield > 0:
            effect_kg = active_yield * (effect_pct / 100.0)
        baseline_value = baseline_inputs.get(input_key)
        scenario_value = scenario_inputs.get(input_key, baseline_value)
        delta_input = None
        if isinstance(baseline_value, (int, float)) and isinstance(scenario_value, (int, float)):
            delta_input = float(scenario_value) - float(baseline_value)
        normalized.append(
            {
                "driver_id": str(raw.get("driver_id") or input_key or f"driver_{index + 1}"),
                "label": label,
                "factor": label,
                "input_key": input_key,
                "input": input_key,
                "baseline_value": baseline_value,
                "scenario_value": scenario_value,
                "delta_input": delta_input,
                "effect_kg_ha": round(float(effect_kg), 2),
                "effect_pct": round(float(effect_pct), 2),
                "direction": "positive" if effect_pct >= 0 else "negative",
                "source": str(raw.get("source") or source),
                "confidence": round(float(raw.get("confidence") or 0.72), 3),
            }
        )
    normalized.sort(key=lambda item: abs(float(item.get("effect_kg_ha") or 0.0)), reverse=True)
    return normalized


class TemporalAnalyticsService:
    """Builds seasonal field analytics and management zones."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_temporal_analytics(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date | None = None,
        date_to: date | None = None,
        crop_code: str | None = None,
    ) -> dict[str, Any]:
        # TTL cache — skip for backfill-required data; keyed by (field, org, dates, crop)
        _cache_key = f"{field_id}::{organization_id}::{date_from}::{date_to}::{crop_code}"
        _now = time.monotonic()
        _cached = _TEMPORAL_ANALYTICS_CACHE.get(_cache_key)
        if _cached and (_now - _cached[0]) < _TEMPORAL_CACHE_TTL:
            return _cached[1]

        field = await self._get_field(field_id, organization_id=organization_id)
        resolved_from, resolved_to = self._resolve_requested_range(date_from, date_to)
        centroid = to_shape(field.geom).centroid
        crop_context = await self._resolve_crop_context(field_id, organization_id=organization_id, crop_code=crop_code)
        weekly_rows = await self._load_weekly_rows_for_range(
            field_id,
            organization_id=organization_id,
            date_from=resolved_from,
            date_to=resolved_to,
        )
        weekly_series = self._build_weekly_metric_series_from_rows(weekly_rows)
        primary_ndvi = weekly_series.get("ndvi") or []
        phenology = self._build_phenology(primary_ndvi, crop_context.get("crop_code"))
        anomalies = self._build_anomalies(weekly_series, phenology)
        history_trend = await self._build_history_trend(
            field_id,
            organization_id=organization_id,
            crop_code=crop_context.get("crop_code"),
        )
        water_balance = await self._build_water_balance(
            field_id,
            organization_id=organization_id,
            crop_code=crop_context.get("crop_code"),
            phenology=phenology,
            weekly_rows=weekly_rows,
        )
        risk = self._build_risk_summary(
            crop_code=crop_context.get("crop_code"),
            phenology=phenology,
            water_balance=water_balance,
            anomalies=anomalies,
        )
        data_status = await self._build_data_status(
            field_id,
            organization_id=organization_id,
            date_from=resolved_from,
            date_to=resolved_to,
            weekly_series=weekly_series,
            weekly_rows=weekly_rows,
        )
        result = {
            "field_id": str(field_id),
            "crop_context": crop_context,
            "seasonal_series": {
                "metrics": [
                    {
                        "metric": metric,
                        "label": METRIC_META.get(metric, {}).get("label", metric.upper()),
                        "unit": METRIC_META.get(metric, {}).get("unit", ""),
                        "points": points,
                    }
                    for metric, points in weekly_series.items()
                ],
                "date_from": resolved_from.isoformat(),
                "date_to": resolved_to.isoformat(),
            },
            "phenology": phenology,
            "anomalies": anomalies,
            "water_balance": water_balance,
            "risk": risk,
            "history_trend": history_trend,
            "data_status": data_status,
            "supported_sections": {
                "series": bool(weekly_series),
                "phenology": bool(phenology.get("supported")),
                "anomalies": bool(anomalies),
                "water_balance": bool(water_balance.get("supported")),
                "risk": bool(risk.get("supported")),
                "history_trend": bool(history_trend.get("supported")),
            },
            "analytics_summary": {
                "active_alert_count": len([item for item in anomalies if item.get("severity") in {"warning", "critical"}]),
                "current_stage": phenology.get("stage_label"),
                "lag_weeks_vs_norm": phenology.get("lag_weeks_vs_norm"),
                "history_trend_slope": history_trend.get("trend_slope_kg_ha_per_year"),
                "water_stress_class": water_balance.get("summary", {}).get("stress_class"),
                "head_count": GEOMETRY_FOUNDATION["head_count"],
                "heads": list(GEOMETRY_FOUNDATION["heads"]),
                "tta_standard": GEOMETRY_FOUNDATION["tta_standard"],
                "tta_quality": GEOMETRY_FOUNDATION["tta_quality"],
                "retrain_description": GEOMETRY_FOUNDATION["retrain_description"],
                "centroid": {"lat": round(float(centroid.y), 6), "lon": round(float(centroid.x), 6)},
            },
            "freshness": build_freshness(
                provider="seasonal_analytics",
                fetched_at=datetime.now(timezone.utc),
                cache_written_at=datetime.now(timezone.utc),
                model_version=None,
                dataset_version=None,
            ),
        }
        # Cache result only when data is complete (no backfill needed)
        if not data_status.get("backfill_required"):
            _TEMPORAL_ANALYTICS_CACHE[_cache_key] = (time.monotonic(), result)
            # Evict stale entries to prevent unbounded memory growth (keep max 200)
            if len(_TEMPORAL_ANALYTICS_CACHE) > 200:
                _oldest = min(_TEMPORAL_ANALYTICS_CACHE, key=lambda k: _TEMPORAL_ANALYTICS_CACHE[k][0])
                _TEMPORAL_ANALYTICS_CACHE.pop(_oldest, None)
        return result

    async def materialize_temporal_range(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date,
        date_to: date,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        resolved_from, resolved_to = self._resolve_requested_range(date_from, date_to, allow_default=False)
        span_days = (resolved_to - resolved_from).days + 1
        if span_days > TEMPORAL_BACKFILL_MAX_DAYS:
            raise ValueError(f"Период materialization ограничен {TEMPORAL_BACKFILL_MAX_DAYS} днями")

        field = await self._get_field(field_id, organization_id=organization_id)
        years = self._season_years_for_range(resolved_from, resolved_to)
        season_windows: dict[int, list[tuple[date, date]]] = {}
        for season_year in years:
            season_from, season_to = self._season_bounds(season_year)
            backfill_from = max(resolved_from, season_from)
            backfill_to = min(resolved_to, season_to)
            season_windows[season_year] = [] if backfill_from > backfill_to else self._biweekly_windows(backfill_from, backfill_to)
        satellite_windows_total = sum(len(windows) for windows in season_windows.values())
        await self._emit_progress(
            progress_callback,
            progress=8,
            stage_code="prepare",
            stage_label="prepare",
            stage_detail="planning requested temporal range",
            stage_detail_code="planning_temporal_range",
            stage_detail_params={
                "date_from": resolved_from.isoformat(),
                "date_to": resolved_to.isoformat(),
                "season_years": years,
            },
            estimated_remaining_s=max(4, len(years) * 3),
        )

        satellite_windows_saved = 0
        weather_rows_created = 0
        sentinel_client = SentinelHubClient()
        shared_client = await sentinel_client.get_shared_client()

        try:
            for year_index, season_year in enumerate(years, start=1):
                season_from, season_to = self._season_bounds(season_year)
                backfill_from = max(resolved_from, season_from)
                backfill_to = min(resolved_to, season_to)
                if backfill_from > backfill_to:
                    continue
                windows = season_windows.get(season_year) or []
                await self._emit_progress(
                    progress_callback,
                    progress=min(72, 10 + round((year_index - 1) / max(len(years), 1) * 48)),
                    stage_code="satellite_backfill",
                    stage_label="satellite-backfill",
                    stage_detail=f"season {season_year}",
                    stage_detail_code="season_backfill",
                    stage_detail_params={
                        "season_year": season_year,
                        "window_count": len(windows),
                    },
                    estimated_remaining_s=max(2, (len(years) - year_index + 1) * 2),
                )
                saved = await self._backfill_satellite_metric_series(
                    field=field,
                    organization_id=organization_id,
                    date_from=backfill_from,
                    date_to=backfill_to,
                    season_year=season_year,
                    client=sentinel_client,
                    shared_client=shared_client,
                    progress_callback=progress_callback,
                    window_offset=satellite_windows_saved,
                    window_total=satellite_windows_total,
                )
                satellite_windows_saved += saved
                rows = await ensure_weekly_profile(
                    self.db,
                    organization_id=organization_id,
                    field_id=field_id,
                    season_year=season_year,
                    min_rows=1,
                    force=True,
                )
                weather_rows_created += len(rows)

            await self._emit_progress(
                progress_callback,
                progress=92,
                stage_code="weekly_profile",
                stage_label="weekly-profile",
                stage_detail="rebuilding weekly temporal profile",
                stage_detail_code="rebuilding_weekly_profile",
                stage_detail_params={
                    "date_from": resolved_from.isoformat(),
                    "date_to": resolved_to.isoformat(),
                },
                estimated_remaining_s=1,
            )
            payload = await self.get_temporal_analytics(
                field_id,
                organization_id=organization_id,
                date_from=resolved_from,
                date_to=resolved_to,
            )
            return {
                "field_id": str(field_id),
                "date_from": resolved_from.isoformat(),
                "date_to": resolved_to.isoformat(),
                "season_years": years,
                "satellite_windows_total": satellite_windows_total,
                "satellite_windows_saved": satellite_windows_saved,
                "weekly_rows_materialized": weather_rows_created,
                "result": payload,
            }
        except TemporalBackfillError:
            await self.db.rollback()
            raise
        except Exception as exc:
            await self.db.rollback()
            classified = self._classify_backfill_error(exc)
            raise TemporalBackfillError(classified, str(exc)) from exc

    async def get_management_zones(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        prediction_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        field = await self._get_field(field_id, organization_id=organization_id)
        field_geom_subquery = select(Field.geom).where(Field.id == field.id).scalar_subquery()
        max_zoom_result = await self.db.execute(
            select(func.max(GridCell.zoom_level))
            .where(GridCell.organization_id == organization_id)
            .where(GridCell.aoi_run_id == field.aoi_run_id)
        )
        max_zoom = _result_scalar_one_or_none(max_zoom_result)
        if max_zoom is None:
            return {
                "field_id": str(field_id),
                "mode": "unsupported",
                "zones": [],
                "geojson": {"type": "FeatureCollection", "features": []},
                "summary": {"supported": False, "reason": "Нет grid-метрик для построения зон."},
            }
        result = await self.db.execute(
            select(GridCell)
            .where(GridCell.organization_id == organization_id)
            .where(GridCell.aoi_run_id == field.aoi_run_id)
            .where(GridCell.zoom_level == max_zoom)
            .where(GridCell.geom.ST_Intersects(field_geom_subquery))
        )
        cells = _result_scalars_all(result)
        if len(cells) < 6:
            return {
                "field_id": str(field_id),
                "mode": "unsupported",
                "zones": [],
                "geojson": {"type": "FeatureCollection", "features": []},
                "summary": {"supported": False, "reason": "Недостаточно grid-ячеек для зонирования."},
            }

        scores = self._build_zone_scores(cells)
        if not scores:
            return {
                "field_id": str(field_id),
                "mode": "unsupported",
                "zones": [],
                "geojson": {"type": "FeatureCollection", "features": []},
                "summary": {"supported": False, "reason": "Недостаточно валидных метрик внутри поля."},
            }

        baseline_prediction = float(prediction_payload.get("estimated_yield_kg_ha") or 0.0) if prediction_payload else 0.0
        baseline_confidence = float(prediction_payload.get("confidence") or 0.0) if prediction_payload else 0.0
        exact_yield_mode = baseline_prediction > 0 and baseline_confidence >= 0.58
        zone_items, features = self._build_zone_payloads(
            field=field,
            cells=cells,
            scores=scores,
            baseline_prediction=baseline_prediction,
            exact_yield_mode=exact_yield_mode,
            baseline_confidence=baseline_confidence,
        )
        return {
            "field_id": str(field_id),
            "mode": "yield" if exact_yield_mode else "yield_potential",
            "zones": zone_items,
            "geojson": {"type": "FeatureCollection", "features": features},
            "summary": {
                "supported": True,
                "mode": "yield" if exact_yield_mode else "yield_potential",
                "field_mean_prediction_kg_ha": round(baseline_prediction, 2) if baseline_prediction > 0 else None,
                "zone_count": len(zone_items),
            },
        }

    async def _get_field(self, field_id: UUID, *, organization_id: UUID) -> Field:
        field_result = await self.db.execute(
            select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
        )
        field = _result_scalar_one_or_none(field_result)
        if field is None:
            raise ValueError("Поле не найдено")
        return field

    async def _resolve_crop_context(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
    ) -> dict[str, Any]:
        if crop_code:
            return {"crop_code": crop_code, "source": "explicit"}
        stmt = (
            select(CropAssignment.crop_code, FieldSeason.season_year)
            .join(FieldSeason, CropAssignment.field_season_id == FieldSeason.id)
            .where(CropAssignment.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .order_by(desc(FieldSeason.season_year))
            .limit(1)
        )
        row = (await self.db.execute(stmt)).first()
        if row is None:
            return {"crop_code": None, "source": "unknown"}
        return {"crop_code": str(row[0]), "season_year": int(row[1]), "source": "field_season"}

    def _resolve_requested_range(
        self,
        date_from: date | None,
        date_to: date | None,
        *,
        allow_default: bool = True,
    ) -> tuple[date, date]:
        if date_from and date_to:
            if date_to < date_from:
                raise ValueError("Дата окончания должна быть позже даты начала")
            return date_from, date_to
        if not allow_default:
            raise ValueError("Нужно указать date_from и date_to")
        today = datetime.now(timezone.utc).date()
        if today.month < 3:
            return date(today.year - 1, 3, 1), date(today.year - 1, 10, 31)
        return date(today.year, 3, 1), min(today, date(today.year, 10, 31))

    @staticmethod
    def _season_bounds(season_year: int) -> tuple[date, date]:
        return date(season_year, 3, 1), date(season_year, 10, 31)

    def _season_years_for_range(self, date_from: date, date_to: date) -> list[int]:
        years = sorted({day.year for day in (date_from, date_to)} | set(range(date_from.year, date_to.year + 1)))
        season_years: list[int] = []
        for year in years:
            season_start, season_end = self._season_bounds(year)
            if date_to < season_start or date_from > season_end:
                continue
            season_years.append(year)
        return season_years or [date_from.year]

    @staticmethod
    def _biweekly_windows(date_from: date, date_to: date) -> list[tuple[date, date]]:
        windows: list[tuple[date, date]] = []
        current = date_from
        while current <= date_to:
            window_end = min(current + timedelta(days=TEMPORAL_BACKFILL_WINDOW_DAYS - 1), date_to)
            windows.append((current, window_end))
            current = window_end + timedelta(days=1)
        return windows

    async def _emit_progress(self, callback: Any | None, **payload: Any) -> None:
        if callback is None:
            return
        result = callback(**payload)
        if hasattr(result, "__await__"):
            await result

    async def _load_weekly_rows_for_range(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date,
        date_to: date,
    ) -> list[FieldFeatureWeekly]:
        rows: list[FieldFeatureWeekly] = []
        for season_year in self._season_years_for_range(date_from, date_to):
            existing = await load_weekly_feature_rows(
                self.db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=season_year,
            )
            if not existing or any(str(getattr(row, "feature_schema_version", "") or "") != FEATURE_SCHEMA_VERSION for row in existing):
                existing = await ensure_weekly_profile(
                    self.db,
                    organization_id=organization_id,
                    field_id=field_id,
                    season_year=season_year,
                    min_rows=1,
                    force=bool(existing),
                )
            rows.extend(existing)
        return [row for row in rows if row.week_start and date_from <= row.week_start <= date_to]

    def _build_weekly_metric_series_from_rows(
        self,
        rows: list[FieldFeatureWeekly],
    ) -> dict[str, list[dict[str, Any]]]:
        metric_specs = {
            "ndvi": ("ndvi_mean", "satellite_coverage"),
            "ndmi": ("ndmi_mean", "satellite_coverage"),
            "ndwi": ("ndwi_mean", "satellite_coverage"),
            "bsi": ("bsi_mean", "satellite_coverage"),
            "gdd": ("gdd", "weather_coverage"),
            "vpd": ("vpd_kpa", "weather_coverage"),
            "soil_moisture": ("soil_moisture", "weather_coverage"),
            "precipitation": ("precipitation_mm", "weather_coverage"),
            "wind": ("wind_speed_m_s", "weather_coverage"),
        }
        result: dict[str, list[dict[str, Any]]] = {}
        for metric, (value_attr, coverage_attr) in metric_specs.items():
            points: list[dict[str, Any]] = []
            for row in rows:
                value = getattr(row, value_attr, None)
                if value is None:
                    continue
                coverage = getattr(row, coverage_attr, None)
                points.append(
                    {
                        "observed_at": row.week_start.isoformat(),
                        "value": round(float(value), 4),
                        "coverage": round(float(coverage), 3) if coverage is not None else None,
                        "source": getattr(row, "source", None),
                    }
                )
            if not points:
                continue
            smoothed = self._smooth_points([float(point["value"]) for point in points], window=3)
            for point, smooth in zip(points, smoothed):
                point["smoothed"] = round(float(smooth), 4)
            result[metric] = points

        # Cumulative GDD series from weekly GDD values
        if "gdd" in result:
            cumulative = 0.0
            cumulative_points: list[dict[str, Any]] = []
            for point in result["gdd"]:
                cumulative += float(point["value"])
                cumulative_points.append({
                    "observed_at": point["observed_at"],
                    "value": round(cumulative, 1),
                    "smoothed": round(cumulative, 1),
                    "coverage": point.get("coverage"),
                    "source": point.get("source"),
                })
            result["gdd_cumulative"] = cumulative_points

        return result

    async def _build_data_status(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date,
        date_to: date,
        weekly_series: dict[str, list[dict[str, Any]]],
        weekly_rows: list[FieldFeatureWeekly],
    ) -> dict[str, Any]:
        metric_point_counts = {
            metric: len(points)
            for metric, points in weekly_series.items()
        }
        actual_range = self._actual_range_from_series(weekly_series)
        range_days = (date_to - date_from).days + 1
        max_points = max(metric_point_counts.values(), default=0)
        has_backfill_rows = await self._has_backfill_rows(
            field_id,
            organization_id=organization_id,
            date_from=date_from,
            date_to=date_to,
        )
        code = "ready"
        message_code = "temporal_ready"
        if range_days > TEMPORAL_BACKFILL_MAX_DAYS:
            code = "range_exceeds_limit"
            message_code = "temporal_range_exceeds_limit"
        elif max_points < TEMPORAL_RANGE_READY_MIN_POINTS:
            if range_days >= 45 and not has_backfill_rows:
                code = "backfill_required"
                message_code = "temporal_backfill_required"
            elif has_backfill_rows:
                code = "historical_data_sparse"
                message_code = "temporal_historical_data_sparse"
            else:
                code = "insufficient_points_current_season"
                message_code = "temporal_insufficient_points_current_season"
        elif not weekly_rows:
            code = "no_history_available"
            message_code = "temporal_no_history_available"
        # Count satellite observation days from NDVI series (proxy for clear-sky coverage)
        ndvi_points = weekly_series.get("ndvi") or []
        days_with_observations = len(ndvi_points)

        return {
            "code": code,
            "message_code": message_code,
            "requested_range": {
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
                "days": range_days,
            },
            "actual_range": actual_range,
            "backfill_required": code == "backfill_required",
            "backfill_in_progress": False,
            "metric_point_counts": metric_point_counts,
            "days_with_observations": days_with_observations,
        }

    @staticmethod
    def _actual_range_from_series(weekly_series: dict[str, list[dict[str, Any]]]) -> dict[str, str | None]:
        all_dates = [
            str(point.get("observed_at"))
            for points in weekly_series.values()
            for point in points
            if point.get("observed_at")
        ]
        if not all_dates:
            return {"date_from": None, "date_to": None}
        return {"date_from": min(all_dates), "date_to": max(all_dates)}

    async def _has_backfill_rows(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date,
        date_to: date,
    ) -> bool:
        stmt = (
            select(func.count(FieldMetricSeries.id))
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id == field_id)
            .where(FieldMetricSeries.source == TEMPORAL_BACKFILL_SOURCE)
            .where(FieldMetricSeries.observed_at >= date_from)
            .where(FieldMetricSeries.observed_at <= date_to + timedelta(days=1))
        )
        return bool(int(_result_scalar_one_or_none(await self.db.execute(stmt)) or 0))

    async def _backfill_satellite_metric_series(
        self,
        *,
        field: Field,
        organization_id: UUID,
        date_from: date,
        date_to: date,
        season_year: int,
        client: SentinelHubClient,
        shared_client: Any,
        progress_callback: Any | None = None,
        window_offset: int = 0,
        window_total: int = 0,
    ) -> int:
        await self.db.execute(
            delete(FieldMetricSeries)
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id == field.id)
            .where(FieldMetricSeries.source == TEMPORAL_BACKFILL_SOURCE)
            .where(FieldMetricSeries.observed_at >= date_from)
            .where(FieldMetricSeries.observed_at <= date_to + timedelta(days=1))
        )

        field_geom = to_shape(field.geom)
        bbox = tuple(field_geom.bounds)
        windows = self._biweekly_windows(date_from, date_to)
        saved_windows = 0

        for index, (window_from, window_to) in enumerate(windows, start=1):
            absolute_index = window_offset + index
            if window_total:
                progress = min(88, 16 + round((absolute_index / max(window_total, 1)) * 60))
            else:
                progress = min(88, 16 + round((index / max(len(windows), 1)) * 60))
            await self._emit_progress(
                progress_callback,
                progress=progress,
                stage_code="satellite_backfill",
                stage_label="satellite-backfill",
                stage_detail=f"window {index}/{len(windows)}",
                stage_detail_code="materializing_biweekly_window",
                stage_detail_params={
                    "season_year": season_year,
                    "window_index": index,
                    "window_total": len(windows),
                    "window_from": window_from.isoformat(),
                    "window_to": window_to.isoformat(),
                },
                estimated_remaining_s=max(1, len(windows) - index),
            )
            window_metrics = await self._compute_field_window_metrics(
                field_geom=field_geom,
                bbox=bbox,
                date_from=window_from,
                date_to=window_to,
                client=client,
                shared_client=shared_client,
            )
            if not window_metrics:
                continue
            observed_at = datetime.combine(
                window_from + timedelta(days=(window_to - window_from).days // 2),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            for metric, payload in window_metrics.items():
                self.db.add(
                    FieldMetricSeries(
                        organization_id=organization_id,
                        field_id=field.id,
                        aoi_run_id=field.aoi_run_id,
                        metric=metric,
                        observed_at=observed_at,
                        value_mean=payload.get("mean"),
                        value_min=payload.get("min"),
                        value_max=payload.get("max"),
                        value_median=payload.get("median"),
                        value_p25=payload.get("p25"),
                        value_p75=payload.get("p75"),
                        coverage=payload.get("coverage"),
                        source=TEMPORAL_BACKFILL_SOURCE,
                        meta={
                            "season_year": season_year,
                            "window_from": window_from.isoformat(),
                            "window_to": window_to.isoformat(),
                            "provider_account": client.last_account_alias,
                        },
                    )
                )
            saved_windows += 1

        await self.db.commit()
        return saved_windows

    async def _compute_field_window_metrics(
        self,
        *,
        field_geom: Any,
        bbox: tuple[float, float, float, float],
        date_from: date,
        date_to: date,
        client: SentinelHubClient,
        shared_client: Any,
    ) -> dict[str, dict[str, float]]:
        minx, miny, maxx, maxy = bbox
        width = TEMPORAL_RASTER_SIZE
        height = TEMPORAL_RASTER_SIZE
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        field_mask = rasterize(
            [(mapping(field_geom), 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        ).astype(bool)
        if not np.any(field_mask):
            return {}

        try:
            payload = await client.fetch_tile_harmonized(
                bbox=bbox,
                time_from=f"{date_from.isoformat()}T00:00:00Z",
                time_to=f"{date_to.isoformat()}T23:59:59Z",
                width=width,
                height=height,
                max_cloud_pct=TEMPORAL_DEFAULT_CLOUD_PCT,
                client=shared_client,
            )
        except Exception as exc:
            raise TemporalBackfillError(self._classify_backfill_error(exc), str(exc)) from exc

        scl = np.asarray(payload.get("SCL"), dtype=np.uint8)
        valid_mask = field_mask & np.isin(scl, list(TEMPORAL_VALID_SCL_CLASSES))
        valid_pixels = int(np.count_nonzero(valid_mask))
        if valid_pixels < 12:
            return {}

        metric_map = {
            "ndvi": payload.get("NDVI_idx"),
            "ndmi": payload.get("NDMI_idx"),
            "ndwi": payload.get("NDWI_idx"),
            "bsi": payload.get("BSI_idx"),
        }
        result: dict[str, dict[str, float]] = {}
        for metric, raster in metric_map.items():
            if raster is None:
                continue
            values = np.asarray(raster, dtype=np.float32)[valid_mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            result[metric] = {
                "mean": round(float(np.mean(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "median": round(float(np.median(values)), 4),
                "p25": round(float(np.percentile(values, 25)), 4),
                "p75": round(float(np.percentile(values, 75)), 4),
                "coverage": round(valid_pixels / max(int(np.count_nonzero(field_mask)), 1), 4),
            }
        return result

    @staticmethod
    def _classify_backfill_error(exc: Exception) -> str:
        text = str(exc).lower()
        if "insufficient processing units" in text or "access_insufficient_processing_units" in text:
            return "source_unavailable_quota"
        if "rate limit" in text or "rate_limit_exceeded" in text or "429" in text:
            return "backfill_delayed"
        return "backfill_failed"

    async def _load_field_metric_series(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date | None,
        date_to: date | None,
    ) -> dict[str, list[WeeklyPoint]]:
        stmt = (
            select(FieldMetricSeries)
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id == field_id)
            .order_by(FieldMetricSeries.metric.asc(), FieldMetricSeries.observed_at.asc())
        )
        if date_from is not None:
            stmt = stmt.where(FieldMetricSeries.observed_at >= date_from)
        if date_to is not None:
            stmt = stmt.where(FieldMetricSeries.observed_at <= date_to)
        rows = _result_scalars_all(await self.db.execute(stmt))
        grouped: dict[str, list[WeeklyPoint]] = defaultdict(list)
        for row in rows:
            value = row.value_mean if row.value_mean is not None else row.value_median
            if value is None:
                continue
            metric = str(row.metric)
            grouped[metric].append(
                WeeklyPoint(
                    observed_at=row.observed_at.date(),
                    value=float(value),
                    source=str(row.source),
                    coverage=float(row.coverage) if row.coverage is not None else None,
                )
            )
        return grouped

    def _build_weekly_metric_series(self, grouped: dict[str, list[WeeklyPoint]]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for metric, items in grouped.items():
            by_week: dict[date, list[WeeklyPoint]] = defaultdict(list)
            for item in items:
                week_start = item.observed_at - timedelta(days=item.observed_at.weekday())
                by_week[week_start].append(item)
            points = []
            for week_start, values in sorted(by_week.items(), key=lambda pair: pair[0]):
                mean_value = float(np.mean([row.value for row in values]))
                mean_coverage = (
                    float(np.mean([row.coverage for row in values if row.coverage is not None]))
                    if any(row.coverage is not None for row in values)
                    else None
                )
                points.append(
                    {
                        "observed_at": week_start.isoformat(),
                        "value": round(mean_value, 4),
                        "coverage": round(mean_coverage, 2) if mean_coverage is not None else None,
                        "source": values[-1].source,
                    }
                )
            smoothed = self._smooth_points([float(point["value"]) for point in points], window=3)
            for point, smooth in zip(points, smoothed):
                point["smoothed"] = round(float(smooth), 4)
            result[metric] = points
        return result

    async def compute_ndvi_auc(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
    ) -> dict[str, float | None]:
        """Compute cumulative NDVI (area under curve) for the current growing season.

        Returns dict with:
          ndvi_auc — trapezoidal integral of weekly smoothed NDVI over the season
          ndvi_peak — peak smoothed NDVI value
          ndvi_season_weeks — number of weeks with data
          ndvi_mean_season — mean smoothed NDVI over the season
        """
        now = datetime.now(timezone.utc).date()
        # Growing season: approx March 1 to current date (or Oct 31 at latest)
        season_start = date(now.year, 3, 1)
        season_end = min(now, date(now.year, 10, 31))
        if now.month < 3:
            # Before March — use previous year's season
            season_start = date(now.year - 1, 3, 1)
            season_end = date(now.year - 1, 10, 31)

        weekly_rows = await self._load_weekly_rows_for_range(
            field_id,
            organization_id=organization_id,
            date_from=season_start,
            date_to=season_end,
        )
        weekly = self._build_weekly_metric_series_from_rows(weekly_rows)
        ndvi_points = weekly.get("ndvi") or []

        if not ndvi_points:
            return {
                "ndvi_auc": None,
                "ndvi_peak": None,
                "ndvi_season_weeks": 0,
                "ndvi_mean_season": None,
            }

        values = [float(p.get("smoothed") or p.get("value") or 0.0) for p in ndvi_points]
        dates = [date.fromisoformat(str(p["observed_at"])) for p in ndvi_points]
        peak = round(float(max(values)), 4)
        mean_val = round(float(np.mean(values)), 4)

        if len(values) < 2:
            return {
                "ndvi_auc": None,
                "ndvi_peak": peak,
                "ndvi_season_weeks": len(values),
                "ndvi_mean_season": mean_val,
            }

        # Trapezoidal integration (weeks as unit)
        auc = 0.0
        for i in range(1, len(values)):
            dt_weeks = (dates[i] - dates[i - 1]).days / 7.0
            auc += 0.5 * (values[i - 1] + values[i]) * dt_weeks

        return {
            "ndvi_auc": round(auc, 4),
            "ndvi_peak": peak,
            "ndvi_season_weeks": len(values),
            "ndvi_mean_season": mean_val,
        }

    def _build_phenology(self, ndvi_points: list[dict[str, Any]], crop_code: str | None) -> dict[str, Any]:
        if len(ndvi_points) < 5:
            return {"supported": False, "stage": "unsupported", "stage_label": "Недостаточно точек NDVI"}
        values = np.asarray([float(point.get("smoothed") or point.get("value") or 0.0) for point in ndvi_points], dtype=float)
        dates = [date.fromisoformat(str(point["observed_at"])) for point in ndvi_points]
        peak_idx = int(np.argmax(values))
        peak_val = float(values[peak_idx])
        base_val = float(np.min(values))
        amplitude = max(peak_val - base_val, 1e-6)
        sos_threshold = base_val + amplitude * 0.2
        eos_threshold = base_val + amplitude * 0.25
        sos_idx = next((idx for idx, value in enumerate(values) if value >= sos_threshold), 0)
        eos_idx = next((idx for idx in range(len(values) - 1, peak_idx - 1, -1) if values[idx] >= eos_threshold), len(values) - 1)
        current_idx = len(values) - 1
        stage = "vegetative"
        if current_idx <= sos_idx + 1:
            stage = "emergence"
        elif current_idx < peak_idx - 1:
            stage = "vegetative"
        elif current_idx <= peak_idx + 1:
            stage = "reproductive"
        elif current_idx < eos_idx:
            stage = "grain_fill"
        else:
            stage = "senescence"
        profile = _crop_profile(crop_code)
        peak_week = peak_idx + 1
        lag_weeks = round(float(peak_week - profile["norm_peak_week"]), 2)
        return {
            "supported": True,
            "crop_code": crop_code,
            "stage": stage,
            "stage_label": _phenology_stage_label(stage),
            "sos": dates[sos_idx].isoformat(),
            "peak": dates[peak_idx].isoformat(),
            "eos": dates[eos_idx].isoformat(),
            "peak_value": round(peak_val, 4),
            "seasonal_amplitude": round(amplitude, 4),
            "season_length_weeks": int(max(eos_idx - sos_idx + 1, 1)),
            "lag_weeks_vs_norm": lag_weeks,
            "norm_peak_week": profile["norm_peak_week"],
        }

    def _build_anomalies(self, weekly_series: dict[str, list[dict[str, Any]]], phenology: dict[str, Any]) -> list[dict[str, Any]]:
        ndvi_points = weekly_series.get("ndvi") or []
        if len(ndvi_points) < 5:
            return []
        values = np.asarray([float(point.get("smoothed") or point.get("value") or 0.0) for point in ndvi_points], dtype=float)
        diffs = np.diff(values)
        value_z = _robust_zscores(values)
        diff_z = _robust_zscores(diffs) if diffs.size else np.asarray([])
        anomalies: list[dict[str, Any]] = []
        for index in range(1, len(ndvi_points)):
            dz = float(diff_z[index - 1]) if diff_z.size >= index else 0.0
            vz = float(value_z[index])
            severity = None
            kind = None
            reason = None
            ndmi_points = weekly_series.get("ndmi") or []
            ndmi_val = float(ndmi_points[index].get("smoothed") or ndmi_points[index].get("value") or 0.0) if len(ndmi_points) > index else 0.0
            if dz <= -1.8 and vz <= -1.25:
                kind = "rapid_canopy_loss"
                severity = "critical" if dz <= -2.4 else "warning"
                reason = "NDVI упал быстрее нормы на недельной производной."
                if ndmi_val < -0.05:
                    kind = "possible_drought_stress"
                    reason = "Падение NDVI сопровождается сухим canopy-сигналом NDMI."
            elif dz >= 1.4 and phenology.get("stage") == "emergence":
                kind = "delayed_development"
                severity = "info"
                reason = "Развитие культуры смещено относительно ранней фенологической нормы."
            if kind:
                anomalies.append(
                    {
                        "metric": "ndvi",
                        "kind": kind,
                        "label": _anomaly_label(kind),
                        "severity": severity,
                        "observed_at": ndvi_points[index]["observed_at"],
                        "z_score": round(vz, 3),
                        "delta_z_score": round(dz, 3),
                        "reason": reason,
                    }
                )
        if anomalies and len(anomalies) >= 2:
            last_two = anomalies[-2:]
            if last_two[0]["kind"] == last_two[1]["kind"]:
                last_two[1]["persistent"] = True
        return anomalies

    async def _build_history_trend(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
    ) -> dict[str, Any]:
        if not crop_code:
            return {"supported": False, "points": []}
        stmt = (
            select(FieldSeason.season_year, YieldObservation.yield_kg_ha)
            .join(FieldSeason, YieldObservation.field_season_id == FieldSeason.id)
            .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            .where(YieldObservation.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(CropAssignment.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .where(CropAssignment.crop_code == crop_code)
            .order_by(FieldSeason.season_year.asc())
        )
        rows = (await self.db.execute(stmt)).all()
        if not rows:
            return {"supported": False, "points": []}
        points: list[dict[str, Any]] = []
        yields: list[float] = []
        years: list[float] = []
        for season_year, yield_kg_ha in rows:
            years.append(float(season_year))
            yields.append(float(yield_kg_ha))
        rolling = []
        for index, value in enumerate(yields):
            window = yields[max(0, index - 2): index + 1]
            rolling.append(round(float(np.mean(window)), 2))
        for index, (season_year, yield_kg_ha) in enumerate(rows):
            points.append(
                {
                    "year": int(season_year),
                    "observed_yield_kg_ha": round(float(yield_kg_ha), 2),
                    "rolling_mean_kg_ha": rolling[index],
                }
            )
        slope = 0.0
        if len(years) >= 2:
            x = np.asarray(years, dtype=float)
            y = np.asarray(yields, dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
        mean_yield = float(np.mean(yields))
        stability = float(np.std(yields) / mean_yield) if mean_yield > 0 else 0.0
        return {
            "supported": True,
            "points": points,
            "trend_slope_kg_ha_per_year": round(slope, 2),
            "stability_cv": round(stability, 4),
            "latest_yield_kg_ha": round(float(yields[-1]), 2),
            "rolling_mean_latest_kg_ha": rolling[-1],
        }

    async def _build_water_balance(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
        phenology: dict[str, Any],
        weekly_rows: list[FieldFeatureWeekly],
    ) -> dict[str, Any]:
        if not crop_code:
            return {"supported": False, "model": "FAO-lite", "series": [], "summary": {"reason": "Культура не определена"}}
        weather_rows = [
            row for row in weekly_rows
            if any(
                value is not None
                for value in (row.tmean_c, row.precipitation_mm, row.vpd_kpa, row.wind_speed_m_s, row.gdd)
            )
        ]
        if len(weather_rows) < 2:
            return {"supported": False, "model": "FAO-lite", "series": [], "summary": {"reason": "Недостаточно погодных рядов"}}
        soil = await self._latest_soil(field_id, organization_id=organization_id)
        taw_mm, raw_fraction = _soil_capacity(soil.texture_class if soil else None)
        profile = _crop_profile(crop_code)
        stage = str(phenology.get("stage") or "vegetative")
        kc = profile["kc"].get(stage, 1.0)
        storage = taw_mm * 0.62
        series = []
        et_model_used = "empirical"
        for row in weather_rows:
            temp = float(row.tmean_c or 0.0)
            vpd = float(row.vpd_kpa or 0.0)
            wind = float(row.wind_speed_m_s or 2.0)
            gdd = float(row.gdd or 0.0) / 7.0
            precip = float(row.precipitation_mm or 0.0)
            solar_rad = float(row.solar_radiation_mj or 0.0)  # weekly MJ/m²
            if solar_rad > 0.0:
                # Priestley-Taylor (1972) reference ET, adapted for weekly data.
                # ET_PT = 1.26 * (Δ/(Δ+γ)) * Rn/λ   [mm/day]
                # Δ = slope of saturation vapour pressure curve (kPa/°C)
                # γ = 0.067 kPa/°C (psychrometric constant at sea level)
                # λ = 2.45 MJ/kg  (latent heat of vaporisation)
                # Rn ≈ 0.77 * Rs  (net radiation; albedo ~0.23 for vegetated surface)
                # Source: Allen et al. 1998 (FAO-56), Priestley & Taylor 1972
                es = 0.6108 * float(np.exp(17.27 * temp / max(temp + 237.3, 1e-6)))
                delta = 4098.0 * es / max((temp + 237.3) ** 2, 1e-6)
                gamma = 0.067
                lam = 2.45
                rn_daily = solar_rad * 0.77 / 7.0  # MJ/m²/day
                et_pt_daily = 1.26 * (delta / max(delta + gamma, 1e-6)) * (rn_daily / lam)
                et_proxy = max(1.0, et_pt_daily)
                et_model_used = "priestley-taylor"
            else:
                # Fallback empirical proxy when solar radiation is missing
                et_proxy = max(1.0, 1.5 + 0.12 * temp + 2.8 * vpd + 0.18 * wind + 0.003 * gdd)
            demand = et_proxy * kc * 7.0
            storage = storage + precip * 0.82 - demand
            drainage = 0.0
            if storage > taw_mm:
                drainage = storage - taw_mm
                storage = taw_mm
            storage = max(0.0, storage)
            deficit = max(0.0, taw_mm - storage)
            stress_ratio = deficit / max(taw_mm, 1.0)
            point: dict[str, Any] = {
                "observed_at": row.week_start.isoformat(),
                "storage_mm": round(storage, 2),
                "deficit_mm": round(deficit, 2),
                "demand_mm": round(demand, 2),
                "precipitation_mm": round(precip, 2),
                "drainage_mm": round(drainage, 2),
                "stress_ratio": round(stress_ratio, 3),
                "et_model": et_model_used,
            }
            if solar_rad > 0.0:
                point["solar_radiation_mj_day"] = round(solar_rad / 7.0, 2)
            series.append(point)
        latest = series[-1]
        irrigation_need_class = "низкая"
        if latest["stress_ratio"] >= 0.55:
            irrigation_need_class = "высокая"
        elif latest["stress_ratio"] >= 0.3:
            irrigation_need_class = "умеренная"
        stress_class = "норма"
        if latest["stress_ratio"] >= 0.6:
            stress_class = "сильный дефицит"
        elif latest["stress_ratio"] >= 0.35:
            stress_class = "умеренный дефицит"
        return {
            "supported": True,
            "model": f"FAO-lite/{et_model_used}",
            "series": series,
            "summary": {
                "root_zone_storage_mm": latest["storage_mm"],
                "deficit_mm": latest["deficit_mm"],
                "stress_ratio": latest["stress_ratio"],
                "stress_class": stress_class,
                "irrigation_need_class": irrigation_need_class,
                "taw_mm": round(taw_mm, 2),
                "raw_mm": round(taw_mm * raw_fraction, 2),
                "kc_proxy": round(kc, 3),
            },
            "et_method": et_model_used,
            "solar_radiation_available": any(pt.get("solar_radiation_mj_day") is not None for pt in series),
            "notes": [
                "FAO-lite root-zone bucket: ET рассчитывается как погодный proxy.",
                (
                    "ET рассчитан по методу Priestley-Taylor (1972) с использованием солнечной радиации ERA5."
                    if et_model_used == "priestley-taylor"
                    else "Солнечная радиация ERA5 недоступна — используется эмпирический proxy ET (температура+VPD+ветер)."
                ),
            ],
        }

    def _build_risk_summary(
        self,
        *,
        crop_code: str | None,
        phenology: dict[str, Any],
        water_balance: dict[str, Any],
        anomalies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        items = []
        if water_balance.get("supported"):
            stress_ratio = float((water_balance.get("summary") or {}).get("stress_ratio") or 0.0)
            if stress_ratio >= 0.55:
                items.append(
                    {
                        "id": "drought_risk",
                        "label": "Риск засухи",
                        "level": "высокий",
                        "score": round(min(1.0, stress_ratio), 3),
                        "reason": "Корневой слой по FAO-lite находится в выраженном дефиците.",
                    }
                )
            elif stress_ratio >= 0.32:
                items.append(
                    {
                        "id": "drought_risk",
                        "label": "Риск засухи",
                        "level": "умеренный",
                        "score": round(min(1.0, stress_ratio), 3),
                        "reason": "Нарастает водный дефицит в корневом слое.",
                    }
                )
        if any(item.get("kind") == "possible_drought_stress" for item in anomalies):
            items.append(
                {
                    "id": "vegetation_stress",
                    "label": "Растительный стресс",
                    "level": "умеренный",
                    "score": 0.64,
                    "reason": "NDVI и NDMI показывают негативную совместную динамику.",
                }
            )
        stage = str(phenology.get("stage") or "")
        if crop_code in {"wheat", "barley"} and stage in {"reproductive", "grain_fill"}:
            humidity_score = 0.0
            wb_series = water_balance.get("series") or []
            if wb_series:
                wet_days = sum(1 for row in wb_series[-10:] if float(row.get("precipitation_mm") or 0.0) >= 4.0)
                humidity_score = min(1.0, wet_days / 6.0)
            if humidity_score >= 0.5:
                items.append(
                    {
                        "id": "fhb_risk",
                        "label": "FHB risk",
                        "level": "умеренный",
                        "score": round(humidity_score, 3),
                        "reason": "Зерновая культура находится в чувствительной фазе при влажном сценарии.",
                    }
                )
        return {"supported": bool(items), "items": items}

    async def _latest_weather_rows(self, field_id: UUID, *, organization_id: UUID) -> list[WeatherDaily]:
        season_stmt = (
            select(FieldSeason.id)
            .where(FieldSeason.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .order_by(desc(FieldSeason.season_year))
            .limit(1)
        )
        season_id = _result_scalar_one_or_none(await self.db.execute(season_stmt))
        if season_id is None:
            return []
        stmt = (
            select(WeatherDaily)
            .where(WeatherDaily.organization_id == organization_id)
            .where(WeatherDaily.field_season_id == season_id)
            .order_by(WeatherDaily.observed_on.asc())
        )
        return _result_scalars_all(await self.db.execute(stmt))

    async def _latest_soil(self, field_id: UUID, *, organization_id: UUID) -> SoilProfile | None:
        stmt = (
            select(SoilProfile)
            .where(SoilProfile.organization_id == organization_id)
            .where(SoilProfile.field_id == field_id)
            .order_by(desc(SoilProfile.sampled_at))
            .limit(1)
        )
        return _result_scalar_one_or_none(await self.db.execute(stmt))

    def _build_zone_scores(self, cells: list[GridCell]) -> list[dict[str, Any]]:
        metric_vectors = {
            "ndvi": np.asarray([_safe_float(cell.ndvi_mean) for cell in cells], dtype=float),
            "ndmi": np.asarray([_safe_float(cell.ndmi_mean) for cell in cells], dtype=float),
            "bsi": np.asarray([_safe_float(cell.bsi_mean) for cell in cells], dtype=float),
            "soil_moisture": np.asarray([_safe_float(cell.soil_moist) for cell in cells], dtype=float),
            "vpd": np.asarray([_safe_float(cell.vpd_mean) for cell in cells], dtype=float),
            "wind": np.asarray([_safe_float(cell.wind_speed_m_s) for cell in cells], dtype=float),
        }
        ndvi_z = _robust_zscores(metric_vectors["ndvi"])
        ndmi_z = _robust_zscores(metric_vectors["ndmi"])
        bsi_z = _robust_zscores(metric_vectors["bsi"])
        soil_z = _robust_zscores(metric_vectors["soil_moisture"])
        vpd_z = _robust_zscores(metric_vectors["vpd"])
        wind_z = _robust_zscores(metric_vectors["wind"])
        scores = []
        for index, cell in enumerate(cells):
            score = (
                0.45 * float(ndvi_z[index])
                + 0.18 * float(ndmi_z[index])
                - 0.17 * float(bsi_z[index])
                + 0.12 * float(soil_z[index])
                - 0.05 * float(vpd_z[index])
                - 0.03 * float(wind_z[index])
            )
            scores.append({"cell": cell, "score": float(score)})
        return scores

    def _build_zone_payloads(
        self,
        *,
        field: Field,
        cells: list[GridCell],
        scores: list[dict[str, Any]],
        baseline_prediction: float,
        exact_yield_mode: bool,
        baseline_confidence: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw_scores = np.asarray([item["score"] for item in scores], dtype=float)
        q1, q2 = np.percentile(raw_scores, [33.333, 66.666])
        zone_keys = []
        for item in scores:
            score = float(item["score"])
            if score <= q1:
                zone_keys.append("low")
            elif score <= q2:
                zone_keys.append("medium")
            else:
                zone_keys.append("high")

        raw_multipliers = []
        for score in raw_scores:
            raw_multipliers.append(float(np.clip(1.0 + score * 0.12, 0.75, 1.25)))
        area_weights = np.asarray([max(_safe_float(item["cell"].field_coverage), 0.01) for item in scores], dtype=float)
        if exact_yield_mode and baseline_prediction > 0:
            weighted_mean = float(np.average(raw_multipliers, weights=area_weights))
            normalized_multipliers = [mult / max(weighted_mean, 1e-6) for mult in raw_multipliers]
        else:
            normalized_multipliers = raw_multipliers

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        total_weight = float(np.sum(area_weights))
        field_area_m2 = float(field.area_m2 or 0.0)
        features = []
        for index, item in enumerate(scores):
            cell = item["cell"]
            zone_code = zone_keys[index]
            multiplier = float(normalized_multipliers[index])
            predicted_yield = round(baseline_prediction * multiplier, 2) if exact_yield_mode and baseline_prediction > 0 else None
            score = float(item["score"])
            cell_weight = max(_safe_float(cell.field_coverage), 0.01)
            cell_area_m2 = field_area_m2 * cell_weight / max(total_weight, 1e-6)
            grouped[zone_code].append(
                {
                    "cell": cell,
                    "score": score,
                    "multiplier": multiplier,
                    "predicted_yield": predicted_yield,
                    "area_m2": cell_area_m2,
                }
            )
            feature_geom = mapping(to_shape(cell.geom))
            features.append(
                {
                    "type": "Feature",
                    "geometry": feature_geom,
                    "properties": {
                        "kind": "management_zone",
                        "zone_code": zone_code,
                        "zone_label": ZONE_LABELS[zone_code],
                        "zone_score": round(score, 4),
                        "area_m2": round(cell_area_m2, 2),
                        "predicted_yield_kg_ha": predicted_yield,
                        "yield_potential_kg_ha": predicted_yield,
                        "confidence": round(max(0.42, min(0.92, baseline_confidence if exact_yield_mode else 0.58)), 3),
                    },
                }
            )

        zone_payloads = []
        for zone_code in ("high", "medium", "low"):
            items = grouped.get(zone_code) or []
            if not items:
                continue
            zone_weight = sum(max(_safe_float(item["cell"].field_coverage), 0.01) for item in items)
            zone_area_m2 = field_area_m2 * zone_weight / max(total_weight, 1e-6)
            zone_payloads.append(
                {
                    "zone_code": zone_code,
                    "label": ZONE_LABELS[zone_code],
                    "cell_count": len(items),
                    "area_m2": round(zone_area_m2, 2),
                    "area_share_pct": round(zone_weight / max(total_weight, 1e-6) * 100.0, 2),
                    "mean_score": round(float(np.mean([item["score"] for item in items])), 4),
                    "predicted_yield_kg_ha": (
                        round(float(np.mean([item["predicted_yield"] for item in items if item["predicted_yield"] is not None])), 2)
                        if exact_yield_mode and any(item["predicted_yield"] is not None for item in items)
                        else None
                    ),
                    "confidence": round(max(0.42, min(0.92, baseline_confidence if exact_yield_mode else 0.58)), 3),
                }
            )
        return zone_payloads, features

    @staticmethod
    def _smooth_points(values: list[float], *, window: int = 3) -> list[float]:
        if len(values) <= 2:
            return list(values)
        arr = np.asarray(values, dtype=float)
        smoothed: list[float] = []
        radius = max(1, window // 2)
        for index in range(arr.size):
            start = max(0, index - radius)
            end = min(arr.size, index + radius + 1)
            smoothed.append(float(np.mean(arr[start:end])))
        return smoothed


def _crop_profile(crop_code: str | None) -> dict[str, Any]:
    code = str(crop_code or "").lower()
    if code in {"wheat", "barley", "rye"}:
        return {"norm_peak_week": 9, "kc": {"emergence": 0.35, "vegetative": 0.78, "reproductive": 1.08, "grain_fill": 0.92, "senescence": 0.55}}
    if code in {"corn", "maize"}:
        return {"norm_peak_week": 10, "kc": {"emergence": 0.3, "vegetative": 0.82, "reproductive": 1.12, "grain_fill": 0.96, "senescence": 0.55}}
    if code in {"soy", "soybean"}:
        return {"norm_peak_week": 9, "kc": {"emergence": 0.28, "vegetative": 0.8, "reproductive": 1.02, "grain_fill": 0.9, "senescence": 0.5}}
    if code in {"sunflower"}:
        return {"norm_peak_week": 8, "kc": {"emergence": 0.32, "vegetative": 0.74, "reproductive": 1.0, "grain_fill": 0.86, "senescence": 0.52}}
    return {"norm_peak_week": 9, "kc": {"emergence": 0.3, "vegetative": 0.75, "reproductive": 1.0, "grain_fill": 0.88, "senescence": 0.5}}


def _phenology_stage_label(stage: str) -> str:
    return {
        "emergence": "Всходы",
        "vegetative": "Вегетативный рост",
        "reproductive": "Репродуктивная фаза",
        "grain_fill": "Налив",
        "senescence": "Сенесценция",
    }.get(stage, "Неизвестная стадия")


def _anomaly_label(kind: str) -> str:
    return {
        "rapid_canopy_loss": "Быстрая потеря зелёной массы",
        "possible_drought_stress": "Возможный стресс засухи",
        "possible_waterlogging": "Возможное переувлажнение",
        "possible_disease": "Возможное заболевание/вредитель",
        "delayed_development": "Смещение развития",
    }.get(kind, kind)


def _robust_zscores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    if mad <= 1e-9:
        std = float(np.nanstd(arr))
        if std <= 1e-9:
            return np.zeros_like(arr)
        return (arr - float(np.nanmean(arr))) / std
    return 0.6745 * (arr - median) / mad


def _soil_capacity(texture_class: str | None) -> tuple[float, float]:
    texture = str(texture_class or "").strip().lower()
    if "sand" in texture:
        return 75.0, 0.45
    if "clay" in texture:
        return 155.0, 0.38
    if "silt" in texture:
        return 135.0, 0.42
    return 120.0, 0.4


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _result_scalar_one_or_none(result: Any) -> Any | None:
    if hasattr(result, "scalar_one_or_none"):
        return result.scalar_one_or_none()
    if hasattr(result, "first"):
        row = result.first()
        if isinstance(row, tuple):
            return row[0] if row else None
        return row
    return None


def _result_scalars_all(result: Any) -> list[Any]:
    if hasattr(result, "scalars"):
        return list(result.scalars().all())
    if hasattr(result, "all"):
        rows = list(result.all() or [])
        if rows and isinstance(rows[0], tuple) and len(rows[0]) == 1:
            return [row[0] for row in rows]
        return rows
    return []
