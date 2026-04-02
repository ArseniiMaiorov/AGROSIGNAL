"""Сервис аналитики по полям и группам полей."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

import numpy as np
from geoalchemy2.shape import to_shape
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.forecast_curve import build_forecast_curve_points
from services.field_quality import describe_field_quality, extract_runtime_geometry_quality
from services.message_codes import classify_support_reason
from services.temporal_analytics_service import GEOMETRY_FOUNDATION, TemporalAnalyticsService, normalize_driver_breakdown
from services.trust_service import describe_prediction_operational_tier
from services.payload_meta import build_freshness
from services.weather_service import WeatherService
from storage.db import AoiRun, ArchiveEntry, Crop, Field, FieldFeatureWeekly, FieldMetricSeries, GridCell, ScenarioRun, YieldPrediction

METRIC_LABELS: dict[str, str] = {
    "ndvi": "NDVI",
    "ndmi": "NDMI",
    "ndwi": "NDWI",
    "bsi": "BSI",
    "ndre": "NDRE",
    "gdd": "GDD",
    "vpd": "VPD",
    "soil_moisture": "Влага почвы",
    "precipitation": "Осадки",
    "wind": "Ветер",
    "solar_radiation": "Солнечная радиация",
}

GRID_METRIC_COLUMNS: dict[str, str] = {
    "ndvi": "ndvi_mean",
    "ndmi": "ndmi_mean",
    "ndwi": "ndwi_mean",
    "bsi": "bsi_mean",
    "ndre": "ndre_mean",
    "gdd": "gdd_sum",
    "vpd": "vpd_mean",
    "soil_moisture": "soil_moist",
    "precipitation": "precipitation_mm",
    "wind": "wind_speed_m_s",
}

HISTOGRAM_METRICS: tuple[str, ...] = ("ndvi", "ndmi", "soil_moisture", "vpd")


class FieldAnalyticsService:
    """Агрегация метрик, рядов и архивных срезов для UI поля."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.weather_service = WeatherService(db)

    async def get_field_dashboard(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date | None = None,
        date_to: date | None = None,
        lite: bool = False,
    ) -> dict[str, Any]:
        field = (
            await self.db.execute(
                select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
            )
        ).scalar_one_or_none()
        if field is None:
            raise ValueError("Поле не найдено")

        runtime_map = await self._load_run_runtime_map([field.aoi_run_id], organization_id=organization_id)
        archives = await self._list_archives(field.id, organization_id=organization_id)
        scenarios = await self._list_scenarios(field.id, organization_id=organization_id)

        if lite:
            current_metrics = await self._load_snapshot_metrics(
                field.id,
                organization_id=organization_id,
                run_id=field.aoi_run_id,
            )
            prediction_ready = await self._has_prediction(field.id, organization_id=organization_id)
            observation_cells = self._observation_cells_from_metrics(current_metrics)
            return {
                "mode": "single",
                "field": self._field_to_dict(field, runtime_map=runtime_map),
                "kpis": {
                    "prediction_ready": prediction_ready,
                    "archive_count": len(archives),
                    "scenario_count": len(scenarios),
                    "observation_cells": observation_cells,
                },
                "current_metrics": current_metrics,
                "prediction": None,
                "analytics_summary": {},
                "supported_sections": {},
                "zones_summary": {},
                "archives": archives,
                "scenarios": scenarios,
                "data_quality": {
                    "observation_cells": observation_cells,
                    "metrics_available": sorted(current_metrics.keys()),
                    "has_time_series": bool(current_metrics),
                    "has_prediction": prediction_ready,
                    "has_solar_radiation": False,
                },
            }

        current_metrics, raw_values = await self._collect_field_metrics(field)
        await self._ensure_series(field, current_metrics)

        latest_prediction = await self._get_latest_prediction(
            field.id,
            organization_id=organization_id,
            field=field,
        )
        if latest_prediction is not None:
            analytics_summary = {
                "current_stage": (latest_prediction.get("phenology") or {}).get("stage_label"),
                "lag_weeks_vs_norm": (latest_prediction.get("phenology") or {}).get("lag_weeks_vs_norm"),
                "history_trend_slope": (latest_prediction.get("history_trend") or {}).get("trend_slope_kg_ha_per_year"),
                "water_stress_class": (latest_prediction.get("water_balance") or {}).get("summary", {}).get("stress_class"),
                "active_alert_count": len(latest_prediction.get("anomalies") or []),
                **GEOMETRY_FOUNDATION,
            }
            supported_sections = {
                "series": bool((latest_prediction.get("seasonal_series") or {}).get("metrics")),
                "phenology": bool((latest_prediction.get("phenology") or {}).get("supported")),
                "anomalies": bool(latest_prediction.get("anomalies")),
                "water_balance": bool((latest_prediction.get("water_balance") or {}).get("supported")),
                "risk": bool((latest_prediction.get("risk") or {}).get("supported")),
                "history_trend": bool((latest_prediction.get("history_trend") or {}).get("supported")),
                "zones": bool((latest_prediction.get("management_zone_summary") or {}).get("supported")),
            }
            zones_summary = dict(latest_prediction.get("management_zone_summary") or {})
        else:
            temporal_service = TemporalAnalyticsService(self.db)
            temporal_analytics = await temporal_service.get_temporal_analytics(
                field.id,
                organization_id=organization_id,
                date_from=date_from,
                date_to=date_to,
            )
            zones_summary = (await temporal_service.get_management_zones(
                field.id,
                organization_id=organization_id,
                prediction_payload=None,
            )).get("summary") or {}
            analytics_summary = dict(temporal_analytics.get("analytics_summary") or {})
            supported_sections = dict(temporal_analytics.get("supported_sections") or {})
        series = await self._load_series(field.id, organization_id=organization_id, date_from=date_from, date_to=date_to)
        histograms = self._build_histograms(raw_values)
        solar_series = await self._load_weekly_solar_series(field.id, organization_id=organization_id)

        observation_cells = self._observation_cells_from_metrics(current_metrics)

        return {
            "mode": "single",
            "field": self._field_to_dict(field, runtime_map=runtime_map),
            "kpis": {
                "prediction_ready": latest_prediction is not None,
                "archive_count": len(archives),
                "scenario_count": len(scenarios),
                "observation_cells": observation_cells,
            },
            "current_metrics": current_metrics,
            "series": series,
            "solar_radiation_weekly": solar_series,
            "histograms": histograms,
            "prediction": latest_prediction,
            "analytics_summary": analytics_summary,
            "supported_sections": supported_sections,
            "zones_summary": zones_summary,
            "archives": archives,
            "scenarios": scenarios,
            "data_quality": {
                "observation_cells": observation_cells,
                "metrics_available": sorted(current_metrics.keys()),
                "has_time_series": any(series.values()),
                "has_prediction": latest_prediction is not None,
                "has_solar_radiation": bool(solar_series),
            },
        }

    async def _has_prediction(self, field_id: UUID, *, organization_id: UUID) -> bool:
        result = await self.db.execute(
            select(YieldPrediction.id)
            .where(YieldPrediction.organization_id == organization_id)
            .where(YieldPrediction.field_id == field_id)
            .order_by(desc(YieldPrediction.prediction_date))
            .limit(1)
        )
        return result.first() is not None

    async def _load_snapshot_metrics(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        run_id: UUID | None = None,
    ) -> dict[str, Any]:
        stmt = (
            select(FieldMetricSeries)
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id == field_id)
            .where(FieldMetricSeries.source == "run_snapshot")
            .order_by(FieldMetricSeries.metric.asc(), FieldMetricSeries.observed_at.desc())
        )
        if run_id is not None:
            stmt = stmt.where(FieldMetricSeries.aoi_run_id == run_id)
        rows = (await self.db.execute(stmt)).scalars().all()
        metrics: dict[str, Any] = {}
        for row in rows:
            if row.metric in metrics:
                continue
            metrics[row.metric] = {
                "label": METRIC_LABELS.get(row.metric, row.metric.upper()),
                "coverage": round(float(row.coverage or 0.0), 4) if row.coverage is not None else None,
                "mean": row.value_mean,
                "min": row.value_min,
                "max": row.value_max,
                "median": row.value_median,
                "p25": row.value_p25,
                "p75": row.value_p75,
            }
        return metrics

    async def get_group_dashboard(self, field_ids: list[UUID], *, organization_id: UUID) -> dict[str, Any]:
        ordered_ids = list(dict.fromkeys(field_ids))
        if not ordered_ids:
            raise ValueError("Не выбраны поля для групповой аналитики")

        result = await self.db.execute(
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.id.in_(ordered_ids))
            .where(Field.source != "merged_hidden")
        )
        field_map = {field.id: field for field in result.scalars().all()}
        fields = [field_map[field_id] for field_id in ordered_ids if field_id in field_map]
        if not fields:
            raise ValueError("Не удалось загрузить выбранные поля")
        runtime_map = await self._load_run_runtime_map(
            [field.aoi_run_id for field in fields],
            organization_id=organization_id,
        )

        all_metric_values: dict[str, list[float]] = defaultdict(list)
        field_items: list[dict[str, Any]] = []
        observation_counts: list[float] = []

        batch_metrics = await self._collect_batch_field_metrics(fields)

        for field in fields:
            current_metrics, raw_values = batch_metrics.get(field.id, ({}, {}))
            field_items.append(
                {
                    **self._field_to_dict(field, runtime_map=runtime_map),
                    "current_metrics": current_metrics,
                }
            )
            observation_counts.extend([
                float(payload.get("coverage") or 0.0)
                for payload in current_metrics.values()
            ])
            for metric, values in raw_values.items():
                all_metric_values[metric].extend(values)

        group_metrics = {
            metric: self._summarize_values(metric, values)
            for metric, values in all_metric_values.items()
            if values
        }
        series = await self._load_group_series([field.id for field in fields], organization_id=organization_id)
        observation_cells = int(round(float(np.mean(observation_counts or [0.0]))))

        return {
            "mode": "group",
            "selection": {
                "field_count": len(fields),
                "field_ids": [str(field.id) for field in fields],
                "total_area_m2": round(sum(float(field.area_m2 or 0.0) for field in fields), 2),
            },
            "kpis": {
                "field_count": len(fields),
                "observation_cells": observation_cells,
                "metric_count": len(group_metrics),
            },
            "current_metrics": group_metrics,
            "series": series,
            "histograms": self._build_histograms(all_metric_values),
            "fields": field_items,
            "data_quality": {
                "observation_cells": observation_cells,
                "metrics_available": sorted(group_metrics.keys()),
                "has_time_series": any(series.values()),
            },
        }

    async def build_archive_snapshots(self, field_id: UUID, *, organization_id: UUID) -> dict[str, Any]:
        dashboard = await self.get_field_dashboard(field_id, organization_id=organization_id)
        prediction = dashboard.get("prediction") or {}
        return {
            "field_snapshot": dashboard.get("field") or {},
            "prediction_snapshot": prediction,
            "metrics_snapshot": {
                "current_metrics": dashboard.get("current_metrics") or {},
                "histograms": dashboard.get("histograms") or {},
            },
            "scenario_snapshot": {
                "items": dashboard.get("scenarios") or [],
            },
            "model_meta": {
                "model_version": prediction.get("model_version"),
                "explained": bool(prediction.get("explanation")),
            },
        }

    async def attach_archive_series(
        self,
        field_id: UUID,
        archive_entry_id: int,
        observed_at: datetime,
        *,
        organization_id: UUID,
    ) -> None:
        field = (
            await self.db.execute(
                select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
            )
        ).scalar_one_or_none()
        if field is None:
            return
        current_metrics, _raw_values = await self._collect_field_metrics(field)
        for metric, payload in current_metrics.items():
            self.db.add(
                FieldMetricSeries(
                    organization_id=organization_id,
                    field_id=field.id,
                    aoi_run_id=field.aoi_run_id,
                    archive_entry_id=archive_entry_id,
                    metric=metric,
                    observed_at=observed_at,
                    value_mean=payload.get("mean"),
                    value_min=payload.get("min"),
                    value_max=payload.get("max"),
                    value_median=payload.get("median"),
                    value_p25=payload.get("p25"),
                    value_p75=payload.get("p75"),
                    coverage=payload.get("coverage"),
                    source="archive_snapshot",
                    meta={"label": METRIC_LABELS.get(metric, metric.upper())},
                )
            )
        await self.db.flush()

    async def _collect_batch_field_metrics(
        self, fields: list[Field]
    ) -> dict[UUID, tuple[dict[str, Any], dict[str, list[float]]]]:
        """Batch-collect metrics for multiple fields in a single query per run group."""
        run_groups: dict[UUID, list[Field]] = defaultdict(list)
        for field in fields:
            run_groups[field.aoi_run_id].append(field)

        result_map: dict[UUID, tuple[dict[str, Any], dict[str, list[float]]]] = {}

        for run_id, group_fields in run_groups.items():
            max_zoom_result = await self.db.execute(
                select(func.max(GridCell.zoom_level))
                .where(GridCell.organization_id == group_fields[0].organization_id)
                .where(GridCell.aoi_run_id == run_id)
            )
            max_zoom = max_zoom_result.scalar_one_or_none()
            if max_zoom is None:
                for field in group_fields:
                    result_map[field.id] = ({}, {})
                continue

            field_ids_in_group = [f.id for f in group_fields]
            group_org_ids = list({f.organization_id for f in group_fields})
            cells_result = await self.db.execute(
                select(GridCell, Field.id.label("matched_field_id"))
                .join(Field, GridCell.geom.ST_Intersects(Field.geom))
                .where(GridCell.organization_id.in_(group_org_ids))
                .where(GridCell.aoi_run_id == run_id)
                .where(GridCell.zoom_level == max_zoom)
                .where(Field.id.in_(field_ids_in_group))
            )
            rows = cells_result.all()

            field_cells: dict[UUID, list] = defaultdict(list)
            for row in rows:
                cell = row[0]
                fid = row[1]
                field_cells[fid].append(cell)

            for field in group_fields:
                cells = field_cells.get(field.id, [])
                raw_values: dict[str, list[float]] = defaultdict(list)
                for cell in cells:
                    for metric, column_name in GRID_METRIC_COLUMNS.items():
                        value = getattr(cell, column_name, None)
                        if value is not None:
                            raw_values[metric].append(float(value))
                metrics = {
                    metric: self._summarize_values(metric, values)
                    for metric, values in raw_values.items()
                    if values
                }
                result_map[field.id] = (metrics, raw_values)

        return result_map

    async def _collect_field_metrics(self, field: Field) -> tuple[dict[str, Any], dict[str, list[float]]]:
        max_zoom_result = await self.db.execute(
            select(func.max(GridCell.zoom_level))
            .where(GridCell.organization_id == field.organization_id)
            .where(GridCell.aoi_run_id == field.aoi_run_id)
        )
        max_zoom = max_zoom_result.scalar_one_or_none()
        if max_zoom is None:
            return {}, {}

        field_geom_subquery = select(Field.geom).where(Field.id == field.id).scalar_subquery()
        result = await self.db.execute(
            select(GridCell)
            .where(GridCell.organization_id == field.organization_id)
            .where(GridCell.aoi_run_id == field.aoi_run_id)
            .where(GridCell.zoom_level == max_zoom)
            .where(GridCell.geom.ST_Intersects(field_geom_subquery))
        )
        cells = list(result.scalars().all())
        raw_values: dict[str, list[float]] = defaultdict(list)
        for cell in cells:
            for metric, column_name in GRID_METRIC_COLUMNS.items():
                value = getattr(cell, column_name, None)
                if value is None:
                    continue
                raw_values[metric].append(float(value))

        metrics = {
            metric: self._summarize_values(metric, values)
            for metric, values in raw_values.items()
            if values
        }
        return metrics, raw_values

    async def _ensure_series(self, field: Field, metrics: dict[str, Any]) -> None:
        if not metrics:
            return
        existing = await self.db.execute(
            select(func.count(FieldMetricSeries.id))
            .where(FieldMetricSeries.organization_id == field.organization_id)
            .where(FieldMetricSeries.field_id == field.id)
            .where(FieldMetricSeries.aoi_run_id == field.aoi_run_id)
            .where(FieldMetricSeries.source == "run_snapshot")
        )
        if int(existing.scalar_one() or 0) > 0:
            return

        observed_at = field.created_at or datetime.now(timezone.utc)
        for metric, payload in metrics.items():
            self.db.add(
                FieldMetricSeries(
                    organization_id=field.organization_id,
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
                    source="run_snapshot",
                    meta={"label": METRIC_LABELS.get(metric, metric.upper())},
                )
            )
        await self.db.flush()

    async def _load_weekly_solar_series(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        season_year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return weekly solar radiation series (MJ/m²/day) from FieldFeatureWeekly.

        Solar radiation is derived from ERA5 cloud_cover via the Ångström-Prescott
        formula and stored in FieldFeatureWeekly.solar_radiation_mj (weekly mean MJ/m²/day).
        This series is separate from the GridCell-based FieldMetricSeries because
        solar radiation is not captured at detection time — it comes from ERA5.
        """
        from datetime import date as _date
        from services.weekly_profile_service import current_season_year
        year = season_year or current_season_year()
        season_from = _date(year, 1, 1)
        season_to = _date(year, 12, 31)
        stmt = (
            select(
                FieldFeatureWeekly.week_start,
                FieldFeatureWeekly.solar_radiation_mj,
                FieldFeatureWeekly.tmean_c,
                FieldFeatureWeekly.weather_coverage,
            )
            .where(FieldFeatureWeekly.organization_id == organization_id)
            .where(FieldFeatureWeekly.field_id == field_id)
            .where(FieldFeatureWeekly.season_year == year)
            .where(FieldFeatureWeekly.week_start >= season_from)
            .where(FieldFeatureWeekly.week_start <= season_to)
            .where(FieldFeatureWeekly.solar_radiation_mj.isnot(None))
            .order_by(FieldFeatureWeekly.week_start.asc())
        )
        rows = (await self.db.execute(stmt)).all()
        return [
            {
                "week_start": row.week_start.isoformat(),
                "solar_radiation_mj_day": round(float(row.solar_radiation_mj), 2),
                "tmean_c": round(float(row.tmean_c), 1) if row.tmean_c is not None else None,
                "weather_coverage": round(float(row.weather_coverage), 2) if row.weather_coverage is not None else None,
            }
            for row in rows
            if row.solar_radiation_mj is not None
        ]

    async def _load_series(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        stmt = (
            select(FieldMetricSeries)
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id == field_id)
        )
        if date_from is not None:
            stmt = stmt.where(FieldMetricSeries.observed_at >= date_from)
        if date_to is not None:
            stmt = stmt.where(FieldMetricSeries.observed_at <= date_to)
        stmt = stmt.order_by(FieldMetricSeries.metric.asc(), FieldMetricSeries.observed_at.asc())
        result = await self.db.execute(stmt)
        series: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in result.scalars().all():
            series[row.metric].append(
                {
                    "observed_at": row.observed_at.isoformat(),
                    "mean": row.value_mean,
                    "min": row.value_min,
                    "max": row.value_max,
                    "median": row.value_median,
                    "p25": row.value_p25,
                    "p75": row.value_p75,
                    "coverage": row.coverage,
                    "source": row.source,
                }
            )
        return dict(series)

    async def _load_group_series(self, field_ids: list[UUID], *, organization_id: UUID) -> dict[str, list[dict[str, Any]]]:
        if not field_ids:
            return {}
        result = await self.db.execute(
            select(FieldMetricSeries)
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id.in_(field_ids))
            .order_by(FieldMetricSeries.metric.asc(), FieldMetricSeries.observed_at.asc())
        )
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in result.scalars().all():
            key = f"{row.metric}|{row.observed_at.isoformat()}"
            grouped[key]["mean"].append(float(row.value_mean or 0.0))
            grouped[key]["min"].append(float(row.value_min or 0.0))
            grouped[key]["max"].append(float(row.value_max or 0.0))
            grouped[key]["median"].append(float(row.value_median or 0.0))
            grouped[key]["coverage"].append(float(row.coverage or 0.0))

        aggregated: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key, payload in grouped.items():
            metric, observed_at = key.split("|", 1)
            aggregated[metric].append(
                {
                    "observed_at": observed_at,
                    "mean": round(float(np.mean(payload["mean"])), 4),
                    "min": round(float(np.min(payload["min"])), 4),
                    "max": round(float(np.max(payload["max"])), 4),
                    "median": round(float(np.mean(payload["median"])), 4),
                    "coverage": round(float(np.mean(payload["coverage"])), 4),
                }
            )
        return dict(aggregated)

    async def _build_forecast_curve(
        self,
        field: Field,
        crop: Crop,
        *,
        organization_id: UUID,
    ) -> dict[str, Any]:
        geom = to_shape(field.geom)
        centroid = geom.centroid
        forecast_payload = await self.weather_service.get_forecast(
            centroid.y,
            centroid.x,
            days=10,
            organization_id=organization_id,
        )
        base_temp_c = float(getattr(crop, "base_temp_c", None) or 5.0)
        return {
            "provider": forecast_payload.get("provider"),
            "days": int(forecast_payload.get("days") or 10),
            "base_temp_c": base_temp_c,
            "freshness": forecast_payload.get("freshness") or {},
            "error": forecast_payload.get("error"),
            "points": build_forecast_curve_points(
                list(forecast_payload.get("forecast") or []),
                base_temp_c=base_temp_c,
            ),
        }

    async def _get_latest_prediction(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        field: Field | None = None,
    ) -> dict[str, Any] | None:
        result = await self.db.execute(
            select(YieldPrediction, Crop)
            .where(YieldPrediction.organization_id == organization_id)
            .where(YieldPrediction.field_id == field_id)
            .join(Crop, YieldPrediction.crop_id == Crop.id, isouter=True)
            .order_by(desc(YieldPrediction.prediction_date))
            .limit(1)
        )
        row = result.first()
        if row is None:
            return None
        prediction, crop = row
        crop = crop or Crop(id=0, code="unknown", name="Unknown", category="unknown", yield_baseline_kg_ha=0.0, ndvi_target=0.0, base_temp_c=0.0)
        forecast_curve = {}
        if field is not None:
            forecast_curve = await self._build_forecast_curve(field, crop, organization_id=organization_id)
        details = dict(prediction.details or {})
        data_quality = dict(prediction.data_quality or {})
        input_features = dict(prediction.input_features or {})
        supported = bool(details.get("supported"))
        confidence_tier = str(
            details.get("confidence_tier")
            or data_quality.get("confidence_tier")
            or ("tenant_calibrated" if supported else "unsupported")
        )
        support_reason = None if supported else details.get("support_reason") or data_quality.get("confidence_reason")
        support_reason_code, support_reason_params = classify_support_reason(support_reason)
        crop_suitability = dict(details.get("crop_suitability") or {})
        trust_meta = describe_prediction_operational_tier(
            supported=supported,
            confidence_tier=confidence_tier,
            crop_suitability=crop_suitability,
            support_reason=support_reason,
        )
        temporal_service = TemporalAnalyticsService(self.db)
        analytics = await temporal_service.get_temporal_analytics(
            field_id,
            organization_id=organization_id,
            crop_code=crop.code,
        )
        zones = await temporal_service.get_management_zones(
            field_id,
            organization_id=organization_id,
            prediction_payload={
                "estimated_yield_kg_ha": prediction.estimated_yield_kg_ha,
                "confidence": prediction.confidence,
            },
        )
        normalized_drivers = normalize_driver_breakdown(
            list((prediction.explanation or {}).get("drivers") or []),
            baseline_yield_kg_ha=float(prediction.estimated_yield_kg_ha or 0.0),
            baseline_inputs=input_features,
            source="yield_model",
        )
        explanation = dict(prediction.explanation or {})
        explanation["drivers"] = normalized_drivers
        return {
            "id": prediction.id,
            "field_id": str(prediction.field_id),
            "crop": {"id": crop.id, "code": crop.code, "name": crop.name},
            "prediction_date": prediction.prediction_date.isoformat(),
            "estimated_yield_kg_ha": prediction.estimated_yield_kg_ha,
            "confidence": prediction.confidence,
            "confidence_tier": confidence_tier,
            "model_version": prediction.model_version,
            "details": details,
            "input_features": input_features,
            "explanation": explanation,
            "data_quality": data_quality,
            "prediction_interval": dict(data_quality.get("prediction_interval") or details.get("prediction_interval") or {}),
            "model_applicability": {
                "supported": supported,
                "coverage_score": float(data_quality.get("valid_feature_count") or 0.0) / 14.0,
                "feature_gaps": list(details.get("applicability_feature_gaps") or []),
                "confidence_tier": confidence_tier,
            },
            "training_domain": {
                "samples": int(details.get("training_samples") or 0),
                "crop_code": crop.code,
                "confidence_tier": confidence_tier,
            },
            "forecast_curve": forecast_curve,
            "feature_coverage": {
                "available": list(details.get("applicability_feature_available") or []),
                "missing": list(details.get("applicability_feature_gaps") or []),
            },
            "crop_suitability": crop_suitability,
            "seasonal_series": dict(analytics.get("seasonal_series") or {}),
            "phenology": dict(analytics.get("phenology") or {}),
            "anomalies": list(analytics.get("anomalies") or []),
            "water_balance": dict(analytics.get("water_balance") or {}),
            "risk": dict(analytics.get("risk") or {}),
            "history_trend": dict(analytics.get("history_trend") or {}),
            "management_zone_summary": dict(zones.get("summary") or {}),
            "driver_breakdown": normalized_drivers,
            "support_reason": support_reason,
            "support_reason_code": support_reason_code,
            "support_reason_params": support_reason_params,
            "operational_tier": trust_meta.get("operational_tier"),
            "review_required": bool(trust_meta.get("review_required")),
            "review_reason": trust_meta.get("review_reason"),
            "review_reason_code": trust_meta.get("review_reason_code"),
            "review_reason_params": trust_meta.get("review_reason_params") or {},
            "freshness": build_freshness(
                provider="yield_model",
                fetched_at=prediction.prediction_date,
                cache_written_at=prediction.prediction_date,
                model_version=prediction.model_version,
                dataset_version=details.get("dataset_version"),
            ),
        }

    async def _list_archives(self, field_id: UUID, *, organization_id: UUID) -> list[dict[str, Any]]:
        result = await self.db.execute(
            select(ArchiveEntry)
            .where(ArchiveEntry.organization_id == organization_id)
            .where(ArchiveEntry.field_id == field_id)
            .order_by(desc(ArchiveEntry.created_at))
            .limit(10)
        )
        items = []
        for row in result.scalars().all():
            items.append(
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "date_from": row.date_from.isoformat(),
                    "date_to": row.date_to.isoformat(),
                    "status": row.status,
                    "meta": dict(row.meta or {}),
                }
            )
        return items

    async def _list_scenarios(self, field_id: UUID, *, organization_id: UUID) -> list[dict[str, Any]]:
        result = await self.db.execute(
            select(ScenarioRun)
            .where(ScenarioRun.organization_id == organization_id)
            .where(ScenarioRun.field_id == field_id)
            .order_by(desc(ScenarioRun.created_at))
            .limit(12)
        )
        items = []
        for row in result.scalars().all():
            items.append(
                {
                    "id": row.id,
                    "scenario_name": row.scenario_name,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "model_version": row.model_version,
                    "parameters": dict(row.parameters or {}),
                    "baseline_snapshot": dict(row.baseline_snapshot or {}),
                    "result_snapshot": dict(row.result_snapshot or {}),
                    "delta_pct": row.delta_pct,
                }
            )
        return items

    @staticmethod
    def _summarize_values(metric: str, values: list[float]) -> dict[str, Any]:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "label": METRIC_LABELS.get(metric, metric.upper()),
            "coverage": round(float(arr.size), 4),
            "mean": round(float(np.mean(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
        }

    def _build_histograms(self, raw_values: dict[str, list[float]]) -> dict[str, Any]:
        histograms: dict[str, Any] = {}
        for metric in HISTOGRAM_METRICS:
            values = raw_values.get(metric) or []
            if len(values) < 2:
                continue
            arr_hist = np.asarray(values, dtype=np.float64)
            if np.max(arr_hist) == np.min(arr_hist):
                # All values identical — single bin histogram
                histograms[metric] = {
                    "bins": [round(float(arr_hist[0]) - 0.01, 4), round(float(arr_hist[0]) + 0.01, 4)],
                    "counts": [len(values)],
                }
                continue
            counts, edges = np.histogram(arr_hist, bins=min(8, max(4, len(values) // 4)))
            histograms[metric] = {
                "bins": [round(float(edge), 4) for edge in edges.tolist()],
                "counts": [int(item) for item in counts.tolist()],
            }
        return histograms

    @staticmethod
    def _observation_cells_from_metrics(current_metrics: dict[str, Any]) -> int:
        if not current_metrics:
            return 0
        return int(round(float(np.mean([
            metric_payload["coverage"]
            for metric_payload in current_metrics.values()
            if metric_payload.get("coverage") is not None
        ] or [0.0]))))

    async def _load_run_runtime_map(
        self,
        run_ids: list[UUID],
        *,
        organization_id: UUID,
    ) -> dict[UUID, dict[str, Any]]:
        unique_ids = [run_id for run_id in dict.fromkeys(run_ids) if run_id is not None]
        if not unique_ids:
            return {}
        result = await self.db.execute(
            select(AoiRun.id, AoiRun.params)
            .where(AoiRun.organization_id == organization_id)
            .where(AoiRun.id.in_(unique_ids))
        )
        runtime_map: dict[UUID, dict[str, Any]] = {}
        for run_id, params in result.all():
            payload = dict(params or {})
            runtime_map[run_id] = dict(payload.get("runtime") or {})
        return runtime_map

    @staticmethod
    def _field_to_dict(field: Field, *, runtime_map: dict[UUID, dict[str, Any]] | None = None) -> dict[str, Any]:
        geom = to_shape(field.geom)
        centroid = geom.centroid
        runtime_quality = extract_runtime_geometry_quality(
            (runtime_map or {}).get(field.aoi_run_id),
            lon=float(centroid.x),
            lat=float(centroid.y),
        )
        quality_meta = describe_field_quality(
            field.quality_score,
            field.source,
            geometry_confidence=runtime_quality.get("geometry_confidence"),
            tta_consensus=runtime_quality.get("tta_consensus"),
            boundary_uncertainty=runtime_quality.get("boundary_uncertainty"),
            uncertainty_source=runtime_quality.get("uncertainty_source"),
        )
        return {
            "field_id": str(field.id),
            "aoi_run_id": str(field.aoi_run_id),
            "area_m2": float(field.area_m2 or 0.0),
            "perimeter_m": float(field.perimeter_m or 0.0),
            "quality_score": field.quality_score,
            "quality_confidence": quality_meta["confidence"],
            "geometry_confidence": quality_meta.get("geometry_confidence"),
            "tta_consensus": quality_meta.get("tta_consensus"),
            "boundary_uncertainty": quality_meta.get("boundary_uncertainty"),
            "quality_band": quality_meta["band"],
            "quality_label": quality_meta["label"],
            "quality_reason": quality_meta["reason"],
            "quality_reason_code": quality_meta.get("reason_code"),
            "quality_reason_params": quality_meta.get("reason_params") or {},
            "operational_tier": quality_meta.get("operational_tier"),
            "review_required": bool(quality_meta.get("review_required")),
            "review_reason": quality_meta.get("review_reason"),
            "review_reason_code": quality_meta.get("review_reason_code"),
            "review_reason_params": quality_meta.get("review_reason_params") or {},
            "source": field.source,
            "created_at": field.created_at.isoformat() if field.created_at else None,
            "centroid": {"lat": round(float(centroid.y), 6), "lon": round(float(centroid.x), 6)},
        }
