"""Canonical loading and conversion of weekly field profiles."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID

from geoalchemy2.shape import to_shape
import httpx
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from providers.era5.client import ERA5Client
from services.field_quality import describe_field_quality, extract_runtime_geometry_quality
from services.mechanistic_engine import WeeklyInput, run_mechanistic_baseline
from storage.db import (
    AoiRun,
    Crop,
    CropAssignment,
    Field,
    FieldCropPosterior,
    FieldFeatureWeekly,
    FieldMetricSeries,
    FieldSeason,
    ManagementEvent,
    WeatherDaily,
)


FEATURE_SCHEMA_VERSION = "weekly_v4"
logger = get_logger(__name__)


def _coerce_day(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except Exception:
        return None


def _week_start(day_value: date) -> date:
    return day_value - timedelta(days=day_value.weekday())


def _season_week_starts(start: date, end: date) -> list[date]:
    starts: dict[tuple[int, int], date] = {}
    current = start
    while current <= end:
        iso = current.isocalendar()
        starts[(iso.year, iso.week)] = _week_start(current)
        current += timedelta(days=1)
    return [starts[key] for key in sorted(starts)]


def _safe_mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _safe_sum(values: list[float]) -> float | None:
    return float(sum(values)) if values else None


def _metric_value(row: Any) -> float | None:
    for attr in ("value_mean", "value_median", "value_max"):
        value = getattr(row, attr, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                return None
    return None


def _is_irrigation_event(event_type: str, unit: str) -> bool:
    text = f"{event_type} {unit}".lower()
    return any(token in text for token in ("irrig", "полив", "water"))


def _is_nitrogen_event(event_type: str, unit: str) -> bool:
    text = f"{event_type} {unit}".lower()
    return any(token in text for token in ("fert", "nitrogen", "азот", "urea", "ammon", "селитр", "карбамид"))


def _normalize_irrigation_amount(amount: float, unit: str) -> float:
    unit_value = str(unit or "").strip().lower()
    if unit_value in {"cm", "centimeter", "centimeters"}:
        return amount * 10.0
    return amount


def _normalize_nitrogen_amount(amount: float, unit: str) -> float:
    unit_value = str(unit or "").strip().lower()
    if "g/" in unit_value:
        return amount / 1000.0
    return amount


def _merge_weather_daily(primary: dict[date, dict[str, float]], fallback: dict[date, dict[str, float]]) -> dict[date, dict[str, float]]:
    merged: dict[date, dict[str, float]] = {}
    for key in sorted(set(primary) | set(fallback)):
        payload = dict(fallback.get(key) or {})
        payload.update({name: value for name, value in (primary.get(key) or {}).items() if value is not None})
        merged[key] = payload
    return merged


def _parse_era5_timeseries(payload: dict[str, Any]) -> dict[date, dict[str, float]]:
    series_by_day: dict[date, dict[str, float]] = defaultdict(dict)
    temp_series = payload.get("temperature_2m") or []
    dew_series = payload.get("dewpoint_2m") or []
    precip_series = payload.get("total_precipitation") or []
    soil_series = payload.get("soil_water_l1") or []
    u_series = payload.get("u_wind_10m") or []
    v_series = payload.get("v_wind_10m") or []
    cloud_series = payload.get("total_cloud_cover") or []

    for item in temp_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            series_by_day[day_value]["tmean_c"] = ERA5Client.kelvin_to_celsius(float(value))

    dew_map: dict[date, float] = {}
    for item in dew_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            dew_map[day_value] = ERA5Client.kelvin_to_celsius(float(value))

    for day_value, payload_day in series_by_day.items():
        tmean = payload_day.get("tmean_c")
        dew = dew_map.get(day_value)
        if tmean is not None and dew is not None:
            payload_day["vpd_kpa"] = ERA5Client.compute_vpd(float(tmean), float(dew))

    for item in precip_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            series_by_day[day_value]["precipitation_mm"] = float(value) * 1000.0

    for item in soil_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            series_by_day[day_value]["soil_moisture"] = float(value)

    u_map: dict[date, float] = {}
    for item in u_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            u_map[day_value] = float(value)

    v_map: dict[date, float] = {}
    for item in v_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            v_map[day_value] = float(value)

    for day_value in set(u_map) | set(v_map):
        if day_value in u_map and day_value in v_map:
            series_by_day[day_value]["wind_speed_m_s"] = ERA5Client.compute_wind_speed(u_map[day_value], v_map[day_value])

    for item in cloud_series:
        day_value = _coerce_day((item or {}).get("date"))
        value = (item or {}).get("value")
        if day_value is not None and value is not None:
            series_by_day[day_value]["cloud_cover"] = float(value)

    for day_value, payload_day in series_by_day.items():
        tmean = payload_day.get("tmean_c")
        if tmean is not None:
            payload_day["gdd"] = ERA5Client.compute_gdd(float(tmean) + 4.0, float(tmean) - 4.0, t_base=10.0)

    return dict(series_by_day)


def _parse_openmeteo_archive_hourly(payload: dict[str, Any]) -> dict[date, dict[str, float]]:
    hourly = payload.get("hourly") or {}
    timestamps = hourly.get("time") or []
    metric_columns = {
        "tmean_c": "temperature_2m",
        "precipitation_mm": "precipitation",
        "vpd_kpa": "vapour_pressure_deficit",
        "soil_moisture": "soil_moisture_0_to_7cm",
        "wind_speed_m_s": "wind_speed_10m",
        "cloud_cover": "cloud_cover",
    }
    day_buckets: dict[date, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for index, timestamp in enumerate(timestamps):
        day_value = _coerce_day(timestamp)
        if day_value is None:
            continue
        for metric, column in metric_columns.items():
            values = hourly.get(column) or []
            if index >= len(values):
                continue
            value = values[index]
            if value is None:
                continue
            try:
                day_buckets[day_value][metric].append(float(value))
            except Exception:
                continue

    parsed: dict[date, dict[str, float]] = {}
    for day_value, buckets in day_buckets.items():
        payload_day: dict[str, float] = {}
        tmean = _safe_mean(buckets.get("tmean_c", []))
        if tmean is not None:
            payload_day["tmean_c"] = float(tmean)
            payload_day["gdd"] = ERA5Client.compute_gdd(float(tmean) + 4.0, float(tmean) - 4.0, t_base=10.0)

        precipitation = _safe_sum(buckets.get("precipitation_mm", []))
        if precipitation is not None:
            payload_day["precipitation_mm"] = float(precipitation)

        for metric in ("vpd_kpa", "soil_moisture", "wind_speed_m_s"):
            value = _safe_mean(buckets.get(metric, []))
            if value is not None:
                payload_day[metric] = float(value)

        cloud_cover = _safe_mean(buckets.get("cloud_cover", []))
        if cloud_cover is not None:
            payload_day["cloud_cover"] = float(cloud_cover) / 100.0 if cloud_cover > 1.0 else float(cloud_cover)

        if payload_day:
            parsed[day_value] = payload_day
    return parsed


async def _fetch_openmeteo_archive_daily(
    *,
    lat: float,
    lon: float,
    date_from: date,
    date_to: date,
) -> dict[date, dict[str, float]]:
    settings = get_settings()
    params = {
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
        "start_date": date_from.isoformat(),
        "end_date": date_to.isoformat(),
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "vapour_pressure_deficit",
                "soil_moisture_0_to_7cm",
                "wind_speed_10m",
                "cloud_cover",
            ]
        ),
        "timezone": "UTC",
    }
    timeout = httpx.Timeout(45.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(settings.OPENMETEO_ARCHIVE_BASE_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    archive_daily = _parse_openmeteo_archive_hourly(payload)
    for day_value, payload_day in archive_daily.items():
        if "cloud_cover" in payload_day and "solar_radiation_mj" not in payload_day:
            try:
                doy = day_value.timetuple().tm_yday
                ra = _extraterrestrial_radiation(float(lat), doy)
                payload_day["solar_radiation_mj"] = _solar_from_cloud(ra, float(payload_day["cloud_cover"]))
            except Exception:
                continue
    return archive_daily


def _extraterrestrial_radiation(lat_deg: float, doy: int) -> float:
    """FAO-56 Eq 21: extraterrestrial radiation Ra, MJ/m²/day.

    Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56, Eq 21.
    Valid for latitudes −90 to +90°.
    """
    Gsc = 0.082  # solar constant MJ/m²/min
    phi = math.radians(float(lat_deg))
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    # Clamp to avoid math domain error at extreme (polar) latitudes
    arg = max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta)))
    omega_s = math.acos(arg)
    Ra = (
        (24.0 * 60.0 / math.pi)
        * Gsc
        * dr
        * (
            omega_s * math.sin(phi) * math.sin(delta)
            + math.cos(phi) * math.cos(delta) * math.sin(omega_s)
        )
    )
    return max(0.0, Ra)


def _solar_from_cloud(ra_mj: float, cloud_cover: float) -> float:
    """Ångström-Prescott (1940) formula: Rs = Ra × (as + bs × (1 − cloud_cover)).

    Converts ERA5 total cloud cover fraction (0–1) to surface solar radiation.
    Coefficients as=0.25, bs=0.50 are FAO-56 defaults for humid mid-latitude
    climates (Allen et al. 1998, Table 2).

    For agricultural purposes these coefficients are well-validated for
    temperate wheat/maize growing regions (50–60°N) and are the standard
    choice when pyranometer data is absent.
    """
    cloud_frac = max(0.0, min(1.0, float(cloud_cover)))
    return max(0.0, float(ra_mj) * (0.25 + 0.50 * (1.0 - cloud_frac)))


def current_season_year() -> int:
    return datetime.now(timezone.utc).year


async def load_weekly_feature_rows(
    db: AsyncSession,
    *,
    organization_id: UUID,
    field_id: UUID,
    season_year: int | None = None,
) -> list[FieldFeatureWeekly]:
    resolved_year = int(season_year or current_season_year())
    stmt = (
        select(FieldFeatureWeekly)
        .where(FieldFeatureWeekly.organization_id == organization_id)
        .where(FieldFeatureWeekly.field_id == field_id)
        .where(FieldFeatureWeekly.season_year == resolved_year)
        .order_by(FieldFeatureWeekly.week_number.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


def rows_to_weekly_inputs(rows: Sequence[FieldFeatureWeekly]) -> list[WeeklyInput]:
    inputs: list[WeeklyInput] = []
    for row in rows:
        inputs.append(
            WeeklyInput(
                week=int(getattr(row, "week_number")),
                tmean_c=float(getattr(row, "tmean_c", None) or 15.0),
                tmax_c=float(getattr(row, "tmax_c", None) or getattr(row, "tmean_c", None) or 22.0),
                tmin_c=float(getattr(row, "tmin_c", None) or getattr(row, "tmean_c", None) or 8.0),
                precipitation_mm=float(getattr(row, "precipitation_mm", None) or 10.0),
                vpd_kpa=float(getattr(row, "vpd_kpa", None) or 1.0),
                solar_radiation_mj=float(getattr(row, "solar_radiation_mj", None) or 15.0),
                soil_moisture=float(getattr(row, "soil_moisture")) if getattr(row, "soil_moisture", None) is not None else None,
                wind_speed_m_s=float(getattr(row, "wind_speed_m_s", None) or 2.0),
                ndvi=float(getattr(row, "ndvi_mean")) if getattr(row, "ndvi_mean", None) is not None else None,
                ndre=float(getattr(row, "ndre_mean")) if getattr(row, "ndre_mean", None) is not None else None,
                ndmi=float(getattr(row, "ndmi_mean")) if getattr(row, "ndmi_mean", None) is not None else None,
                irrigation_mm=float(getattr(row, "irrigation_mm", None) or 0.0),
                n_applied_kg_ha=float(getattr(row, "n_applied_kg_ha", None) or 0.0),
                week_start=getattr(row, "week_start", None),
                season_year=int(getattr(row, "season_year", 0) or 0) or None,
                previous_crop_code=getattr(row, "previous_crop_code", None),
            )
        )
    return inputs


def profile_has_signal(rows: Sequence[FieldFeatureWeekly]) -> bool:
    for row in rows:
        if float(row.weather_coverage or 0.0) > 0.15:
            return True
        if float(row.satellite_coverage or 0.0) > 0.05:
            return True
        if any(
            value is not None
            for value in (
                row.ndvi_mean,
                row.ndre_mean,
                row.ndmi_mean,
                row.precipitation_mm,
                row.tmean_c,
            )
        ):
            return True
    return False


def summarize_geometry_quality(rows: Sequence[FieldFeatureWeekly]) -> dict[str, float | None]:
    if not rows:
        return {
            "geometry_confidence": None,
            "tta_consensus": None,
            "boundary_uncertainty": None,
        }

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    geometry_values = [float(row.geometry_confidence) for row in rows if row.geometry_confidence is not None]
    consensus_values = [float(row.tta_consensus) for row in rows if row.tta_consensus is not None]
    uncertainty_values = [float(row.boundary_uncertainty) for row in rows if row.boundary_uncertainty is not None]
    return {
        "geometry_confidence": _mean(geometry_values),
        "tta_consensus": _mean(consensus_values),
        "boundary_uncertainty": _mean(uncertainty_values),
    }


async def load_crop_hint(
    db: AsyncSession,
    *,
    organization_id: UUID,
    field_id: UUID,
    season_year: int | None = None,
) -> dict[str, Any]:
    resolved_year = int(season_year or current_season_year())
    stmt = (
        select(FieldCropPosterior)
        .where(FieldCropPosterior.organization_id == organization_id)
        .where(FieldCropPosterior.field_id == field_id)
        .where(FieldCropPosterior.season_year == resolved_year)
        .order_by(FieldCropPosterior.probability.desc(), desc(FieldCropPosterior.created_at))
    )
    rows = list((await db.execute(stmt)).scalars().all())
    if not rows:
        return {}
    top = rows[0]
    return {
        "season_year": resolved_year,
        "top_crop_code": top.crop_code,
        "top_probability": round(float(top.probability or 0.0), 4),
        "source": top.source,
        "model_version": top.model_version,
        "distribution": [
            {
                "crop_code": row.crop_code,
                "probability": round(float(row.probability or 0.0), 4),
                "source": row.source,
            }
            for row in rows
        ],
    }


async def materialize_weekly_profile(
    db: AsyncSession,
    *,
    organization_id: UUID,
    field_id: UUID,
    season_year: int | None = None,
) -> dict[str, Any]:
    resolved_year = int(season_year or current_season_year())
    season_start = date(resolved_year, 3, 1)
    season_end = date(resolved_year, 10, 31)

    field = (
        await db.execute(
            select(Field)
            .where(Field.id == field_id)
            .where(Field.organization_id == organization_id)
        )
    ).scalar_one_or_none()
    if field is None:
        return {
            "field_id": str(field_id),
            "season_year": resolved_year,
            "weeks_created": 0,
            "status": "field_not_found",
        }

    field_geom = to_shape(field.geom)
    centroid = field_geom.centroid

    season_row = (
        await db.execute(
            select(FieldSeason)
            .where(FieldSeason.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .where(FieldSeason.season_year == resolved_year)
        )
    ).scalar_one_or_none()

    runtime_params = (
        await db.execute(
            select(AoiRun.params)
            .where(AoiRun.organization_id == organization_id)
            .where(AoiRun.id == field.aoi_run_id)
        )
    ).scalar_one_or_none()
    runtime_quality = extract_runtime_geometry_quality(
        dict((runtime_params or {}).get("runtime") or {}),
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

    latest_assignment = None
    previous_crop_code: str | None = None
    crop_baseline_kg_ha: float | None = None

    sat_stmt = (
        select(FieldMetricSeries)
        .where(FieldMetricSeries.organization_id == organization_id)
        .where(FieldMetricSeries.field_id == field_id)
        .where(FieldMetricSeries.observed_at >= season_start)
        .where(FieldMetricSeries.observed_at <= season_end)
        .order_by(FieldMetricSeries.observed_at)
    )
    sat_rows = (await db.execute(sat_stmt)).scalars().all()

    sat_by_week: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in sat_rows:
        obs_date = _coerce_day(row.observed_at)
        if obs_date is None:
            continue
        week_num = _week_start(obs_date).isocalendar()[1]
        metric = str(row.metric).lower()
        value = _metric_value(row)
        if value is not None:
            sat_by_week[int(week_num)][metric].append(value)

    imported_weather_daily: dict[date, dict[str, float]] = {}
    if season_row is not None:
        weather_stmt = (
            select(WeatherDaily)
            .where(WeatherDaily.organization_id == organization_id)
            .where(WeatherDaily.field_season_id == season_row.id)
            .where(WeatherDaily.observed_on >= season_start)
            .where(WeatherDaily.observed_on <= season_end)
            .order_by(WeatherDaily.observed_on.asc())
        )
        for wr in (await db.execute(weather_stmt)).scalars().all():
            day_value = _coerce_day(wr.observed_on)
            if day_value is None:
                continue
            imported_weather_daily[day_value] = {
                "tmean_c": float(wr.temperature_mean_c) if wr.temperature_mean_c is not None else None,
                "precipitation_mm": float(wr.precipitation_mm) if wr.precipitation_mm is not None else None,
                "gdd": float(wr.gdd) if wr.gdd is not None else None,
                "vpd_kpa": float(wr.vpd) if wr.vpd is not None else None,
                "soil_moisture": float(wr.soil_moisture) if wr.soil_moisture is not None else None,
            }

    imported_weather_days = len(imported_weather_daily)
    openmeteo_archive_daily: dict[date, dict[str, float]] = {}
    era5_daily: dict[date, dict[str, float]] = {}
    if imported_weather_days < 21:
        try:
            openmeteo_archive_daily = await _fetch_openmeteo_archive_daily(
                lat=float(centroid.y),
                lon=float(centroid.x),
                date_from=season_start,
                date_to=season_end,
            )
            if openmeteo_archive_daily:
                logger.info(
                    "weekly_feature_openmeteo_archive_fallback_used",
                    field_id=str(field_id),
                    days=len(openmeteo_archive_daily),
                )
        except Exception as exc:
            logger.warning("weekly_feature_openmeteo_archive_fallback_failed", field_id=str(field_id), error=str(exc))
        if not openmeteo_archive_daily:
            try:
                era5_payload = await ERA5Client().get_timeseries(
                    lat=float(centroid.y),
                    lon=float(centroid.x),
                    variables=[
                        "temperature_2m",
                        "dewpoint_2m",
                        "u_wind_10m",
                        "v_wind_10m",
                        "total_precipitation",
                        "soil_water_l1",
                        "total_cloud_cover",
                    ],
                    date_from=season_start,
                    date_to=season_end,
                )
                era5_daily = _parse_era5_timeseries(era5_payload)
                # Derive solar_radiation_mj from ERA5 cloud cover using Ångström-Prescott
                # formula (FAO-56, Allen et al. 1998). ERA5 provides total_cloud_cover
                # (0–1 fraction) which we convert to surface solar radiation via:
                #   Rs = Ra × (0.25 + 0.50 × (1 – cloud_cover))
                # where Ra is extraterrestrial radiation from latitude + day-of-year.
                _lat = float(centroid.y)
                for _day, _dp in era5_daily.items():
                    if "cloud_cover" in _dp and "solar_radiation_mj" not in _dp:
                        try:
                            _doy = _day.timetuple().tm_yday
                            _ra = _extraterrestrial_radiation(_lat, _doy)
                            _dp["solar_radiation_mj"] = _solar_from_cloud(_ra, float(_dp["cloud_cover"]))
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("weekly_feature_era5_fallback_failed", field_id=str(field_id), error=str(exc))

    weather_fallback = _merge_weather_daily(era5_daily, openmeteo_archive_daily)
    weather_daily = _merge_weather_daily(imported_weather_daily, weather_fallback)
    weather_by_week: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for day_value, payload in weather_daily.items():
        if not (season_start <= day_value <= season_end):
            continue
        week_num = _week_start(day_value).isocalendar()[1]
        for key, value in payload.items():
            if value is not None:
                weather_by_week[int(week_num)][key].append(float(value))

    management_by_week: dict[int, dict[str, float]] = defaultdict(lambda: {"irrigation_mm": 0.0, "n_applied_kg_ha": 0.0})
    if season_row is not None:
        latest_assignment = (
            await db.execute(
                select(CropAssignment)
                .where(CropAssignment.organization_id == organization_id)
                .where(CropAssignment.field_season_id == season_row.id)
                .order_by(CropAssignment.assigned_at.desc())
            )
        ).scalars().first()
        mgmt_stmt = (
            select(ManagementEvent)
            .where(ManagementEvent.organization_id == organization_id)
            .where(ManagementEvent.field_season_id == season_row.id)
            .where(ManagementEvent.event_date >= season_start)
            .where(ManagementEvent.event_date <= season_end)
            .order_by(ManagementEvent.event_date.asc())
        )
        for event in (await db.execute(mgmt_stmt)).scalars().all():
            event_day = _coerce_day(event.event_date)
            if event_day is None:
                continue
            week_num = _week_start(event_day).isocalendar()[1]
            amount = float(event.amount or 0.0)
            event_type = str(event.event_type or "")
            unit = str(event.unit or "")
            if _is_irrigation_event(event_type, unit):
                management_by_week[int(week_num)]["irrigation_mm"] += _normalize_irrigation_amount(amount, unit)
            if _is_nitrogen_event(event_type, unit):
                management_by_week[int(week_num)]["n_applied_kg_ha"] += _normalize_nitrogen_amount(amount, unit)

        previous_season = (
            await db.execute(
                select(FieldSeason)
                .where(FieldSeason.organization_id == organization_id)
                .where(FieldSeason.field_id == field_id)
                .where(FieldSeason.season_year == resolved_year - 1)
            )
        ).scalars().first()
        if previous_season is not None:
            previous_assignment = (
                await db.execute(
                    select(CropAssignment)
                    .where(CropAssignment.organization_id == organization_id)
                    .where(CropAssignment.field_season_id == previous_season.id)
                    .order_by(CropAssignment.assigned_at.desc())
                )
            ).scalars().first()
            if previous_assignment is not None:
                previous_crop_code = str(previous_assignment.crop_code or "").strip() or None

    if latest_assignment is not None:
        crop_row = (
            await db.execute(
                select(Crop)
                .where(Crop.code == str(latest_assignment.crop_code))
            )
        ).scalars().first()
        if crop_row is not None and crop_row.yield_baseline_kg_ha is not None:
            crop_baseline_kg_ha = float(crop_row.yield_baseline_kg_ha)

    week_starts = _season_week_starts(season_start, season_end)
    await db.execute(
        delete(FieldFeatureWeekly)
        .where(FieldFeatureWeekly.organization_id == organization_id)
        .where(FieldFeatureWeekly.field_id == field_id)
        .where(FieldFeatureWeekly.season_year == resolved_year)
    )
    created = 0
    created_rows: list[FieldFeatureWeekly] = []
    weather_sources: list[str] = []
    if imported_weather_daily:
        weather_sources.append("weather_daily")
    if era5_daily:
        weather_sources.append("era5")
    if openmeteo_archive_daily:
        weather_sources.append("openmeteo_archive")
    weather_source = "+".join(weather_sources) or "weather_daily"

    for week_start in week_starts:
        week_number = int(week_start.isocalendar()[1])
        sat = sat_by_week.get(week_number, {})
        weather_week = weather_by_week.get(week_number, {})
        management = management_by_week.get(week_number, {})
        if not sat and not weather_week and not any(float(v or 0.0) > 0.0 for v in management.values()):
            continue

        feature = FieldFeatureWeekly(
            organization_id=organization_id,
            field_id=field_id,
            season_year=resolved_year,
            week_number=week_number,
            week_start=week_start,
            ndvi_mean=_safe_mean(sat.get("ndvi", [])),
            ndvi_max=_safe_max(sat.get("ndvi", [])),
            ndre_mean=_safe_mean(sat.get("ndre", [])),
            ndmi_mean=_safe_mean(sat.get("ndmi", [])),
            ndwi_mean=_safe_mean(sat.get("ndwi", [])),
            bsi_mean=_safe_mean(sat.get("bsi", [])),
            tmean_c=_safe_mean(weather_week.get("tmean_c", [])),
            tmax_c=_safe_max(weather_week.get("tmean_c", [])),
            tmin_c=min(weather_week.get("tmean_c", [])) if weather_week.get("tmean_c") else None,
            precipitation_mm=_safe_sum(weather_week.get("precipitation_mm", [])),
            vpd_kpa=_safe_mean(weather_week.get("vpd_kpa", [])),
            solar_radiation_mj=_safe_mean(weather_week.get("solar_radiation_mj", [])),
            soil_moisture=_safe_mean(weather_week.get("soil_moisture", [])),
            wind_speed_m_s=_safe_mean(weather_week.get("wind_speed_m_s", [])),
            gdd=_safe_sum(weather_week.get("gdd", [])),
            irrigation_mm=float(management.get("irrigation_mm") or 0.0),
            n_applied_kg_ha=float(management.get("n_applied_kg_ha") or 0.0),
            previous_crop_code=previous_crop_code,
            geometry_confidence=quality_meta.get("geometry_confidence"),
            tta_consensus=quality_meta.get("tta_consensus"),
            boundary_uncertainty=quality_meta.get("boundary_uncertainty"),
            satellite_coverage=min(float(len(sat.get("ndvi", []))) / 7.0, 1.0) if sat.get("ndvi") else 0.0,
            weather_coverage=min(float(len(weather_week.get("tmean_c", []))) / 7.0, 1.0) if weather_week.get("tmean_c") else 0.0,
            source=f"backfill:{weather_source}",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        db.add(feature)
        created_rows.append(feature)
        created += 1

    if created_rows and latest_assignment is not None and crop_baseline_kg_ha is not None:
        try:
            mechanistic = run_mechanistic_baseline(
                crop_code=str(latest_assignment.crop_code),
                crop_baseline_kg_ha=crop_baseline_kg_ha,
                weekly_inputs=rows_to_weekly_inputs(created_rows),
                field_area_ha=float(field.area_m2 or 0.0) / 10000.0,
                latitude=float(centroid.y),
                previous_crop_code=previous_crop_code,
            )
            for feature, trace_row in zip(created_rows, mechanistic.trace):
                feature.stage = int(trace_row.get("stage")) if trace_row.get("stage") is not None else None
                feature.canopy_cover = float(trace_row.get("canopy_cover")) if trace_row.get("canopy_cover") is not None else None
                feature.water_stress = float(trace_row.get("water_stress")) if trace_row.get("water_stress") is not None else None
                feature.heat_stress = float(trace_row.get("heat_stress")) if trace_row.get("heat_stress") is not None else None
                feature.nutrient_stress = float(trace_row.get("nutrient_stress")) if trace_row.get("nutrient_stress") is not None else None
                feature.biomass_proxy = float(trace_row.get("biomass_proxy")) if trace_row.get("biomass_proxy") is not None else None
        except Exception as exc:
            logger.warning("weekly_feature_mechanistic_backfill_skipped", field_id=str(field_id), error=str(exc))

    if season_row is not None:
        await db.execute(
            delete(FieldCropPosterior)
            .where(FieldCropPosterior.organization_id == organization_id)
            .where(FieldCropPosterior.field_id == field_id)
            .where(FieldCropPosterior.season_year == resolved_year)
            .where(FieldCropPosterior.source == "manual_selection")
        )
        if latest_assignment is not None:
            db.add(
                FieldCropPosterior(
                    organization_id=organization_id,
                    field_id=field_id,
                    season_year=resolved_year,
                    crop_code=str(latest_assignment.crop_code),
                    probability=1.0,
                    source="manual_selection",
                    model_version="manual_selection_v1",
                    payload={
                        "field_season_id": int(season_row.id),
                        "crop_assignment_id": int(latest_assignment.id),
                    },
                )
            )

    await db.commit()
    return {
        "field_id": str(field_id),
        "season_year": resolved_year,
        "weeks_created": created,
        "weather_days_imported": imported_weather_days,
        "weather_days_era5": len(era5_daily),
        "previous_crop_code": previous_crop_code,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "status": "ok",
    }


async def ensure_weekly_profile(
    db: AsyncSession,
    *,
    organization_id: UUID,
    field_id: UUID,
    season_year: int | None = None,
    min_rows: int = 4,
    force: bool = False,
) -> list[FieldFeatureWeekly]:
    resolved_year = int(season_year or current_season_year())
    rows = await load_weekly_feature_rows(
        db,
        organization_id=organization_id,
        field_id=field_id,
        season_year=resolved_year,
    )
    has_current_schema = all(
        str(getattr(row, "feature_schema_version", "") or "") == FEATURE_SCHEMA_VERSION
        for row in rows
    )
    if force or len(rows) < min_rows or not has_current_schema:
        await materialize_weekly_profile(
            db,
            organization_id=organization_id,
            field_id=field_id,
            season_year=resolved_year,
        )
        rows = await load_weekly_feature_rows(
            db,
            organization_id=organization_id,
            field_id=field_id,
            season_year=resolved_year,
        )
    return rows


def serialize_weekly_feature_rows(rows: Sequence[FieldFeatureWeekly]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "week_number": int(row.week_number),
                "week_start": row.week_start.isoformat() if row.week_start else None,
                "ndvi_mean": row.ndvi_mean,
                "ndvi_max": row.ndvi_max,
                "ndre_mean": row.ndre_mean,
                "ndmi_mean": row.ndmi_mean,
                "ndwi_mean": row.ndwi_mean,
                "bsi_mean": row.bsi_mean,
                "tmean_c": row.tmean_c,
                "tmax_c": row.tmax_c,
                "tmin_c": row.tmin_c,
                "precipitation_mm": row.precipitation_mm,
                "vpd_kpa": row.vpd_kpa,
                "solar_radiation_mj": row.solar_radiation_mj,
                "soil_moisture": row.soil_moisture,
                "wind_speed_m_s": row.wind_speed_m_s,
                "gdd": row.gdd,
                "irrigation_mm": row.irrigation_mm,
                "n_applied_kg_ha": row.n_applied_kg_ha,
                "previous_crop_code": row.previous_crop_code,
                "geometry_confidence": row.geometry_confidence,
                "tta_consensus": row.tta_consensus,
                "boundary_uncertainty": row.boundary_uncertainty,
                "stage": row.stage,
                "canopy_cover": row.canopy_cover,
                "water_stress": row.water_stress,
                "heat_stress": row.heat_stress,
                "nutrient_stress": row.nutrient_stress,
                "biomass_proxy": row.biomass_proxy,
                "satellite_coverage": row.satellite_coverage,
                "weather_coverage": row.weather_coverage,
                "source": row.source,
                "feature_schema_version": row.feature_schema_version,
            }
        )
    return payload
