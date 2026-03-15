"""Helpers for future weather and cumulative GDD curves."""
from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _daily_mean_temperature(day: dict[str, Any]) -> float | None:
    explicit = _as_float(day.get("temp_mean_c"))
    if explicit is not None:
        return explicit
    tmax = _as_float(day.get("temp_max_c"))
    tmin = _as_float(day.get("temp_min_c"))
    if tmax is None and tmin is None:
        return None
    if tmax is None:
        return tmin
    if tmin is None:
        return tmax
    return (tmax + tmin) / 2.0


def build_forecast_curve_points(
    forecast_days: list[dict[str, Any]],
    *,
    base_temp_c: float,
    temperature_delta_c: float = 0.0,
    extra_precip_total_mm: float = 0.0,
    precipitation_factor: float | None = None,
) -> list[dict[str, Any]]:
    """Convert daily forecast rows to additive weather/GDD points."""
    if not forecast_days:
        return []
    points: list[dict[str, Any]] = []
    cumulative = 0.0
    extra_precip_per_day = float(extra_precip_total_mm) / max(len(forecast_days), 1)
    precip_multiplier = float(precipitation_factor) if precipitation_factor is not None else 1.0
    for day in forecast_days:
        date = str(day.get("date") or "")
        temp_mean = _daily_mean_temperature(day)
        adjusted_temp = (temp_mean if temp_mean is not None else 0.0) + float(temperature_delta_c)
        base_precip = _as_float(day.get("precipitation_mm")) or 0.0
        precipitation = max(0.0, base_precip * precip_multiplier + extra_precip_per_day)
        gdd_daily = max(0.0, adjusted_temp - float(base_temp_c))
        cumulative += gdd_daily
        points.append(
            {
                "date": date,
                "temperature_mean_c": round(adjusted_temp, 2) if temp_mean is not None else None,
                "precipitation_mm": round(precipitation, 2),
                "gdd_daily": round(gdd_daily, 2),
                "gdd_cumulative": round(cumulative, 2),
            }
        )
    return points
