from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from services.weekly_profile_service import (
    _parse_openmeteo_archive_hourly,
    profile_has_signal,
    rows_to_weekly_inputs,
    summarize_geometry_quality,
)


def test_rows_to_weekly_inputs_maps_feature_rows():
    rows = [
        SimpleNamespace(
            week_number=18,
            season_year=2026,
            week_start=None,
            tmean_c=16.5,
            tmax_c=24.0,
            tmin_c=9.0,
            precipitation_mm=12.0,
            vpd_kpa=1.4,
            solar_radiation_mj=17.0,
            soil_moisture=0.23,
            wind_speed_m_s=3.8,
            ndvi_mean=0.61,
            ndre_mean=0.19,
            ndmi_mean=0.11,
            irrigation_mm=8.0,
            n_applied_kg_ha=18.0,
            previous_crop_code="soy",
        )
    ]

    inputs = rows_to_weekly_inputs(rows)

    assert len(inputs) == 1
    item = inputs[0]
    assert item.week == 18
    assert item.tmean_c == 16.5
    assert item.solar_radiation_mj == 17.0
    assert item.wind_speed_m_s == 3.8
    assert item.ndvi == 0.61
    assert item.irrigation_mm == 8.0
    assert item.n_applied_kg_ha == 18.0
    assert item.previous_crop_code == "soy"


def test_profile_signal_and_geometry_summary_use_weekly_rows():
    rows = [
        SimpleNamespace(
            weather_coverage=0.0,
            satellite_coverage=0.0,
            ndvi_mean=None,
            ndre_mean=None,
            ndmi_mean=None,
            precipitation_mm=None,
            tmean_c=None,
            geometry_confidence=0.62,
            tta_consensus=0.71,
            boundary_uncertainty=0.19,
        ),
        SimpleNamespace(
            weather_coverage=0.28,
            satellite_coverage=0.14,
            ndvi_mean=0.58,
            ndre_mean=0.17,
            ndmi_mean=0.09,
            precipitation_mm=14.0,
            tmean_c=15.0,
            geometry_confidence=0.78,
            tta_consensus=0.84,
            boundary_uncertainty=0.11,
        ),
    ]

    assert profile_has_signal(rows) is True
    summary = summarize_geometry_quality(rows)
    assert summary["geometry_confidence"] == 0.7
    assert summary["tta_consensus"] == pytest.approx(0.775)
    assert summary["boundary_uncertainty"] == 0.15


def test_parse_openmeteo_archive_hourly_aggregates_weather_daily():
    payload = {
        "hourly": {
            "time": [
                "2025-03-01T00:00",
                "2025-03-01T01:00",
                "2025-03-02T00:00",
            ],
            "temperature_2m": [10.0, 14.0, 20.0],
            "precipitation": [1.2, 0.8, 0.0],
            "vapour_pressure_deficit": [0.4, 0.6, 1.1],
            "soil_moisture_0_to_7cm": [0.21, 0.25, 0.31],
            "wind_speed_10m": [3.0, 5.0, 7.0],
            "cloud_cover": [50.0, 70.0, 10.0],
        }
    }

    parsed = _parse_openmeteo_archive_hourly(payload)

    assert sorted(parsed.keys()) == [date(2025, 3, 1), date(2025, 3, 2)]
    first_day = parsed[date(2025, 3, 1)]
    assert first_day["tmean_c"] == pytest.approx(12.0)
    assert first_day["precipitation_mm"] == pytest.approx(2.0)
    assert first_day["vpd_kpa"] == pytest.approx(0.5)
    assert first_day["soil_moisture"] == pytest.approx(0.23)
    assert first_day["wind_speed_m_s"] == pytest.approx(4.0)
    assert first_day["cloud_cover"] == pytest.approx(0.6)
    assert first_day["gdd"] == pytest.approx(2.0)
