from __future__ import annotations

from datetime import date

import pytest

from services.mechanistic_engine import (
    WeeklyInput,
    _estimate_et0,
    run_mechanistic_baseline,
)


def _weekly_inputs(*, season_year: int = 2026) -> list[WeeklyInput]:
    return [
        WeeklyInput(
            week=week,
            season_year=season_year,
            week_start=date.fromisocalendar(season_year, week, 1),
            tmean_c=16.0 + idx * 0.7,
            tmax_c=24.0 + idx * 0.8,
            tmin_c=8.0 + idx * 0.4,
            precipitation_mm=12.0 - idx * 0.5,
            vpd_kpa=1.1 + idx * 0.08,
            solar_radiation_mj=16.0 + idx * 0.6,
            soil_moisture=0.24 - idx * 0.005,
            wind_speed_m_s=3.2 + idx * 0.15,
            ndvi=0.35 + idx * 0.04,
            ndre=0.10 + idx * 0.012,
            ndmi=0.07 + idx * 0.01,
            irrigation_mm=0.0,
            n_applied_kg_ha=4.0 if idx < 3 else 0.0,
        )
        for idx, week in enumerate(range(16, 24))
    ]


def test_penman_monteith_proxy_responds_to_weather_signal():
    mild = _estimate_et0(
        tmean=18.0,
        tmax=24.0,
        tmin=11.0,
        solar_mj=15.0,
        latitude=46.0,
        wind_speed_m_s=1.5,
        vpd_kpa=0.7,
    )
    hot_dry_windy = _estimate_et0(
        tmean=26.0,
        tmax=34.0,
        tmin=18.0,
        solar_mj=22.0,
        latitude=46.0,
        wind_speed_m_s=4.8,
        vpd_kpa=2.1,
    )

    assert hot_dry_windy > mild
    assert mild > 0.0


def test_previous_crop_effect_improves_wheat_after_legume_break():
    inputs = _weekly_inputs()

    after_soy = run_mechanistic_baseline(
        crop_code="wheat",
        crop_baseline_kg_ha=4200.0,
        weekly_inputs=inputs,
        soil_texture_code=4.0,
        soil_n_ppm=2.0,
        previous_crop_code="soy",
        latitude=50.0,
    )
    monoculture = run_mechanistic_baseline(
        crop_code="wheat",
        crop_baseline_kg_ha=4200.0,
        weekly_inputs=inputs,
        soil_texture_code=4.0,
        soil_n_ppm=2.0,
        previous_crop_code="wheat",
        latitude=50.0,
    )

    assert after_soy.total_biomass_proxy > monoculture.total_biomass_proxy
    assert after_soy.final_state.soil_n_available > monoculture.final_state.soil_n_available


def test_trace_contains_new_weekly_state_fields():
    result = run_mechanistic_baseline(
        crop_code="soy",
        crop_baseline_kg_ha=3100.0,
        weekly_inputs=_weekly_inputs(),
        soil_texture_code=4.5,
        soil_n_ppm=14.0,
        latitude=53.0,
    )

    assert result.trace
    first = result.trace[0]
    assert "photoperiod_hours" in first
    assert "surface_layer_water_mm" in first
    assert "subsoil_water_mm" in first
    assert "root_depth_m" in first
    assert "gross_assimilation" in first
    assert "maintenance_respiration" in first
    assert result.params_used["feature_model"] == "wofost_lite_v1"
