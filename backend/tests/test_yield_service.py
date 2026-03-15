from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

import services.yield_service as yield_service_module
from services.yield_service import (
    TrainingRow,
    YieldService,
    _evaluate_crop_suitability,
    _hydro_factor,
    _management_factor,
)
from storage.db import Crop, Field


class _FakeExecuteResult:
    def __init__(self, value=None) -> None:
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def first(self):
        return self._value

    def all(self):
        return self._value or []


class _FakeDb:
    def __init__(self, field: Field) -> None:
        self._field = field
        self.saved = []

    async def get(self, model, key):
        if model is Field and key == self._field.id:
            return self._field
        return None

    async def execute(self, stmt):
        if "FROM fields" in str(stmt):
            return _FakeExecuteResult(self._field)
        return _FakeExecuteResult(None)

    def add(self, row):
        self.saved.append(row)

    async def flush(self):
        return None


class _QueuedDb:
    def __init__(self, values) -> None:
        self._values = list(values)

    async def execute(self, stmt):
        if not self._values:
            return _FakeExecuteResult(None)
        return _FakeExecuteResult(self._values.pop(0))


@pytest.mark.asyncio
async def test_yield_service_builds_prediction(monkeypatch):
    field_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = Field(
        id=field_id,
        organization_id=uuid4(),
        aoi_run_id=uuid4(),
        geom=from_shape(polygon, srid=4326),
        area_m2=10000.0,
        perimeter_m=400.0,
        quality_score=0.81,
        source='ml',
    )
    db = _FakeDb(field)
    service = YieldService(db)
    crop = Crop(
        id=1,
        code='wheat',
        name='Пшеница',
        category='grain',
        yield_baseline_kg_ha=4200.0,
        ndvi_target=0.72,
        base_temp_c=5.0,
        description='Тестовая культура',
    )
    monkeypatch.setattr(service.crop_service, 'get_crop_by_code', AsyncMock(return_value=crop))
    monkeypatch.setattr(yield_service_module, 'ensure_weekly_profile', AsyncMock(return_value=[]))
    monkeypatch.setattr(yield_service_module, 'load_crop_hint', AsyncMock(return_value={}))
    monkeypatch.setattr(
        service.field_analytics_service,
        '_collect_field_metrics',
        AsyncMock(return_value=({"ndvi": {"mean": 0.62}}, {})),
    )
    monkeypatch.setattr(service, '_seasonal_weather_summary', AsyncMock(return_value={}))
    monkeypatch.setattr(service.temporal_analytics_service, 'get_temporal_analytics', AsyncMock(return_value={}))
    monkeypatch.setattr(service.temporal_analytics_service, 'get_management_zones', AsyncMock(return_value={}))
    monkeypatch.setattr(
        service,
        '_build_current_features',
        AsyncMock(
            return_value=(
                {
                    "crop_baseline": 4200.0,
                    "field_area_ha": 1.0,
                    "compactness": 0.8,
                    "soil_organic_matter_pct": 3.0,
                    "soil_ph": 6.5,
                    "soil_n_ppm": 18.0,
                    "soil_p_ppm": 10.0,
                    "soil_k_ppm": 20.0,
                    "management_total_amount": 12.0,
                    "historical_field_mean_yield": 4100.0,
                },
                set(),
            )
        ),
    )
    monkeypatch.setattr(
        service,
        '_load_training_rows',
        AsyncMock(
            return_value=[
                TrainingRow(
                    features={
                        "crop_baseline": 4200.0,
                        "field_area_ha": 1.0,
                        "compactness": 0.8,
                        "soil_organic_matter_pct": 3.0,
                        "soil_ph": 6.4,
                        "soil_n_ppm": 18.0,
                        "soil_p_ppm": 10.0,
                        "soil_k_ppm": 20.0,
                        "management_total_amount": 12.0,
                        "historical_field_mean_yield": 4050.0,
                    },
                    target=4000.0 + idx * 120.0,
                )
                for idx in range(5)
            ]
        ),
    )
    monkeypatch.setattr(
        service.weather_service,
        'get_current_weather',
        AsyncMock(
            return_value={
                'temperature_c': 21.0,
                'precipitation_mm': 8.0,
                'cloud_cover_pct': 12.0,
                'soil_moisture': 0.22,
            }
        ),
    )
    monkeypatch.setattr(
        service.weather_service,
        'get_forecast',
        AsyncMock(
            return_value={
                'provider': 'openmeteo',
                'days': 2,
                'forecast': [
                    {'date': '2026-03-12', 'temp_mean_c': 12.0, 'precipitation_mm': 3.0},
                    {'date': '2026-03-13', 'temp_mean_c': 13.0, 'precipitation_mm': 1.0},
                ],
                'freshness': {'freshness_state': 'fresh'},
            }
        ),
    )

    payload = await service.get_or_create_prediction(
        field_id,
        organization_id=field.organization_id,
        crop_code='wheat',
        refresh=True,
    )

    assert payload['field_id'] == str(field_id)
    assert payload['crop']['code'] == 'wheat'
    assert payload['estimated_yield_kg_ha'] > 0
    assert payload['confidence'] >= 0.25  # conformal coverage with small sample size
    assert payload['confidence_tier'] == 'tenant_calibrated'
    assert payload['model_applicability']['supported'] is True
    assert payload['operational_tier'] == 'validated_core'
    assert payload['review_required'] is False
    assert payload['geometry_quality_impact']['status'] == 'neutral'
    assert payload['geometry_quality_impact']['penalty_factor'] == 1.0
    assert payload['crop_hint'] == {}
    assert payload['forecast_curve']['provider'] == 'openmeteo'
    assert payload['forecast_curve']['points'][0]['gdd_daily'] == pytest.approx(7.0)
    assert payload['forecast_curve']['points'][1]['gdd_cumulative'] == pytest.approx(15.0)
    assert db.saved, 'Прогноз должен быть сохранен в сессии'


@pytest.mark.asyncio
async def test_yield_service_rejects_preview_contours():
    field_id = uuid4()
    organization_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = Field(
        id=field_id,
        organization_id=organization_id,
        aoi_run_id=uuid4(),
        geom=from_shape(polygon, srid=4326),
        area_m2=10000.0,
        perimeter_m=400.0,
        quality_score=0.81,
        source='autodetect_preview',
    )
    db = _FakeDb(field)
    service = YieldService(db)

    with pytest.raises(ValueError, match='Preview-контур'):
        await service._get_field(field_id, organization_id=organization_id)


@pytest.mark.asyncio
async def test_scenario_adjustments_do_not_create_model_applicability(monkeypatch):
    field_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = Field(
        id=field_id,
        organization_id=uuid4(),
        aoi_run_id=uuid4(),
        geom=from_shape(polygon, srid=4326),
        area_m2=10000.0,
        perimeter_m=400.0,
        quality_score=0.81,
        source='ml',
    )
    db = _FakeDb(field)
    service = YieldService(db)
    crop = Crop(
        id=1,
        code='wheat',
        name='Пшеница',
        category='grain',
        yield_baseline_kg_ha=4200.0,
        ndvi_target=0.72,
        base_temp_c=5.0,
        description='Тестовая культура',
    )

    monkeypatch.setattr(service.crop_service, 'get_crop_by_code', AsyncMock(return_value=crop))
    monkeypatch.setattr(yield_service_module, 'ensure_weekly_profile', AsyncMock(return_value=[]))
    monkeypatch.setattr(yield_service_module, 'load_crop_hint', AsyncMock(return_value={}))
    monkeypatch.setattr(
        service.field_analytics_service,
        '_collect_field_metrics',
        AsyncMock(return_value=({}, {})),
    )
    monkeypatch.setattr(service, '_seasonal_weather_summary', AsyncMock(return_value={}))
    monkeypatch.setattr(service.temporal_analytics_service, 'get_temporal_analytics', AsyncMock(return_value={}))
    monkeypatch.setattr(service.temporal_analytics_service, 'get_management_zones', AsyncMock(return_value={}))
    monkeypatch.setattr(
        service.weather_service,
        'get_current_weather',
        AsyncMock(
            return_value={
                'temperature_c': 21.0,
                'precipitation_mm': 0.0,
                'soil_moisture': 0.23,
                'wind_speed_m_s': 6.0,
            }
        ),
    )

    async def fake_build_current_features(*args, scenario_adjustments=None, **kwargs):
        if scenario_adjustments:
            return (
                {
                    "crop_baseline": 4200.0,
                    "field_area_ha": 1.0,
                    "compactness": 0.8,
                    "soil_organic_matter_pct": 3.0,
                    "soil_ph": 6.5,
                    "soil_n_ppm": 18.0,
                    "soil_p_ppm": 10.0,
                    "soil_k_ppm": 20.0,
                    "management_total_amount": 12.0,
                    "historical_field_mean_yield": None,
                    "current_ndvi_mean": None,
                    "current_ndmi_mean": None,
                    "current_soil_moisture": 0.3,
                    "current_vpd_mean": None,
                    "current_precipitation_mm": 10.0,
                    "current_wind_speed_m_s": 6.0,
                },
                {"historical_field_mean_yield", "current_ndvi_mean", "current_ndmi_mean", "current_vpd_mean"},
            )
        return (
            {
                "crop_baseline": 4200.0,
                "field_area_ha": 1.0,
                "compactness": 0.8,
                "soil_organic_matter_pct": None,
                "soil_ph": None,
                "soil_n_ppm": None,
                "soil_p_ppm": None,
                "soil_k_ppm": None,
                "management_total_amount": None,
                "historical_field_mean_yield": None,
                "current_ndvi_mean": None,
                "current_ndmi_mean": None,
                "current_soil_moisture": 0.23,
                "current_vpd_mean": None,
                "current_precipitation_mm": 0.0,
                "current_wind_speed_m_s": 6.0,
            },
            {
                "soil_organic_matter_pct",
                "soil_ph",
                "soil_n_ppm",
                "soil_p_ppm",
                "soil_k_ppm",
                "management_total_amount",
                "historical_field_mean_yield",
                "current_ndvi_mean",
                "current_ndmi_mean",
                "current_vpd_mean",
            },
        )

    monkeypatch.setattr(service, '_build_current_features', AsyncMock(side_effect=fake_build_current_features))
    monkeypatch.setattr(service, '_load_training_rows', AsyncMock(return_value=[]))

    payload = await service.estimate_prediction(
        field_id,
        organization_id=field.organization_id,
        crop_code='wheat',
        scenario_adjustments={
            'irrigation_pct': 15.0,
            'fertilizer_pct': 20.0,
            'expected_rain_mm': 12.0,
        },
    )

    assert payload['confidence_tier'] == 'unsupported'
    assert payload['model_applicability']['supported'] is False
    assert payload['estimated_yield_kg_ha'] == 0.0
    assert 'soil_organic_matter_pct' in payload['model_applicability']['feature_gaps']
    assert payload['operational_tier'] == 'unsupported'
    assert payload['review_required'] is True


@pytest.mark.asyncio
async def test_seasonal_weather_summary_materializes_weekly_profile_without_field_season(monkeypatch):
    service = YieldService(_QueuedDb([None]))
    weekly_mock = AsyncMock(return_value=[])
    fallback_mock = AsyncMock(return_value={'gdd_sum': 123.0, 'temperature_mean_c': 9.5})
    monkeypatch.setattr(yield_service_module, 'ensure_weekly_profile', weekly_mock)
    monkeypatch.setattr(service, '_seasonal_weather_from_weekly', fallback_mock)

    payload = await service._seasonal_weather_summary(uuid4(), 'wheat', organization_id=uuid4())

    assert payload['gdd_sum'] == 123.0
    weekly_mock.assert_awaited_once()
    fallback_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_seasonal_weather_summary_falls_back_to_weekly_when_weatherdaily_missing(monkeypatch):
    season_year = 2026
    service = YieldService(_QueuedDb([
        (1, season_year),
        (None, None, None, None, None, 7),
    ]))
    weekly_mock = AsyncMock(return_value=[])
    fallback_mock = AsyncMock(
        return_value={
            'gdd_sum': 245.0,
            'precipitation_sum': 18.0,
            'temperature_mean_c': 11.2,
            'vpd_mean': 0.41,
        }
    )
    monkeypatch.setattr(yield_service_module, 'ensure_weekly_profile', weekly_mock)
    monkeypatch.setattr(service, '_seasonal_weather_from_weekly', fallback_mock)

    payload = await service._seasonal_weather_summary(uuid4(), 'wheat', organization_id=uuid4())

    assert payload['season_year'] == season_year
    assert payload['gdd_sum'] == 245.0
    assert payload['precipitation_sum'] == 18.0
    assert payload['temperature_mean_c'] == 11.2
    weekly_mock.assert_awaited_once()
    fallback_mock.assert_awaited_once()


def test_management_factor_has_optimum_instead_of_monotonic_growth():
    low = _management_factor(20.0)
    optimal = _management_factor(200.0)
    excessive = _management_factor(1040.0)

    assert optimal > low
    assert optimal > excessive


def test_hydro_factor_penalizes_drought_and_waterlogging():
    balanced = _hydro_factor(
        {
            "current_soil_moisture": 0.26,
            "current_vpd_mean": 1.1,
            "current_precipitation_mm": 12.0,
        }
    )
    drought = _hydro_factor(
        {
            "current_soil_moisture": 0.07,
            "current_vpd_mean": 3.2,
            "current_precipitation_mm": 0.4,
        }
    )
    waterlogged = _hydro_factor(
        {
            "current_soil_moisture": 0.55,
            "current_vpd_mean": 0.4,
            "current_precipitation_mm": 72.0,
        }
    )

    assert balanced > drought
    assert balanced > waterlogged


def test_crop_suitability_penalizes_warm_season_crop_in_north():
    crop = Crop(
        id=1,
        code='corn',
        name='Кукуруза',
        category='grain',
        yield_baseline_kg_ha=5200.0,
        ndvi_target=0.72,
        base_temp_c=8.0,
        description='Теплолюбивая культура',
    )
    suitability = _evaluate_crop_suitability(
        crop,
        {
            'latitude': 59.6,
            'seasonal_gdd_sum': 1120.0,
            'seasonal_precipitation_mm': 340.0,
            'seasonal_temperature_mean_c': 10.2,
            'seasonal_observed_days': 110.0,
        },
    )

    assert suitability['status'] in {'low', 'unsuitable'}
    assert suitability['score'] < 0.5
    assert suitability['warnings']
