from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

import services.modeling_service as modeling_service_module
import services.mechanistic_engine as mechanistic_engine_module
from services.modeling_service import ModelingService


@pytest.mark.asyncio
async def test_modeling_service_returns_delta():
    field_id = uuid4()
    organization_id = uuid4()
    service = ModelingService(db=None)
    service.yield_service.get_or_create_prediction = AsyncMock(
        return_value={
            'field_id': str(field_id),
            'estimated_yield_kg_ha': 4000.0,
            'model_version': 'agronomy_tabular_v2',
            'confidence_tier': 'tenant_calibrated',
            'model_applicability': {'supported': True},
            'crop_suitability': {'status': 'moderate', 'warnings': []},
        }
    )
    service.yield_service.estimate_prediction = AsyncMock(
        return_value={
            'field_id': str(field_id),
            'estimated_yield_kg_ha': 4300.0,
            'model_version': 'agronomy_tabular_v2',
            'confidence_tier': 'tenant_calibrated',
            'model_applicability': {'supported': True},
            'training_domain': {'samples': 8},
            'feature_coverage': {'available': ['crop_baseline'], 'missing': []},
            'crop_suitability': {'status': 'moderate', 'warnings': []},
            'explanation': {'drivers': [{'label': 'management', 'effect': 0.08}]},
        }
    )

    payload = await service.simulate(
        field_id,
        organization_id=organization_id,
        crop_code='wheat',
        irrigation_pct=10.0,
        fertilizer_pct=5.0,
        expected_rain_mm=20.0,
    )

    assert payload['field_id'] == str(field_id)
    assert payload['scenario_yield_kg_ha'] > payload['baseline_yield_kg_ha']
    assert payload['predicted_yield_change_pct'] > 0
    assert payload['supported'] is True
    assert payload['confidence_tier'] == 'tenant_calibrated'
    assert payload['crop_suitability']['status'] == 'moderate'
    assert payload['crop_hint'] == {}
    assert payload['geometry_quality_impact'] == {}
    assert payload['operational_tier'] == 'validated_core'
    assert payload['review_required'] is False


def test_modeling_service_rejects_preview_contours():
    field = SimpleNamespace(source='autodetect_preview')

    with pytest.raises(ValueError, match='Preview-контур'):
        ModelingService._ensure_operational_field(field)


@pytest.mark.asyncio
async def test_modeling_service_allows_negative_delta_for_excess_inputs():
    field_id = uuid4()
    organization_id = uuid4()
    service = ModelingService(db=None)
    service.yield_service.get_or_create_prediction = AsyncMock(
        return_value={
            'field_id': str(field_id),
            'estimated_yield_kg_ha': 4000.0,
            'model_version': 'agronomy_tabular_v2',
            'confidence_tier': 'tenant_calibrated',
            'model_applicability': {'supported': True},
            'crop_suitability': {'status': 'moderate', 'warnings': []},
        }
    )
    service.yield_service.estimate_prediction = AsyncMock(
        return_value={
            'field_id': str(field_id),
            'estimated_yield_kg_ha': 3720.0,
            'model_version': 'agronomy_tabular_v2',
            'confidence_tier': 'tenant_calibrated',
            'model_applicability': {'supported': True},
            'training_domain': {'samples': 8},
            'feature_coverage': {'available': ['crop_baseline'], 'missing': []},
            'crop_suitability': {'status': 'moderate', 'warnings': ['Переувлажнение усиливает риск полегания.']},
            'explanation': {'drivers': [{'label': 'scenario_rainfall', 'effect': -0.06}]},
        }
    )

    payload = await service.simulate(
        field_id,
        organization_id=organization_id,
        crop_code='wheat',
        irrigation_pct=35.0,
        fertilizer_pct=40.0,
        expected_rain_mm=80.0,
    )

    assert payload['scenario_yield_kg_ha'] < payload['baseline_yield_kg_ha']
    assert payload['predicted_yield_change_pct'] < 0
    assert payload['supported'] is True
    assert payload['constraint_warnings']
    assert payload['operational_tier'] == 'validated_core'


@pytest.mark.asyncio
async def test_modeling_service_does_not_expose_scenario_number_when_baseline_unsupported():
    field_id = uuid4()
    organization_id = uuid4()
    service = ModelingService(db=None)
    service.yield_service.get_or_create_prediction = AsyncMock(
        return_value={
            'field_id': str(field_id),
            'estimated_yield_kg_ha': 0.0,
            'model_version': 'unsupported_v3',
            'confidence_tier': 'unsupported',
            'model_applicability': {'supported': False},
            'support_reason': 'Недостаточно объективных данных.',
        }
    )
    service.yield_service.estimate_prediction = AsyncMock(
        return_value={
            'field_id': str(field_id),
            'estimated_yield_kg_ha': 3860.0,
            'model_version': 'global_baseline_v3',
            'confidence_tier': 'global_baseline',
            'model_applicability': {'supported': False},
            'support_reason': 'Недостаточно объективных данных.',
            'training_domain': {'samples': 0},
            'feature_coverage': {'available': ['current_soil_moisture'], 'missing': ['current_ndvi_mean']},
            'explanation': {'drivers': [{'label': 'scenario_irrigation', 'effect': 0.08}]},
        }
    )

    payload = await service.simulate(
        field_id,
        organization_id=organization_id,
        crop_code='wheat',
        irrigation_pct=20.0,
        fertilizer_pct=10.0,
        expected_rain_mm=25.0,
    )

    assert payload['supported'] is False
    assert payload['confidence_tier'] == 'unsupported'
    assert payload['baseline_yield_kg_ha'] == 0.0
    assert payload['scenario_yield_kg_ha'] == 0.0
    assert payload['predicted_yield_change_pct'] == 0.0
    assert payload['operational_tier'] == 'unsupported'
    assert payload['review_required'] is True


class _FakeExecuteResult:
    def __init__(self, value=None):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def first(self):
        return self._value


class _FakeModelingDb:
    def __init__(self, field, crop):
        self.field = field
        self.crop = crop
        self.added = []

    async def execute(self, stmt):
        text = str(stmt)
        if "FROM fields" in text:
            return _FakeExecuteResult(self.field)
        if "FROM crops" in text:
            return _FakeExecuteResult(self.crop)
        return _FakeExecuteResult(None)

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        if self.added:
            setattr(self.added[-1], "id", 1)
        return None


@pytest.mark.asyncio
async def test_modeling_service_uses_weekly_profile_for_mechanistic_weather_scenarios(monkeypatch):
    field_id = uuid4()
    organization_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = SimpleNamespace(
        id=field_id,
        organization_id=organization_id,
        area_m2=10000.0,
        geom=from_shape(polygon, srid=4326),
    )
    crop = SimpleNamespace(code="wheat", yield_baseline_kg_ha=4200.0)
    db = _FakeModelingDb(field, crop)
    service = ModelingService(db)
    service.yield_service.get_or_create_prediction = AsyncMock(
        return_value={
            "id": 11,
            "field_id": str(field_id),
            "estimated_yield_kg_ha": 4050.0,
            "prediction_date": "2026-03-11T10:00:00+00:00",
            "model_version": "yield_v2",
            "confidence_tier": "global_baseline",
            "model_applicability": {"supported": True},
            "training_domain": {"samples": 0},
            "feature_coverage": {"available": ["weekly_profile"], "missing": []},
            "crop_suitability": {"status": "moderate", "warnings": []},
            "geometry_quality_impact": {"status": "neutral", "penalty_factor": 1.0},
            "crop_hint": {"top_crop_code": "wheat", "top_probability": 1.0},
            "freshness": {"dataset_version": "weekly_v3"},
        }
    )

    weekly_rows = [
        SimpleNamespace(
            week_number=18,
            geometry_confidence=0.74,
            tta_consensus=0.81,
            boundary_uncertainty=0.12,
            weather_coverage=0.9,
            satellite_coverage=0.6,
            ndvi_mean=0.58,
            ndre_mean=0.18,
            ndmi_mean=0.09,
            precipitation_mm=12.0,
            tmean_c=15.0,
        )
        for _ in range(4)
    ]
    monkeypatch.setattr(modeling_service_module, "ensure_weekly_profile", AsyncMock(return_value=weekly_rows))
    monkeypatch.setattr(modeling_service_module, "profile_has_signal", lambda rows: True)
    monkeypatch.setattr(
        modeling_service_module,
        "rows_to_weekly_inputs",
        lambda rows: [
            SimpleNamespace(
                week=18,
                tmean_c=15.0,
                tmax_c=23.0,
                tmin_c=8.0,
                precipitation_mm=10.0,
                vpd_kpa=1.2,
                solar_radiation_mj=16.0,
                soil_moisture=0.24,
                ndvi=0.61,
                ndre=0.18,
                ndmi=0.09,
                irrigation_mm=0.0,
                n_applied_kg_ha=0.0,
            )
            for _ in rows
        ],
    )
    monkeypatch.setattr(
        modeling_service_module,
        "load_crop_hint",
        AsyncMock(return_value={"top_crop_code": "wheat", "top_probability": 1.0}),
    )
    monkeypatch.setattr(
        service.yield_service.weather_service,
        "get_forecast",
        AsyncMock(
            return_value={
                "provider": "openmeteo",
                "days": 2,
                "forecast": [
                    {"date": "2026-03-12", "temp_mean_c": 11.0, "precipitation_mm": 2.0},
                    {"date": "2026-03-13", "temp_mean_c": 12.0, "precipitation_mm": 1.0},
                ],
                "freshness": {"freshness_state": "fresh"},
            }
        ),
    )

    results = iter(
        [
            SimpleNamespace(
                baseline_yield_kg_ha=4000.0,
                harvest_index=0.44,
                trace=[{"week": 18, "root_zone_water_mm": 82.0, "water_stress": 0.22, "heat_stress": 0.05, "vpd_stress": 0.11, "nutrient_stress": 0.07, "yield_potential_remaining": 0.91}],
                final_state=SimpleNamespace(
                    root_zone_water_mm=82.0,
                    water_stress=0.22,
                    heat_stress=0.05,
                    vpd_stress=0.11,
                    nutrient_stress=0.07,
                    canopy_cover=0.68,
                    biomass_proxy=7100.0,
                    yield_potential_remaining=0.91,
                ),
                params_used={},
            ),
            SimpleNamespace(
                baseline_yield_kg_ha=4280.0,
                harvest_index=0.46,
                trace=[{"week": 18, "root_zone_water_mm": 95.0, "water_stress": 0.08, "heat_stress": 0.03, "vpd_stress": 0.09, "nutrient_stress": 0.06, "yield_potential_remaining": 0.95}],
                final_state=SimpleNamespace(
                    root_zone_water_mm=95.0,
                    water_stress=0.08,
                    heat_stress=0.03,
                    vpd_stress=0.09,
                    nutrient_stress=0.06,
                    canopy_cover=0.73,
                    biomass_proxy=7520.0,
                    yield_potential_remaining=0.95,
                ),
                params_used={},
            ),
        ]
    )
    monkeypatch.setattr(
        mechanistic_engine_module,
        "run_mechanistic_baseline",
        lambda **kwargs: next(results),
    )

    payload = await service.simulate(
        field_id,
        organization_id=organization_id,
        crop_code="wheat",
        irrigation_pct=0.0,
        fertilizer_pct=0.0,
        expected_rain_mm=0.0,
        temperature_delta_c=2.0,
        precipitation_factor=0.85,
        save=False,
    )

    assert payload["supported"] is True
    assert payload["trace_supported"] is True
    assert payload["engine_version"] == "mechanistic_v1"
    assert payload["scenario_yield_kg_ha"] > payload["baseline_yield_kg_ha"]
    assert payload["comparison"]["factor_breakdown"]
    assert payload["scenario_time_series"]["baseline"]
    assert payload["forecast_curve"]["baseline_points"]
    assert payload["forecast_curve"]["scenario_points"]
    assert payload["forecast_curve"]["scenario_points"][0]["gdd_daily"] > payload["forecast_curve"]["baseline_points"][0]["gdd_daily"]
    assert payload["scenario_water_balance"]["baseline"]
    assert payload["crop_hint"]["top_crop_code"] == "wheat"
