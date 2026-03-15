from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

from services.field_analytics_service import FieldAnalyticsService
import services.field_analytics_service as field_analytics_module
from storage.db import Crop, Field, YieldPrediction


class _FakeExecuteResult:
    def __init__(self, row=None) -> None:
        self._row = row

    def first(self):
        return self._row


class _FakeDb:
    def __init__(self, row) -> None:
        self._row = row

    async def execute(self, stmt):
        return _FakeExecuteResult(self._row)


@pytest.mark.asyncio
async def test_latest_prediction_includes_forecast_curve(monkeypatch):
    org_id = uuid4()
    field_id = uuid4()
    polygon = Polygon([(36.0, 45.0), (36.1, 45.0), (36.1, 45.1), (36.0, 45.1)])
    field = Field(
        id=field_id,
        organization_id=org_id,
        aoi_run_id=uuid4(),
        geom=from_shape(polygon, srid=4326),
        area_m2=1156300.0,
        perimeter_m=4800.0,
        quality_score=0.91,
        source='autodetect',
    )
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
    prediction = YieldPrediction(
        organization_id=org_id,
        field_id=field_id,
        crop_id=crop.id,
        prediction_date=datetime.now(timezone.utc),
        estimated_yield_kg_ha=3374.0,
        confidence=0.5,
        model_version='global_baseline_v3',
        details={'supported': True, 'confidence_tier': 'global_baseline'},
        input_features={},
        explanation={},
        data_quality={},
    )
    service = FieldAnalyticsService(_FakeDb((prediction, crop)))
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
    monkeypatch.setattr(field_analytics_module.TemporalAnalyticsService, 'get_temporal_analytics', AsyncMock(return_value={}))
    monkeypatch.setattr(field_analytics_module.TemporalAnalyticsService, 'get_management_zones', AsyncMock(return_value={'summary': {}}))

    payload = await service._get_latest_prediction(field_id, organization_id=org_id, field=field)

    assert payload is not None
    assert payload['forecast_curve']['provider'] == 'openmeteo'
    assert len(payload['forecast_curve']['points']) == 2
    assert payload['forecast_curve']['points'][1]['gdd_cumulative'] == pytest.approx(15.0)
