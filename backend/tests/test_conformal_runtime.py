from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from geoalchemy2.shape import from_shape
import pytest
from shapely.geometry import Polygon

import services.yield_service as yield_service_module
from services.conformal_service import ConformalCalibrationSet, ConformalService
from services.yield_service import YieldService
import storage.db as db_module
import tasks.model_tasks as model_tasks
from storage.db import Crop, Field, YieldModel


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        if isinstance(self._value, list):
            return self._value
        return []


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _SessionContext:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _YieldDb:
    def __init__(self, field: Field, models: list[YieldModel] | None = None):
        self._field = field
        self._models = models or []
        self.saved = []

    async def execute(self, stmt):
        text = str(stmt)
        if "FROM fields" in text:
            return _ScalarResult(self._field)
        if "FROM yield_models" in text:
            return _ScalarResult(self._models)
        return _ScalarResult(None)

    def add(self, row):
        self.saved.append(row)

    async def flush(self):
        return None


class _CalibrationDb:
    def __init__(self, model, rows, crop):
        self.model = model
        self.rows = rows
        self.crop = crop
        self.committed = False

    async def execute(self, stmt):
        text = str(stmt)
        if "FROM yield_models" in text:
            return _ScalarResult(self.model)
        if "FROM yield_observations" in text:
            return _RowsResult(self.rows)
        if "FROM crops" in text:
            return _ScalarResult(self.crop)
        return _ScalarResult(None)

    async def commit(self):
        self.committed = True
        return None


@pytest.mark.asyncio
async def test_load_conformal_service_prefers_registry_sets():
    field = Field(
        id=uuid4(),
        organization_id=uuid4(),
        aoi_run_id=uuid4(),
        geom=from_shape(Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)]), srid=4326),
        area_m2=10000.0,
        perimeter_m=400.0,
        quality_score=0.9,
        source="ml",
    )
    org_model = YieldModel(
        organization_id=field.organization_id,
        model_name="global_residual",
        model_version="org_model_v1",
        training_summary={},
        metrics={},
        config_snapshot={
            "conformal_sets": [
                {
                    "crop_code": "wheat",
                    "region_key": "lat_55_62",
                    "residuals": [100.0] * 6,
                    "n_calibration": 6,
                    "model_version": "org_model_v1",
                }
            ]
        },
        status="deployed",
        created_at=datetime.now(timezone.utc),
    )
    global_model = YieldModel(
        organization_id=None,
        model_name="global_residual",
        model_version="global_model_v1",
        training_summary={},
        metrics={},
        config_snapshot={
            "conformal_sets": [
                {
                    "crop_code": "wheat",
                    "region_key": "lat_55_62",
                    "residuals": [250.0] * 6,
                    "n_calibration": 6,
                    "model_version": "global_model_v1",
                }
            ]
        },
        status="validated",
        created_at=datetime.now(timezone.utc),
    )
    service = YieldService(_YieldDb(field, [global_model, org_model]))

    conformal_svc, meta = await service._load_conformal_service(
        organization_id=field.organization_id,
        preferred_model_version="org_model_v1",
    )

    assert conformal_svc is not None
    assert meta["model_version"] == "org_model_v1"
    interval = conformal_svc.compute_interval(
        crop_code="wheat",
        region_key="lat_55_62",
        point_estimate=4000.0,
    )
    assert interval.width == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_yield_service_uses_conformal_interval_from_registry(monkeypatch):
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
        source="ml",
    )
    db = _YieldDb(field)
    service = YieldService(db)
    crop = Crop(
        id=1,
        code="wheat",
        name="Пшеница",
        category="grain",
        yield_baseline_kg_ha=4200.0,
        ndvi_target=0.72,
        base_temp_c=5.0,
        description="Тестовая культура",
    )
    svc = ConformalService()
    svc.register_calibration_set(
        ConformalCalibrationSet(
            crop_code="wheat",
            region_key="lat_55_62",
            residuals=[120.0] * 12,
            model_version="registry_conformal_v1",
            n_calibration=12,
        )
    )

    monkeypatch.setattr(service.crop_service, "get_crop_by_code", AsyncMock(return_value=crop))
    monkeypatch.setattr(yield_service_module, "ensure_weekly_profile", AsyncMock(return_value=[]))
    monkeypatch.setattr(yield_service_module, "load_crop_hint", AsyncMock(return_value={}))
    monkeypatch.setattr(
        service.field_analytics_service,
        "_collect_field_metrics",
        AsyncMock(return_value=({"ndvi": {"mean": 0.62}, "ndmi": {"mean": 0.31}}, {})),
    )
    monkeypatch.setattr(service, "_seasonal_weather_summary", AsyncMock(return_value={}))
    monkeypatch.setattr(service.temporal_analytics_service, "get_temporal_analytics", AsyncMock(return_value={}))
    monkeypatch.setattr(service.temporal_analytics_service, "get_management_zones", AsyncMock(return_value={}))
    monkeypatch.setattr(
        service,
        "_build_current_features",
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
                    "historical_field_mean_yield": None,
                    "current_ndvi_mean": 0.62,
                    "current_ndmi_mean": 0.31,
                },
                set(),
            )
        ),
    )
    monkeypatch.setattr(service, "_load_training_rows", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        service.weather_service,
        "get_current_weather",
        AsyncMock(return_value={"temperature_c": 21.0, "precipitation_mm": 8.0, "soil_moisture": 0.22}),
    )
    monkeypatch.setattr(
        service,
        "_load_conformal_service",
        AsyncMock(return_value=(svc, {"model_version": "registry_conformal_v1", "n_sets": 1})),
    )

    payload = await service.get_or_create_prediction(
        field_id,
        organization_id=field.organization_id,
        crop_code="wheat",
        refresh=True,
    )

    assert payload["prediction_interval"]["lower"] is not None
    assert payload["details"]["conformal_interval"]["method"] == "conformal"
    assert payload["details"]["conformal_interval"]["model_version"] == "registry_conformal_v1"
    assert payload["data_quality"]["interval_method"] == "conformal"


def test_refresh_conformal_calibration_task_populates_model_snapshot(monkeypatch):
    organization_id = uuid4()
    field_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = SimpleNamespace(id=field_id, area_m2=10000.0, geom=from_shape(polygon, srid=4326), quality_score=0.8)
    crop_assignment = SimpleNamespace(crop_code="wheat")
    crop = SimpleNamespace(code="wheat", yield_baseline_kg_ha=4200.0)
    model = YieldModel(
        organization_id=organization_id,
        model_name="global_residual",
        model_version="residual_ridge_v1_6s",
        training_summary={},
        metrics={},
        config_snapshot={"feature_names": ["peak_ndvi"], "coefficients": [0.0]},
        status="validated",
        created_at=datetime.now(timezone.utc),
    )

    rows = []
    for season_id in range(1, 7):
        obs = SimpleNamespace(
            organization_id=organization_id,
            field_season_id=season_id,
            yield_kg_ha=4000.0 + season_id * 10.0,
        )
        season = SimpleNamespace(id=season_id, season_year=2020 + season_id)
        rows.append((obs, season, field, crop_assignment))

    weekly_rows = [
        SimpleNamespace(
            week_number=18,
            ndvi_mean=0.5,
            ndre_mean=0.2,
            ndmi_mean=0.1,
            water_stress=0.0,
            heat_stress=0.0,
            tmean_c=18.0,
            tmax_c=24.0,
            tmin_c=11.0,
            precipitation_mm=10.0,
            vpd_kpa=1.1,
            irrigation_mm=0.0,
            n_applied_kg_ha=0.0,
            feature_schema_version="weekly_v3",
        )
    ] * 6
    db = _CalibrationDb(model, rows, crop)

    monkeypatch.setattr(db_module, "get_session_factory", lambda: (lambda: _SessionContext(db)))
    monkeypatch.setattr("services.weekly_profile_service.ensure_weekly_profile", AsyncMock(return_value=weekly_rows))
    monkeypatch.setattr("services.weekly_profile_service.profile_has_signal", lambda rows: True)
    monkeypatch.setattr(
        "services.weekly_profile_service.rows_to_weekly_inputs",
        lambda rows: [
            SimpleNamespace(
                week=18,
                tmean_c=18.0,
                tmax_c=24.0,
                tmin_c=11.0,
                precipitation_mm=10.0,
                vpd_kpa=1.1,
                ndvi=0.5,
                ndre=0.2,
                ndmi=0.1,
                irrigation_mm=0.0,
                n_applied_kg_ha=0.0,
                week_start=None,
                season_year=2026,
                previous_crop_code=None,
            )
        ] * 6,
    )
    monkeypatch.setattr(
        "services.mechanistic_engine.run_mechanistic_baseline",
        lambda **kwargs: SimpleNamespace(baseline_yield_kg_ha=3900.0),
    )

    payload = model_tasks.refresh_conformal_calibration.run(
        model_version="residual_ridge_v1_6s",
        organization_id_str=str(organization_id),
    )

    assert payload["status"] == "calibrated"
    assert payload["bucket_count"] >= 1
    assert "conformal_sets" in (model.config_snapshot or {})
    assert model.calibration_set_hash
    assert db.committed is True
