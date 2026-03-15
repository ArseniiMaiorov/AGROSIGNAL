from __future__ import annotations

import base64
import importlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np

from api.dependencies import RequestContext, get_current_context
from services.archive_service import ArchiveService
from services.crop_service import CropService
from services.data_import_service import DataImportService
from services.field_analytics_service import FieldAnalyticsService
from services.labeling_service import LabelingService
from services.mlops_service import MlOpsService
from services.modeling_service import ModelingService
from services.status_service import StatusService
from services.weather_service import WeatherService
from services.yield_service import YieldService
from storage.db import get_db
from storage.fields_repo import FieldsRepository


async def _noop() -> None:
    return None


async def _fake_db():
    yield None


def _client(monkeypatch: any) -> TestClient:
    import storage.db as db

    monkeypatch.setattr(db, "init_db", _noop)
    monkeypatch.setattr(db, "seed_defaults", _noop)
    monkeypatch.setattr(db, "seed_layers", _noop)
    main = importlib.import_module("main")
    main.app.dependency_overrides = {}
    main.app.dependency_overrides[get_db] = _fake_db
    main.app.dependency_overrides[get_current_context] = lambda: RequestContext(
        organization_id=uuid4(),
        user_id=uuid4(),
        email="tests@local",
        role_names=("tenant_admin",),
        permissions=frozenset(
            {
                "fields:read",
                "fields:write",
                "weather:read",
                "layers:read",
                "crops:read",
                "predictions:read",
                "predictions:write",
                "scenarios:read",
                "scenarios:write",
                "archive:read",
                "archive:write",
                "labeling:read",
                "labeling:write",
                "labeling:review",
                "imports:read",
                "imports:write",
                "mlops:read",
                "mlops:write",
                "status:read",
            }
        ),
    )
    return TestClient(main.app)


def test_status_route(monkeypatch):
    payload = {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {"database": {"status": "online", "detail": 1}},
        "runs": {"running": 0, "total": 1},
    }
    monkeypatch.setattr(StatusService, "get_system_status", AsyncMock(return_value=payload))
    with _client(monkeypatch) as client:
        response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_weather_routes(monkeypatch):
    monkeypatch.setattr(
        WeatherService,
        "get_current_weather",
        AsyncMock(
            return_value={
                "latitude": 59.0,
                "longitude": 30.0,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "provider": "openmeteo",
                "cached": False,
                "temperature_c": 12.5,
                "apparent_temperature_c": 11.0,
                "precipitation_mm": 0.0,
                "wind_speed_m_s": 4.2,
                "u_wind_10m": -1.5,
                "v_wind_10m": -3.9,
                "wind_direction_deg": 22.0,
                "humidity_pct": 65.0,
                "cloud_cover_pct": 40.0,
                "pressure_hpa": 1006.0,
                "soil_moisture": 0.2,
            }
        ),
    )
    monkeypatch.setattr(
        WeatherService,
        "get_forecast",
        AsyncMock(
            return_value={
                "latitude": 59.0,
                "longitude": 30.0,
                "provider": "openmeteo",
                "days": 3,
                "forecast": [{"date": "2026-03-07", "temp_max_c": 10.0, "temp_min_c": 4.0}],
                "error": None,
            }
        ),
    )
    with _client(monkeypatch) as client:
        current = client.get("/api/v1/weather/current", params={"lat": 59, "lon": 30})
        forecast = client.get("/api/v1/weather/forecast", params={"lat": 59, "lon": 30, "days": 3})
    assert current.status_code == 200
    assert forecast.status_code == 200
    assert current.json()["provider"] == "openmeteo"
    assert forecast.json()["days"] == 3


def test_crops_and_prediction_routes(monkeypatch):
    crop = SimpleNamespace(
        id=1,
        code="wheat",
        name="Пшеница",
        category="grain",
        yield_baseline_kg_ha=4200.0,
        ndvi_target=0.72,
        base_temp_c=5.0,
        description="Описание",
    )
    prediction = {
        "id": 1,
        "field_id": str(uuid4()),
        "crop": {"id": 1, "code": "wheat", "name": "Пшеница"},
        "prediction_date": datetime.now(timezone.utc).isoformat(),
        "estimated_yield_kg_ha": 4300.0,
        "confidence": 0.75,
        "confidence_tier": "global_baseline",
        "model_version": "heuristic_v1",
        "details": {"quality_score": 0.8},
    }
    monkeypatch.setattr(CropService, "list_crops", AsyncMock(return_value=[crop]))
    monkeypatch.setattr(YieldService, "get_or_create_prediction", AsyncMock(return_value=prediction))
    field_id = uuid4()
    with _client(monkeypatch) as client:
        crops = client.get("/api/v1/crops")
        pred = client.get(f"/api/v1/predictions/field/{field_id}")
    assert crops.status_code == 200
    assert pred.status_code == 200
    assert crops.json()["crops"][0]["code"] == "wheat"
    assert pred.json()["estimated_yield_kg_ha"] == 4300.0


def test_prediction_weekly_profile_routes(monkeypatch):
    predictions_api = importlib.import_module("api.predictions")
    field_id = uuid4()
    rows = [
        {
            "week_number": 18,
            "week_start": "2026-05-04",
            "ndvi_mean": 0.61,
            "precipitation_mm": 12.0,
            "geometry_confidence": 0.74,
            "tta_consensus": 0.81,
            "boundary_uncertainty": 0.12,
            "feature_schema_version": "weekly_v3",
        }
    ]
    monkeypatch.setattr(
        predictions_api,
        "ensure_weekly_profile",
        AsyncMock(return_value=[SimpleNamespace(**rows[0])]),
    )
    monkeypatch.setattr(
        predictions_api,
        "load_crop_hint",
        AsyncMock(return_value={"top_crop_code": "wheat", "top_probability": 1.0}),
    )
    monkeypatch.setattr(
        predictions_api,
        "serialize_weekly_feature_rows",
        lambda _rows: rows,
    )
    monkeypatch.setattr(
        predictions_api,
        "summarize_geometry_quality",
        lambda _rows: {
            "geometry_confidence": 0.74,
            "tta_consensus": 0.81,
            "boundary_uncertainty": 0.12,
        },
    )

    with _client(monkeypatch) as client:
        response = client.get(f"/api/v1/predictions/field/{field_id}/weekly-profile")
        rebuilt = client.post(f"/api/v1/predictions/field/{field_id}/weekly-profile/backfill")

    assert response.status_code == 200
    assert rebuilt.status_code == 200
    assert response.json()["feature_schema_version"] == "weekly_v3"
    assert response.json()["crop_hint"]["top_crop_code"] == "wheat"
    assert rebuilt.json()["rows"][0]["week_number"] == 18


def test_prediction_job_routes(monkeypatch):
    predictions_api = importlib.import_module("api.predictions")

    class _DummyTask:
        @staticmethod
        def delay(*_args, **_kwargs):
            return SimpleNamespace(id="prediction-job-1")

    monkeypatch.setattr(predictions_api, "run_prediction_job", _DummyTask())
    monkeypatch.setattr(
        predictions_api,
        "prime_async_job",
        lambda **kwargs: {
            "status": "queued",
            "progress": 0,
            "stage_label": "queued",
            "stage_detail": "waiting for worker",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0,
            "estimated_remaining_s": None,
            "logs": ["queued"],
            "error_msg": None,
        },
    )

    statuses = iter(
        [
            {
                "job_type": "prediction",
                "organization_id": str(uuid4()),
                "status": "running",
                "progress": 42,
                "stage_label": "features",
                "stage_detail": "weather and field analytics",
                "logs": ["queued", "running"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result_ready": False,
            },
            {
                "job_type": "prediction",
                "organization_id": str(uuid4()),
                "status": "done",
                "progress": 100,
                "stage_label": "done",
                "stage_detail": "prediction completed",
                "logs": ["queued", "running", "done"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result_ready": True,
                "result": {
                    "field_id": str(uuid4()),
                    "estimated_yield_kg_ha": 4300.0,
                },
            },
        ]
    )

    def _payload(_task_id):
        payload = next(statuses)
        payload["organization_id"] = str(_client_ctx.organization_id)
        return payload

    # piggyback on the client override context produced below
    _client_ctx = RequestContext(
        organization_id=uuid4(),
        user_id=uuid4(),
        email="tests@local",
        role_names=("tenant_admin",),
        permissions=frozenset({"predictions:read", "predictions:write"}),
    )

    def _client_with_prediction_ctx() -> TestClient:
        import storage.db as db

        monkeypatch.setattr(db, "init_db", _noop)
        monkeypatch.setattr(db, "seed_defaults", _noop)
        monkeypatch.setattr(db, "seed_layers", _noop)
        main = importlib.import_module("main")
        main.app.dependency_overrides = {}
        main.app.dependency_overrides[get_db] = _fake_db
        main.app.dependency_overrides[get_current_context] = lambda: _client_ctx
        return TestClient(main.app)

    monkeypatch.setattr(predictions_api, "get_async_job_payload", _payload)

    with _client_with_prediction_ctx() as client:
        submitted = client.post(f"/api/v1/predictions/field/{uuid4()}/jobs", params={"refresh": True})
        status = client.get("/api/v1/predictions/jobs/prediction-job-1")
        result = client.get("/api/v1/predictions/jobs/prediction-job-1/result")

    assert submitted.status_code == 200
    assert submitted.json()["task_id"] == "prediction-job-1"
    assert status.status_code == 200
    assert status.json()["progress"] == 42
    assert result.status_code == 200
    assert result.json()["result"]["estimated_yield_kg_ha"] == 4300.0


def test_modeling_archive_and_manual_routes(monkeypatch):
    field_id = uuid4()
    aoi_run_id = uuid4()
    monkeypatch.setattr(
        ModelingService,
        "simulate",
        AsyncMock(
            return_value={
                "field_id": str(field_id),
                "baseline_yield_kg_ha": 4000.0,
                "scenario_yield_kg_ha": 4400.0,
                "predicted_yield_change_pct": 10.0,
                "factors": {"irrigation_pct": 10.0, "fertilizer_pct": 5.0, "expected_rain_mm": 20.0},
            }
        ),
    )
    monkeypatch.setattr(
        ArchiveService,
        "create_archive",
        AsyncMock(
            return_value={
                "id": 1,
                "field_id": str(field_id),
                "date_from": datetime.now(timezone.utc).isoformat(),
                "date_to": datetime.now(timezone.utc).isoformat(),
                "layers": ["ndvi"],
                "file_path": "/tmp/test.zip",
                "status": "ready",
                "expires_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "meta": {},
            }
        ),
    )
    monkeypatch.setattr(
        FieldsRepository,
        "create_manual_field",
        AsyncMock(
            return_value=SimpleNamespace(
                id=field_id,
                aoi_run_id=aoi_run_id,
                area_m2=1000.0,
                perimeter_m=150.0,
                quality_score=1.0,
                source="manual",
                created_at=datetime.now(timezone.utc),
            )
        ),
    )
    with _client(monkeypatch) as client:
        modeling = client.post(
            "/api/v1/modeling/simulate",
            json={
                "field_id": str(field_id),
                "crop_code": "wheat",
                "irrigation_pct": 10,
                "fertilizer_pct": 5,
                "expected_rain_mm": 20,
            },
        )
        archive = client.post(
            "/api/v1/archive/create",
            json={
                "field_id": str(field_id),
                "date_from": datetime.now(timezone.utc).isoformat(),
                "date_to": datetime.now(timezone.utc).isoformat(),
                "layers": ["ndvi"],
            },
        )
        manual = client.post(
            "/api/v1/manual/fields",
            json={
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[30.0, 59.0], [30.1, 59.0], [30.1, 59.1], [30.0, 59.0]]],
                },
                "quality_score": 1.0,
            },
        )
    assert modeling.status_code == 200
    assert archive.status_code == 200
    assert manual.status_code == 200


def test_modeling_job_routes(monkeypatch):
    modeling_api = importlib.import_module("api.modeling")

    class _DummyTask:
        @staticmethod
        def delay(*_args, **_kwargs):
            return SimpleNamespace(id="scenario-job-1")

    monkeypatch.setattr(modeling_api, "run_modeling_job", _DummyTask())
    monkeypatch.setattr(
        modeling_api,
        "prime_async_job",
        lambda **kwargs: {
            "status": "queued",
            "progress": 0,
            "stage_label": "queued",
            "stage_detail": "waiting for worker",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": 0,
            "estimated_remaining_s": None,
            "logs": ["queued"],
            "error_msg": None,
        },
    )

    ctx = RequestContext(
        organization_id=uuid4(),
        user_id=uuid4(),
        email="tests@local",
        role_names=("tenant_admin",),
        permissions=frozenset({"scenarios:read", "scenarios:write"}),
    )

    def _client_with_modeling_ctx() -> TestClient:
        import storage.db as db

        monkeypatch.setattr(db, "init_db", _noop)
        monkeypatch.setattr(db, "seed_defaults", _noop)
        monkeypatch.setattr(db, "seed_layers", _noop)
        main = importlib.import_module("main")
        main.app.dependency_overrides = {}
        main.app.dependency_overrides[get_db] = _fake_db
        main.app.dependency_overrides[get_current_context] = lambda: ctx
        return TestClient(main.app)

    statuses = iter(
        [
            {
                "job_type": "scenario",
                "organization_id": str(ctx.organization_id),
                "status": "running",
                "progress": 57,
                "stage_label": "counterfactual",
                "stage_detail": "agronomic response evaluation",
                "logs": ["queued", "running"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result_ready": False,
            },
            {
                "job_type": "scenario",
                "organization_id": str(ctx.organization_id),
                "status": "done",
                "progress": 100,
                "stage_label": "done",
                "stage_detail": "scenario completed",
                "logs": ["queued", "running", "done"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "result_ready": True,
                "result": {
                    "field_id": str(uuid4()),
                    "predicted_yield_change_pct": 4.5,
                },
            },
        ]
    )
    monkeypatch.setattr(modeling_api, "get_async_job_payload", lambda _task_id: next(statuses))

    with _client_with_modeling_ctx() as client:
        submitted = client.post(
            "/api/v1/modeling/jobs",
            json={
                "field_id": str(uuid4()),
                "crop_code": "wheat",
                "irrigation_pct": 10,
                "fertilizer_pct": 5,
                "expected_rain_mm": 20,
            },
        )
        status = client.get("/api/v1/modeling/jobs/scenario-job-1")
        result = client.get("/api/v1/modeling/jobs/scenario-job-1/result")

    assert submitted.status_code == 200
    assert submitted.json()["task_id"] == "scenario-job-1"
    assert status.status_code == 200
    assert status.json()["stage_label"] == "counterfactual"
    assert result.status_code == 200
    assert result.json()["result"]["predicted_yield_change_pct"] == 4.5


def test_fields_geojson_merge_and_split_routes(monkeypatch):
    field_id = uuid4()
    second_id = uuid4()
    run_id = uuid4()
    created_at = datetime.now(timezone.utc)

    monkeypatch.setattr(
        FieldsRepository,
        "get_all_fields_geojson",
        AsyncMock(
            return_value={
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": []},
                        "properties": {
                            "field_id": str(field_id),
                            "aoi_run_id": str(run_id),
                            "area_m2": 1200.0,
                            "perimeter_m": 180.0,
                            "quality_score": 0.7,
                            "source": "autodetect",
                            "created_at": created_at.isoformat(),
                            "has_archive": True,
                        },
                    }
                ],
            }
        ),
    )
    monkeypatch.setattr(
        FieldsRepository,
        "merge_fields",
        AsyncMock(
            return_value=SimpleNamespace(
                id=field_id,
                aoi_run_id=run_id,
                area_m2=2500.0,
                perimeter_m=260.0,
                quality_score=0.8,
                source="manual",
                created_at=created_at,
            )
        ),
    )
    monkeypatch.setattr(
        FieldsRepository,
        "split_field",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=field_id,
                    aoi_run_id=run_id,
                    area_m2=1300.0,
                    perimeter_m=190.0,
                    quality_score=0.8,
                    source="manual",
                    created_at=created_at,
                ),
                SimpleNamespace(
                    id=second_id,
                    aoi_run_id=run_id,
                    area_m2=1200.0,
                    perimeter_m=175.0,
                    quality_score=0.8,
                    source="manual",
                    created_at=created_at,
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        FieldsRepository,
        "delete_field",
        AsyncMock(return_value={"field_id": field_id, "aoi_run_id": run_id, "deleted": True}),
    )

    with _client(monkeypatch) as client:
        geojson = client.get("/api/v1/fields/geojson")
        merged = client.post("/api/v1/fields/merge", json={"field_ids": [str(field_id), str(second_id)]})
        split = client.post(
            "/api/v1/fields/split",
            json={
                "field_id": str(field_id),
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[30.0, 59.0], [30.2, 59.2]],
                },
            },
        )
        deleted = client.delete(f"/api/v1/fields/{field_id}")

    assert geojson.status_code == 200
    assert geojson.json()["features"][0]["properties"]["has_archive"] is True
    assert merged.status_code == 200
    assert merged.json()["source"] == "manual"
    assert split.status_code == 200
    assert len(split.json()["fields"]) == 2
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


def test_dashboard_scenarios_and_archive_view_routes(monkeypatch):
    field_id = uuid4()
    second_id = uuid4()
    archive_id = 7
    timestamp = datetime.now(timezone.utc).isoformat()
    dashboard_payload = {
        "mode": "single",
        "field": {
            "field_id": str(field_id),
            "aoi_run_id": str(uuid4()),
            "area_m2": 12000.0,
            "perimeter_m": 460.0,
            "quality_score": 0.83,
            "source": "autodetect",
            "created_at": timestamp,
        },
        "kpis": {"prediction_ready": True, "archive_count": 1, "scenario_count": 1, "observation_cells": 18},
        "current_metrics": {
            "ndvi": {
                "label": "NDVI",
                "coverage": 18,
                "mean": 0.62,
                "min": 0.41,
                "max": 0.78,
                "median": 0.63,
                "p25": 0.57,
                "p75": 0.68,
            }
        },
        "series": {"ndvi": [{"observed_at": timestamp, "mean": 0.62}]},
        "histograms": {"ndvi": {"bins": [0.4, 0.5, 0.6], "counts": [2, 4]}},
        "prediction": {
            "id": 1,
            "prediction_date": timestamp,
            "estimated_yield_kg_ha": 4100.0,
            "confidence": 0.77,
            "model_version": "heuristic_v1",
            "details": {},
            "input_features": {"ndvi_mean": 0.62},
            "explanation": {"summary": "Тестовое объяснение", "drivers": []},
            "data_quality": {"valid_feature_count": 6},
        },
        "archives": [{"id": archive_id, "created_at": timestamp, "date_from": timestamp, "date_to": timestamp, "status": "ready", "meta": {}}],
        "scenarios": [{"id": 1, "field_id": str(field_id), "scenario_name": "Базовый", "created_at": timestamp, "model_version": "heuristic_v1", "parameters": {}, "baseline_snapshot": {}, "result_snapshot": {}, "delta_pct": 3.1}],
        "data_quality": {"observation_cells": 18, "metrics_available": ["ndvi"], "has_time_series": True, "has_prediction": True},
        "fields": [],
    }
    group_payload = {
        "mode": "group",
        "selection": {"field_count": 2, "field_ids": [str(field_id), str(second_id)], "total_area_m2": 33000.0},
        "kpis": {"field_count": 2, "observation_cells": 16, "metric_count": 3},
        "current_metrics": {"ndvi": {"label": "NDVI", "mean": 0.61, "median": 0.6, "min": 0.4, "max": 0.8, "p25": 0.52, "p75": 0.68, "coverage": 16}},
        "series": {"ndvi": [{"observed_at": timestamp, "mean": 0.61}]},
        "histograms": {},
        "fields": [{"field_id": str(field_id), "aoi_run_id": str(uuid4()), "area_m2": 12000.0, "perimeter_m": 400.0, "quality_score": 0.8, "source": "autodetect", "created_at": timestamp}],
        "data_quality": {"observation_cells": 16, "metrics_available": ["ndvi"], "has_time_series": True},
    }
    archive_view_payload = {
        "archive": {
            "id": archive_id,
            "field_id": str(field_id),
            "date_from": timestamp,
            "date_to": timestamp,
            "layers": ["ndvi"],
            "file_path": "/tmp/test.zip",
            "status": "ready",
            "expires_at": timestamp,
            "created_at": timestamp,
            "meta": {},
            "field_snapshot": {"field_id": str(field_id)},
            "prediction_snapshot": {"estimated_yield_kg_ha": 4100.0},
            "metrics_snapshot": {"current_metrics": {"ndvi": {"mean": 0.62}}},
            "weather_snapshot": {"temperature_c": 12.0},
            "scenario_snapshot": {"items": []},
            "model_meta": {"model_version": "heuristic_v1"},
        },
        "snapshot": {
            "field_snapshot": {"field_id": str(field_id)},
            "prediction_snapshot": {"estimated_yield_kg_ha": 4100.0, "confidence": 0.77},
            "metrics_snapshot": {"current_metrics": {"ndvi": {"mean": 0.62}}},
            "weather_snapshot": {"temperature_c": 12.0},
            "scenario_snapshot": {"items": []},
            "model_meta": {"model_version": "heuristic_v1"},
        },
    }

    monkeypatch.setattr(FieldAnalyticsService, "get_field_dashboard", AsyncMock(return_value=dashboard_payload))
    monkeypatch.setattr(FieldAnalyticsService, "get_group_dashboard", AsyncMock(return_value=group_payload))
    monkeypatch.setattr(ModelingService, "list_scenarios", AsyncMock(return_value=dashboard_payload["scenarios"]))
    monkeypatch.setattr(ArchiveService, "get_archive_view", AsyncMock(return_value=archive_view_payload))

    with _client(monkeypatch) as client:
        dashboard = client.get(f"/api/v1/fields/{field_id}/dashboard")
        group = client.post("/api/v1/fields/dashboard/group", json={"field_ids": [str(field_id), str(second_id)]})
        scenarios = client.get("/api/v1/modeling/scenarios", params={"field_id": str(field_id)})
        archive_view = client.get(f"/api/v1/archive/{archive_id}/view")

    assert dashboard.status_code == 200
    assert dashboard.json()["prediction"]["estimated_yield_kg_ha"] == 4100.0
    assert group.status_code == 200
    assert group.json()["selection"]["field_count"] == 2
    assert scenarios.status_code == 200
    assert scenarios.json()["scenarios"][0]["scenario_name"] == "Базовый"
    assert archive_view.status_code == 200
    assert archive_view.json()["snapshot"]["prediction_snapshot"]["estimated_yield_kg_ha"] == 4100.0


def test_detect_preflight_and_run_list_routes(monkeypatch):
    run_id = uuid4()
    created_at = datetime.now(timezone.utc)
    monkeypatch.setattr(
        FieldsRepository,
        "list_runs",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=run_id,
                    status="done",
                    progress=100,
                    created_at=created_at,
                    params={
                        "aoi": {
                            "type": "point_radius",
                            "lat": 45.2,
                            "lon": 38.7,
                            "radius_km": 4,
                        },
                        "time_range": {
                            "start_date": "2025-05-01",
                            "end_date": "2025-08-31",
                        },
                        "resolution_m": 10,
                        "target_dates": 7,
                        "min_field_area_ha": 0.25,
                        "use_sam": False,
                        "config": {"preset": "standard"},
                    },
                ),
            ]
        ),
    )

    with _client(monkeypatch) as client:
        preflight = client.post(
            "/api/v1/fields/detect/preflight?use_sam=false",
            json={
                "aoi": {"type": "point_radius", "lat": 45.2, "lon": 38.7, "radius_km": 4},
                "time_range": {"start_date": "2025-05-01", "end_date": "2025-08-31"},
                "resolution_m": 10,
                "max_cloud_pct": 40,
                "target_dates": 7,
                "min_field_area_ha": 0.25,
                "seed_mode": "edges",
                "debug": False,
                "config": {"preset": "standard"},
            },
        )
        runs = client.get("/api/v1/fields/runs")

    assert preflight.status_code == 200
    assert preflight.json()["budget_ok"] is True
    assert preflight.json()["preset"] == "standard"
    assert preflight.json()["estimated_tiles"] >= 1
    assert preflight.json()["launch_tier"] == "validated_core"
    assert preflight.json()["review_required"] is False
    assert preflight.json()["output_mode"] == "field_boundaries"
    assert preflight.json()["operational_eligible"] is True
    assert preflight.json()["max_radius_km"] == 20
    assert preflight.json()["recommended_radius_km"] == 20
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["id"] == str(run_id)
    assert runs.json()["runs"][0]["preset"] == "standard"


def test_detect_route_rejects_submit_when_gpu_worker_is_unavailable(monkeypatch):
    fields_api = importlib.import_module("api.fields")
    monkeypatch.setattr(fields_api, "has_live_workers_for_queue", lambda *_args, **_kwargs: False)
    create_run = AsyncMock()
    monkeypatch.setattr(FieldsRepository, "create_run", create_run)

    payload = {
        "aoi": {"type": "point_radius", "lat": 45.2, "lon": 38.7, "radius_km": 4},
        "time_range": {"start_date": "2025-05-01", "end_date": "2025-08-31"},
        "resolution_m": 10,
        "max_cloud_pct": 40,
        "target_dates": 7,
        "min_field_area_ha": 0.25,
        "seed_mode": "edges",
        "debug": False,
        "config": {"preset": "standard"},
    }

    with _client(monkeypatch) as client:
        response = client.post("/api/v1/fields/detect?use_sam=false", json=payload)

    assert response.status_code == 503
    assert "worker" in response.json()["detail"]
    create_run.assert_not_called()


def test_run_status_uses_preflight_contract_before_worker_runtime(monkeypatch):
    fields_api = importlib.import_module("api.fields")
    created_at = datetime.now(timezone.utc)
    run = SimpleNamespace(
        id=uuid4(),
        status="queued",
        progress=0,
        error_msg=None,
        created_at=created_at,
        params={
            "config": {
                "preset": "fast",
                "preflight": {
                    "pipeline_profile": "fast_preview",
                    "preview_only": True,
                    "output_mode": "preview_agri_contours",
                    "operational_eligible": False,
                    "max_radius_km": 40,
                    "recommended_radius_km": 30,
                    "enabled_stages": ["fetch", "candidate_postprocess", "segmentation"],
                },
            }
        },
    )

    payload = fields_api._run_response_payload(run)

    assert payload["status"] == "queued"
    assert payload["pipeline_profile"] == "fast_preview"
    assert payload["preview_only"] is True
    assert payload["output_mode"] == "preview_agri_contours"
    assert payload["operational_eligible"] is False
    assert payload["max_radius_km"] == 40
    assert payload["recommended_radius_km"] == 30
    assert payload["enabled_stages"] == ["fetch", "candidate_postprocess", "segmentation"]


def test_quality_preflight_blocks_radius_above_hard_cap(monkeypatch):
    with _client(monkeypatch) as client:
        preflight = client.post(
            "/api/v1/fields/detect/preflight?use_sam=true",
            json={
                "aoi": {"type": "point_radius", "lat": 45.2, "lon": 38.7, "radius_km": 15},
                "time_range": {"start_date": "2025-05-01", "end_date": "2025-08-31"},
                "resolution_m": 10,
                "max_cloud_pct": 40,
                "target_dates": 9,
                "min_field_area_ha": 0.1,
                "seed_mode": "edges",
                "debug": False,
                "config": {"preset": "quality"},
            },
        )

    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["budget_ok"] is False
    assert payload["hard_block"] is True
    assert payload["estimated_tiles"] >= 1
    assert "максимальный радиус" in payload["reason"]
    assert payload["launch_tier"] == "blocked"
    assert payload["review_required"] is True
    assert payload["output_mode"] == "field_boundaries_hifi"
    assert payload["operational_eligible"] is True
    assert payload["max_radius_km"] == 8
    assert payload["recommended_radius_km"] == 8


def test_quality_preflight_with_fewer_dates_is_warning_only(monkeypatch):
    with _client(monkeypatch) as client:
        preflight = client.post(
            "/api/v1/fields/detect/preflight?use_sam=true",
            json={
                "aoi": {"type": "point_radius", "lat": 59.72, "lon": 54.66, "radius_km": 8},
                "time_range": {"start_date": "2025-05-01", "end_date": "2025-08-31"},
                "resolution_m": 10,
                "max_cloud_pct": 40,
                "target_dates": 7,
                "min_field_area_ha": 0.1,
                "seed_mode": "edges",
                "debug": False,
                "config": {"preset": "quality"},
            },
        )

    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["hard_block"] is False
    assert payload["budget_ok"] is False
    assert any("меньше временных срезов" in item for item in payload["warnings"])
    assert payload["launch_tier"] == "experimental_rest"
    assert payload["review_required"] is True


def test_satellite_true_color_route(monkeypatch):
    class DummySentinel:
        def __init__(self):
            self.last_account_alias = "reserv"
            self.last_failover_level = 1

        async def fetch_tile(self, **kwargs):
            shape = (8, 8)
            return {
                "B2": np.full(shape, 0.18, dtype=np.float32),
                "B3": np.full(shape, 0.24, dtype=np.float32),
                "B4": np.full(shape, 0.31, dtype=np.float32),
                "SCL": np.zeros(shape, dtype=np.uint8),
            }

    satellite = importlib.import_module("api.satellite")
    monkeypatch.setattr(satellite, "SentinelHubClient", DummySentinel)

    with _client(monkeypatch) as client:
        response = client.get(
            "/api/v1/satellite/true-color",
            params={
                "minx": 37.4,
                "miny": 55.6,
                "maxx": 37.6,
                "maxy": 55.8,
                "width": 256,
                "height": 256,
                "scene_date": "2025-07-05",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_account"] == "reserv"
    assert payload["failover_level"] == 1
    assert payload["requested_date"] == "2025-07-05"
    assert payload["image_base64"]
    decoded = base64.b64decode(payload["image_base64"])
    assert decoded.startswith(b"\x89PNG")


def test_labeling_imports_and_mlops_routes(monkeypatch):
    task_payload = {
        "id": 11,
        "aoi_run_id": str(uuid4()),
        "field_id": str(uuid4()),
        "title": "Review autodetect field",
        "status": "queued",
        "source": "active_learning",
        "queue_name": "north",
        "priority_score": 0.81,
        "task_payload": {"field_score": 0.42},
        "claimed_by_user_id": None,
        "latest_version": None,
        "latest_review": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    import_payload = {
        "id": 5,
        "import_type": "yield_history",
        "status": "previewed",
        "source_filename": "yield.csv",
        "preview_summary": {"rows_total": 1, "valid_rows": 1},
        "commit_summary": {},
        "error_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    dataset_payload = {
        "id": 1,
        "dataset_version": "manual_gt_v1",
        "checksum": "abc123",
        "code_sha": "deadbeef",
        "status": "ready",
        "manifest_json": {"items": [{"split": "train"}, {"split": "calibration"}, {"split": "holdout"}]},
        "split_summary": {"holdout": 300},
        "artifact_uri": "s3://artifacts/manual_gt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    deployment_payload = {
        "id": 3,
        "deployment_name": "boundary-prod",
        "model_version": "boundary_unet_v2",
        "dataset_version_id": 1,
        "benchmark_id": 2,
        "model_uri": "models:/boundary_unet_v2/Production",
        "config_snapshot": {"tracking_uri": "http://mlflow:5000"},
        "code_sha": "deadbeef",
        "status": "promoted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rolled_back_at": None,
    }

    monkeypatch.setattr(LabelingService, "list_tasks", AsyncMock(return_value=[task_payload]))
    monkeypatch.setattr(LabelingService, "create_task", AsyncMock(return_value=task_payload))
    monkeypatch.setattr(
        DataImportService,
        "list_jobs",
        AsyncMock(return_value=[import_payload]),
    )
    monkeypatch.setattr(
        DataImportService,
        "create_import",
        AsyncMock(return_value=import_payload),
    )
    monkeypatch.setattr(MlOpsService, "list_datasets", AsyncMock(return_value=[dataset_payload]))
    monkeypatch.setattr(MlOpsService, "promote", AsyncMock(return_value=deployment_payload))

    with _client(monkeypatch) as client:
        labeling = client.get("/api/v1/labeling/tasks")
        create_task = client.post(
            "/api/v1/labeling/tasks",
            json={"title": "Review autodetect field", "source": "active_learning"},
        )
        imports = client.get("/api/v1/data-imports")
        create_import = client.post(
            "/api/v1/data-imports",
            json={
                "import_type": "yield_history",
                "source_filename": "yield.csv",
                "content_base64": base64.b64encode(
                    b"field_external_id,season_year,crop_code,yield_kg_ha\nFIELD-1,2025,wheat,4300\n"
                ).decode("ascii"),
            },
        )
        datasets = client.get("/api/v1/admin/ml/datasets")
        promote = client.post(
            "/api/v1/admin/ml/promote",
            json={
                "deployment_name": "boundary-prod",
                "model_version": "boundary_unet_v2",
                "benchmark_id": 2,
                "dataset_version_id": 1,
                "model_uri": "models:/boundary_unet_v2/Production",
                "code_sha": "deadbeef",
            },
        )

    assert labeling.status_code == 200
    assert labeling.json()["tasks"][0]["source"] == "active_learning"
    assert create_task.status_code == 200
    assert imports.status_code == 200
    assert create_import.status_code == 200
    assert datasets.status_code == 200
    assert datasets.json()["datasets"][0]["dataset_version"] == "manual_gt_v1"
    assert promote.status_code == 200
    assert promote.json()["status"] == "promoted"


def test_debug_tiles_routes(monkeypatch, tmp_path):
    fields_api = importlib.import_module("api.fields")
    debug_root = tmp_path / "debug_runs"
    run_dir = debug_root / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / "tile-1.npz"
    np.savez(
        npz_path,
        step_06_after_grow=np.pad(np.ones((8, 8), dtype=np.uint8), 4),
        step_07_after_gap_close=np.pad(np.ones((6, 6), dtype=np.uint8), 5),
        boundary_prob=np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16),
    )

    run = SimpleNamespace(
        id=uuid4(),
        status="done",
        progress=100,
        error_msg=None,
        created_at=datetime.now(timezone.utc),
        params={
            "runtime": {
                "tiles": [
                    {
                        "tile_id": "tile-1",
                        "bbox": [30.0, 59.0, 30.2, 59.2],
                        "debug_artifact": str(npz_path),
                        "traditional_gpkg": str(run_dir / "traditional.gpkg"),
                        "final_gpkg": str(run_dir / "final.gpkg"),
                        "sam_raw_gpkg": str(run_dir / "sam_raw.gpkg"),
                        "sam_filtered_gpkg": str(run_dir / "sam_filtered.gpkg"),
                        "components_after_grow": 7,
                        "components_after_merge": 2,
                        "components_after_watershed": 2,
                    }
                ]
            }
        },
    )

    monkeypatch.setattr(fields_api, "_allowed_debug_root", lambda: Path(debug_root).resolve())
    monkeypatch.setattr(FieldsRepository, "get_run", AsyncMock(return_value=run))

    with _client(monkeypatch) as client:
        listing = client.get(f"/api/v1/fields/runs/{run.id}/debug/tiles")
        tile = client.get(f"/api/v1/fields/runs/{run.id}/debug/tiles/tile-1")
        layer = client.get(f"/api/v1/fields/runs/{run.id}/debug/tiles/tile-1/layers/after_grow")

    assert listing.status_code == 200
    listing_payload = listing.json()
    assert listing_payload["tiles"][0]["tile_id"] == "tile-1"
    assert any(item["id"] == "after_grow" for item in listing_payload["tiles"][0]["available_layers"])

    assert tile.status_code == 200
    tile_payload = tile.json()
    assert tile_payload["runtime_meta"]["components_after_grow"] == 7
    assert tile_payload["bbox"] == [30.0, 59.0, 30.2, 59.2]

    assert layer.status_code == 200
    layer_payload = layer.json()
    assert layer_payload["layer_name"] == "after_grow"
    assert layer_payload["type"] == "image_static"
    assert layer_payload["image_base64"]
    decoded = base64.b64decode(layer_payload["image_base64"])
    assert decoded.startswith(b"\x89PNG")


def test_detection_candidates_routes(monkeypatch):
    fields_api = importlib.import_module("api.fields")
    run = SimpleNamespace(
        id=uuid4(),
        status="done",
        progress=100,
        error_msg=None,
        created_at=datetime.now(timezone.utc),
        params={
            "runtime": {
                "tiles": [
                    {"tile_id": "tile-1"},
                    {"tile_id": "tile-2"},
                ]
            }
        },
    )
    diagnostic = SimpleNamespace(tile_index=0)
    candidate = SimpleNamespace(
        id=11,
        tile_diagnostic_id=21,
        field_id=uuid4(),
        branch="boundary_first",
        geom=None,
        area_m2=1250.5,
        score=0.87,
        rank=1,
        kept=True,
        reject_reason=None,
        features={"edge_score": 0.63, "ndvi_std": 0.18},
        model_version="boundary_unet_v2",
        created_at=datetime.now(timezone.utc),
    )
    rejected = SimpleNamespace(
        id=12,
        tile_diagnostic_id=21,
        field_id=None,
        branch="crop_region",
        geom=None,
        area_m2=860.0,
        score=0.41,
        rank=2,
        kept=False,
        reject_reason="low_object_score",
        features={"edge_score": 0.21},
        model_version="boundary_unet_v2",
        created_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(FieldsRepository, "get_run", AsyncMock(return_value=run))

    async def _fake_load_detection_candidates(_db, **kwargs):
        tile_index = kwargs.get("tile_index")
        kept = kwargs.get("kept")
        rows = [(candidate, diagnostic), (rejected, diagnostic)]
        if tile_index is not None and tile_index != 0:
            return []
        if kept is None:
            return rows
        return [row for row in rows if bool(row[0].kept) is bool(kept)]

    monkeypatch.setattr(fields_api, "_load_detection_candidates", _fake_load_detection_candidates)

    with _client(monkeypatch) as client:
        listing = client.get(f"/api/v1/fields/runs/{run.id}/candidates")
        kept_only = client.get(f"/api/v1/fields/runs/{run.id}/candidates", params={"kept": "true"})
        tile_listing = client.get(f"/api/v1/fields/runs/{run.id}/debug/tiles/tile-1/candidates")

    assert listing.status_code == 200
    listing_payload = listing.json()
    assert listing_payload["run_id"] == str(run.id)
    assert listing_payload["total"] == 2
    assert listing_payload["candidates"][0]["branch"] == "boundary_first"
    assert listing_payload["candidates"][0]["tile_id"] == "tile-1"
    assert listing_payload["candidates"][1]["reject_reason"] == "low_object_score"

    assert kept_only.status_code == 200
    kept_payload = kept_only.json()
    assert kept_payload["total"] == 1
    assert kept_payload["candidates"][0]["kept"] is True

    assert tile_listing.status_code == 200
    tile_payload = tile_listing.json()
    assert tile_payload["total"] == 2
    assert all(item["tile_id"] == "tile-1" for item in tile_payload["candidates"])
