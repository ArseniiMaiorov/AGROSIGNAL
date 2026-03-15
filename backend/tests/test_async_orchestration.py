from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import services.weekly_profile_service as weekly_profile_service_module
import storage.db as db_module
import tasks.analytics as analytics_tasks
import tasks.prediction_tasks as prediction_tasks


class _AsyncSessionContext:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AnalyticsDb:
    async def commit(self):
        return None

    async def rollback(self):
        return None


class _ScenarioTaskDb:
    def __init__(self, scenario, field, crop):
        self.scenario = scenario
        self.field = field
        self.crop = crop
        self.committed = False

    async def execute(self, stmt):
        text = str(stmt)
        if "FROM scenario_runs" in text:
            return _ScalarResult(self.scenario)
        if "FROM fields" in text:
            return _ScalarResult(self.field)
        if "FROM crops" in text:
            return _ScalarResult(self.crop)
        return _ScalarResult(None)

    async def commit(self):
        self.committed = True
        return None


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeTask:
    def __init__(self):
        self.states: list[dict] = []

    def update_state(self, **kwargs):
        self.states.append(kwargs)


@pytest.mark.asyncio
async def test_run_modeling_impl_passes_extended_weekly_inputs(monkeypatch):
    field_id = uuid4()
    organization_id = uuid4()
    db = _AnalyticsDb()
    task = _FakeTask()
    simulate_mock = AsyncMock(
        return_value={
            "field_id": str(field_id),
            "baseline_yield_kg_ha": 4100.0,
            "scenario_yield_kg_ha": 4380.0,
            "predicted_yield_change_pct": 6.83,
            "supported": True,
            "engine_version": "mechanistic_v1",
        }
    )

    class _FakeModelingService:
        def __init__(self, _db):
            self.db = _db

        simulate = simulate_mock

    monkeypatch.setattr(analytics_tasks, "ModelingService", _FakeModelingService)
    monkeypatch.setattr(analytics_tasks, "_reset_task_db_runtime", AsyncMock(return_value=None))
    monkeypatch.setattr(analytics_tasks, "get_session_factory", lambda: (lambda: _AsyncSessionContext(db)))
    monkeypatch.setattr(weekly_profile_service_module, "current_season_year", lambda: 2026)
    monkeypatch.setattr(
        weekly_profile_service_module,
        "ensure_weekly_profile",
        AsyncMock(return_value=[SimpleNamespace(week_number=18), SimpleNamespace(week_number=19)]),
    )

    payload = await analytics_tasks._run_modeling_impl(
        task,
        organization_id=str(organization_id),
        request_payload={
            "field_id": str(field_id),
            "crop_code": "wheat",
            "scenario_name": "Forward weather",
            "temperature_delta_c": 1.5,
            "precipitation_factor": 0.85,
            "sowing_shift_days": 7,
            "irrigation_events": [{"week": 20, "amount_mm": 25.0}],
            "fertilizer_events": [{"week": 19, "n_kg_ha": 30.0}],
        },
    )

    assert payload["status"] == "done"
    _, kwargs = simulate_mock.call_args
    assert kwargs["precipitation_factor"] == pytest.approx(0.85)
    assert kwargs["sowing_shift_days"] == 7
    assert kwargs["irrigation_events"] == [{"week": 20, "amount_mm": 25.0}]
    assert kwargs["fertilizer_events"] == [{"week": 19, "n_kg_ha": 30.0}]
    assert payload["result"]["feature_schema_version"] == weekly_profile_service_module.FEATURE_SCHEMA_VERSION
    assert payload["result"]["weekly_profile_rows"] == 2


def test_refresh_field_prediction_task_uses_canonical_weekly_profile(monkeypatch):
    field_id = uuid4()
    organization_id = uuid4()
    ensure_mock = AsyncMock(return_value=[SimpleNamespace(week_number=18)] * 6)
    crop_hint = {"top_crop_code": "wheat", "top_probability": 1.0}

    class _FakeYieldService:
        def __init__(self, _db):
            self.db = _db

        async def get_or_create_prediction(self, *args, **kwargs):
            return {
                "estimated_yield_kg_ha": 4321.0,
                "confidence": 0.74,
                "model_version": "yield_v2",
            }

    monkeypatch.setattr(db_module, "get_session_factory", lambda: (lambda: _AsyncSessionContext(SimpleNamespace())))
    monkeypatch.setattr(weekly_profile_service_module, "current_season_year", lambda: 2026)
    monkeypatch.setattr(weekly_profile_service_module, "ensure_weekly_profile", ensure_mock)
    monkeypatch.setattr(weekly_profile_service_module, "load_crop_hint", AsyncMock(return_value=crop_hint))
    monkeypatch.setattr("services.yield_service.YieldService", _FakeYieldService)

    payload = prediction_tasks.refresh_field_prediction.run(
        str(field_id),
        str(organization_id),
        "wheat",
        False,
    )

    assert payload["estimated_yield_kg_ha"] == pytest.approx(4321.0)
    assert payload["feature_schema_version"] == weekly_profile_service_module.FEATURE_SCHEMA_VERSION
    assert payload["weekly_profile_rows"] == 6
    assert payload["crop_hint"] == crop_hint
    assert ensure_mock.await_count == 1


def test_simulate_scenario_forward_uses_unified_mechanistic_service(monkeypatch):
    scenario = SimpleNamespace(
        id=11,
        organization_id=uuid4(),
        field_id=uuid4(),
        crop_id=7,
        scenario_name="Mechanistic run",
        parameters={
            "temperature_delta_c": 2.0,
            "precipitation_factor": 0.9,
            "irrigation_events": [{"week": 21, "amount_mm": 20.0}],
        },
        result_snapshot={},
        baseline_snapshot={},
        delta_pct=0.0,
        model_version=None,
    )
    field = SimpleNamespace(id=scenario.field_id, area_m2=12500.0)
    crop = SimpleNamespace(id=7, code="wheat", yield_baseline_kg_ha=4000.0)
    db = _ScenarioTaskDb(scenario, field, crop)
    ensure_mock = AsyncMock(return_value=[SimpleNamespace(week_number=18)] * 5)
    simulate_mechanistic_mock = AsyncMock(
        return_value={
            "baseline_yield_kg_ha": 3900.0,
            "scenario_yield_kg_ha": 4200.0,
            "predicted_yield_change_pct": 7.69,
            "supported": True,
            "engine_version": "mechanistic_v1",
            "model_version": "mechanistic_scenario_v1",
            "confidence_tier": "global_baseline",
            "trace_supported": True,
            "weeks_simulated": 5,
        }
    )

    class _FakeModelingService:
        def __init__(self, _db):
            self.db = _db

        simulate_mechanistic = simulate_mechanistic_mock

    monkeypatch.setattr(db_module, "get_session_factory", lambda: (lambda: _AsyncSessionContext(db)))
    monkeypatch.setattr(weekly_profile_service_module, "current_season_year", lambda: 2026)
    monkeypatch.setattr(weekly_profile_service_module, "ensure_weekly_profile", ensure_mock)
    monkeypatch.setattr(weekly_profile_service_module, "profile_has_signal", lambda rows: True)
    monkeypatch.setattr("services.modeling_service.ModelingService", _FakeModelingService)

    payload = prediction_tasks.simulate_scenario_forward.run(str(scenario.id), str(scenario.organization_id))

    assert payload["supported"] is True
    assert payload["engine_version"] == "mechanistic_v1"
    _, kwargs = simulate_mechanistic_mock.call_args
    assert kwargs["scenario_events"] == scenario.parameters
    assert kwargs["save"] is False
    assert kwargs["degraded_fallback"] is False
    assert scenario.result_snapshot["feature_schema_version"] == weekly_profile_service_module.FEATURE_SCHEMA_VERSION
    assert scenario.result_snapshot["weekly_profile_rows"] == 5
    assert db.committed is True
