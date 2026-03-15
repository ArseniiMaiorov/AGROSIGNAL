from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from services.status_service import StatusService


class _DummyDb:
    async def execute(self, _query, _params=None):
        return SimpleNamespace(
            scalar_one=lambda: 0,
            scalar_one_or_none=lambda: None,
        )


def test_check_classifier_returns_offline_for_incompatible_pickle(monkeypatch):
    service = StatusService(_DummyDb())

    def _raise(*_args, **_kwargs):
        raise ValueError("Не удалось прочитать classifier pickle 'object_classifier.pkl': несовместимый артефакт")

    monkeypatch.setattr("services.status_service.validate_classifier_file", _raise)

    payload = service._check_classifier()

    assert payload["status"] == "offline"
    assert "несовместимый артефакт" in payload["detail"]


@pytest.mark.asyncio
async def test_get_system_status_includes_sentinel_provider(monkeypatch):
    service = StatusService(_DummyDb())
    now = datetime.now(timezone.utc).isoformat()

    monkeypatch.setattr(service, "_check_database", AsyncMock(return_value={"status": "online", "detail": 1}))
    monkeypatch.setattr(service, "_check_redis", AsyncMock(return_value={"status": "online", "detail": True}))
    monkeypatch.setattr(service, "_check_auth_bootstrap", AsyncMock(return_value={"status": "online", "detail": "ok"}))
    monkeypatch.setattr(service, "_check_sentinel_provider", AsyncMock(return_value={"status": "online", "detail": "account=reserv, failover=1"}))
    monkeypatch.setattr(service, "_check_weather_cache", AsyncMock(return_value={"status": "online", "detail": now}))
    monkeypatch.setattr(service, "_check_layer_catalog", AsyncMock(return_value={"status": "online", "detail": "9 layers"}))
    monkeypatch.setattr(service, "_check_model_file", lambda _path: {"status": "online", "detail": "model"})
    monkeypatch.setattr(service, "_check_classifier", lambda: {"status": "online", "detail": "classifier"})

    payload = await service.get_system_status(organization_id=uuid4())

    assert payload["components"]["sentinel_provider"]["status"] == "online"
    assert "account=reserv" in payload["components"]["sentinel_provider"]["detail"]


@pytest.mark.asyncio
async def test_bootstrap_status_includes_release_modules(monkeypatch):
    service = StatusService(_DummyDb())

    monkeypatch.setattr(service, "_check_database", AsyncMock(return_value={"status": "online", "detail": 1}))
    monkeypatch.setattr(service, "_check_redis", AsyncMock(return_value={"status": "online", "detail": True}))
    monkeypatch.setattr(service, "_check_auth_bootstrap", AsyncMock(return_value={"status": "online", "detail": "ok"}))
    monkeypatch.setattr(service, "_check_weather_provider", AsyncMock(return_value={"status": "online", "detail": "openmeteo"}))
    monkeypatch.setattr(service, "_check_layer_catalog", AsyncMock(return_value={"status": "online", "detail": "9 layers"}))
    monkeypatch.setattr(service, "_check_model_file", lambda _path: {"status": "online", "detail": "model"})
    monkeypatch.setattr(service, "_check_classifier", lambda: {"status": "online", "detail": "classifier"})
    monkeypatch.setattr(service, "_check_scene_cache", lambda: {"status": "online", "detail": "scene-cache"})
    monkeypatch.setattr(service, "_check_satellite_browse", lambda: {"status": "online", "detail": "primary=yes"})
    monkeypatch.setattr(service, "_check_archive_runtime", lambda: {"status": "online", "detail": "/tmp/archive"})
    monkeypatch.setattr(service, "_check_runtime_module", lambda module_path, detail=None: {"status": "online", "detail": detail or module_path})
    monkeypatch.setattr(service, "_check_project_artifact", lambda _path, optional=False: {"status": "online", "detail": _path})

    payload = await service.get_bootstrap_status()

    assert payload["components"]["satellite_browse"]["status"] == "online"
    assert payload["components"]["prediction_engine"]["status"] == "online"
    assert payload["components"]["scenario_engine"]["status"] == "online"
    assert payload["components"]["archive_service"]["status"] == "online"
    assert payload["components"]["async_jobs"]["status"] == "online"
    assert payload["components"]["qa_matrix"]["status"] == "online"
