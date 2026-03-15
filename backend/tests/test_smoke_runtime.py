from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from processing.fields.object_classifier import FEATURE_COLUMNS
from processing.fields.objectclassifier import FEATURE_NAMES as LEGACY_FEATURE_NAMES
from utils.classifier_schema import validate_classifier_file


def test_import_main_and_health(monkeypatch):
    import storage.db as db

    async def _noop() -> None:
        return None

    monkeypatch.setattr(db, "init_db", _noop)
    monkeypatch.setattr(db, "seed_defaults", _noop)

    main = importlib.import_module("main")
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["checks"]["database"] == "ok"


def test_object_classifier_artifact_schema_matches_runtime():
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "backend/models/object_classifier.pkl"
    meta = validate_classifier_file(model_path, FEATURE_COLUMNS)

    assert meta["feature_count"] == len(FEATURE_COLUMNS)
    assert meta["pipeline_feature_count"] in {None, len(FEATURE_COLUMNS)}
    assert len(LEGACY_FEATURE_NAMES) == 12
