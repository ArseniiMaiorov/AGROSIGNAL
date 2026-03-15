from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.data_import_service import DataImportService
from services.mlops_service import MlOpsService
from storage.db import DataImportJob, YieldObservation


class _DummyResult:
    def __init__(self, value=None):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalar_one(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _FakeDb:
    def __init__(self) -> None:
        self.saved = []
        self.executed = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        return _DummyResult(None)

    def add(self, row):
        self.saved.append(row)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_data_import_preview_collects_validation_errors():
    db = _FakeDb()
    service = DataImportService(db)
    job = DataImportJob(
        id=1,
        organization_id=uuid4(),
        created_by_user_id=uuid4(),
        import_type="yield_history",
        status="uploaded",
        source_filename="yield.csv",
        source_path="/tmp/yield.csv",
    )

    summary = await service._preview_csv(
        job,
        rows=[
            {
                "field_external_id": "",
                "season_year": "2025",
                "crop_code": "wheat",
                "yield_kg_ha": "4300",
            }
        ],
    )

    assert summary["invalid_rows"] == 1
    assert job.error_count == 1
    assert any(item.__class__.__name__ == "DataImportError" for item in db.saved)


@pytest.mark.asyncio
async def test_data_import_commit_yield_history_creates_observation(monkeypatch):
    db = _FakeDb()
    service = DataImportService(db)
    organization_id = uuid4()
    field_id = uuid4()
    job = DataImportJob(
        id=2,
        organization_id=organization_id,
        created_by_user_id=uuid4(),
        import_type="yield_history",
        status="previewed",
        source_filename="yield.csv",
        source_path="/tmp/yield.csv",
    )

    async def _resolve_field(*, organization_id, field_external_id):
        return SimpleNamespace(id=field_id, external_field_id=field_external_id)

    async def _get_or_create_field_season(*, organization_id, field, season_year):
        return SimpleNamespace(id=101, field_id=field.id, season_year=season_year), True

    async def _upsert_crop_assignment(**_kwargs):
        return SimpleNamespace(id=77)

    async def _crop_map():
        return {"wheat": 1}

    monkeypatch.setattr(service, "_resolve_field", _resolve_field)
    monkeypatch.setattr(service, "_get_or_create_field_season", _get_or_create_field_season)
    monkeypatch.setattr(service, "_upsert_crop_assignment", _upsert_crop_assignment)
    monkeypatch.setattr(service, "_crop_map", _crop_map)

    summary = await service._commit_csv(
        job,
        rows=[
            {
                "field_external_id": "FIELD-1",
                "season_year": "2025",
                "crop_code": "wheat",
                "yield_kg_ha": "4300",
                "observed_at": "2025-09-01",
            }
        ],
    )

    assert summary["inserted"] >= 2
    assert summary["error_count"] == 0
    assert any(isinstance(item, YieldObservation) for item in db.saved)


def test_mlops_promotion_gates_accept_valid_holdout():
    service = MlOpsService(db=None)
    report = service._evaluate_promotion_gates(
        {
            "manual_holdout_size": 320,
            "iou_geo": 0.81,
            "baseline_iou_geo": 0.80,
            "hd95_m_p90": 4.5,
            "baseline_hd95_m_p90": 4.7,
            "field_recall_south": 0.93,
            "missed_fields_rate_south": 0.07,
            "oversegmented_fields_rate_south": 0.11,
            "mean_components_per_gt_field_south": 1.1,
            "boundary_iou_south_median": 0.80,
            "boundary_iou_north_median": 0.84,
            "contour_shrink_ratio_north_median": 1.0,
            "centroid_shift_m_north_p90": 4.0,
            "north_inward_shrink_obvious_rate": 0.05,
        }
    )

    assert report["passed"] is True
    assert report["reasons"] == []


def test_mlops_promotion_gates_reject_regressions():
    service = MlOpsService(db=None)
    report = service._evaluate_promotion_gates(
        {
            "manual_holdout_size": 120,
            "iou_geo": 0.70,
            "baseline_iou_geo": 0.75,
            "hd95_m_p90": 8.0,
            "baseline_hd95_m_p90": 6.0,
            "field_recall_south": 0.80,
            "missed_fields_rate_south": 0.20,
            "oversegmented_fields_rate_south": 0.20,
            "mean_components_per_gt_field_south": 1.5,
            "boundary_iou_south_median": 0.70,
            "boundary_iou_north_median": 0.75,
            "contour_shrink_ratio_north_median": 1.2,
            "centroid_shift_m_north_p90": 8.0,
            "north_inward_shrink_obvious_rate": 0.30,
        }
    )

    assert report["passed"] is False
    assert any("manual holdout" in reason for reason in report["reasons"])
    assert any("geo IoU" in reason for reason in report["reasons"])
