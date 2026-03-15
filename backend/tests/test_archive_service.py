from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon

from services.archive_service import ArchiveService
from storage.db import ArchiveEntry, Field


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _ExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _ScalarResult(self._rows)


class _FakeDb:
    def __init__(self, field: Field, entries: list[ArchiveEntry] | None = None) -> None:
        self._field = field
        self._entries = entries or []
        self.saved = []
        self.deleted = []

    async def get(self, model, key):
        if model is Field and key == self._field.id:
            return self._field
        if model is ArchiveEntry:
            for entry in self._entries + self.saved:
                if entry.id == key:
                    return entry
        return None

    async def execute(self, _stmt):
        return _ExecuteResult(self._entries)

    def add(self, row):
        row.id = row.id or len(self.saved) + 1
        self.saved.append(row)

    async def delete(self, row):
        self.deleted.append(row)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_archive_service_creates_zip(monkeypatch, tmp_path):
    field_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = Field(
        id=field_id,
        aoi_run_id=uuid4(),
        geom=from_shape(polygon, srid=4326),
        area_m2=10000.0,
        perimeter_m=400.0,
        quality_score=0.9,
        source='manual',
    )
    db = _FakeDb(field)
    service = ArchiveService(db)
    service.archive_dir = tmp_path
    monkeypatch.setattr(
        service.weather_service,
        'get_current_weather',
        AsyncMock(return_value={'temperature_c': 15.0, 'provider': 'openmeteo'}),
    )
    monkeypatch.setattr(
        service.field_analytics_service,
        'build_archive_snapshots',
        AsyncMock(return_value={
            'field_snapshot': {'field_id': str(field_id), 'area_m2': 10000.0},
            'prediction_snapshot': {'estimated_yield_kg_ha': 4200.0},
            'metrics_snapshot': {'current_metrics': {}},
            'scenario_snapshot': {'items': []},
            'model_meta': {'model_version': 'heuristic_v1'},
        }),
    )
    monkeypatch.setattr(
        service.field_analytics_service,
        'attach_archive_series',
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        service.modeling_service,
        'list_scenarios',
        AsyncMock(return_value=[]),
    )

    payload = await service.create_archive(
        field_id=field_id,
        date_from=datetime(2026, 3, 1, tzinfo=timezone.utc),
        date_to=datetime(2026, 3, 7, tzinfo=timezone.utc),
        layers=['ndvi', 'weather'],
        organization_id=field_id,
    )

    archive_path = Path(payload['file_path'])
    assert archive_path.exists()
    assert payload['status'] == 'ready'
    assert db.saved


@pytest.mark.asyncio
async def test_archive_service_rejects_preview_contours(tmp_path):
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
        quality_score=0.9,
        source='autodetect_preview',
    )
    db = _FakeDb(field)
    service = ArchiveService(db)
    service.archive_dir = tmp_path

    with pytest.raises(ValueError, match='Preview-контур'):
        await service.create_archive(
            field_id=field_id,
            date_from=datetime(2026, 3, 1, tzinfo=timezone.utc),
            date_to=datetime(2026, 3, 7, tzinfo=timezone.utc),
            layers=['ndvi'],
            organization_id=organization_id,
        )


@pytest.mark.asyncio
async def test_archive_cleanup_removes_expired_files(tmp_path):
    field_id = uuid4()
    polygon = Polygon([(30.0, 59.0), (30.1, 59.0), (30.1, 59.1), (30.0, 59.1)])
    field = Field(
        id=field_id,
        aoi_run_id=uuid4(),
        geom=from_shape(polygon, srid=4326),
        area_m2=10000.0,
        perimeter_m=400.0,
        quality_score=0.9,
        source='manual',
    )
    expired_file = tmp_path / 'expired.zip'
    expired_file.write_bytes(b'test')
    expired_entry = ArchiveEntry(
        id=7,
        field_id=field_id,
        date_from=datetime(2026, 3, 1, tzinfo=timezone.utc),
        date_to=datetime(2026, 3, 7, tzinfo=timezone.utc),
        layers=['ndvi'],
        file_path=str(expired_file),
        status='ready',
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        meta={},
    )
    db = _FakeDb(field, entries=[expired_entry])
    service = ArchiveService(db)
    report = await service.cleanup_expired()

    assert report == {'removed_rows': 1, 'removed_files': 1}
    assert not expired_file.exists()
    assert db.deleted == [expired_entry]
