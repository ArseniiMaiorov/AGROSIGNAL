from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from storage.db import AoiRun, Field
from storage.fields_repo import FieldsRepository


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        return None

    async def execute(self, *_args, **_kwargs) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def refresh(self, field: Field) -> None:
        field.created_at = datetime.now(timezone.utc)
        field.area_m2 = 1000.0
        field.perimeter_m = 140.0


@pytest.mark.asyncio
async def test_create_manual_field_uses_naive_utc_run_timestamps() -> None:
    repo = FieldsRepository(_FakeSession())
    field = await repo.create_manual_field(
        {
            "type": "Polygon",
            "coordinates": [[[30.0, 59.0], [30.01, 59.0], [30.01, 59.01], [30.0, 59.01], [30.0, 59.0]]],
        },
        organization_id=uuid4(),
        created_by_user_id=uuid4(),
        quality_score=1.0,
    )

    run = next(item for item in repo.session.added if isinstance(item, AoiRun))
    created_field = next(item for item in repo.session.added if isinstance(item, Field))

    assert run.time_start.tzinfo is None
    assert run.time_end.tzinfo is None
    assert created_field.source == "manual"
    assert field.area_m2 == 1000.0
