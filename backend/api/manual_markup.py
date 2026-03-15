"""API ручной разметки полей."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import FieldSummary, ManualFieldCreateRequest, ManualFieldResponse
from storage.db import get_db
from storage.fields_repo import FieldsRepository

router = APIRouter(prefix="/manual", tags=["manual"])


@router.post("/fields", response_model=ManualFieldResponse)
async def create_manual_field(
    request: ManualFieldCreateRequest,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> ManualFieldResponse:
    repo = FieldsRepository(db)
    try:
        field = await repo.create_manual_field(
            request.geometry,
            organization_id=ctx.organization_id,
            created_by_user_id=ctx.user_id,
            quality_score=request.quality_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ManualFieldResponse(
        field=FieldSummary(
            id=field.id,
            aoi_run_id=field.aoi_run_id,
            area_m2=field.area_m2,
            perimeter_m=field.perimeter_m,
            quality_score=field.quality_score,
            source=field.source,
            created_at=field.created_at,
        )
    )


@router.get("/fields/geojson")
async def get_manual_fields_geojson(
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict:
    repo = FieldsRepository(db)
    return await repo.get_manual_fields_geojson(organization_id=ctx.organization_id)
