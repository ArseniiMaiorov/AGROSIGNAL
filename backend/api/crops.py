"""API справочника культур."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import CropInfoResponse, CropListResponse
from services.crop_service import CropService
from storage.db import get_db

router = APIRouter(prefix="/crops", tags=["crops"])


@router.get("", response_model=CropListResponse)
async def list_crops(
    _ctx: RequestContext = Depends(require_permissions("crops:read")),
    db: AsyncSession = Depends(get_db),
) -> CropListResponse:
    service = CropService(db)
    crops = await service.list_crops()
    return CropListResponse(
        crops=[
            CropInfoResponse(
                id=crop.id,
                code=crop.code,
                name=crop.name,
                category=crop.category,
                yield_baseline_kg_ha=crop.yield_baseline_kg_ha,
                ndvi_target=crop.ndvi_target,
                base_temp_c=crop.base_temp_c,
                description=crop.description,
            )
            for crop in crops
        ]
    )
