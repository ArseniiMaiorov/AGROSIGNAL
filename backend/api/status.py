"""API статуса системы."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import SystemStatusResponse
from services.status_service import StatusService
from storage.db import get_db

router = APIRouter(prefix="/status", tags=["status"])


@router.get("", response_model=SystemStatusResponse)
async def get_system_status(
    ctx: RequestContext = Depends(require_permissions("status:read")),
    db: AsyncSession = Depends(get_db),
) -> SystemStatusResponse:
    service = StatusService(db)
    payload = await service.get_system_status(organization_id=ctx.organization_id)
    return SystemStatusResponse(**payload)
