"""Anonymous bootstrap status for pre-login shell."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import BootstrapResponse
from services.status_service import StatusService
from storage.db import get_db

router = APIRouter(prefix="/bootstrap", tags=["bootstrap"])


@router.get("", response_model=BootstrapResponse)
async def get_bootstrap_status(
    db: AsyncSession = Depends(get_db),
) -> BootstrapResponse:
    service = StatusService(db)
    payload = await service.get_bootstrap_status()
    return BootstrapResponse(**payload)
