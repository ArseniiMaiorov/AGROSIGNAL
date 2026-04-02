"""API настройки локального и облачного хранения."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import RequestContext, require_permissions
from api.schemas import StorageConfigRequest, StorageConfigResponse
from services.storage_service import StorageService

router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("", response_model=StorageConfigResponse)
async def get_storage_config(
    ctx: RequestContext = Depends(require_permissions("storage:read")),
) -> StorageConfigResponse:
    service = StorageService()
    return StorageConfigResponse(**service.get_config(ctx.organization_id))


@router.post("", response_model=StorageConfigResponse)
async def update_storage_config(
    request: StorageConfigRequest,
    ctx: RequestContext = Depends(require_permissions("storage:write")),
) -> StorageConfigResponse:
    service = StorageService()
    return StorageConfigResponse(
        **service.update_config(
            ctx.organization_id,
            {"storage_mode": request.storage_mode, "cloud_url": request.cloud_url},
        )
    )


@router.post("/connect", response_model=StorageConfigResponse)
async def connect_storage_cloud(
    ctx: RequestContext = Depends(require_permissions("storage:write")),
) -> StorageConfigResponse:
    service = StorageService()
    return StorageConfigResponse(**service.connect_cloud(ctx.organization_id))
