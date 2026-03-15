from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    DataImportCreateRequest,
    DataImportErrorsListResponse,
    DataImportErrorResponse,
    DataImportListResponse,
    DataImportPreviewResponse,
)
from services.data_import_service import DataImportService
from storage.db import get_db

router = APIRouter(prefix="/data-imports", tags=["data-imports"])


@router.get("", response_model=DataImportListResponse)
async def list_jobs(
    ctx: RequestContext = Depends(require_permissions("imports:read")),
    db: AsyncSession = Depends(get_db),
) -> DataImportListResponse:
    service = DataImportService(db)
    items = await service.list_jobs(organization_id=ctx.organization_id)
    return DataImportListResponse(jobs=[DataImportPreviewResponse(**item) for item in items])


@router.post("", response_model=DataImportPreviewResponse)
async def create_job(
    payload: DataImportCreateRequest,
    ctx: RequestContext = Depends(require_permissions("imports:write")),
    db: AsyncSession = Depends(get_db),
) -> DataImportPreviewResponse:
    service = DataImportService(db)
    try:
        content = base64.b64decode(payload.content_base64.encode("ascii"), validate=True)
        item = await service.create_import(
            organization_id=ctx.organization_id,
            created_by_user_id=ctx.user_id,
            import_type=payload.import_type,
            source_filename=payload.source_filename,
            content=content,
        )
    except (base64.binascii.Error, UnicodeEncodeError) as exc:
        raise HTTPException(status_code=422, detail="content_base64 must be valid base64") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DataImportPreviewResponse(**item)


@router.get("/{job_id}", response_model=DataImportPreviewResponse)
async def get_job(
    job_id: int,
    ctx: RequestContext = Depends(require_permissions("imports:read")),
    db: AsyncSession = Depends(get_db),
) -> DataImportPreviewResponse:
    service = DataImportService(db)
    try:
        item = await service.get_job(job_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DataImportPreviewResponse(**item)


@router.post("/{job_id}/preview", response_model=DataImportPreviewResponse)
async def preview_job(
    job_id: int,
    ctx: RequestContext = Depends(require_permissions("imports:write")),
    db: AsyncSession = Depends(get_db),
) -> DataImportPreviewResponse:
    service = DataImportService(db)
    try:
        item = await service.preview_job(job_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DataImportPreviewResponse(**item)


@router.post("/{job_id}/commit", response_model=DataImportPreviewResponse)
async def commit_job(
    job_id: int,
    ctx: RequestContext = Depends(require_permissions("imports:write")),
    db: AsyncSession = Depends(get_db),
) -> DataImportPreviewResponse:
    service = DataImportService(db)
    try:
        item = await service.commit_job(job_id, organization_id=ctx.organization_id, actor_user_id=ctx.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DataImportPreviewResponse(**item)


@router.get("/{job_id}/errors", response_model=DataImportErrorsListResponse)
async def list_errors(
    job_id: int,
    ctx: RequestContext = Depends(require_permissions("imports:read")),
    db: AsyncSession = Depends(get_db),
) -> DataImportErrorsListResponse:
    service = DataImportService(db)
    try:
        items = await service.list_errors(job_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DataImportErrorsListResponse(errors=[DataImportErrorResponse(**item) for item in items])
