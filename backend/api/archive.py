"""API архивов по полям."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import ArchiveCreateRequest, ArchiveEntryResponse, ArchiveListResponse, ArchiveSnapshotResponse
from services.archive_service import ArchiveService
from storage.db import get_db

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("", response_model=ArchiveListResponse)
async def list_archives(
    field_id: UUID | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("archive:read")),
    db: AsyncSession = Depends(get_db),
) -> ArchiveListResponse:
    service = ArchiveService(db)
    items = await service.list_archives(field_id=field_id, organization_id=ctx.organization_id)
    return ArchiveListResponse(archives=[ArchiveEntryResponse(**item) for item in items])


@router.post("/create", response_model=ArchiveEntryResponse)
async def create_archive(
    request: ArchiveCreateRequest,
    ctx: RequestContext = Depends(require_permissions("archive:write")),
    db: AsyncSession = Depends(get_db),
) -> ArchiveEntryResponse:
    service = ArchiveService(db)
    try:
        item = await service.create_archive(
            field_id=request.field_id,
            date_from=request.date_from,
            date_to=request.date_to,
            layers=request.layers,
            organization_id=ctx.organization_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ArchiveEntryResponse(**item)


@router.get("/{archive_id}/download")
async def download_archive(
    archive_id: int,
    ctx: RequestContext = Depends(require_permissions("archive:read")),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    service = ArchiveService(db)
    try:
        file_path = await service.get_archive_path(archive_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = Path(file_path).resolve()
    archive_dir = service.archive_dir.resolve()
    try:
        path.relative_to(archive_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Недопустимый путь к архиву")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл архива не найден")
    return FileResponse(path=path, media_type="application/zip", filename=path.name)


@router.get("/{archive_id}/view", response_model=ArchiveSnapshotResponse)
async def get_archive_view(
    archive_id: int,
    ctx: RequestContext = Depends(require_permissions("archive:read")),
    db: AsyncSession = Depends(get_db),
) -> ArchiveSnapshotResponse:
    service = ArchiveService(db)
    try:
        payload = await service.get_archive_view(archive_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ArchiveSnapshotResponse(**payload)
