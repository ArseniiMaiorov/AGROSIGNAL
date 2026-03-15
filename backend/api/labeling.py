from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    DatasetManifestResponse,
    LabelReviewDecisionRequest,
    LabelTaskCreateRequest,
    LabelTaskListResponse,
    LabelTaskResponse,
    LabelVersionCreateRequest,
)
from services.labeling_service import LabelingService
from storage.db import get_db

router = APIRouter(prefix="/labeling", tags=["labeling"])


@router.get("/tasks", response_model=LabelTaskListResponse)
async def list_tasks(
    status: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("labeling:read")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskListResponse:
    service = LabelingService(db)
    items = await service.list_tasks(organization_id=ctx.organization_id, status=status)
    return LabelTaskListResponse(tasks=[LabelTaskResponse(**item) for item in items])


@router.post("/tasks", response_model=LabelTaskResponse)
async def create_task(
    payload: LabelTaskCreateRequest,
    ctx: RequestContext = Depends(require_permissions("labeling:write")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskResponse:
    service = LabelingService(db)
    item = await service.create_task(
        organization_id=ctx.organization_id,
        created_by_user_id=ctx.user_id,
        aoi_run_id=payload.aoi_run_id,
        field_id=payload.field_id,
        title=payload.title,
        source=payload.source,
        queue_name=payload.queue_name,
        priority_score=payload.priority_score,
        task_payload=payload.task_payload,
        geometry=payload.geometry,
    )
    return LabelTaskResponse(**item)


@router.get("/tasks/{task_id}", response_model=LabelTaskResponse)
async def get_task(
    task_id: int,
    ctx: RequestContext = Depends(require_permissions("labeling:read")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskResponse:
    service = LabelingService(db)
    try:
        item = await service.get_task(task_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LabelTaskResponse(**item)


@router.post("/tasks/{task_id}/claim", response_model=LabelTaskResponse)
async def claim_task(
    task_id: int,
    ctx: RequestContext = Depends(require_permissions("labeling:write")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskResponse:
    service = LabelingService(db)
    try:
        item = await service.claim_task(task_id, organization_id=ctx.organization_id, user_id=ctx.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LabelTaskResponse(**item)


@router.post("/tasks/{task_id}/versions", response_model=LabelTaskResponse)
async def add_version(
    task_id: int,
    payload: LabelVersionCreateRequest,
    ctx: RequestContext = Depends(require_permissions("labeling:write")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskResponse:
    service = LabelingService(db)
    try:
        item = await service.add_version(
            task_id=task_id,
            organization_id=ctx.organization_id,
            created_by_user_id=ctx.user_id,
            geometry=payload.geometry,
            notes=payload.notes,
            quality_tier=payload.quality_tier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LabelTaskResponse(**item)


@router.post("/reviews/{review_id}/approve", response_model=LabelTaskResponse)
async def approve_review(
    review_id: int,
    payload: LabelReviewDecisionRequest,
    ctx: RequestContext = Depends(require_permissions("labeling:review")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskResponse:
    service = LabelingService(db)
    try:
        item = await service.approve_review(
            review_id,
            organization_id=ctx.organization_id,
            reviewer_user_id=ctx.user_id,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LabelTaskResponse(**item)


@router.post("/reviews/{review_id}/reject", response_model=LabelTaskResponse)
async def reject_review(
    review_id: int,
    payload: LabelReviewDecisionRequest,
    ctx: RequestContext = Depends(require_permissions("labeling:review")),
    db: AsyncSession = Depends(get_db),
) -> LabelTaskResponse:
    service = LabelingService(db)
    try:
        item = await service.reject_review(
            review_id,
            organization_id=ctx.organization_id,
            reviewer_user_id=ctx.user_id,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return LabelTaskResponse(**item)


@router.get("/export-manifest", response_model=DatasetManifestResponse)
async def export_manifest(
    dataset_version: str = Query(..., min_length=3),
    ctx: RequestContext = Depends(require_permissions("labeling:review", "mlops:write")),
    db: AsyncSession = Depends(get_db),
) -> DatasetManifestResponse:
    service = LabelingService(db)
    item = await service.export_manifest(organization_id=ctx.organization_id, dataset_version=dataset_version)
    return DatasetManifestResponse(**item)
