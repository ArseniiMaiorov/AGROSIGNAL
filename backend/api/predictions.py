"""API прогнозов урожайности."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    AsyncJobResultResponse,
    AsyncJobStatusResponse,
    AsyncJobSubmitResponse,
    WeeklyFeatureRowResponse,
    WeeklyProfileResponse,
    YieldPredictionResponse,
)
from services.async_job_service import (
    build_async_job_submit_payload,
    get_async_job_payload,
    prime_async_job,
    require_job_access,
)
from services.yield_service import YieldService
from services.weekly_profile_service import (
    FEATURE_SCHEMA_VERSION,
    current_season_year,
    ensure_weekly_profile,
    load_crop_hint,
    serialize_weekly_feature_rows,
    summarize_geometry_quality,
)
from tasks.analytics import run_prediction_job, run_weekly_backfill_job
from storage.db import get_db

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/field/{field_id}", response_model=YieldPredictionResponse)
async def get_field_prediction(
    field_id: UUID,
    crop_code: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("predictions:read")),
    db: AsyncSession = Depends(get_db),
) -> YieldPredictionResponse:
    service = YieldService(db)
    try:
        payload = await service.get_or_create_prediction(
            field_id,
            crop_code=crop_code,
            refresh=False,
            organization_id=ctx.organization_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return YieldPredictionResponse(**payload)


@router.get("/field/{field_id}/weekly-profile", response_model=WeeklyProfileResponse)
async def get_field_weekly_profile(
    field_id: UUID,
    season_year: int | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("predictions:read")),
    db: AsyncSession = Depends(get_db),
) -> WeeklyProfileResponse:
    resolved_year = int(season_year or current_season_year())
    rows = await ensure_weekly_profile(
        db,
        organization_id=ctx.organization_id,
        field_id=field_id,
        season_year=resolved_year,
    )
    crop_hint = await load_crop_hint(
        db,
        organization_id=ctx.organization_id,
        field_id=field_id,
        season_year=resolved_year,
    )
    payload_rows = serialize_weekly_feature_rows(rows)
    return WeeklyProfileResponse(
        field_id=str(field_id),
        season_year=resolved_year,
        weeks_count=len(payload_rows),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        geometry_quality_summary=summarize_geometry_quality(rows),
        crop_hint=crop_hint,
        rows=[WeeklyFeatureRowResponse(**row) for row in payload_rows],
    )


@router.post("/field/{field_id}/weekly-profile/backfill", response_model=AsyncJobSubmitResponse)
async def backfill_field_weekly_profile(
    field_id: UUID,
    season_year: int | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("predictions:write")),
) -> AsyncJobSubmitResponse:
    resolved_year = int(season_year or current_season_year())
    task = run_weekly_backfill_job.delay(
        str(field_id),
        str(ctx.organization_id),
        resolved_year,
    )
    meta = prime_async_job(
        task_id=task.id,
        job_type="weekly_backfill",
        organization_id=ctx.organization_id,
        field_id=str(field_id),
        stage_code="waiting_for_worker",
        stage_label="queued",
        stage_detail="Задача backfill недельного профиля поставлена в очередь.",
        stage_detail_code="waiting_for_worker",
        logs=[f"Backfill недельного профиля для поля {field_id}, сезон {resolved_year} поставлен в очередь."],
    )
    return AsyncJobSubmitResponse(task_id=task.id, **build_async_job_submit_payload(meta))


@router.post("/field/{field_id}/refresh", response_model=YieldPredictionResponse)
async def refresh_field_prediction(
    field_id: UUID,
    crop_code: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("predictions:write")),
    db: AsyncSession = Depends(get_db),
) -> YieldPredictionResponse:
    service = YieldService(db)
    try:
        payload = await service.get_or_create_prediction(
            field_id,
            crop_code=crop_code,
            refresh=True,
            organization_id=ctx.organization_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return YieldPredictionResponse(**payload)


@router.post("/field/{field_id}/jobs", response_model=AsyncJobSubmitResponse)
async def submit_field_prediction_job(
    field_id: UUID,
    crop_code: str | None = Query(None),
    refresh: bool = Query(True),
    ctx: RequestContext = Depends(require_permissions("predictions:write")),
) -> AsyncJobSubmitResponse:
    task = run_prediction_job.delay(
        str(field_id),
        str(ctx.organization_id),
        crop_code,
        refresh,
    )
    meta = prime_async_job(
        task_id=task.id,
        job_type="prediction",
        organization_id=ctx.organization_id,
        field_id=field_id,
        stage_code="queued",
        stage_label="queued",
        stage_detail="waiting for worker",
        stage_detail_code="waiting_for_worker",
        logs=[f"Задача прогноза для поля {field_id} поставлена в очередь."],
    )
    return AsyncJobSubmitResponse(task_id=task.id, **build_async_job_submit_payload(meta))


@router.get("/jobs/{task_id}", response_model=AsyncJobStatusResponse)
async def get_prediction_job_status(
    task_id: str,
    ctx: RequestContext = Depends(require_permissions("predictions:read")),
) -> AsyncJobStatusResponse:
    payload = get_async_job_payload(task_id)
    try:
        require_job_access(payload, ctx.organization_id, job_type="prediction")
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AsyncJobStatusResponse(
        task_id=task_id,
        job_type="prediction",
        status=payload.get("status", "queued"),
        progress=int(payload.get("progress", 0) or 0),
        progress_pct=float(payload.get("progress_pct", payload.get("progress", 0)) or 0.0),
        stage_code=payload.get("stage_code"),
        stage_label=payload.get("stage_label"),
        stage_detail=payload.get("stage_detail"),
        stage_detail_code=payload.get("stage_detail_code"),
        stage_detail_params=dict(payload.get("stage_detail_params") or {}),
        started_at=payload.get("started_at"),
        updated_at=payload.get("updated_at"),
        elapsed_s=payload.get("elapsed_s"),
        estimated_remaining_s=payload.get("estimated_remaining_s"),
        logs=list(payload.get("logs") or []),
        error_msg=payload.get("error_msg"),
        result_ready=bool(payload.get("result_ready")),
    )


@router.get("/jobs/{task_id}/result", response_model=AsyncJobResultResponse)
async def get_prediction_job_result(
    task_id: str,
    ctx: RequestContext = Depends(require_permissions("predictions:read")),
) -> AsyncJobResultResponse:
    payload = get_async_job_payload(task_id)
    try:
        require_job_access(payload, ctx.organization_id, job_type="prediction")
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AsyncJobResultResponse(
        task_id=task_id,
        job_type="prediction",
        status=payload.get("status", "queued"),
        progress=int(payload.get("progress", 0) or 0),
        progress_pct=float(payload.get("progress_pct", payload.get("progress", 0)) or 0.0),
        stage_code=payload.get("stage_code"),
        stage_label=payload.get("stage_label"),
        stage_detail=payload.get("stage_detail"),
        stage_detail_code=payload.get("stage_detail_code"),
        stage_detail_params=dict(payload.get("stage_detail_params") or {}),
        started_at=payload.get("started_at"),
        updated_at=payload.get("updated_at"),
        elapsed_s=payload.get("elapsed_s"),
        estimated_remaining_s=payload.get("estimated_remaining_s"),
        logs=list(payload.get("logs") or []),
        error_msg=payload.get("error_msg"),
        result_ready=bool(payload.get("result_ready")),
        result=payload.get("result"),
    )
