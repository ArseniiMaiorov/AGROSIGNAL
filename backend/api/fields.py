import base64
import io
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from geoalchemy2.shape import to_shape
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    AsyncJobResultResponse,
    AsyncJobStatusResponse,
    AsyncJobSubmitResponse,
    DetectRequest,
    DetectPreflightResponse,
    DetectResponse,
    DetectionCandidatesResponse,
    DetectionCandidateInfo,
    FieldDashboardResponse,
    FieldDeleteResponse,
    FieldGroupDashboardRequest,
    FieldMergeRequest,
    FieldSplitRequest,
    FieldsListResponse,
    FieldSummary,
    ManagementEventCreate,
    ManagementEventUpdate,
    ManagementEventResponse,
    ManagementEventsListResponse,
    RunListResponse,
    RunResult,
    RunSummary,
    RunStatus,
)
from core.celery_app import has_live_workers_for_queue
from core.config import get_adaptive_season_window, get_settings
from core.metrics import DETECT_REQUESTS
from core.rate_limit import limiter
from core.region import resolve_region_band, resolve_region_boundary_profile
from processing.fields.tiling import bbox_to_polygon, make_tiles, point_radius_to_polygon, polygon_coords_to_polygon
from services.async_job_service import (
    build_async_job_submit_payload,
    get_async_job_payload,
    prime_async_job,
    require_job_access,
)
from services.field_analytics_service import FieldAnalyticsService
from services.temporal_analytics_service import GEOMETRY_FOUNDATION, TemporalAnalyticsService
from tasks.analytics import run_temporal_analytics_job
from services.trust_service import describe_detect_launch
from storage.db import AoiRun, Field, FieldSeason, ManagementEvent, get_db
from storage.fields_repo import FieldsRepository

_settings = get_settings()

router = APIRouter(prefix="/fields", tags=["fields"])

_STAGE_LABELS = {
    "queued": "queued",
    "fetch": "fetch",
    "tiling": "tiling",
    "date_selection": "date selection",
    "candidate_postprocess": "boundary fill",
    "model_inference": "model inference",
    "segmentation": "segmentation",
    "boundary_refine": "boundary refine",
    "sam_refine": "sam refine",
    "tile_finalize": "tile finalize",
    "merge": "merge",
    "object_classifier": "object classifier",
    "db_insert": "db insert",
    "topology": "topology",
    "done": "complete",
    "failed": "failed",
}

_DETAIL_PROGRESS_RE = re.compile(
    r"^(?P<label>[a-z ]+?)\s+(?P<current>\d+)/(?P<total>\d+)(?:\s+[·-]\s+(?P<extra>.+))?$",
    re.IGNORECASE,
)

_DETECT_PRESET_CONFIGS = {
    "fast": {
        "resolution_m": 10,
        "target_dates": 4,
        "use_sam": False,
        "min_field_area_ha": 0.5,
        # Larger preview tiles reduce tile-count overhead and return coarse agri contours.
        "tile_size_px": 1024,
        "max_tiles": 48,
        "max_complexity": 280.0,
        "max_radius_km": 40,
        "recommended_radius_km": 30,
        "tta_mode": "none",
        "s1_policy": "off",
        "multi_scale": False,
        "min_good_dates": 4,
        "pipeline_profile": "fast_preview",
        "preview_only": True,
        "output_mode": "preview_agri_contours",
        "operational_eligible": False,
        "enabled_stages": [
            "fetch",
            "candidate_postprocess",
            "segmentation",
            "boundary_refine",
            "tile_finalize",
            "merge",
            "db_insert",
        ],
    },
    "standard": {
        "resolution_m": 10,
        "target_dates": 7,
        "use_sam": False,
        "min_field_area_ha": 0.25,
        # 896px → 8.96 km tiles; ~25% fewer candidates per tile vs 1024px,
        # fits 20 km radius in ~23 tiles (well within limit)
        "tile_size_px": 896,
        "max_tiles": 36,          # up to ~20 km radius at 896px/8.46km step
        "max_complexity": 260.0,
        "max_radius_km": 20,
        "recommended_radius_km": 20,
        "tta_mode": "flip2",
        "s1_policy": "north_or_opt_in",
        "multi_scale": False,
        "min_good_dates": 6,
        "pipeline_profile": "standard_balanced",
        "preview_only": False,
        "output_mode": "field_boundaries",
        "operational_eligible": True,
        "enabled_stages": [
            "fetch",
            "candidate_postprocess",
            "model_inference",
            "segmentation",
            "boundary_refine",
            "object_classifier",
            "tile_finalize",
            "merge",
            "db_insert",
        ],
    },
    "quality": {
        "resolution_m": 10,
        "target_dates": 9,
        "use_sam": True,
        "min_field_area_ha": 0.1,
        # 768px → 7.68 km tiles; smallest tiles for highest candidate control.
        "tile_size_px": 768,
        "max_tiles": 24,          # up to ~8 km radius at 768px/7.18km step
        "max_complexity": 300.0,
        "max_radius_km": 8,
        "recommended_radius_km": 8,
        "tta_mode": "rotate4",
        "s1_policy": "on",
        "multi_scale": True,
        "min_good_dates": 9,
        "pipeline_profile": "quality_full",
        "preview_only": False,
        "output_mode": "field_boundaries_hifi",
        "operational_eligible": True,
        "enabled_stages": [
            "fetch",
            "candidate_postprocess",
            "model_inference",
            "segmentation",
            "boundary_refine",
            "sam_refine",
            "object_classifier",
            "tile_finalize",
            "merge",
            "db_insert",
        ],
    },
}

_HARD_TILE_LIMIT = 72
_HARD_COMPLEXITY_LIMIT = 1_450.0
_HARD_RAM_LIMIT_MB = 6_500
_DEBUG_LAYER_KEY_MAP = {
    "candidate_initial": "step_00_candidate_initial",
    "after_grow": "step_06_after_grow",
    "after_gap_close": "step_07_after_gap_close",
    "after_infill": "step_08_after_infill",
    "after_merge": "step_09_after_merge",
    "after_watershed": "step_10_after_watershed",
    "barrier_mask": "step_03_barrier_mask",
    "boundary_prob": "boundary_prob",
    "owt_edge": "owt_edge",
    "field_candidate": "step_03b_field_candidate",
}
_DEBUG_LAYER_STYLE = {
    "candidate_initial": {"color": "#f2cf3d", "opacity_default": 0.42, "label": "Candidate initial"},
    "after_grow": {"color": "#d6902a", "opacity_default": 0.45, "label": "After grow"},
    "after_gap_close": {"color": "#df7c28", "opacity_default": 0.48, "label": "After gap close"},
    "after_infill": {"color": "#58a05e", "opacity_default": 0.46, "label": "After infill"},
    "after_merge": {"color": "#3c8dc8", "opacity_default": 0.44, "label": "After merge"},
    "after_watershed": {"color": "#8f5bd2", "opacity_default": 0.44, "label": "After watershed"},
    "barrier_mask": {"color": "#ca493d", "opacity_default": 0.5, "label": "Barrier mask"},
    "field_candidate": {"color": "#d9c44d", "opacity_default": 0.42, "label": "Field candidate"},
    "boundary_prob": {"color": "#ffffff", "opacity_default": 0.68, "label": "Boundary probability"},
    "owt_edge": {"color": "#9be7ff", "opacity_default": 0.68, "label": "OWT edge"},
}





@router.get("", response_model=FieldsListResponse)
async def list_fields(
    aoi_run_id: UUID | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> FieldsListResponse:
    repo = FieldsRepository(db)
    fields = await repo.list_fields(organization_id=ctx.organization_id, run_id=aoi_run_id)
    return FieldsListResponse(
        fields=[
            FieldSummary(
                id=field.id,
                aoi_run_id=field.aoi_run_id,
                area_m2=field.area_m2,
                perimeter_m=field.perimeter_m,
                quality_score=field.quality_score,
                source=field.source,
                created_at=field.created_at,
            )
            for field in fields
        ]
    )


@router.get("/geojson")
async def list_fields_geojson(
    aoi_run_id: UUID | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict:
    repo = FieldsRepository(db)
    return await repo.get_all_fields_geojson(organization_id=ctx.organization_id, run_id=aoi_run_id)


@router.get("/{field_id}/dashboard", response_model=FieldDashboardResponse)
async def get_field_dashboard(
    field_id: UUID,
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    lite: bool = Query(False),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> FieldDashboardResponse:
    service = FieldAnalyticsService(db)
    try:
        payload = await service.get_field_dashboard(
            field_id,
            organization_id=ctx.organization_id,
            date_from=date_from,
            date_to=date_to,
            lite=lite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FieldDashboardResponse(**payload)


@router.get("/{field_id}/temporal-analytics", response_model=dict[str, object])
async def get_field_temporal_analytics(
    field_id: UUID,
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    crop_code: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    service = TemporalAnalyticsService(db)
    try:
        return await service.get_temporal_analytics(
            field_id,
            organization_id=ctx.organization_id,
            date_from=date_from,
            date_to=date_to,
            crop_code=crop_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{field_id}/temporal-analytics/jobs", response_model=AsyncJobSubmitResponse)
async def rebuild_field_temporal_analytics(
    field_id: UUID,
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    crop_code: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> AsyncJobSubmitResponse:
    service = TemporalAnalyticsService(db)
    try:
        resolved_from, resolved_to = service._resolve_requested_range(date_from, date_to)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    task = run_temporal_analytics_job.delay(
        str(field_id),
        str(ctx.organization_id),
        resolved_from.isoformat(),
        resolved_to.isoformat(),
    )
    meta = prime_async_job(
        task_id=task.id,
        job_type="temporal_analytics",
        organization_id=ctx.organization_id,
        field_id=field_id,
        stage_code="queued",
        stage_label="queued",
        stage_detail="waiting for worker",
        stage_detail_code="waiting_for_worker",
        stage_detail_params={
            "date_from": resolved_from.isoformat(),
            "date_to": resolved_to.isoformat(),
            "crop_code": crop_code,
        },
        logs=[f"Задача materialization сезонной аналитики для поля {field_id} поставлена в очередь."],
    )
    return AsyncJobSubmitResponse(task_id=task.id, **build_async_job_submit_payload(meta))


@router.get("/temporal-analytics/jobs/{task_id}", response_model=AsyncJobStatusResponse)
async def get_temporal_analytics_job_status(
    task_id: str,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
) -> AsyncJobStatusResponse:
    payload = get_async_job_payload(task_id)
    try:
        require_job_access(payload, ctx.organization_id, job_type="temporal_analytics")
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AsyncJobStatusResponse(
        task_id=task_id,
        job_type="temporal_analytics",
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


@router.get("/temporal-analytics/jobs/{task_id}/result", response_model=AsyncJobResultResponse)
async def get_temporal_analytics_job_result(
    task_id: str,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
) -> AsyncJobResultResponse:
    payload = get_async_job_payload(task_id)
    try:
        require_job_access(payload, ctx.organization_id, job_type="temporal_analytics")
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AsyncJobResultResponse(
        task_id=task_id,
        job_type="temporal_analytics",
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


@router.get("/{field_id}/management-zones", response_model=dict[str, object])
async def get_field_management_zones(
    field_id: UUID,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    temporal_service = TemporalAnalyticsService(db)
    analytics_service = FieldAnalyticsService(db)
    prediction_payload = await analytics_service._get_latest_prediction(
        field_id,
        organization_id=ctx.organization_id,
    )
    try:
        return await temporal_service.get_management_zones(
            field_id,
            organization_id=ctx.organization_id,
            prediction_payload=prediction_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{field_id}/events", response_model=ManagementEventsListResponse)
async def list_field_events(
    field_id: UUID,
    season_year: int | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> ManagementEventsListResponse:
    stmt = (
        select(ManagementEvent, FieldSeason.season_year)
        .join(FieldSeason, ManagementEvent.field_season_id == FieldSeason.id)
        .where(ManagementEvent.organization_id == ctx.organization_id)
        .where(FieldSeason.field_id == field_id)
        .where(FieldSeason.organization_id == ctx.organization_id)
    )
    if season_year is not None:
        stmt = stmt.where(FieldSeason.season_year == season_year)
    count_stmt = stmt.with_only_columns(func.count()).order_by(None)
    total = (await db.execute(count_stmt)).scalar_one_or_none() or 0
    stmt = stmt.order_by(ManagementEvent.event_date.desc()).offset(offset).limit(limit)
    rows = (await db.execute(stmt)).all()
    events = [
        ManagementEventResponse(
            id=ev.id,
            field_season_id=ev.field_season_id,
            season_year=sy,
            event_date=ev.event_date,
            event_type=ev.event_type,
            amount=ev.amount,
            unit=ev.unit,
            source=ev.source,
            payload=dict(ev.payload or {}),
        )
        for ev, sy in rows
    ]
    return ManagementEventsListResponse(events=events, total=total)


@router.post("/{field_id}/events", response_model=ManagementEventResponse, status_code=201)
async def create_field_event(
    field_id: UUID,
    body: ManagementEventCreate,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> ManagementEventResponse:
    # Verify field belongs to org
    field_stmt = select(Field).where(Field.id == field_id).where(Field.organization_id == ctx.organization_id)
    field = (await db.execute(field_stmt)).scalar_one_or_none()
    if field is None:
        raise HTTPException(status_code=404, detail="Field not found")
    # Get or create FieldSeason
    season_stmt = (
        select(FieldSeason)
        .where(FieldSeason.organization_id == ctx.organization_id)
        .where(FieldSeason.field_id == field_id)
        .where(FieldSeason.season_year == body.season_year)
    )
    season = (await db.execute(season_stmt)).scalar_one_or_none()
    if season is None:
        season = FieldSeason(
            organization_id=ctx.organization_id,
            field_id=field_id,
            season_year=body.season_year,
            label=str(body.season_year),
            external_field_id=field.external_field_id,
        )
        db.add(season)
        await db.flush()
    event = ManagementEvent(
        organization_id=ctx.organization_id,
        field_season_id=season.id,
        event_date=body.event_date,
        event_type=body.event_type,
        amount=body.amount,
        unit=body.unit,
        source="manual",
        payload=body.payload,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return ManagementEventResponse(
        id=event.id,
        field_season_id=event.field_season_id,
        season_year=season.season_year,
        event_date=event.event_date,
        event_type=event.event_type,
        amount=event.amount,
        unit=event.unit,
        source=event.source,
        payload=dict(event.payload or {}),
    )


@router.patch("/{field_id}/events/{event_id}", response_model=ManagementEventResponse)
async def update_field_event(
    field_id: UUID,
    event_id: int,
    body: ManagementEventUpdate,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> ManagementEventResponse:
    stmt = (
        select(ManagementEvent, FieldSeason.season_year)
        .join(FieldSeason, ManagementEvent.field_season_id == FieldSeason.id)
        .where(ManagementEvent.id == event_id)
        .where(ManagementEvent.organization_id == ctx.organization_id)
        .where(FieldSeason.field_id == field_id)
        .where(FieldSeason.organization_id == ctx.organization_id)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    event, season_year = row
    if body.event_date is not None:
        event.event_date = body.event_date
    if body.event_type is not None:
        event.event_type = body.event_type
    if body.amount is not None:
        event.amount = body.amount
    if body.unit is not None:
        event.unit = body.unit
    if body.payload is not None:
        event.payload = body.payload
    await db.commit()
    await db.refresh(event)
    return ManagementEventResponse(
        id=event.id,
        field_season_id=event.field_season_id,
        season_year=season_year,
        event_date=event.event_date,
        event_type=event.event_type,
        amount=event.amount,
        unit=event.unit,
        source=event.source,
        payload=dict(event.payload or {}),
    )


@router.delete("/{field_id}/events/{event_id}", status_code=204)
async def delete_field_event(
    field_id: UUID,
    event_id: int,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> None:
    stmt = (
        select(ManagementEvent)
        .join(FieldSeason, ManagementEvent.field_season_id == FieldSeason.id)
        .where(ManagementEvent.id == event_id)
        .where(ManagementEvent.organization_id == ctx.organization_id)
        .where(FieldSeason.field_id == field_id)
        .where(FieldSeason.organization_id == ctx.organization_id)
    )
    event = (await db.execute(stmt)).scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    await db.delete(event)
    await db.commit()


@router.post("/dashboard/group", response_model=FieldDashboardResponse)
async def get_group_dashboard(
    request: FieldGroupDashboardRequest,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> FieldDashboardResponse:
    service = FieldAnalyticsService(db)
    try:
        payload = await service.get_group_dashboard(list(request.field_ids), organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FieldDashboardResponse(**payload)


@router.post("/merge", response_model=FieldSummary)
async def merge_fields(
    request: FieldMergeRequest,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> FieldSummary:
    repo = FieldsRepository(db)
    try:
        field = await repo.merge_fields(list(request.field_ids), organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FieldSummary(
        id=field.id,
        aoi_run_id=field.aoi_run_id,
        area_m2=field.area_m2,
        perimeter_m=field.perimeter_m,
        quality_score=field.quality_score,
        source=field.source,
        created_at=field.created_at,
    )


@router.post("/split", response_model=FieldsListResponse)
async def split_field(
    request: FieldSplitRequest,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> FieldsListResponse:
    repo = FieldsRepository(db)
    try:
        fields = await repo.split_field(request.field_id, request.geometry, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FieldsListResponse(
        fields=[
            FieldSummary(
                id=field.id,
                aoi_run_id=field.aoi_run_id,
                area_m2=field.area_m2,
                perimeter_m=field.perimeter_m,
                quality_score=field.quality_score,
                source=field.source,
                created_at=field.created_at,
            )
            for field in fields
        ]
    )


@router.delete("/{field_id}", response_model=FieldDeleteResponse)
async def delete_field(
    field_id: UUID,
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
) -> FieldDeleteResponse:
    repo = FieldsRepository(db)
    try:
        payload = await repo.delete_field(field_id, organization_id=ctx.organization_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FieldDeleteResponse(**payload)


