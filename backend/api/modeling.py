"""API моделирования сценариев."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    AsyncJobResultResponse,
    AsyncJobStatusResponse,
    AsyncJobSubmitResponse,
    ModelingRequest,
    ModelingResponse,
    ScenarioListResponse,
    ScenarioRunResponse,
    SensitivitySweepRequest,
    SensitivitySweepResponse,
)
from services.async_job_service import (
    build_async_job_submit_payload,
    get_async_job_payload,
    prime_async_job,
    require_job_access,
)
from services.modeling_service import ModelingService
from tasks.analytics import run_modeling_job
from storage.db import get_db

router = APIRouter(prefix="/modeling", tags=["modeling"])


@router.post("/simulate", response_model=ModelingResponse)
async def simulate_modeling(
    request: ModelingRequest,
    ctx: RequestContext = Depends(require_permissions("scenarios:write")),
    db: AsyncSession = Depends(get_db),
) -> ModelingResponse:
    service = ModelingService(db)
    try:
        payload = await service.simulate(
            request.field_id,
            organization_id=ctx.organization_id,
            crop_code=request.crop_code,
            scenario_name=request.scenario_name,
            irrigation_pct=request.irrigation_pct,
            fertilizer_pct=request.fertilizer_pct,
            expected_rain_mm=request.expected_rain_mm,
            temperature_delta_c=request.temperature_delta_c,
            precipitation_factor=request.precipitation_factor,
            planting_density_pct=request.planting_density_pct,
            sowing_shift_days=request.sowing_shift_days,
            tillage_type=request.tillage_type,
            pest_pressure=request.pest_pressure,
            soil_compaction=request.soil_compaction,
            cloud_cover_factor=request.cloud_cover_factor,
            irrigation_events=[item.model_dump(exclude_none=True) for item in request.irrigation_events],
            fertilizer_events=[item.model_dump(exclude_none=True) for item in request.fertilizer_events],
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelingResponse(**payload)


@router.post("/jobs", response_model=AsyncJobSubmitResponse)
async def submit_modeling_job(
    request: ModelingRequest,
    ctx: RequestContext = Depends(require_permissions("scenarios:write")),
) -> AsyncJobSubmitResponse:
    task = run_modeling_job.delay(str(ctx.organization_id), request.model_dump(mode="json"))
    meta = prime_async_job(
        task_id=task.id,
        job_type="scenario",
        organization_id=ctx.organization_id,
        field_id=request.field_id,
        stage_code="queued",
        stage_label="queued",
        stage_detail="waiting for worker",
        stage_detail_code="waiting_for_worker",
        logs=[f"Задача сценарного моделирования для поля {request.field_id} поставлена в очередь."],
    )
    return AsyncJobSubmitResponse(task_id=task.id, **build_async_job_submit_payload(meta))


@router.get("/jobs/{task_id}", response_model=AsyncJobStatusResponse)
async def get_modeling_job_status(
    task_id: str,
    ctx: RequestContext = Depends(require_permissions("scenarios:read")),
) -> AsyncJobStatusResponse:
    payload = get_async_job_payload(task_id)
    try:
        require_job_access(payload, ctx.organization_id, job_type="scenario")
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AsyncJobStatusResponse(
        task_id=task_id,
        job_type="scenario",
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
async def get_modeling_job_result(
    task_id: str,
    ctx: RequestContext = Depends(require_permissions("scenarios:read")),
) -> AsyncJobResultResponse:
    payload = get_async_job_payload(task_id)
    try:
        require_job_access(payload, ctx.organization_id, job_type="scenario")
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AsyncJobResultResponse(
        task_id=task_id,
        job_type="scenario",
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


@router.post("/sensitivity", response_model=SensitivitySweepResponse)
async def sensitivity_sweep(
    request: SensitivitySweepRequest,
    ctx: RequestContext = Depends(require_permissions("scenarios:read")),
    db: AsyncSession = Depends(get_db),
) -> SensitivitySweepResponse:
    """Run sensitivity sweep: vary one parameter across a range, return response curve."""
    import numpy as _np
    sweep_values = list(_np.linspace(request.sweep_min, request.sweep_max, request.sweep_steps))
    service = ModelingService(db)
    try:
        result = await service.sensitivity_sweep(
            request.field_id,
            organization_id=ctx.organization_id,
            crop_code=request.crop_code,
            base_adjustments=request.base_adjustments,
            sweep_param=request.sweep_param,
            sweep_values=sweep_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Sensitivity sweep failed: {exc}") from exc
    return SensitivitySweepResponse(**result)


@router.get("/scenarios", response_model=ScenarioListResponse)
async def list_scenarios(
    field_id: UUID = Query(...),
    ctx: RequestContext = Depends(require_permissions("scenarios:read")),
    db: AsyncSession = Depends(get_db),
) -> ScenarioListResponse:
    service = ModelingService(db)
    items = await service.list_scenarios(field_id, organization_id=ctx.organization_id)
    return ScenarioListResponse(scenarios=[ScenarioRunResponse(**item) for item in items])
