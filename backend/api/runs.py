"""Run status/result/list API routes: /fields/runs, /fields/status, /fields/result.

Extracted from the monolithic api/fields.py for maintainability.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api._fields_shared import (
    aggregate_runtime_mode,
    candidate_summary,
    effective_run_status,
    elapsed_s,
    estimated_remaining_s,
    geometry_summary,
    infer_detect_preset,
    progress_pct,
    runtime_int_metric,
    runtime_progress_metric,
    runtime_with_stale_flag,
    stage_code,
    stage_label,
    updated_at_iso,
)
from api._fields_shared import _DETAIL_PROGRESS_RE  # noqa: PLC2701
from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    DetectRequest,
    RunListResponse,
    RunResult,
    RunStatus,
    RunSummary,
)
from storage.db import AoiRun, get_db
from storage.fields_repo import FieldsRepository
from typing import Any

router = APIRouter(prefix="/fields", tags=["fields"])


# ---------------------------------------------------------------------------
# Internal helpers used only in this module
# ---------------------------------------------------------------------------


def _explicit_stage_detail_payload(
    detail: str | None,
    *,
    stage_code_value: str | None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    explicit_detail = str(detail or "").strip()
    if not explicit_detail:
        return None, None, {}
    lowered = explicit_detail.lower()
    if lowered == "postprocess start":
        return explicit_detail, "postprocess_start", {}
    if lowered == "watershed":
        return explicit_detail, "watershed", {}
    match = _DETAIL_PROGRESS_RE.match(explicit_detail)
    if match:
        label = str(match.group("label") or "").strip().lower().replace(" ", "_")
        extra = str(match.group("extra") or "").strip() or None
        return (
            explicit_detail,
            f"{label}_progress" if label else "progress_detail",
            {
                "current": int(match.group("current")),
                "total": int(match.group("total")),
                **({   "extra": extra} if extra else {}),
            },
        )
    return explicit_detail, None, {}


def _stage_detail_payload(
    runtime: dict | None, s_code: str | None, status: str
) -> tuple[str | None, str | None, dict[str, Any]]:
    runtime = dict(runtime or {})
    if not runtime and not s_code:
        return None, None, {}

    explicit_detail = runtime.get("progress_detail")
    detail_text, detail_code, detail_params = _explicit_stage_detail_payload(
        explicit_detail, stage_code_value=s_code
    )
    if detail_text:
        return detail_text, detail_code, detail_params

    progress_stage = str(runtime.get("progress_stage") or "").strip().lower()
    tile_count = int(runtime.get("tile_count") or max(len(runtime.get("tiles") or []), 0))
    current_tile_index = int(runtime.get("current_tile_index") or 0)
    time_windows = list(runtime.get("time_windows") or [])
    selected_dates = list(runtime.get("selected_dates") or [])
    n_valid_scenes = int(runtime.get("n_valid_scenes") or 0)

    if progress_stage in {"fetch", "date_selection"} and time_windows:
        completed = max(len(selected_dates), n_valid_scenes)
        safe_completed = min(completed, len(time_windows))
        return (
            f"windows {safe_completed}/{len(time_windows)}",
            "windows_progress",
            {"current": safe_completed, "total": len(time_windows)},
        )
    if (
        progress_stage in {
            "candidate_postprocess",
            "model_inference",
            "segmentation",
            "boundary_refine",
            "sam_refine",
            "tile_finalize",
        }
        and tile_count > 0
    ):
        display_tile = current_tile_index or min(len(runtime.get("tiles") or []) + 1, tile_count)
        safe_tile = min(display_tile, tile_count)
        return f"tile {safe_tile}/{tile_count}", "tile_progress", {"current": safe_tile, "total": tile_count}
    if progress_stage == "merge":
        merged_tiles = len(runtime.get("tiles") or [])
        return f"tiles merged: {merged_tiles}", "tiles_merged", {"merged_tiles": merged_tiles}
    if progress_stage == "db_insert":
        grid_cells = int(runtime.get("grid_cells") or 0)
        candidates = int(runtime.get("active_learning_candidates") or 0)
        if grid_cells or candidates:
            return (
                f"grid {grid_cells}, candidates {candidates}",
                "db_insert_counts",
                {"grid_cells": grid_cells, "candidates": candidates},
            )
    if progress_stage == "failed":
        failure_stage = str(runtime.get("failure_stage") or "").strip()
        if failure_stage:
            return f"failure stage: {failure_stage}", "failure_stage", {"failure_stage": failure_stage}
    return None, None, {}


def _run_response_payload(run) -> dict:
    params = dict(run.params or {})
    config = dict(params.get("config") or {})
    preflight = dict(config.get("preflight") or {})
    runtime = runtime_with_stale_flag(run)
    status = effective_run_status(run, runtime)
    prog = progress_pct(run, runtime)
    s_code = stage_code(runtime, status)
    s_label = stage_label(runtime, status)
    stage_detail, stage_detail_code, stage_detail_params = _stage_detail_payload(runtime, s_code, status)
    cand_summary = candidate_summary(runtime)
    return {
        "aoi_run_id": run.id,
        "status": status,
        "progress": int(run.progress or 0),
        "progress_pct": prog,
        "error_msg": run.error_msg,
        "stage_code": s_code,
        "stage_label": s_label,
        "stage_detail": stage_detail,
        "stage_detail_code": stage_detail_code,
        "stage_detail_params": stage_detail_params,
        "stage_progress_pct": runtime_progress_metric(runtime, "stage_progress_pct"),
        "tile_progress_pct": runtime_progress_metric(runtime, "tile_progress_pct"),
        "started_at": (
            (run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=__import__("datetime").timezone.utc)).isoformat()
            if run.created_at is not None
            else None
        ),
        "updated_at": updated_at_iso(run, runtime),
        "last_heartbeat_ts": (runtime or {}).get("last_heartbeat_ts"),
        "stale_running": bool((runtime or {}).get("stale_running")),
        "queue_ahead": 0,
        "blocking_run_id": None,
        "blocking_status": None,
        "elapsed_s": elapsed_s(run, runtime),
        "estimated_remaining_s": estimated_remaining_s(run, runtime, prog, status),
        "qc_mode": aggregate_runtime_mode(runtime, "qc_mode"),
        "processing_profile": aggregate_runtime_mode(runtime, "processing_profile"),
        "pipeline_profile": str((runtime or {}).get("pipeline_profile") or preflight.get("pipeline_profile") or "") or None,
        "preview_only": bool((runtime or {}).get("preview_only", preflight.get("preview_only", False))),
        "output_mode": str((runtime or {}).get("output_mode") or preflight.get("output_mode") or "") or None,
        "operational_eligible": bool((runtime or {}).get("operational_eligible", preflight.get("operational_eligible", True))),
        "max_radius_km": runtime_int_metric(runtime, "max_radius_km") or runtime_int_metric(preflight, "max_radius_km"),
        "recommended_radius_km": runtime_int_metric(runtime, "recommended_radius_km") or runtime_int_metric(preflight, "recommended_radius_km"),
        "enabled_stages": list((runtime or {}).get("enabled_stages") or preflight.get("enabled_stages") or []),
        "candidate_branch_counts": cand_summary["candidate_branch_counts"],
        "candidate_reject_summary": cand_summary["candidate_reject_summary"],
        "candidates_total": cand_summary["candidates_total"],
        "candidates_kept": cand_summary["candidates_kept"],
        "geometry_summary": geometry_summary(runtime),
        "runtime": runtime,
    }


def _run_summary_payload(run) -> dict:
    params = dict(run.params or {})
    config = dict(params.get("config") or {})
    runtime = dict(params.get("runtime") or {})
    cand_summary = candidate_summary(runtime)
    aoi = params.get("aoi") if isinstance(params.get("aoi"), dict) else None
    preset = str(config.get("preset") or runtime.get("preset") or "").strip().lower() or None
    if preset is None and {"aoi", "time_range"}.issubset(params):
        try:
            preset = infer_detect_preset(DetectRequest(**params), use_sam=bool(params.get("use_sam")))
        except Exception:
            preset = None
    from datetime import timezone
    created_at = (
        (run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)).isoformat()
        if run.created_at is not None
        else None
    )
    return {
        "id": run.id,
        "status": str(run.status),
        "progress": int(run.progress or 0),
        "created_at": created_at,
        "preset": preset,
        "aoi": aoi,
        "use_sam": bool(params.get("use_sam")),
        "resolution_m": int(params.get("resolution_m") or 0) or None,
        "target_dates": int(params.get("target_dates") or 0) or None,
        "qc_mode": aggregate_runtime_mode(runtime, "qc_mode"),
        "processing_profile": aggregate_runtime_mode(runtime, "processing_profile"),
        "candidates_total": cand_summary["candidates_total"],
        "candidates_kept": cand_summary["candidates_kept"],
    }


async def _queue_wait_payload(db: AsyncSession, run) -> dict[str, Any]:
    from datetime import timezone
    if str(run.status) != "queued" or run.organization_id is None or run.created_at is None:
        return {"queue_ahead": 0, "blocking_run_id": None, "blocking_status": None}
    created_at = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    result = await db.execute(
        select(AoiRun.id, AoiRun.status, AoiRun.created_at)
        .where(AoiRun.organization_id == run.organization_id)
        .where(AoiRun.id != run.id)
        .where(AoiRun.status.in_(("queued", "running")))
        .where(AoiRun.created_at < created_at)
        .order_by(AoiRun.created_at.asc())
    )
    active_before = list(result.all())
    blocking_row = next((row for row in active_before if str(row.status) == "running"), None)
    if blocking_row is None and active_before:
        blocking_row = active_before[-1]
    return {
        "queue_ahead": len(active_before),
        "blocking_run_id": blocking_row[0] if blocking_row else None,
        "blocking_status": str(blocking_row[1]) if blocking_row else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> RunListResponse:
    repo = FieldsRepository(db)
    runs = await repo.list_runs(organization_id=ctx.organization_id, limit=limit)
    detect_runs = [run for run in runs if isinstance((run.params or {}).get("aoi"), dict)]
    return RunListResponse(runs=[RunSummary(**_run_summary_payload(run)) for run in detect_runs])


@router.get("/status/{aoi_run_id}", response_model=RunStatus)
async def get_run_status(
    aoi_run_id: UUID,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
):
    repo = FieldsRepository(db)
    run = await repo.get_run(aoi_run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    payload = _run_response_payload(run)
    payload.update(await _queue_wait_payload(db, run))
    return RunStatus(**payload)


@router.get("/result/{aoi_run_id}", response_model=RunResult)
async def get_run_result(
    aoi_run_id: UUID,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
):
    repo = FieldsRepository(db)
    run = await repo.get_run(aoi_run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    payload = _run_response_payload(run)
    payload.update(await _queue_wait_payload(db, run))
    if payload["status"] in {"queued", "running", "failed", "stale", "cancelled"}:
        return RunResult(**payload)
    geojson = await repo.get_fields_geojson(aoi_run_id, organization_id=ctx.organization_id)
    payload["geojson"] = geojson
    if payload["status"] == "done":
        payload["progress"] = 100
        payload["progress_pct"] = 100.0
        payload["estimated_remaining_s"] = 0
    return RunResult(**payload)
