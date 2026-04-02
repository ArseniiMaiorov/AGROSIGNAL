"""Detection-related API routes: /fields/detect/preflight, /fields/detect.

Extracted from the monolithic api/fields.py for maintainability.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api._fields_shared import (
    DETECT_PRESET_CONFIGS,
    HARD_COMPLEXITY_LIMIT,
    HARD_RAM_LIMIT_MB,
    HARD_TILE_LIMIT,
    estimate_runtime_class,
    infer_detect_preset,
    resolve_aoi,
)
from api.dependencies import RequestContext, require_permissions
from api.schemas import DetectPreflightResponse, DetectRequest, DetectResponse
from core.celery_app import has_live_workers_for_queue
from core.config import get_adaptive_season_window, get_settings
from core.metrics import DETECT_REQUESTS
from core.rate_limit import limiter
from core.region import resolve_region_band, resolve_region_boundary_profile
from processing.fields.tiling import make_tiles
from services.trust_service import describe_detect_launch
from storage.db import get_db
from storage.fields_repo import FieldsRepository
from datetime import datetime, timezone

_settings = get_settings()

router = APIRouter(prefix="/fields", tags=["fields"])


def _build_detect_preflight(req: DetectRequest, *, use_sam: bool) -> dict:
    preset = infer_detect_preset(req, use_sam=use_sam)
    budget = DETECT_PRESET_CONFIGS[preset]
    aoi_geom = resolve_aoi(req)
    centroid = getattr(aoi_geom, "centroid", None)
    aoi_lat = float(getattr(centroid, "y", req.aoi.lat))
    region_band = resolve_region_band(
        aoi_lat,
        south_max=float(getattr(_settings, "REGION_LAT_SOUTH_MAX", 48.0)),
        north_min=float(getattr(_settings, "REGION_LAT_NORTH_MIN", 57.0)),
    )
    regional_profile = resolve_region_boundary_profile(aoi_lat, _settings)
    season_start, season_end = get_adaptive_season_window(aoi_lat, _settings)
    preset_tile_size_px = int(budget.get("tile_size_px", _settings.TILE_SIZE_PX))
    resolution_m = max(10.0, float(req.resolution_m))
    tiles = make_tiles(
        aoi_geom,
        tile_size_m=preset_tile_size_px * resolution_m,
        overlap_m=_settings.TILE_OVERLAP_M,
        resolution_m=resolution_m,
    )
    estimated_tiles = len(tiles)
    target_dates = max(1, int(req.target_dates))
    requested_radius_km = None
    if getattr(req.aoi, "type", None) == "point_radius":
        requested_radius_km = float(getattr(req.aoi, "radius_km", 0.0) or 0.0)
    preset_max_radius_km = int(budget.get("max_radius_km") or 0)
    preset_recommended_radius_km = int(budget.get("recommended_radius_km") or preset_max_radius_km or 0)
    sam_penalty = 1.5 if use_sam else 1.0
    tta_penalty = 1.35 if budget["tta_mode"] == "flip2" else 1.9 if budget["tta_mode"] == "rotate4" else 1.0
    multi_scale_penalty = 1.25 if bool(budget["multi_scale"]) else 1.0
    s1_planned = (
        budget["s1_policy"] == "on"
        or (budget["s1_policy"] == "north_or_opt_in" and region_band == "north")
    )
    s1_penalty = 1.2 if s1_planned else 1.0
    complexity_score = estimated_tiles * target_dates * sam_penalty * tta_penalty * multi_scale_penalty * s1_penalty
    expected_dates_ok = target_dates >= int(budget["min_good_dates"])
    approx_feature_channels = 16 + (2 if s1_planned else 0)
    approx_temporal_planes = target_dates * 7
    working_planes = approx_feature_channels + approx_temporal_planes + 12
    estimated_tile_ram_mb = int(
        round((preset_tile_size_px * preset_tile_size_px * working_planes * 4 * 2.2) / (1024 * 1024))
    )
    estimated_ram_mb = max(512, min(14_000, estimated_tile_ram_mb))
    hard_block = False
    warnings: list[str] = []
    budget_ok = (
        estimated_tiles <= int(budget["max_tiles"])
        and complexity_score <= float(budget["max_complexity"])
        and expected_dates_ok
    )
    if req.resolution_m > int(budget["resolution_m"]):
        warnings.append("Профиль ориентирован на 10 м детализацию; текущее разрешение грубее рекомендованного.")
    if req.target_dates != int(budget["target_dates"]):
        warnings.append(
            f"Профиль '{preset}' рассчитан на {int(budget['target_dates'])} временных срезов; сейчас запрошено {int(req.target_dates)}."
        )
    if bool(use_sam) != bool(budget["use_sam"]):
        warnings.append(
            "SAM-параметр отличается от штатного режима профиля; время выполнения может отличаться от ожидаемого."
        )
    if region_band == "north" and preset != "quality":
        warnings.append("Северный регион: рекомендуется Standard/Quality с расширенным фенологическим окном и S1-enrichment.")
    if not expected_dates_ok:
        warnings.append(
            f"Для профиля '{preset}' запрошено меньше временных срезов, чем рекомендовано: минимум {int(budget['min_good_dates'])}, сейчас {target_dates}."
        )
    if requested_radius_km is not None and requested_radius_km < 1.0:
        warnings.append(
            f"Радиус {requested_radius_km:.2f} км очень мал — детекция работает, но полей внутри AOI может быть мало. Рекомендуется ≥ 1 км."
        )
    if requested_radius_km is not None and preset_max_radius_km > 0 and requested_radius_km > preset_max_radius_km:
        hard_block = True
        warnings.append(
            f"Для профиля '{preset}' максимальный радиус на этом хосте — {preset_max_radius_km} км, сейчас {requested_radius_km:.0f} км."
        )
    if estimated_tiles == 0:
        hard_block = True
        warnings.append(
            "AOI слишком мал: не удалось разбить на тайлы. Увеличьте радиус или используйте bbox/polygon большего размера."
        )
    if estimated_tiles > HARD_TILE_LIMIT:
        hard_block = True
        warnings.append(
            f"AOI слишком большой для одной задачи на текущем хосте: {estimated_tiles} тайлов при жёстком лимите {HARD_TILE_LIMIT}."
        )
    if complexity_score > HARD_COMPLEXITY_LIMIT:
        warnings.append(
            f"Оценка сложности {complexity_score:.1f} превышает рекомендуемый compute envelope worker-а ({HARD_COMPLEXITY_LIMIT:.0f}); запуск будет длинным."
        )
    if estimated_ram_mb >= HARD_RAM_LIMIT_MB:
        hard_block = True
        warnings.append(
            f"Оценка памяти {estimated_ram_mb} MB подходит к пределу текущего хоста ({HARD_RAM_LIMIT_MB} MB)."
        )

    recommended_preset = preset
    reason = None
    budget_reason = None
    if not budget_ok:
        recommended_preset = None
        for candidate in ("standard", "fast"):
            if candidate == preset:
                continue
            cb = DETECT_PRESET_CONFIGS[candidate]
            c_sam = 1.5 if cb["use_sam"] else 1.0
            c_tta = 1.35 if cb["tta_mode"] == "flip2" else 1.9 if cb["tta_mode"] == "rotate4" else 1.0
            c_ms = 1.25 if bool(cb["multi_scale"]) else 1.0
            c_s1_planned = cb["s1_policy"] == "on" or (cb["s1_policy"] == "north_or_opt_in" and region_band == "north")
            c_s1 = 1.2 if c_s1_planned else 1.0
            c_complexity = estimated_tiles * int(cb["target_dates"]) * c_sam * c_tta * c_ms * c_s1
            if (
                estimated_tiles <= int(cb["max_tiles"])
                and c_complexity <= float(cb["max_complexity"])
                and int(req.target_dates) >= int(cb["min_good_dates"])
            ):
                recommended_preset = candidate
                break
        if hard_block and estimated_ram_mb >= HARD_RAM_LIMIT_MB:
            budget_reason = f"Запуск выходит за memory-safe envelope: ~{estimated_ram_mb} MB на текущем хосте."
        elif hard_block and estimated_tiles > HARD_TILE_LIMIT:
            budget_reason = f"AOI слишком большой для текущего хоста: {estimated_tiles} тайлов при safety limit {HARD_TILE_LIMIT}."
        elif requested_radius_km is not None and preset_max_radius_km > 0 and requested_radius_km > preset_max_radius_km:
            budget_reason = (
                f"Для профиля '{preset}' максимальный радиус на этом хосте — {preset_max_radius_km} км, "
                f"сейчас {requested_radius_km:.0f} км."
            )
        elif not expected_dates_ok:
            budget_reason = f"Для профиля '{preset}' рекомендовано минимум {int(budget['min_good_dates'])} временных срезов, сейчас {target_dates}."
        else:
            budget_reason = (
                f"Профиль '{preset}' будет долгим для текущего AOI: "
                f"{estimated_tiles} тайлов, runtime-класс {estimate_runtime_class(complexity_score)}."
            )
        if hard_block:
            reason = f"{budget_reason} Уменьшите радиус или используйте профиль '{recommended_preset or 'fast'}'."
        else:
            reason = (
                f"{budget_reason} Запуск разрешён, но расчёт может занять заметно больше времени; при необходимости используйте профиль "
                f"'{recommended_preset or 'standard'}'."
            )

    return {
        "budget_ok": budget_ok,
        "hard_block": hard_block,
        "estimated_tiles": estimated_tiles,
        "estimated_ram_mb": estimated_ram_mb,
        "estimated_runtime_class": estimate_runtime_class(complexity_score),
        "pipeline_profile": str(budget.get("pipeline_profile") or preset),
        "preview_only": bool(budget.get("preview_only")),
        "output_mode": str(budget.get("output_mode") or "field_boundaries"),
        "operational_eligible": bool(budget.get("operational_eligible", True)),
        "max_radius_km": preset_max_radius_km or None,
        "recommended_radius_km": preset_recommended_radius_km or None,
        "enabled_stages": list(budget.get("enabled_stages") or []),
        "expected_dates_ok": expected_dates_ok,
        "regional_profile": regional_profile,
        "s1_planned": s1_planned,
        "tta_mode": str(budget["tta_mode"]),
        "budget_reason": budget_reason,
        "recommended_preset": recommended_preset,
        "reason": reason,
        "warnings": warnings,
        "preset": preset,
        "tile_size_px": preset_tile_size_px,
        "season_window": {"start": season_start, "end": season_end},
        "region_band": region_band,
        **describe_detect_launch(
            region_band=region_band,
            preset=preset,
            budget_ok=budget_ok,
            hard_block=hard_block,
            warnings=warnings,
        ),
    }


@router.post("/detect/preflight", response_model=DetectPreflightResponse)
async def detect_preflight(
    req: DetectRequest,
    use_sam: bool = Query(False, description="Enable SAM2 boundary refinement (A/B test)."),
    _ctx: RequestContext = Depends(require_permissions("fields:write")),
):
    try:
        payload = _build_detect_preflight(req, use_sam=bool(use_sam))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DetectPreflightResponse(**payload)


@router.post("/detect", response_model=DetectResponse, status_code=202)
@limiter.limit(_settings.RATE_LIMIT_DETECT)
async def detect_fields(
    request: Request,
    req: DetectRequest,
    use_sam: bool = Query(
        False,
        description="Enable SAM2 boundary refinement (A/B test). Default is baseline without SAM.",
    ),
    ctx: RequestContext = Depends(require_permissions("fields:write")),
    db: AsyncSession = Depends(get_db),
):
    DETECT_REQUESTS.inc()

    try:
        preflight = _build_detect_preflight(req, use_sam=bool(use_sam))
        if preflight.get("hard_block"):
            raise HTTPException(
                status_code=422,
                detail=preflight["reason"] or "Параметры детекта выходят за лимиты worker-а.",
            )
        aoi_geom = resolve_aoi(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    aoi_wkt = aoi_geom.wkt

    repo = FieldsRepository(db)
    params = req.model_dump(mode="json")
    params["use_sam"] = bool(use_sam)
    params.setdefault("config", {})
    params["config"]["preset"] = preflight["preset"]
    params["config"]["preflight"] = {
        "estimated_tiles": preflight["estimated_tiles"],
        "estimated_ram_mb": preflight.get("estimated_ram_mb"),
        "estimated_runtime_class": preflight["estimated_runtime_class"],
        "pipeline_profile": preflight.get("pipeline_profile"),
        "preview_only": preflight.get("preview_only"),
        "output_mode": preflight.get("output_mode"),
        "operational_eligible": preflight.get("operational_eligible"),
        "max_radius_km": preflight.get("max_radius_km"),
        "recommended_radius_km": preflight.get("recommended_radius_km"),
        "enabled_stages": preflight.get("enabled_stages") or [],
        "expected_dates_ok": preflight.get("expected_dates_ok"),
        "hard_block": preflight.get("hard_block"),
        "regional_profile": preflight.get("regional_profile"),
        "s1_planned": preflight.get("s1_planned"),
        "tta_mode": preflight.get("tta_mode"),
        "budget_reason": preflight.get("budget_reason"),
        "recommended_preset": preflight["recommended_preset"],
        "warnings": preflight["warnings"],
        "season_window": preflight.get("season_window"),
        "region_band": preflight.get("region_band"),
        "tile_size_px": preflight.get("tile_size_px"),
    }

    if not has_live_workers_for_queue("gpu", timeout=1.5):
        raise HTTPException(
            status_code=503,
            detail="Очередь детекта недоступна: worker для queue 'gpu' не отвечает. Перезапустите backend-api/celery-worker.",
        )

    run = await repo.create_run(
        aoi_wkt=aoi_wkt,
        time_start=req.time_range.start_date,
        time_end=req.time_range.end_date,
        params=params,
        organization_id=ctx.organization_id,
        created_by_user_id=ctx.user_id,
    )
    await db.commit()

    try:
        from tasks.autodetect import run_autodetect  # noqa: PLC0415 — lazy import
        task = run_autodetect.apply_async(
            args=[str(run.id)],
            kwargs={"use_sam": bool(use_sam)},
            queue="gpu",
            priority=9,
        )
        runtime = dict(params.get("runtime") or {})
        runtime["celery_task_id"] = str(task.id)
        runtime["queued_at"] = datetime.now(timezone.utc).isoformat()
        params["runtime"] = runtime
        run.params = params
        await db.commit()
    except Exception as exc:
        await repo.update_run_status(run.id, "failed", error_msg=f"Не удалось отправить задачу в очередь: {exc}")
        raise HTTPException(status_code=503, detail=f"Очередь задач недоступна: {exc}") from exc

    return DetectResponse(aoi_run_id=run.id, status="queued")
