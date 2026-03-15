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
from storage.db import Field, FieldSeason, ManagementEvent, get_db
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


def _resolve_aoi(req: DetectRequest):
    aoi = req.aoi
    if aoi.type == "point_radius":
        return point_radius_to_polygon(aoi.lat, aoi.lon, aoi.radius_km)
    if aoi.type == "bbox":
        if aoi.bbox is None:
            raise ValueError("bbox is required for aoi.type='bbox'")
        return bbox_to_polygon(aoi.bbox)
    if aoi.type == "polygon":
        if aoi.polygon is None:
            raise ValueError("polygon is required for aoi.type='polygon'")
        return polygon_coords_to_polygon(aoi.polygon)
    raise ValueError(f"Unknown AOI type: {aoi.type}")


def _infer_detect_preset(req: DetectRequest, *, use_sam: bool) -> str:
    raw = str((req.config or {}).get("preset") or "").strip().lower()
    if raw in _DETECT_PRESET_CONFIGS:
        return raw
    if use_sam or req.target_dates >= 9 or req.min_field_area_ha <= 0.11:
        return "quality"
    if req.target_dates <= 4 and req.min_field_area_ha >= 0.5:
        return "fast"
    return "standard"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.lstrip("#")
    if len(normalized) != 6:
        return (255, 255, 255)
    return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))


def _allowed_debug_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "debug_runs").resolve()


def _resolve_debug_artifact_path(path_value: str | None) -> Path:
    if not path_value:
        raise HTTPException(status_code=404, detail="debug artifact is not available for this tile")
    candidate = Path(path_value).resolve()
    debug_root = _allowed_debug_root()
    if debug_root not in candidate.parents:
        raise HTTPException(status_code=404, detail="debug artifact path is outside debug storage")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="debug artifact file does not exist")
    return candidate


def _normalize_debug_array(layer_name: str, raw: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
    array = np.asarray(raw)
    meta = {"min": None, "max": None}
    if array.ndim != 2:
        raise HTTPException(status_code=422, detail=f"debug layer '{layer_name}' is not 2D")
    if layer_name in {"boundary_prob", "owt_edge"}:
        work = np.asarray(array, dtype=np.float32)
        finite = work[np.isfinite(work)]
        if finite.size == 0:
            return np.zeros_like(work, dtype=np.float32), meta
        min_value = float(np.nanpercentile(finite, 5))
        max_value = float(np.nanpercentile(finite, 95))
        if max_value <= min_value:
            max_value = min_value + 1e-6
        normalized = np.clip((work - min_value) / (max_value - min_value), 0.0, 1.0)
        meta["min"] = round(min_value, 4)
        meta["max"] = round(max_value, 4)
        return normalized.astype(np.float32, copy=False), meta
    normalized = np.asarray(array > 0, dtype=np.float32)
    meta["min"] = 0.0
    meta["max"] = 1.0
    return normalized, meta


def _colorize_debug_array(layer_name: str, normalized: np.ndarray) -> np.ndarray:
    style = _DEBUG_LAYER_STYLE.get(layer_name) or {}
    if layer_name == "boundary_prob":
        red = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        green = np.clip((normalized ** 1.6) * 120.0, 0, 255).astype(np.uint8)
        blue = np.clip((1.0 - normalized) * 90.0, 0, 255).astype(np.uint8)
        alpha = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        return np.stack([red, green, blue, alpha], axis=-1)
    if layer_name == "owt_edge":
        cyan = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        white = np.clip((normalized ** 0.75) * 255.0, 0, 255).astype(np.uint8)
        alpha = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        return np.stack([white, cyan, cyan, alpha], axis=-1)
    red, green, blue = _hex_to_rgb(str(style.get("color") or "#ffffff"))
    alpha = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    out = np.zeros((*normalized.shape, 4), dtype=np.uint8)
    out[..., 0] = red
    out[..., 1] = green
    out[..., 2] = blue
    out[..., 3] = alpha
    return out


def _runtime_tile_maps(run) -> tuple[dict[int, str], dict[str, int]]:
    runtime = dict((run.params or {}).get("runtime") or {})
    tiles = [tile for tile in list(runtime.get("tiles") or []) if isinstance(tile, dict)]
    index_to_id: dict[int, str] = {}
    id_to_index: dict[str, int] = {}
    for index, tile in enumerate(tiles):
        tile_id = str(tile.get("tile_id") or f"tile-{index}")
        index_to_id[int(index)] = tile_id
        id_to_index[tile_id] = int(index)
    return index_to_id, id_to_index


async def _load_detection_candidates(
    db: AsyncSession,
    *,
    organization_id: UUID,
    run_id: UUID,
    limit: int,
    kept: bool | None = None,
    branch: str | None = None,
    tile_index: int | None = None,
):
    from storage.db import FieldDetectionCandidate, TileDiagnostic

    stmt = (
        select(FieldDetectionCandidate, TileDiagnostic)
        .outerjoin(TileDiagnostic, FieldDetectionCandidate.tile_diagnostic_id == TileDiagnostic.id)
        .where(FieldDetectionCandidate.organization_id == organization_id)
        .where(FieldDetectionCandidate.aoi_run_id == run_id)
        .order_by(
            FieldDetectionCandidate.kept.desc(),
            FieldDetectionCandidate.score.desc(),
            FieldDetectionCandidate.rank.asc().nullslast(),
            FieldDetectionCandidate.id.asc(),
        )
        .limit(max(1, min(int(limit), 500)))
    )
    if kept is not None:
        stmt = stmt.where(FieldDetectionCandidate.kept.is_(bool(kept)))
    if branch:
        stmt = stmt.where(FieldDetectionCandidate.branch == str(branch))
    if tile_index is not None:
        stmt = stmt.where(TileDiagnostic.tile_index == int(tile_index))
    result = await db.execute(stmt)
    return list(result.all())


def _detection_candidate_payload(candidate, diagnostic, *, tile_id: str | None = None) -> dict[str, Any]:
    geometry = None
    if getattr(candidate, "geom", None) is not None:
        try:
            geometry = dict(to_shape(candidate.geom).__geo_interface__)
        except Exception:
            geometry = None
    created_at = getattr(candidate, "created_at", None)
    return {
        "id": int(candidate.id),
        "tile_diagnostic_id": int(candidate.tile_diagnostic_id) if candidate.tile_diagnostic_id is not None else None,
        "tile_index": int(getattr(diagnostic, "tile_index", 0)) if diagnostic is not None else None,
        "tile_id": tile_id,
        "field_id": candidate.field_id,
        "branch": str(candidate.branch or "unknown"),
        "area_m2": float(candidate.area_m2) if candidate.area_m2 is not None else None,
        "score": float(candidate.score or 0.0),
        "rank": int(candidate.rank) if candidate.rank is not None else None,
        "kept": bool(candidate.kept),
        "reject_reason": str(candidate.reject_reason) if candidate.reject_reason else None,
        "features": dict(candidate.features or {}),
        "model_version": str(candidate.model_version) if candidate.model_version else None,
        "created_at": (
            (created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=timezone.utc)).isoformat()
            if created_at is not None
            else None
        ),
        "geometry": geometry,
    }


def _encode_rgba_png(image: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"PNG encoder unavailable: {exc}") from exc
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _runtime_debug_tiles(run) -> list[dict]:
    runtime = _runtime_with_stale_flag(run)
    tiles = runtime.get("tiles") if isinstance(runtime, dict) else None
    return list(tiles or [])


def _tile_debug_payload(tile: dict) -> dict[str, object]:
    available_layers: list[dict[str, object]] = []
    artifact_path = tile.get("debug_artifact")
    if artifact_path:
        try:
            debug_npz_path = _resolve_debug_artifact_path(str(artifact_path))
            with np.load(debug_npz_path, allow_pickle=False) as bundle:
                keys = set(bundle.files)
            for public_name, internal_name in _DEBUG_LAYER_KEY_MAP.items():
                if internal_name in keys:
                    style = _DEBUG_LAYER_STYLE.get(public_name) or {}
                    available_layers.append(
                        {
                            "id": public_name,
                            "label": style.get("label") or public_name,
                            "opacity_default": float(style.get("opacity_default") or 0.5),
                        }
                    )
        except HTTPException:
            pass
    return {
        "tile_id": tile.get("tile_id"),
        "bbox": tile.get("bbox"),
        "runtime_meta": tile,
        "available_layers": available_layers,
        "traditional_gpkg": tile.get("traditional_gpkg"),
        "final_gpkg": tile.get("final_gpkg"),
        "sam_raw_gpkg": tile.get("sam_raw_gpkg"),
        "sam_filtered_gpkg": tile.get("sam_filtered_gpkg"),
    }


def _estimate_runtime_class(complexity_score: float) -> str:
    if complexity_score <= 24:
        return "short"
    if complexity_score <= 60:
        return "medium"
    if complexity_score <= 120:
        return "long"
    return "extreme"


def _build_detect_preflight(req: DetectRequest, *, use_sam: bool) -> dict:
    preset = _infer_detect_preset(req, use_sam=use_sam)
    budget = _DETECT_PRESET_CONFIGS[preset]
    aoi_geom = _resolve_aoi(req)
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
    if estimated_tiles > _HARD_TILE_LIMIT:
        hard_block = True
        warnings.append(
            f"AOI слишком большой для одной задачи на текущем хосте: {estimated_tiles} тайлов при жёстком лимите {_HARD_TILE_LIMIT}."
        )
    if complexity_score > _HARD_COMPLEXITY_LIMIT:
        warnings.append(
            f"Оценка сложности {complexity_score:.1f} превышает рекомендуемый compute envelope worker-а ({_HARD_COMPLEXITY_LIMIT:.0f}); запуск будет длинным."
        )
    if estimated_ram_mb >= _HARD_RAM_LIMIT_MB:
        hard_block = True
        warnings.append(
            f"Оценка памяти {estimated_ram_mb} MB подходит к пределу текущего хоста ({_HARD_RAM_LIMIT_MB} MB)."
        )

    recommended_preset = preset
    reason = None
    budget_reason = None
    if not budget_ok:
        recommended_preset = None
        for candidate in ("standard", "fast"):
            if candidate == preset:
                continue
            candidate_budget = _DETECT_PRESET_CONFIGS[candidate]
            candidate_sam_penalty = 1.5 if candidate_budget["use_sam"] else 1.0
            candidate_tta_penalty = 1.35 if candidate_budget["tta_mode"] == "flip2" else 1.9 if candidate_budget["tta_mode"] == "rotate4" else 1.0
            candidate_multi_scale_penalty = 1.25 if bool(candidate_budget["multi_scale"]) else 1.0
            candidate_s1_planned = (
                candidate_budget["s1_policy"] == "on"
                or (candidate_budget["s1_policy"] == "north_or_opt_in" and region_band == "north")
            )
            candidate_s1_penalty = 1.2 if candidate_s1_planned else 1.0
            candidate_complexity = (
                estimated_tiles
                * int(candidate_budget["target_dates"])
                * candidate_sam_penalty
                * candidate_tta_penalty
                * candidate_multi_scale_penalty
                * candidate_s1_penalty
            )
            if (
                estimated_tiles <= int(candidate_budget["max_tiles"])
                and candidate_complexity <= float(candidate_budget["max_complexity"])
                and int(req.target_dates) >= int(candidate_budget["min_good_dates"])
            ):
                recommended_preset = candidate
                break
        if hard_block and estimated_ram_mb >= _HARD_RAM_LIMIT_MB:
            budget_reason = (
                f"Запуск выходит за memory-safe envelope: ~{estimated_ram_mb} MB на текущем хосте."
            )
        elif hard_block and estimated_tiles > _HARD_TILE_LIMIT:
            budget_reason = (
                f"AOI слишком большой для текущего хоста: {estimated_tiles} тайлов при safety limit {_HARD_TILE_LIMIT}."
            )
        elif requested_radius_km is not None and preset_max_radius_km > 0 and requested_radius_km > preset_max_radius_km:
            budget_reason = (
                f"Для профиля '{preset}' максимальный радиус на этом хосте — {preset_max_radius_km} км, "
                f"сейчас {requested_radius_km:.0f} км."
            )
        elif not expected_dates_ok:
            budget_reason = (
                f"Для профиля '{preset}' рекомендовано минимум {int(budget['min_good_dates'])} временных срезов, сейчас {target_dates}."
            )
        else:
            budget_reason = (
                f"Профиль '{preset}' будет долгим для текущего AOI: "
                f"{estimated_tiles} тайлов, runtime-класс {_estimate_runtime_class(complexity_score)}."
            )
        if hard_block:
            reason = (
                f"{budget_reason} "
                f"Уменьшите радиус или используйте профиль '{recommended_preset or 'fast'}'."
            )
        else:
            reason = (
                f"{budget_reason} "
                f"Запуск разрешён, но расчёт может занять заметно больше времени; при необходимости используйте профиль "
                f"'{recommended_preset or 'standard'}'."
            )

    return {
        "budget_ok": budget_ok,
        "hard_block": hard_block,
        "estimated_tiles": estimated_tiles,
        "estimated_ram_mb": estimated_ram_mb,
        "estimated_runtime_class": _estimate_runtime_class(complexity_score),
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


def _runtime_with_stale_flag(run) -> dict | None:
    """Attach stale_running to runtime payload when a worker heartbeat is stale."""
    runtime = dict((run.params or {}).get("runtime") or {})
    if not runtime:
        return None

    stale_running = False
    heartbeat_raw = runtime.get("last_heartbeat_ts")
    if run.status == "running" and isinstance(heartbeat_raw, str):
        try:
            heartbeat = datetime.fromisoformat(heartbeat_raw.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - heartbeat).total_seconds()
            stale_running = age_s > float(get_settings().WORKER_HEARTBEAT_STALE_S)
        except ValueError:
            stale_running = False
    runtime["stale_running"] = stale_running
    return runtime


def _effective_run_status(run, runtime: dict | None) -> str:
    if runtime and runtime.get("stale_running") and run.status == "running":
        return "stale"
    return str(run.status)


def _aggregate_runtime_mode(runtime: dict | None, key: str) -> str | None:
    runtime = dict(runtime or {})
    direct = runtime.get(key)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    values: dict[str, int] = {}
    for tile in list(runtime.get("tiles") or []):
        if not isinstance(tile, dict):
            continue
        value = tile.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        values[value.strip()] = int(values.get(value.strip()) or 0) + 1
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values.keys()))
    return "mixed"


def _candidate_summary(runtime: dict | None) -> dict[str, Any]:
    runtime = dict(runtime or {})
    branch_counts = dict(runtime.get("candidate_branch_counts") or {})
    reject_summary = dict(runtime.get("candidate_reject_summary") or {})
    candidates_total = int(runtime.get("candidates_total") or 0)
    candidates_kept = int(runtime.get("candidates_kept") or 0)
    if branch_counts or reject_summary or candidates_total or candidates_kept:
        return {
            "candidate_branch_counts": branch_counts,
            "candidate_reject_summary": reject_summary,
            "candidates_total": candidates_total,
            "candidates_kept": candidates_kept,
        }

    aggregated_branch_counts: dict[str, dict[str, int]] = {}
    aggregated_rejects: dict[str, int] = {}
    total = 0
    kept = 0
    for tile in list(runtime.get("tiles") or []):
        if not isinstance(tile, dict):
            continue
        total += int(tile.get("candidates_total") or 0)
        kept += int(tile.get("candidates_kept") or 0)
        for branch, counts in dict(tile.get("candidate_branch_counts") or {}).items():
            existing = dict(aggregated_branch_counts.get(branch) or {"total": 0, "kept": 0})
            existing["total"] = int(existing.get("total") or 0) + int(counts.get("total") or 0)
            existing["kept"] = int(existing.get("kept") or 0) + int(counts.get("kept") or 0)
            aggregated_branch_counts[str(branch)] = existing
        for reason, count in dict(tile.get("candidate_reject_summary") or {}).items():
            aggregated_rejects[str(reason)] = int(aggregated_rejects.get(str(reason)) or 0) + int(count or 0)
    return {
        "candidate_branch_counts": aggregated_branch_counts,
        "candidate_reject_summary": aggregated_rejects,
        "candidates_total": int(total),
        "candidates_kept": int(kept),
    }


def _stage_code(runtime: dict | None, status: str) -> str | None:
    stage = str((runtime or {}).get("progress_stage") or status or "").strip().lower()
    if not stage:
        return None
    return stage


def _stage_label(runtime: dict | None, status: str) -> str | None:
    stage_code = _stage_code(runtime, status)
    if not stage_code:
        return None
    return _STAGE_LABELS.get(stage_code, stage_code.replace("_", " "))


def _explicit_stage_detail_payload(
    detail: str | None,
    *,
    stage_code: str | None,
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
                **({"extra": extra} if extra else {}),
            },
        )

    return explicit_detail, None, {}


def _stage_detail_payload(runtime: dict | None, stage_code: str | None, status: str) -> tuple[str | None, str | None, dict[str, Any]]:
    runtime = dict(runtime or {})
    if not runtime and not stage_code:
        return None, None, {}

    explicit_detail = runtime.get("progress_detail")
    detail_text, detail_code, detail_params = _explicit_stage_detail_payload(explicit_detail, stage_code=stage_code)
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
    if progress_stage in {
        "candidate_postprocess",
        "model_inference",
        "segmentation",
        "boundary_refine",
        "sam_refine",
        "tile_finalize",
    } and tile_count > 0:
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


def _progress_pct(run, runtime: dict | None) -> float:
    runtime_value = (runtime or {}).get("progress_pct")
    if isinstance(runtime_value, (int, float)):
        return round(float(runtime_value), 2)
    return round(float(run.progress or 0), 2)


def _runtime_progress_metric(runtime: dict | None, key: str) -> float | None:
    runtime_value = (runtime or {}).get(key)
    if isinstance(runtime_value, (int, float)):
        return round(float(runtime_value), 2)
    return None


def _runtime_int_metric(runtime: dict | None, key: str) -> int | None:
    runtime_value = (runtime or {}).get(key)
    if isinstance(runtime_value, (int, float)):
        return max(0, int(round(float(runtime_value))))
    return None


def _estimated_remaining_s(run, runtime: dict | None, progress_pct: float, status: str) -> int | None:
    if status == "done":
        return 0
    if status not in {"running", "stale"} or progress_pct <= 0 or progress_pct >= 100 or run.created_at is None:
        return None
    updated_at = _updated_at_iso(run, runtime)
    if updated_at is None:
        return None
    try:
        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    started_dt = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    elapsed_s = max((updated_dt - started_dt).total_seconds(), 1.0)
    estimate = int(round(elapsed_s * (100 - progress_pct) / progress_pct))
    return max(0, estimate)


def _elapsed_s(run, runtime: dict | None) -> int | None:
    updated_at = _updated_at_iso(run, runtime)
    if updated_at is None or run.created_at is None:
        return None
    try:
        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    started_dt = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    return max(0, int(round((updated_dt - started_dt).total_seconds())))


def _updated_at_iso(run, runtime: dict | None) -> str | None:
    heartbeat = (runtime or {}).get("last_heartbeat_ts")
    if isinstance(heartbeat, str) and heartbeat.strip():
        return heartbeat
    if run.created_at is None:
        return None
    created_at = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    return created_at.isoformat()


def _run_response_payload(run) -> dict:
    params = dict(run.params or {})
    config = dict(params.get("config") or {})
    preflight = dict(config.get("preflight") or {})
    runtime = _runtime_with_stale_flag(run)
    status = _effective_run_status(run, runtime)
    progress = int(run.progress or 0)
    progress_pct = _progress_pct(run, runtime)
    stage_code = _stage_code(runtime, status)
    stage_label = _stage_label(runtime, status)
    stage_detail, stage_detail_code, stage_detail_params = _stage_detail_payload(runtime, stage_code, status)
    candidate_summary = _candidate_summary(runtime)
    return {
        "aoi_run_id": run.id,
        "status": status,
        "progress": progress,
        "progress_pct": progress_pct,
        "error_msg": run.error_msg,
        "stage_code": stage_code,
        "stage_label": stage_label,
        "stage_detail": stage_detail,
        "stage_detail_code": stage_detail_code,
        "stage_detail_params": stage_detail_params,
        "stage_progress_pct": _runtime_progress_metric(runtime, "stage_progress_pct"),
        "tile_progress_pct": _runtime_progress_metric(runtime, "tile_progress_pct"),
        "started_at": (
            (run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)).isoformat()
            if run.created_at is not None
            else None
        ),
        "updated_at": _updated_at_iso(run, runtime),
        "last_heartbeat_ts": (runtime or {}).get("last_heartbeat_ts"),
        "stale_running": bool((runtime or {}).get("stale_running")),
        "elapsed_s": _elapsed_s(run, runtime),
        "estimated_remaining_s": _estimated_remaining_s(run, runtime, progress_pct, status),
        "qc_mode": _aggregate_runtime_mode(runtime, "qc_mode"),
        "processing_profile": _aggregate_runtime_mode(runtime, "processing_profile"),
        "pipeline_profile": str((runtime or {}).get("pipeline_profile") or preflight.get("pipeline_profile") or "") or None,
        "preview_only": bool((runtime or {}).get("preview_only", preflight.get("preview_only", False))),
        "output_mode": str((runtime or {}).get("output_mode") or preflight.get("output_mode") or "") or None,
        "operational_eligible": bool((runtime or {}).get("operational_eligible", preflight.get("operational_eligible", True))),
        "max_radius_km": _runtime_int_metric(runtime, "max_radius_km")
        or _runtime_int_metric(preflight, "max_radius_km"),
        "recommended_radius_km": _runtime_int_metric(runtime, "recommended_radius_km")
        or _runtime_int_metric(preflight, "recommended_radius_km"),
        "enabled_stages": list((runtime or {}).get("enabled_stages") or preflight.get("enabled_stages") or []),
        "candidate_branch_counts": candidate_summary["candidate_branch_counts"],
        "candidate_reject_summary": candidate_summary["candidate_reject_summary"],
        "candidates_total": candidate_summary["candidates_total"],
        "candidates_kept": candidate_summary["candidates_kept"],
        "geometry_summary": _geometry_summary(runtime),
        "runtime": runtime,
    }


def _geometry_summary(runtime: dict | None) -> dict:
    runtime = dict(runtime or {})
    tiles = [tile for tile in list(runtime.get("tiles") or []) if isinstance(tile, dict)]

    def _aggregate_numeric(key: str, *, as_int: bool = False):
        raw = runtime.get(key)
        if isinstance(raw, (int, float)):
            value = float(raw)
            return int(round(value)) if as_int else round(value, 3)
        values: list[float] = []
        for tile in tiles:
            tile_value = tile.get(key)
            if isinstance(tile_value, (int, float)):
                values.append(float(tile_value))
        if not values:
            return None
        median_value = float(np.median(values))
        return int(round(median_value)) if as_int else round(median_value, 3)

    def _aggregate_reason(key: str) -> str | None:
        direct = runtime.get(key)
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        reasons: list[str] = []
        for tile in tiles:
            reason = tile.get(key)
            if isinstance(reason, str) and reason.strip() and reason.strip() not in reasons:
                reasons.append(reason.strip())
        if not reasons:
            return None
        return "; ".join(reasons[:3])

    direct_applied = runtime.get("watershed_applied")
    if isinstance(direct_applied, bool):
        watershed_applied = direct_applied
    else:
        applied_values = [bool(tile.get("watershed_applied")) for tile in tiles if "watershed_applied" in tile]
        watershed_applied = any(applied_values) if applied_values else None

    return {
        "head_count": int(GEOMETRY_FOUNDATION.get("head_count") or 3),
        "heads": list(GEOMETRY_FOUNDATION.get("heads") or []),
        "tta_standard": str(GEOMETRY_FOUNDATION.get("tta_standard") or "flip2"),
        "tta_quality": str(GEOMETRY_FOUNDATION.get("tta_quality") or "rotate4"),
        "retrain_description": str(GEOMETRY_FOUNDATION.get("retrain_description") or ""),
        "geometry_confidence": _aggregate_numeric("geometry_confidence"),
        "tta_consensus": _aggregate_numeric("tta_consensus"),
        "boundary_uncertainty": _aggregate_numeric("boundary_uncertainty"),
        "tta_extent_disagreement": _aggregate_numeric("tta_extent_disagreement"),
        "tta_boundary_disagreement": _aggregate_numeric("tta_boundary_disagreement"),
        "uncertainty_source": _aggregate_reason("uncertainty_source"),
        "watershed_applied": watershed_applied,
        "watershed_skipped_reason": _aggregate_reason("watershed_skipped_reason"),
        "watershed_rollback_reason": _aggregate_reason("watershed_rollback_reason"),
        "components_after_grow": _aggregate_numeric("components_after_grow", as_int=True),
        "components_after_gap_close": _aggregate_numeric("components_after_gap_close", as_int=True),
        "components_after_infill": _aggregate_numeric("components_after_infill", as_int=True),
        "components_after_merge": _aggregate_numeric("components_after_merge", as_int=True),
        "components_after_watershed": _aggregate_numeric("components_after_watershed", as_int=True),
        "split_score_p50": _aggregate_numeric("split_score_p50"),
        "split_score_p90": _aggregate_numeric("split_score_p90"),
        "tiles_summarized": len(tiles),
    }


def _run_summary_payload(run) -> dict:
    params = dict(run.params or {})
    config = dict(params.get("config") or {})
    runtime = dict(params.get("runtime") or {})
    candidate_summary = _candidate_summary(runtime)
    aoi = params.get("aoi") if isinstance(params.get("aoi"), dict) else None
    preset = str(config.get("preset") or runtime.get("preset") or "").strip().lower() or None
    if preset is None and {"aoi", "time_range"}.issubset(params):
        try:
            preset = _infer_detect_preset(DetectRequest(**params), use_sam=bool(params.get("use_sam")))
        except Exception:
            preset = None
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
        "qc_mode": _aggregate_runtime_mode(runtime, "qc_mode"),
        "processing_profile": _aggregate_runtime_mode(runtime, "processing_profile"),
        "candidates_total": candidate_summary["candidates_total"],
        "candidates_kept": candidate_summary["candidates_kept"],
    }


@router.post("/detect/preflight", response_model=DetectPreflightResponse)
async def detect_preflight(
    req: DetectRequest,
    use_sam: bool = Query(
        False,
        description="Enable SAM2 boundary refinement (A/B test).",
    ),
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
            raise HTTPException(status_code=422, detail=preflight["reason"] or "Параметры детекта выходят за лимиты worker-а.")
        aoi_geom = _resolve_aoi(req)
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
        from tasks.autodetect import run_autodetect
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


@router.get("/runs/{run_id}/debug/tiles")
async def list_run_debug_tiles(
    run_id: UUID,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    repo = FieldsRepository(db)
    run = await repo.get_run(run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    tiles = [_tile_debug_payload(tile) for tile in _runtime_debug_tiles(run) if tile.get("tile_id")]
    return {
        "run_id": str(run.id),
        "status": str(run.status),
        "tiles": tiles,
    }


@router.get("/runs/{run_id}/candidates", response_model=DetectionCandidatesResponse)
async def list_run_detection_candidates(
    run_id: UUID,
    limit: int = Query(200, ge=1, le=500),
    kept: bool | None = Query(None),
    branch: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> DetectionCandidatesResponse:
    repo = FieldsRepository(db)
    run = await repo.get_run(run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    index_to_id, _ = _runtime_tile_maps(run)
    rows = await _load_detection_candidates(
        db,
        organization_id=ctx.organization_id,
        run_id=run_id,
        limit=limit,
        kept=kept,
        branch=branch,
    )
    return DetectionCandidatesResponse(
        run_id=run.id,
        total=len(rows),
        candidates=[
            DetectionCandidateInfo(
                **_detection_candidate_payload(
                    candidate,
                    diagnostic,
                    tile_id=index_to_id.get(int(getattr(diagnostic, "tile_index", -1)))
                    if diagnostic is not None
                    else None,
                )
            )
            for candidate, diagnostic in rows
        ],
    )


@router.get("/runs/{run_id}/debug/tiles/{tile_id}")
async def get_run_debug_tile(
    run_id: UUID,
    tile_id: str,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    repo = FieldsRepository(db)
    run = await repo.get_run(run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    tile = next((item for item in _runtime_debug_tiles(run) if str(item.get("tile_id")) == tile_id), None)
    if tile is None:
        raise HTTPException(status_code=404, detail="debug tile not found")
    return {
        "run_id": str(run.id),
        **_tile_debug_payload(tile),
    }


@router.get("/runs/{run_id}/debug/tiles/{tile_id}/candidates", response_model=DetectionCandidatesResponse)
async def list_run_debug_tile_candidates(
    run_id: UUID,
    tile_id: str,
    limit: int = Query(200, ge=1, le=500),
    kept: bool | None = Query(None),
    branch: str | None = Query(None),
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> DetectionCandidatesResponse:
    repo = FieldsRepository(db)
    run = await repo.get_run(run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    index_to_id, id_to_index = _runtime_tile_maps(run)
    tile_index = id_to_index.get(tile_id)
    if tile_index is None:
        raise HTTPException(status_code=404, detail="debug tile not found")
    rows = await _load_detection_candidates(
        db,
        organization_id=ctx.organization_id,
        run_id=run_id,
        limit=limit,
        kept=kept,
        branch=branch,
        tile_index=tile_index,
    )
    return DetectionCandidatesResponse(
        run_id=run.id,
        total=len(rows),
        candidates=[
            DetectionCandidateInfo(
                **_detection_candidate_payload(
                    candidate,
                    diagnostic,
                    tile_id=index_to_id.get(tile_index),
                )
            )
            for candidate, diagnostic in rows
        ],
    )


@router.get("/runs/{run_id}/debug/tiles/{tile_id}/layers/{layer_name}")
async def get_run_debug_tile_layer(
    run_id: UUID,
    tile_id: str,
    layer_name: str,
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    internal_name = _DEBUG_LAYER_KEY_MAP.get(layer_name)
    if internal_name is None:
        raise HTTPException(status_code=404, detail="debug layer is not supported")

    repo = FieldsRepository(db)
    run = await repo.get_run(run_id, organization_id=ctx.organization_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    tile = next((item for item in _runtime_debug_tiles(run) if str(item.get("tile_id")) == tile_id), None)
    if tile is None:
        raise HTTPException(status_code=404, detail="debug tile not found")
    if not tile.get("bbox"):
        raise HTTPException(status_code=404, detail="debug tile bounds are not available")

    debug_npz_path = _resolve_debug_artifact_path(str(tile.get("debug_artifact")))
    with np.load(debug_npz_path, allow_pickle=False) as bundle:
        if internal_name not in bundle.files:
            raise HTTPException(status_code=404, detail=f"debug layer '{layer_name}' is not available for this tile")
        normalized, range_meta = _normalize_debug_array(layer_name, bundle[internal_name])
    rgba = _colorize_debug_array(layer_name, normalized)
    style = _DEBUG_LAYER_STYLE.get(layer_name) or {}
    return {
        "run_id": str(run.id),
        "tile_id": tile_id,
        "layer_name": layer_name,
        "type": "image_static",
        "bounds": tile.get("bbox"),
        "width": int(normalized.shape[1]),
        "height": int(normalized.shape[0]),
        "opacity_default": float(style.get("opacity_default") or 0.5),
        "legend": {
            "label": style.get("label") or layer_name,
            "min": range_meta.get("min"),
            "max": range_meta.get("max"),
        },
        "image_base64": _encode_rgba_png(rgba),
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
    ctx: RequestContext = Depends(require_permissions("fields:read")),
    db: AsyncSession = Depends(get_db),
) -> FieldDashboardResponse:
    service = FieldAnalyticsService(db)
    try:
        payload = await service.get_field_dashboard(
            field_id, organization_id=ctx.organization_id, date_from=date_from, date_to=date_to,
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
    return RunStatus(**_run_response_payload(run))


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
    if payload["status"] in {"queued", "running", "failed", "stale", "cancelled"}:
        return RunResult(**payload)

    geojson = await repo.get_fields_geojson(aoi_run_id, organization_id=ctx.organization_id)
    payload["geojson"] = geojson
    if payload["status"] == "done":
        payload["progress"] = 100
        payload["progress_pct"] = 100.0
        payload["estimated_remaining_s"] = 0
    return RunResult(**payload)
