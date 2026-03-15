"""Celery task for autodetect pipeline."""
import asyncio
import gc
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any
import uuid

import numpy as np

from core.celery_app import celery
from core.config import get_adaptive_season_window, get_px_area_m2, get_settings
from core.logging import configure_logging, get_logger
from core.metrics import (
    ACTIVE_RUNS,
    DETECT_DURATION,
    GPU_MEMORY_USAGE,
    S1_FETCH_TIME,
    SAM2_INFERENCE_TIME,
    STEP_DURATION,
    TILES_PROCESSED,
    UNET_INFERENCE_TIME,
)
from utils.nan_safe import nanmax_safe, nanmean_safe, nanmedian_safe

logger = get_logger(__name__)

_TILE_PROGRESS_START = 10
_TILE_PROGRESS_END = 72
_POST_PROGRESS_START = 72
_POST_PROGRESS_END = 97


def _safe_valid_fraction(valid_count: np.ndarray, valid_scene_total: int | float | None) -> np.ndarray:
    denominator = max(1.0, float(valid_scene_total or 0))
    return np.clip(valid_count.astype(np.float32) / denominator, 0.0, 1.0)


def _format_utc_day_boundary(value: date | datetime | str, *, end_of_day: bool) -> str:
    """Normalize date-like input to Process API UTC day boundary timestamp."""
    if isinstance(value, datetime):
        day = value.date()
    elif isinstance(value, date):
        day = value
    else:
        # Handles values like "2025-05-01" and "2025-05-01 00:00:00".
        day_token = str(value).strip().split("T", 1)[0].split(" ", 1)[0]
        day = date.fromisoformat(day_token)

    suffix = "23:59:59Z" if end_of_day else "00:00:00Z"
    return f"{day.isoformat()}T{suffix}"


def _to_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    day_token = str(value).strip().split("T", 1)[0].split(" ", 1)[0]
    return date.fromisoformat(day_token)


def _build_time_windows(
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    target_slices: int,
    season_start_mmdd: str | None = None,
    season_end_mmdd: str | None = None,
) -> list[tuple[str, str]]:
    """Split [start, end] into non-overlapping windows in UTC day boundaries."""
    day_start = _to_date(start)
    day_end = _to_date(end)
    if day_end < day_start:
        raise ValueError("time_end must be >= time_start")

    ranges = [(day_start, day_end)]
    if season_start_mmdd and season_end_mmdd:
        ranges = []
        season_start_month, season_start_day = map(int, season_start_mmdd.split("-"))
        season_end_month, season_end_day = map(int, season_end_mmdd.split("-"))
        crosses_year = (season_end_month, season_end_day) < (season_start_month, season_start_day)

        for year in range(day_start.year, day_end.year + 1):
            season_start = date(year, season_start_month, season_start_day)
            if crosses_year:
                season_end = date(year + 1, season_end_month, season_end_day)
            else:
                season_end = date(year, season_end_month, season_end_day)

            clipped_start = max(day_start, season_start)
            clipped_end = min(day_end, season_end)
            if clipped_start <= clipped_end:
                ranges.append((clipped_start, clipped_end))

    included_days: list[date] = []
    for range_start, range_end in ranges:
        span_days = (range_end - range_start).days + 1
        included_days.extend(range_start + timedelta(days=offset) for offset in range(span_days))

    if not included_days:
        return []

    n_slices = max(1, min(target_slices, len(included_days)))
    windows: list[tuple[str, str]] = []

    for i in range(n_slices):
        rel_start = (i * len(included_days)) // n_slices
        rel_end = ((i + 1) * len(included_days)) // n_slices - 1
        chunk = included_days[rel_start : max(rel_start, rel_end) + 1]
        if not chunk:
            continue

        chunk_start = chunk[0]
        chunk_prev = chunk[0]
        for current_day in chunk[1:]:
            if (current_day - chunk_prev).days > 1:
                windows.append(
                    (
                        _format_utc_day_boundary(chunk_start, end_of_day=False),
                        _format_utc_day_boundary(chunk_prev, end_of_day=True),
                    )
                )
                chunk_start = current_day
            chunk_prev = current_day

        windows.append(
            (
                _format_utc_day_boundary(chunk_start, end_of_day=False),
                _format_utc_day_boundary(chunk_prev, end_of_day=True),
            )
        )

    return windows


def _ensure_time_stack(arr: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize raster array to (T, H, W)."""
    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"{name} must be 2D or 3D, got shape={arr.shape}")


def _sanitize_json_floats(value: Any) -> Any:
    """Recursively replace NaN/Inf values so JSONB persistence cannot fail."""
    if isinstance(value, dict):
        return {str(k): _sanitize_json_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json_floats(v) for v in value]
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, np.generic):
        return _sanitize_json_floats(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _persist_runtime_meta(run, runtime_meta: dict) -> None:
    runtime_meta["last_heartbeat_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    params = dict(run.params or {})
    params["runtime"] = _sanitize_json_floats(runtime_meta)
    run.params = params


_last_commit_time: float = 0.0
_MIN_COMMIT_INTERVAL_S: float = 1.5  # Commit at most every 1.5s for progress


def _set_progress_stage(
    run,
    runtime_meta: dict,
    session,
    stage: str,
    *,
    progress: int | None = None,
    progress_pct: float | None = None,
    detail: str | None = None,
    stage_progress_pct: float | None = None,
    tile_progress_pct: float | None = None,
    force_commit: bool = False,
) -> None:
    global _last_commit_time
    runtime_meta["progress_stage"] = str(stage)
    if detail:
        runtime_meta["progress_detail"] = str(detail)
    else:
        runtime_meta.pop("progress_detail", None)
    _persist_runtime_meta(run, runtime_meta)
    if progress is not None:
        run.progress = int(max(int(run.progress or 0), int(progress)))
    if progress_pct is None and progress is not None:
        progress_pct = float(progress)
    if progress_pct is not None:
        current_pct = float(runtime_meta.get("progress_pct") or 0.0)
        runtime_meta["progress_pct"] = round(max(current_pct, float(progress_pct)), 2)
        _persist_runtime_meta(run, runtime_meta)
    if stage_progress_pct is not None:
        runtime_meta["stage_progress_pct"] = round(
            min(max(float(stage_progress_pct), 0.0), 100.0),
            2,
        )
        _persist_runtime_meta(run, runtime_meta)
    if tile_progress_pct is not None:
        runtime_meta["tile_progress_pct"] = round(
            min(max(float(tile_progress_pct), 0.0), 100.0),
            2,
        )
        _persist_runtime_meta(run, runtime_meta)

    now = time.time()
    if force_commit or (now - _last_commit_time) >= _MIN_COMMIT_INTERVAL_S:
        session.commit()
        _last_commit_time = now


def _interpolate_progress(start: int, end: int, fraction: float) -> int:
    bounded_fraction = min(max(float(fraction), 0.0), 1.0)
    return int(round(start + (end - start) * bounded_fraction))


def _interpolate_progress_pct(start: int, end: int, fraction: float) -> float:
    bounded_fraction = min(max(float(fraction), 0.0), 1.0)
    return round(start + (end - start) * bounded_fraction, 2)


def _set_tile_progress(
    run,
    runtime_meta: dict,
    session,
    stage: str,
    *,
    tile_index: int,
    tile_count: int,
    phase_fraction: float,
    detail: str | None = None,
    stage_progress_pct: float | None = None,
    force_commit: bool = False,
) -> None:
    safe_tile_count = max(int(tile_count), 1)
    runtime_meta["tile_count"] = safe_tile_count
    runtime_meta["current_tile_index"] = min(max(int(tile_index) + 1, 1), safe_tile_count)
    overall_fraction = (float(tile_index) + min(max(float(phase_fraction), 0.0), 1.0)) / safe_tile_count
    progress = _interpolate_progress(_TILE_PROGRESS_START, _TILE_PROGRESS_END, overall_fraction)
    progress_pct = _interpolate_progress_pct(_TILE_PROGRESS_START, _TILE_PROGRESS_END, overall_fraction)
    _set_progress_stage(
        run,
        runtime_meta,
        session,
        stage,
        progress=progress,
        progress_pct=progress_pct,
        detail=detail,
        stage_progress_pct=stage_progress_pct,
        tile_progress_pct=min(max(float(phase_fraction), 0.0), 1.0) * 100.0,
        force_commit=force_commit,
    )


def _set_post_progress(
    run,
    runtime_meta: dict,
    session,
    stage: str,
    *,
    phase_fraction: float,
    detail: str | None = None,
    stage_progress_pct: float | None = None,
    force_commit: bool = False,
) -> None:
    progress = _interpolate_progress(_POST_PROGRESS_START, _POST_PROGRESS_END, phase_fraction)
    progress_pct = _interpolate_progress_pct(_POST_PROGRESS_START, _POST_PROGRESS_END, phase_fraction)
    _set_progress_stage(
        run,
        runtime_meta,
        session,
        stage,
        progress=progress,
        progress_pct=progress_pct,
        detail=detail,
        stage_progress_pct=stage_progress_pct if stage_progress_pct is not None else float(phase_fraction) * 100.0,
        tile_progress_pct=100.0,
        force_commit=force_commit,
    )


def _make_tile_stage_progress_callback(
    *,
    run,
    runtime_meta: dict,
    session,
    stage: str,
    tile_index: int,
    tile_count: int,
    phase_map: dict[str, tuple[float, float, str]],
    default_phase: tuple[float, float, str],
    emit_interval_s: float = 1.0,
):
    last_emit = 0.0

    def _callback(token: str, completed: int, total: int) -> None:
        nonlocal last_emit
        phase_start, phase_span, detail_prefix = phase_map.get(str(token), default_phase)
        safe_total = max(int(total), 1)
        safe_completed = min(max(int(completed), 0), safe_total)
        now = time.time()
        if safe_completed < safe_total and (now - last_emit) < emit_interval_s:
            return
        last_emit = now
        ratio = safe_completed / safe_total
        phase_fraction = phase_start + (phase_span * ratio)
        detail = f"{detail_prefix} {safe_completed}/{safe_total}".strip()
        _set_tile_progress(
            run,
            runtime_meta,
            session,
            stage,
            tile_index=tile_index,
            tile_count=tile_count,
            phase_fraction=phase_fraction,
            detail=detail,
            stage_progress_pct=ratio * 100.0,
        )

    return _callback


def _make_post_stage_progress_callback(
    *,
    run,
    runtime_meta: dict,
    session,
    stage: str,
    phase_map: dict[str, tuple[float, float, str]],
    default_phase: tuple[float, float, str],
    emit_interval_s: float = 1.0,
):
    last_emit = 0.0

    def _callback(token: str, completed: int, total: int) -> None:
        nonlocal last_emit
        phase_start, phase_span, detail_prefix = phase_map.get(str(token), default_phase)
        safe_total = max(int(total), 1)
        safe_completed = min(max(int(completed), 0), safe_total)
        now = time.time()
        if safe_completed < safe_total and (now - last_emit) < emit_interval_s:
            return
        last_emit = now
        ratio = safe_completed / safe_total
        phase_fraction = phase_start + (phase_span * ratio)
        detail = f"{detail_prefix} {safe_completed}/{safe_total}".strip()
        _set_post_progress(
            run,
            runtime_meta,
            session,
            stage,
            phase_fraction=phase_fraction,
            detail=detail,
            stage_progress_pct=ratio * 100.0,
        )

    return _callback


def _run_tile_gc(settings) -> None:
    """Best-effort per-tile garbage collection to cap worker RSS growth."""
    if not bool(getattr(settings, "TILE_MEMORY_CLEANUP_ENABLED", True)):
        return
    if bool(getattr(settings, "TILE_MEMORY_GC_EVERY_TILE", True)):
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.read().strip().split()
        if len(fields) >= 2:
            rss_pages = int(fields[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return float(rss_pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    if rss > 10_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _candidate_limit_for_preset(settings, detect_preset: str) -> int:
    attr = {
        "fast": "MAX_CANDIDATES_PER_TILE_FAST",
        "standard": "MAX_CANDIDATES_PER_TILE_STANDARD",
        "quality": "MAX_CANDIDATES_PER_TILE_QUALITY",
    }.get(str(detect_preset).strip().lower(), "MAX_CANDIDATES_PER_TILE_STANDARD")
    return max(32, int(getattr(settings, attr, getattr(settings, "MAX_CANDIDATES_PER_TILE", 1800))))


def _memory_limit_for_preset_mb(settings, detect_preset: str) -> float:
    attr = {
        "fast": "FAST_TILE_MEMORY_SOFT_LIMIT_MB",
        "standard": "STANDARD_TILE_MEMORY_SOFT_LIMIT_MB",
        "quality": "QUALITY_TILE_MEMORY_SOFT_LIMIT_MB",
    }.get(str(detect_preset).strip().lower(), "STANDARD_TILE_MEMORY_SOFT_LIMIT_MB")
    return float(getattr(settings, attr, 2600))


def _resolve_detect_pipeline_profile(settings, detect_preset: str) -> dict[str, Any]:
    preset = str(detect_preset or "standard").strip().lower()
    candidate_limit = _candidate_limit_for_preset(settings, preset)
    if preset == "fast":
        return {
            "name": "fast_preview",
            "preview_only": True,
            "output_mode": "preview_agri_contours",
            "operational_eligible": False,
            "field_source": "autodetect_preview",
            "enabled_stages": [
                "fetch",
                "candidate_postprocess",
                "segmentation",
                "boundary_refine",
                "tile_finalize",
                "merge",
                "db_insert",
            ],
            "enable_model_inference": False,
            "enable_candidate_ranker": False,
            "enable_selective_split": False,
            "enable_snake_refine": False,
            "enable_object_classifier": False,
            "enable_post_merge_smooth": False,
            "enable_active_learning": False,
            "enable_sam": False,
            "memory_fallback": "simplify",
            "max_candidates_per_tile": candidate_limit,
            "post_merge_max_components": max(
                64,
                int(getattr(settings, "POST_MERGE_MAX_COMPONENTS_FAST", 512)),
            ),
        }
    if preset == "quality":
        base_ndvi = float(getattr(settings, "OBIA_MIN_NDVI_DELTA", 0.12))
        base_shape = float(getattr(settings, "OBIA_MAX_SHAPE_INDEX", 2.4))
        base_score = float(getattr(settings, "CANDIDATE_MIN_SCORE", 0.18))
        return {
            "name": "quality_full",
            "preview_only": False,
            "output_mode": "field_boundaries_hifi",
            "operational_eligible": True,
            "field_source": "autodetect",
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
            "enable_model_inference": True,
            "enable_candidate_ranker": bool(getattr(settings, "ENABLE_CANDIDATE_RANKER", True)),
            "enable_selective_split": True,
            "enable_snake_refine": True,
            "enable_object_classifier": True,
            "enable_post_merge_smooth": True,
            "enable_active_learning": True,
            "enable_sam": True,
            "memory_fallback": "simplify",
            "max_candidates_per_tile": candidate_limit,
            "post_merge_max_components": max(
                128,
                int(getattr(settings, "POST_MERGE_MAX_COMPONENTS", 2000)),
            ),
            # Quality mode: extra-permissive OBIA and candidate filters for maximum recall
            "config_overrides": {
                "OBIA_MIN_NDVI_DELTA": round(max(0.06, base_ndvi * 0.75), 4),
                "OBIA_MAX_SHAPE_INDEX": round(base_shape * 1.15, 2),
                "CANDIDATE_MIN_SCORE": round(max(0.10, base_score - 0.04), 4),
                "OBIA_RELAX_MIN_BOUNDARY_CONF": 0.55,
                "POST_MERGE_NDVI_DIFF_MAX": round(float(getattr(settings, "POST_MERGE_NDVI_DIFF_MAX", 0.12)) + 0.04, 4),
            },
        }
    return {
        "name": "standard_balanced",
        "preview_only": False,
        "output_mode": "field_boundaries",
        "operational_eligible": True,
        "field_source": "autodetect",
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
        "enable_model_inference": True,
        "enable_candidate_ranker": bool(getattr(settings, "ENABLE_CANDIDATE_RANKER", True)),
        "enable_selective_split": True,
        "enable_snake_refine": True,
        "enable_object_classifier": True,
        "enable_post_merge_smooth": True,
        "enable_active_learning": True,
        "enable_sam": False,
        "memory_fallback": "simplify",
        "max_candidates_per_tile": candidate_limit,
        "post_merge_max_components": max(
            128,
            int(getattr(settings, "POST_MERGE_MAX_COMPONENTS", 2000)),
        ),
    }


def _should_downgrade_for_memory(
    settings,
    detect_preset: str,
    *,
    stage: str,
) -> tuple[bool, float, float]:
    rss_mb = _current_rss_mb()
    limit_mb = _memory_limit_for_preset_mb(settings, detect_preset)
    if rss_mb >= limit_mb:
        logger.warning(
            "tile_memory_guard_triggered",
            preset=str(detect_preset),
            stage=str(stage),
            rss_mb=round(float(rss_mb), 2),
            limit_mb=round(float(limit_mb), 2),
        )
        return True, rss_mb, limit_mb
    return False, rss_mb, limit_mb


def _limited_component_labels(
    mask: np.ndarray,
    *,
    max_components: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    from scipy.ndimage import label as nd_label

    binary_mask = np.asarray(mask, dtype=bool)
    if not np.any(binary_mask):
        return np.zeros(binary_mask.shape, dtype=np.int32), {
            "component_count": 0,
            "kept_components": 0,
            "dropped_components": 0,
        }

    labeled, component_count = nd_label(binary_mask)
    component_count = int(component_count)
    if component_count <= 0:
        return np.zeros(binary_mask.shape, dtype=np.int32), {
            "component_count": 0,
            "kept_components": 0,
            "dropped_components": 0,
        }

    kept_components = component_count
    if max_components is not None and component_count > int(max_components):
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        selected_labels = np.argsort(sizes)[::-1][: int(max_components)]
        selected_labels = selected_labels[selected_labels > 0]
        keep_mask = np.isin(labeled, selected_labels)
        labeled, kept_components = nd_label(keep_mask)
        kept_components = int(kept_components)

    return labeled.astype(np.int32, copy=False), {
        "component_count": int(component_count),
        "kept_components": int(kept_components),
        "dropped_components": int(max(component_count - kept_components, 0)),
    }


def _simple_candidate_select(mask: np.ndarray, *, max_components: int) -> tuple[np.ndarray, dict[str, int]]:
    labeled, meta = _limited_component_labels(mask, max_components=max_components)
    return labeled, meta


def _empty_fields_gdf():
    import geopandas as gpd

    return gpd.GeoDataFrame(
        columns=["label", "geometry", "area_m2", "perimeter_m"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _enrich_runtime_meta(
    runtime_meta: dict,
    *,
    latency_breakdown: dict[str, float],
    t_start: float,
) -> dict:
    tiles = list(runtime_meta.get("tiles", []))
    runtime_meta["ml_primary_used"] = bool(
        any(bool(tile.get("ml_primary_used")) for tile in tiles)
    )
    fallback_values = [
        float(tile.get("fallback_rate_tile"))
        for tile in tiles
        if isinstance(tile.get("fallback_rate_tile"), (int, float))
    ]
    runtime_meta["fallback_rate_tile"] = (
        round(float(np.mean(fallback_values)), 4) if fallback_values else 0.0
    )
    qg_values = [
        bool(tile.get("quality_gate_failed"))
        for tile in tiles
        if isinstance(tile.get("quality_gate_failed"), (bool, int, float))
    ]
    runtime_meta["quality_gate_failed"] = bool(any(qg_values)) if qg_values else False

    profiles = [
        str(tile.get("geometry_refine_profile")).strip()
        for tile in tiles
        if isinstance(tile.get("geometry_refine_profile"), str) and str(tile.get("geometry_refine_profile")).strip()
    ]
    if profiles:
        uniq_profiles = sorted(set(profiles))
        runtime_meta["geometry_refine_profile"] = (
            uniq_profiles[0] if len(uniq_profiles) == 1 else "mixed"
        )
    else:
        runtime_meta["geometry_refine_profile"] = str(
            runtime_meta.get("geometry_refine_profile") or "unknown"
        )

    extent_thresholds = [
        float(tile.get("ml_extent_threshold"))
        for tile in tiles
        if isinstance(tile.get("ml_extent_threshold"), (int, float))
    ]
    runtime_meta["ml_extent_threshold"] = (
        round(float(np.mean(extent_thresholds)), 4) if extent_thresholds else None
    )

    shrink_values = [
        float(tile.get("contour_shrink_ratio"))
        for tile in tiles
        if isinstance(tile.get("contour_shrink_ratio"), (int, float))
    ]
    if shrink_values:
        runtime_meta["contour_shrink_ratio"] = round(float(np.mean(shrink_values)), 4)

    centroid_values = [
        float(tile.get("centroid_shift_m"))
        for tile in tiles
        if isinstance(tile.get("centroid_shift_m"), (int, float))
    ]
    if centroid_values:
        runtime_meta["centroid_shift_m"] = round(float(np.mean(centroid_values)), 4)

    selected_confidences = [
        float(tile.get("selected_date_confidence_mean"))
        for tile in tiles
        if isinstance(tile.get("selected_date_confidence_mean"), (int, float))
    ]
    if selected_confidences:
        runtime_meta["selected_date_confidence_mean"] = round(
            float(np.mean(selected_confidences)),
            4,
        )
    region_actions = [
        str(action)
        for tile in tiles
        for action in (tile.get("region_profile_actions") or [])
        if str(action).strip()
    ]
    runtime_meta["region_profile_actions"] = sorted(set(region_actions))
    scene_signatures = [
        str(tile.get("selected_scene_signature")).strip()
        for tile in tiles
        if isinstance(tile.get("selected_scene_signature"), str) and str(tile.get("selected_scene_signature")).strip()
    ]
    if scene_signatures:
        joined = "|".join(sorted(scene_signatures))
        runtime_meta["selected_scene_signature"] = hashlib.sha1(
            joined.encode("utf-8")
        ).hexdigest()[:16]
    else:
        runtime_meta["selected_scene_signature"] = ""
    valid_scene_counts = [
        int(tile.get("n_valid_scenes"))
        for tile in tiles
        if isinstance(tile.get("n_valid_scenes"), (int, float))
    ]
    runtime_meta["n_valid_scenes"] = (
        int(round(float(np.mean(valid_scene_counts)))) if valid_scene_counts else 0
    )
    edge_signal_values = [
        float(tile.get("edge_signal_p90"))
        for tile in tiles
        if isinstance(tile.get("edge_signal_p90"), (int, float))
    ]
    runtime_meta["edge_signal_p90"] = (
        round(float(np.mean(edge_signal_values)), 4) if edge_signal_values else 0.0
    )
    ml_quality_values = [
        float(tile.get("ml_quality_score"))
        for tile in tiles
        if isinstance(tile.get("ml_quality_score"), (int, float))
    ]
    runtime_meta["ml_quality_score"] = (
        round(float(np.mean(ml_quality_values)), 4) if ml_quality_values else 0.0
    )
    tta_consensus_values = [
        float(tile.get("tta_consensus"))
        for tile in tiles
        if isinstance(tile.get("tta_consensus"), (int, float))
    ]
    runtime_meta["tta_consensus"] = (
        round(float(np.mean(tta_consensus_values)), 4) if tta_consensus_values else None
    )
    boundary_uncertainty_values = [
        float(tile.get("boundary_uncertainty"))
        for tile in tiles
        if isinstance(tile.get("boundary_uncertainty"), (int, float))
    ]
    runtime_meta["boundary_uncertainty"] = (
        round(float(np.mean(boundary_uncertainty_values)), 4)
        if boundary_uncertainty_values
        else None
    )
    geometry_confidence_values = [
        float(tile.get("geometry_confidence"))
        for tile in tiles
        if isinstance(tile.get("geometry_confidence"), (int, float))
    ]
    runtime_meta["geometry_confidence"] = (
        round(float(np.mean(geometry_confidence_values)), 4)
        if geometry_confidence_values
        else None
    )
    extent_disagreement_values = [
        float(tile.get("tta_extent_disagreement"))
        for tile in tiles
        if isinstance(tile.get("tta_extent_disagreement"), (int, float))
    ]
    runtime_meta["tta_extent_disagreement"] = (
        round(float(np.mean(extent_disagreement_values)), 4)
        if extent_disagreement_values
        else None
    )
    boundary_disagreement_values = [
        float(tile.get("tta_boundary_disagreement"))
        for tile in tiles
        if isinstance(tile.get("tta_boundary_disagreement"), (int, float))
    ]
    runtime_meta["tta_boundary_disagreement"] = (
        round(float(np.mean(boundary_disagreement_values)), 4)
        if boundary_disagreement_values
        else None
    )
    uncertainty_sources = [
        str(tile.get("uncertainty_source")).strip()
        for tile in tiles
        if isinstance(tile.get("uncertainty_source"), str) and str(tile.get("uncertainty_source")).strip()
    ]
    if uncertainty_sources:
        uniq_uncertainty_sources = sorted(set(uncertainty_sources))
        runtime_meta["uncertainty_source"] = (
            uniq_uncertainty_sources[0]
            if len(uniq_uncertainty_sources) == 1
            else "mixed"
        )
    fusion_profiles = [
        str(tile.get("fusion_profile")).strip()
        for tile in tiles
        if isinstance(tile.get("fusion_profile"), str) and str(tile.get("fusion_profile")).strip()
    ]
    if fusion_profiles:
        uniq_fusion = sorted(set(fusion_profiles))
        runtime_meta["fusion_profile"] = uniq_fusion[0] if len(uniq_fusion) == 1 else "mixed"
    else:
        runtime_meta["fusion_profile"] = str(runtime_meta.get("fusion_profile") or "none")
    runtime_meta["date_selection_low_confidence"] = bool(
        runtime_meta.get("date_selection_low_confidence")
        or runtime_meta.get("low_quality_input")
        or any(bool(tile.get("low_quality_input")) for tile in tiles)
    )

    runtime_meta["road_barrier_retry_used"] = bool(
        any(bool(tile.get("road_barrier_retry_used")) for tile in tiles)
    )
    runtime_meta["water_edge_risk_detected"] = bool(
        any(float(tile.get("water_edge_overlap_ratio") or 0.0) > 0.05 for tile in tiles)
    )
    runtime_meta["road_drift_risk_detected"] = bool(
        any(
            float(tile.get("road_edge_overlap_ratio") or 0.0)
            > 0.08
            for tile in tiles
        )
    )

    sam_modes = [
        str(tile.get("sam_runtime_mode")).strip()
        for tile in tiles
        if isinstance(tile.get("sam_runtime_mode"), str) and str(tile.get("sam_runtime_mode")).strip()
    ]
    if sam_modes:
        uniq_sam = sorted(set(sam_modes))
        runtime_meta["sam_runtime_mode"] = uniq_sam[0] if len(uniq_sam) == 1 else "mixed"
    else:
        runtime_meta["sam_runtime_mode"] = str(
            runtime_meta.get("sam_runtime_mode") or "fallback_non_sam"
        )

    model_backends = [
        str(tile.get("model_backend")).strip()
        for tile in tiles
        if isinstance(tile.get("model_backend"), str) and str(tile.get("model_backend")).strip()
    ]
    if model_backends:
        uniq = sorted(set(model_backends))
        runtime_meta["model_backend"] = uniq[0] if len(uniq) == 1 else "mixed"
    else:
        runtime_meta["model_backend"] = str(runtime_meta.get("model_backend") or "unknown")

    if runtime_meta.get("weak_label_source") is None:
        runtime_meta["weak_label_source"] = "unknown"
    runtime_meta["latency_breakdown_s"] = {
        **{k: round(float(v), 3) for k, v in latency_breakdown.items()},
        "total_s": round(float(time.time() - t_start), 3),
    }
    return runtime_meta


def _extract_postprocess_stage_pixels(postprocess_stats: dict | None) -> dict[str, int]:
    if not isinstance(postprocess_stats, dict):
        return {}
    step_map = {
        "after_ml_seed": "step_00_candidate_initial",
        "after_barrier": "step_04_after_barrier",
        "after_clean": "step_05_after_clean",
        "after_grow": "step_06_after_grow",
        "after_gap_close": "step_07_after_gap_close",
        "after_infill": "step_08_after_infill",
        "after_merge": "step_09_after_merge",
        "after_watershed": "step_10_after_watershed",
        "after_small_remove": "step_11_after_small_remove",
    }
    out: dict[str, int] = {}
    for public_name, internal_name in step_map.items():
        step = postprocess_stats.get(internal_name)
        if isinstance(step, dict) and isinstance(step.get("pixels"), (int, float)):
            out[public_name] = int(step["pixels"])
    return out


def _estimate_sam_memory_mb(
    composite_uint8: np.ndarray,
    candidate_mask: np.ndarray,
    component_count: int,
) -> float:
    tile_pixels = int(candidate_mask.size)
    clamped_components = max(1, min(int(component_count), 256))
    estimated_bytes = (
        int(composite_uint8.nbytes)
        + int(candidate_mask.nbytes)
        + (tile_pixels * 4)  # float32 scratch
        + (tile_pixels * min(clamped_components, 64))  # candidate masks
    )
    return float(estimated_bytes / (1024.0 * 1024.0))


def _sam_preflight_budget(
    *,
    composite_uint8: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_coverage_pct: float,
    component_count: int,
    cfg,
) -> tuple[bool, str | None, float]:
    estimated_mb = _estimate_sam_memory_mb(composite_uint8, candidate_mask, component_count)
    tile_pixels = int(candidate_mask.size)
    if tile_pixels > int(getattr(cfg, "SAM_MAX_TILE_PIXELS", 600000)):
        return False, "tile_pixels", estimated_mb
    if candidate_coverage_pct > float(getattr(cfg, "SAM_MAX_CANDIDATE_COVERAGE_PCT", 18.0)):
        return False, "candidate_coverage", estimated_mb
    if component_count > int(getattr(cfg, "SAM_MAX_COMPONENTS", 64)):
        return False, "component_count", estimated_mb
    if estimated_mb > float(getattr(cfg, "SAM_MAX_EST_MEMORY_MB", 2200)):
        return False, "memory_budget", estimated_mb
    return True, None, estimated_mb


def _boundary_guided_ml_seed(
    *,
    extent_prob: np.ndarray,
    boundary_prob: np.ndarray,
    ndvi: np.ndarray,
    ndvi_std: np.ndarray | None,
    cfg,
    extent_threshold_override: float | None = None,
    dilation_px_override: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    from processing.fields.ml_fusion import boundary_guided_ml_seed

    return boundary_guided_ml_seed(
        extent_prob=extent_prob,
        boundary_prob=boundary_prob,
        ndvi=ndvi,
        ndvi_std=ndvi_std,
        cfg=cfg,
        extent_threshold_override=extent_threshold_override,
        dilation_px_override=dilation_px_override,
    )


def _summarize_geometry_diagnostics(before_gdf, after_gdf, diagnostics: list[dict] | None = None) -> dict[str, float | int | list[str]]:
    if before_gdf is None or after_gdf is None or getattr(before_gdf, "empty", True) or getattr(after_gdf, "empty", True):
        return {
            "count_compared": 0,
            "contour_shrink_ratio": 1.0,
            "centroid_shift_m": 0.0,
            "max_centroid_shift_m": 0.0,
            "rejected_reasons": [],
        }
    count = min(len(before_gdf), len(after_gdf))
    if count <= 0:
        return {
            "count_compared": 0,
            "contour_shrink_ratio": 1.0,
            "centroid_shift_m": 0.0,
            "max_centroid_shift_m": 0.0,
            "rejected_reasons": [],
        }
    ratios: list[float] = []
    shifts: list[float] = []
    for idx in range(count):
        before_geom = before_gdf.geometry.iloc[idx]
        after_geom = after_gdf.geometry.iloc[idx]
        before_area = float(getattr(before_geom, "area", 0.0))
        after_area = float(getattr(after_geom, "area", 0.0))
        ratios.append(after_area / max(before_area, 1e-6))
        before_c = before_geom.centroid
        after_c = after_geom.centroid
        shifts.append(float(np.hypot(after_c.x - before_c.x, after_c.y - before_c.y)))
    rejected_reasons = sorted(
        {
            str(item.get("rejected_reason"))
            for item in (diagnostics or [])
            if str(item.get("rejected_reason") or "").strip()
            and str(item.get("rejected_reason")) != "disabled"
        }
    )
    return {
        "count_compared": int(count),
        "contour_shrink_ratio": float(np.mean(ratios)) if ratios else 1.0,
        "centroid_shift_m": float(np.mean(shifts)) if shifts else 0.0,
        "max_centroid_shift_m": float(np.max(shifts)) if shifts else 0.0,
        "rejected_reasons": rejected_reasons[:5],
    }


def resolve_region_band(lat: float, cfg) -> str:
    from core.region import resolve_region_band as _resolve

    return _resolve(
        lat,
        south_max=float(getattr(cfg, "REGION_LAT_SOUTH_MAX", 48.0)),
        north_min=float(getattr(cfg, "REGION_LAT_NORTH_MIN", 57.0)),
    )


def resolve_region_boundary_profile(lat: float, cfg) -> str:
    from core.region import resolve_region_boundary_profile as _resolve

    return _resolve(lat, cfg)


def _regional_quality_target(region_boundary_profile: str) -> str:
    token = str(region_boundary_profile).strip().lower()
    if token == "south_recall":
        return "anti_fragmentation"
    if token == "north_boundary":
        return "anti_shrink"
    return "neutral"


def _resolve_regional_extent_threshold(cfg, region_boundary_profile: str) -> float:
    token = str(region_boundary_profile).strip().lower()
    if token == "south_recall":
        return float(
            getattr(cfg, "SOUTH_ML_EXTENT_BIN_THRESHOLD", getattr(cfg, "ML_EXTENT_BIN_THRESHOLD", 0.42))
        )
    if token == "north_boundary":
        return float(
            getattr(cfg, "NORTH_ML_EXTENT_BIN_THRESHOLD", getattr(cfg, "ML_EXTENT_BIN_THRESHOLD", 0.42))
        )
    return float(getattr(cfg, "ML_EXTENT_BIN_THRESHOLD", 0.42))


def _resolve_regional_dilation_px(cfg, region_boundary_profile: str) -> int | None:
    token = str(region_boundary_profile).strip().lower()
    if token == "north_boundary":
        return int(
            getattr(cfg, "NORTH_BOUNDARY_OUTER_DILATION_PX", getattr(cfg, "POST_BOUNDARY_DILATION_PX", 1))
        )
    return None


def _fuse_ml_primary_candidate(
    ml_seed_mask: np.ndarray,
    pre_ml_candidate_mask: np.ndarray,
    region_boundary_profile: str | None,
) -> tuple[np.ndarray, list[str]]:
    """Backward-compatible wrapper for the extracted fusion helper."""
    from processing.fields.ml_fusion import fuse_ml_primary_candidate

    return fuse_ml_primary_candidate(
        ml_seed_mask=ml_seed_mask,
        pre_ml_candidate_mask=pre_ml_candidate_mask,
        region_boundary_profile=region_boundary_profile,
    )


def _resolve_vectorize_simplify_tol(cfg, region_boundary_profile: str) -> float:
    token = str(region_boundary_profile).strip().lower()
    if token == "north_boundary":
        return float(
            getattr(cfg, "NORTH_VECTORIZE_SIMPLIFY_TOL_M", getattr(cfg, "VECTORIZE_SIMPLIFY_TOL_M", 1.0))
        )
    return float(getattr(cfg, "VECTORIZE_SIMPLIFY_TOL_M", getattr(cfg, "SIMPLIFY_TOL_M", 5.0)))


def _apply_runtime_config(base_settings: Any, overrides: dict | None) -> Any:
    """Return a settings copy with validated runtime overrides."""
    overrides = dict(overrides or {})
    if not overrides:
        return base_settings

    alias_map = {
        "WCTREEHARD_EXCLUSION": "WC_TREE_HARD_EXCLUSION",
        "UNETEDGEMODEL": "UNET_EDGE_MODEL",
        "UNETEDGETHRESHOLD": "UNET_EDGE_THRESHOLD",
        "UNETDEVICE": "UNET_DEVICE",
        "SAMCHECKPOINT": "SAM2_CHECKPOINT",
        "SAMPOINTSPACING": "SAM_POINT_SPACING",
        "SAMPREDIOUTHRESH": "SAM_PRED_IOU_THRESHOLD",
        "S1ENABLED": "S1_ENABLED",
        "S1ACQUISITIONMODE": "S1_ACQUISITION_MODE",
        "S1POLARIZATION": "S1_POLARIZATION",
        "S1LEEFILTERENABLE": "S1_LEE_FILTER_ENABLE",
        "S1LEEWINDOWSIZE": "S1_LEE_WINDOW_SIZE",
        "SNICREFINEENABLED": "SNIC_REFINE_ENABLED",
        "SNICNSEGMENTS": "SNIC_N_SEGMENTS",
        "SNICCOMPACTNESS": "SNIC_COMPACTNESS",
        "SNICMERGENDVITHRESH": "SNIC_MERGE_NDVI_THRESH",
        "FEATUREUNETEDGE": "FEATURE_UNET_EDGE",
        "FEATURESAM2PRIMARY": "FEATURE_SAM2_PRIMARY",
        "FEATURES1FUSION": "FEATURE_S1_FUSION",
        "FEATURESNICREFINE": "FEATURE_SNIC_REFINE",
        "FEATUREMLPRIMARY": "FEATURE_ML_PRIMARY",
        "MODE": "MODE",
        "AUTODETECTVERSION": "AUTO_DETECT_VERSION",
        "MLFEATUREPROFILE": "ML_FEATURE_PROFILE",
        "MLMODELPATH": "ML_MODEL_PATH",
        "MLMODELNORMSTATSPATH": "ML_MODEL_NORM_STATS_PATH",
        "MLINFERENCEDEVICE": "ML_INFERENCE_DEVICE",
        "MLFALLBACKONLOWSCORE": "ML_FALLBACK_ON_LOW_SCORE",
        "MLSCORETHRESHOLD": "ML_SCORE_THRESHOLD",
        "MLTILESIZE": "ML_TILE_SIZE",
        "MLOVERLAP": "ML_OVERLAP",
        "MLUSEONNX": "ML_USE_ONNX",
    }
    valid_keys = set(base_settings.model_fields.keys())
    applied = {}
    for key, value in overrides.items():
        normalized = alias_map.get(key, key)
        if normalized in valid_keys:
            applied[normalized] = value
    if not applied:
        return base_settings
    return base_settings.model_copy(update=applied)


def _resolve_detection_model_version(settings: Any, auto_detect_version: int) -> str:
    model_path = str(getattr(settings, "ML_MODEL_PATH", "") or "").strip()
    feature_profile = str(getattr(settings, "ML_FEATURE_PROFILE", "") or "").strip()
    model_token = Path(model_path).stem if model_path else f"autodetect_v{int(auto_detect_version)}"
    return f"{model_token}:{feature_profile}" if feature_profile else model_token


def _resolve_processing_profile(qc_mode: str | None, cfg: Any) -> dict[str, Any]:
    mode = str(qc_mode or "normal").strip().lower()
    base_grow_relax = float(getattr(cfg, "POST_GROW_NDVI_RELAX", 0.11))
    base_merge_ndvi = float(getattr(cfg, "POST_MERGE_NDVI_DIFF_MAX", 0.12))
    base_obia_delta = float(getattr(cfg, "OBIA_MIN_NDVI_DELTA", 0.20))
    base_candidate_min_score = float(getattr(cfg, "CANDIDATE_MIN_SCORE", 0.25))
    base_split_score = float(getattr(cfg, "SELECTIVE_SPLIT_SCORE_MIN", 0.70))
    base_min_distance = int(getattr(cfg, "WATERSHED_MIN_DISTANCE", 14))
    base_boundary_boost = float(getattr(cfg, "BOUNDARY_RECOVERY_BRANCH_SCORE_BOOST", 0.08))
    base_recovery_boost = float(getattr(cfg, "RECOVERY_SECOND_PASS_SCORE_BOOST", 0.05))
    base_edge_seed_threshold = float(getattr(cfg, "BOUNDARY_RECOVERY_EDGE_SEED_THRESHOLD", 0.18))
    base_edge_seed_percentile = float(getattr(cfg, "BOUNDARY_RECOVERY_EDGE_SEED_PERCENTILE", 72.0))
    base_recovery_dilation = int(getattr(cfg, "BOUNDARY_RECOVERY_DILATION_PX", 2))
    base_boundary_halo = int(getattr(cfg, "BOUNDARY_RECOVERY_BOUNDARY_HALO_PX", 1))

    if mode == "boundary_recovery":
        return {
            "name": "boundary_recovery",
            "enable_second_pass": True,
            "allow_degraded_output": False,
            "prefer_boundary_branch": True,
            "force_boundary_union": True,
            "boundary_branch_score_boost": round(base_boundary_boost, 4),
            "recovery_second_pass_score_boost": round(base_recovery_boost, 4),
            "recovery_edge_seed_threshold": round(base_edge_seed_threshold, 4),
            "recovery_edge_seed_percentile": round(base_edge_seed_percentile, 2),
            "recovery_dilation_px": int(max(base_recovery_dilation, 1)),
            "boundary_halo_px": int(max(base_boundary_halo, 1)),
            "config_overrides": {
                "POST_GROW_NDVI_RELAX": round(base_grow_relax + 0.03, 4),
                "POST_MERGE_NDVI_DIFF_MAX": round(base_merge_ndvi + 0.03, 4),
                "OBIA_MIN_NDVI_DELTA": round(max(0.05, base_obia_delta * 0.65), 4),
                "CANDIDATE_MIN_SCORE": round(max(0.10, base_candidate_min_score - 0.05), 4),
                "SELECTIVE_SPLIT_SCORE_MIN": round(base_split_score + 0.06, 4),
                "WATERSHED_MIN_DISTANCE": int(base_min_distance + 2),
                "SKIP_WEAK_EDGE_TILES": False,
                "RECOVERY_BOUNDARY_ANCHOR_ENABLED": True,
                "RECOVERY_BOUNDARY_ANCHOR_DILATION_PX": int(max(base_recovery_dilation + 1, 2)),
                "RECOVERY_TEMPORAL_COHERENCE_RELAXED": True,
                "RECOVERY_TEMPORAL_GROWTH_AMPLITUDE_MIN": 0.14,
                "RECOVERY_TEMPORAL_ENTROPY_MAX": 3.1,
            },
        }
    if mode == "degraded_output":
        return {
            "name": "degraded_output",
            "enable_second_pass": True,
            "allow_degraded_output": True,
            "prefer_boundary_branch": True,
            "force_boundary_union": True,
            "boundary_branch_score_boost": round(base_boundary_boost + 0.02, 4),
            "recovery_second_pass_score_boost": round(base_recovery_boost + 0.02, 4),
            "recovery_edge_seed_threshold": round(max(0.12, base_edge_seed_threshold - 0.02), 4),
            "recovery_edge_seed_percentile": round(max(60.0, base_edge_seed_percentile - 4.0), 2),
            "recovery_dilation_px": int(max(base_recovery_dilation + 1, 2)),
            "boundary_halo_px": int(max(base_boundary_halo + 1, 1)),
            "config_overrides": {
                "POST_GROW_NDVI_RELAX": round(base_grow_relax + 0.05, 4),
                "POST_MERGE_NDVI_DIFF_MAX": round(base_merge_ndvi + 0.05, 4),
                "OBIA_MIN_NDVI_DELTA": round(max(0.03, base_obia_delta * 0.50), 4),
                "CANDIDATE_MIN_SCORE": round(max(0.08, base_candidate_min_score - 0.10), 4),
                "SELECTIVE_SPLIT_SCORE_MIN": round(base_split_score + 0.10, 4),
                "WATERSHED_MIN_DISTANCE": int(base_min_distance + 4),
                "SKIP_WEAK_EDGE_TILES": False,
                "RECOVERY_BOUNDARY_ANCHOR_ENABLED": True,
                "RECOVERY_BOUNDARY_ANCHOR_DILATION_PX": int(max(base_recovery_dilation + 2, 3)),
                "RECOVERY_EDGE_GUIDE_HALO_FACTOR": 0.8,
                "RECOVERY_TEMPORAL_COHERENCE_RELAXED": True,
                "RECOVERY_TEMPORAL_GROWTH_AMPLITUDE_MIN": 0.12,
                "RECOVERY_TEMPORAL_ENTROPY_MAX": 3.35,
            },
        }
    if mode == "skip_tile":
        return {
            "name": "skip_tile",
            "enable_second_pass": False,
            "allow_degraded_output": False,
            "prefer_boundary_branch": False,
            "force_boundary_union": False,
            "boundary_branch_score_boost": 0.0,
            "recovery_second_pass_score_boost": 0.0,
            "recovery_edge_seed_threshold": 1.0,
            "recovery_edge_seed_percentile": 100.0,
            "recovery_dilation_px": 0,
            "boundary_halo_px": 0,
            "config_overrides": {},
        }
    return {
        "name": "normal",
        "enable_second_pass": False,
        "allow_degraded_output": False,
        "prefer_boundary_branch": False,
        "force_boundary_union": False,
        "boundary_branch_score_boost": 0.0,
        "recovery_second_pass_score_boost": 0.0,
        "recovery_edge_seed_threshold": 1.0,
        "recovery_edge_seed_percentile": 100.0,
        "recovery_dilation_px": 0,
        "boundary_halo_px": 0,
        "config_overrides": {},
    }


def _should_skip_low_observation_tile(low_obs_pct: float, cfg: Any) -> bool:
    base_threshold = float(getattr(cfg, "TILE_MAX_LOW_OBS_PCT", 0.5))
    hard_threshold = max(0.85, base_threshold + 0.25)
    return float(low_obs_pct) >= float(hard_threshold)


def _candidate_tile_quality_score(tile_runtime: dict[str, Any]) -> float:
    coverage = float(tile_runtime.get("qc_coverage") or 0.0)
    edge_p90 = float(tile_runtime.get("qc_edge_p90") or 0.0)
    ndvi_std = float(tile_runtime.get("qc_ndvi_std") or 0.0)
    score = (
        0.45 * np.clip(coverage, 0.0, 1.0)
        + 0.35 * np.clip(edge_p90 * 4.0, 0.0, 1.0)
        + 0.20 * np.clip(ndvi_std * 3.0, 0.0, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _connected_component_candidates(
    mask: np.ndarray,
    branch: str,
    *,
    max_components: int | None = None,
) -> list[Any]:
    from processing.fields.candidate_ranker import CandidatePolygon

    labeled, meta = _limited_component_labels(mask, max_components=max_components)
    if int(meta.get("kept_components") or 0) <= 0:
        return []
    out: list[Any] = []
    for label_id in range(1, int(meta.get("kept_components") or 0) + 1):
        component_mask = labeled == label_id
        if np.any(component_mask):
            out.append(
                CandidatePolygon(
                    mask=component_mask,
                    branch=str(branch),
                    label_id=int(label_id),
                )
            )
    return out


def _apply_branch_score_bias(candidates: list[Any], processing_profile: dict[str, Any]) -> None:
    boundary_boost = float(processing_profile.get("boundary_branch_score_boost") or 0.0)
    recovery_boost = float(processing_profile.get("recovery_second_pass_score_boost") or 0.0)
    if boundary_boost <= 0.0 and recovery_boost <= 0.0:
        return

    for candidate in candidates:
        branch = str(getattr(candidate, "branch", "") or "")
        boost = 0.0
        if branch == "boundary_first":
            boost = boundary_boost
        elif branch == "recovery_second_pass":
            boost = recovery_boost
        if boost <= 0.0:
            continue
        candidate.score = float(np.clip(float(candidate.score or 0.0) + boost, 0.0, 1.0))
        feature_payload = dict(getattr(candidate, "features", {}) or {})
        feature_payload["branch_score_boost"] = round(float(boost), 4)
        feature_payload["score_post_bias"] = round(float(candidate.score), 4)
        candidate.features = feature_payload


def _build_recovery_missed_mask(
    *,
    candidate_masks_payload: dict[str, Any],
    candidate_mask: np.ndarray,
    processing_profile: dict[str, Any],
    edge_source: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.ndimage import binary_dilation

    seed_masks: list[np.ndarray] = []
    for key in ("field_candidate", "crop_soft_mask", "boundary_field_mask"):
        seed = candidate_masks_payload.get(key)
        if seed is None:
            continue
        seed_bool = np.asarray(seed, dtype=bool)
        if np.any(seed_bool):
            seed_masks.append(seed_bool)

    shape = np.asarray(candidate_mask, dtype=bool).shape
    if not seed_masks:
        return np.zeros(shape, dtype=bool), {
            "base_seed_pixels": 0,
            "edge_seed_pixels": 0,
            "boundary_halo_pixels": 0,
            "edge_seed_threshold": None,
        }

    base_union = np.logical_or.reduce(seed_masks)
    guide_mask = base_union
    dilation_px = int(processing_profile.get("recovery_dilation_px") or 0)
    if dilation_px > 0:
        guide_mask = binary_dilation(guide_mask, iterations=dilation_px)

    missed_mask = base_union & ~np.asarray(candidate_mask, dtype=bool)
    boundary_halo_pixels = 0
    boundary_mask = candidate_masks_payload.get("boundary_field_mask")
    if bool(processing_profile.get("prefer_boundary_branch")) and boundary_mask is not None:
        boundary_mask = np.asarray(boundary_mask, dtype=bool)
        if np.any(boundary_mask):
            halo_px = int(processing_profile.get("boundary_halo_px") or 0)
            if halo_px > 0:
                boundary_halo = (
                    binary_dilation(boundary_mask, iterations=halo_px) & guide_mask & ~np.asarray(candidate_mask, dtype=bool)
                )
                if np.any(boundary_halo):
                    missed_mask |= boundary_halo
                    boundary_halo_pixels = int(np.count_nonzero(boundary_halo))

    edge_seed_pixels = 0
    guide_edge_halo_pixels = 0
    edge_threshold = None
    if edge_source is not None:
        edge_arr = np.asarray(edge_source, dtype=np.float32)
        finite = edge_arr[np.isfinite(edge_arr)]
        if finite.size > 0:
            percentile = float(processing_profile.get("recovery_edge_seed_percentile") or 72.0)
            percentile = float(np.clip(percentile, 0.0, 100.0))
            percentile_threshold = float(np.nanpercentile(finite, percentile))
            base_threshold = float(processing_profile.get("recovery_edge_seed_threshold") or 0.0)
            edge_threshold = float(np.clip(max(base_threshold, percentile_threshold), base_threshold, 0.85))
            edge_seed = (
                np.isfinite(edge_arr)
                & (edge_arr >= edge_threshold)
                & guide_mask
                & ~np.asarray(candidate_mask, dtype=bool)
            )
            if np.any(edge_seed):
                missed_mask |= edge_seed
                edge_seed_pixels = int(np.count_nonzero(edge_seed))
            guide_edge_factor = float(
                np.clip(
                    processing_profile.get("recovery_edge_guide_halo_factor") or 0.85,
                    0.5,
                    1.0,
                )
            )
            guide_edge_threshold = float(edge_threshold * guide_edge_factor)
            guide_edge_halo = (
                np.isfinite(edge_arr)
                & (edge_arr >= guide_edge_threshold)
                & guide_mask
                & ~np.asarray(candidate_mask, dtype=bool)
            )
            if np.any(guide_edge_halo):
                missed_mask |= guide_edge_halo
                guide_edge_halo_pixels = int(np.count_nonzero(guide_edge_halo))

    return missed_mask.astype(bool, copy=False), {
        "base_seed_pixels": int(np.count_nonzero(base_union)),
        "edge_seed_pixels": int(edge_seed_pixels),
        "guide_edge_halo_pixels": int(guide_edge_halo_pixels),
        "boundary_halo_pixels": int(boundary_halo_pixels),
        "edge_seed_threshold": None if edge_threshold is None else round(float(edge_threshold), 4),
    }


def _build_temporal_coherence_mask(
    *,
    growth_amplitude: np.ndarray,
    has_growth_peak: np.ndarray,
    ndvi_entropy: np.ndarray,
    candidate_masks_payload: dict[str, Any] | None,
    processing_profile: dict[str, Any],
    cfg: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    relaxed = bool(
        processing_profile.get("name") != "normal"
        or getattr(cfg, "RECOVERY_TEMPORAL_COHERENCE_RELAXED", False)
    )
    amplitude_min = 0.20
    entropy_max = 2.5
    if relaxed:
        amplitude_min = float(
            getattr(cfg, "RECOVERY_TEMPORAL_GROWTH_AMPLITUDE_MIN", amplitude_min)
        )
        entropy_max = float(getattr(cfg, "RECOVERY_TEMPORAL_ENTROPY_MAX", entropy_max))

    tc_mask = (
        np.asarray(has_growth_peak, dtype=np.float32) > 0.5
    ) | (
        (np.asarray(growth_amplitude, dtype=np.float32) >= amplitude_min)
        & (np.asarray(ndvi_entropy, dtype=np.float32) <= entropy_max)
    )

    boundary_keep_pixels = 0
    if relaxed and candidate_masks_payload:
        boundary_seed = candidate_masks_payload.get("boundary_field_mask")
        if boundary_seed is not None:
            boundary_seed = np.asarray(boundary_seed, dtype=bool)
            if np.any(boundary_seed):
                tc_mask |= boundary_seed
                boundary_keep_pixels = int(np.count_nonzero(boundary_seed))

    return tc_mask.astype(bool, copy=False), {
        "relaxed": relaxed,
        "growth_amplitude_min": float(amplitude_min),
        "entropy_max": float(entropy_max),
        "boundary_keep_pixels": int(boundary_keep_pixels),
    }


def _summarize_ranked_candidates(ranked_candidates: list[Any]) -> dict[str, Any]:
    branch_counts: dict[str, dict[str, int]] = {}
    reject_summary: dict[str, int] = {}
    kept_total = 0
    for ranked in ranked_candidates:
        branch = str(ranked.candidate.branch or "unknown")
        branch_counts.setdefault(branch, {"total": 0, "kept": 0})
        branch_counts[branch]["total"] += 1
        if bool(ranked.keep):
            branch_counts[branch]["kept"] += 1
            kept_total += 1
        else:
            reason = str(ranked.candidate.reject_reason or "unclassified_reject")
            reject_summary[reason] = reject_summary.get(reason, 0) + 1
    return {
        "candidate_branch_counts": branch_counts,
        "candidate_reject_summary": reject_summary,
        "candidates_total": int(len(ranked_candidates)),
        "candidates_kept": int(kept_total),
    }


def _accumulate_candidate_summary(runtime_meta: dict[str, Any], summary: dict[str, Any]) -> None:
    runtime_meta["candidates_total"] = int(runtime_meta.get("candidates_total") or 0) + int(
        summary.get("candidates_total") or 0
    )
    runtime_meta["candidates_kept"] = int(runtime_meta.get("candidates_kept") or 0) + int(
        summary.get("candidates_kept") or 0
    )
    branch_counts = dict(runtime_meta.get("candidate_branch_counts") or {})
    for branch, counts in dict(summary.get("candidate_branch_counts") or {}).items():
        merged = dict(branch_counts.get(branch) or {"total": 0, "kept": 0})
        merged["total"] = int(merged.get("total") or 0) + int(counts.get("total") or 0)
        merged["kept"] = int(merged.get("kept") or 0) + int(counts.get("kept") or 0)
        branch_counts[str(branch)] = merged
    runtime_meta["candidate_branch_counts"] = branch_counts
    reject_summary = dict(runtime_meta.get("candidate_reject_summary") or {})
    for reason, count in dict(summary.get("candidate_reject_summary") or {}).items():
        reject_summary[str(reason)] = int(reject_summary.get(str(reason)) or 0) + int(count or 0)
    runtime_meta["candidate_reject_summary"] = reject_summary


def _mask_to_candidate_geometry(mask: np.ndarray, tile_transform: Any, tile_crs: Any):
    import rasterio.features
    from pyproj import Transformer
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform, unary_union

    binary_mask = np.asarray(mask, dtype=bool)
    if not np.any(binary_mask):
        return None
    polygons = []
    for geom_mapping, value in rasterio.features.shapes(
        binary_mask.astype(np.uint8),
        mask=binary_mask.astype(np.uint8),
        transform=tile_transform,
    ):
        if not value:
            continue
        geom = shapely_shape(geom_mapping)
        if geom.is_empty:
            continue
        polygons.append(geom)
    if not polygons:
        return None
    merged = unary_union(polygons)
    transformer = Transformer.from_crs(tile_crs, "EPSG:4326", always_xy=True)
    merged = shapely_transform(transformer.transform, merged)
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: float(g.area), default=None)
    if merged is None or merged.is_empty or merged.geom_type != "Polygon":
        return None
    return merged


def _persist_detection_candidates(
    *,
    session: Any,
    run: Any,
    run_id: uuid.UUID,
    tile_diagnostic_id: int | None,
    ranked_candidates: list[Any],
    tile_transform: Any,
    tile_crs: Any,
    resolution_m: float,
    model_version: str,
    FieldDetectionCandidateModel: Any,
) -> int:
    from geoalchemy2.shape import from_shape

    inserted = 0
    for ranked in ranked_candidates:
        geometry = _mask_to_candidate_geometry(ranked.candidate.mask, tile_transform, tile_crs)
        if geometry is None:
            continue
        area_m2 = float(np.count_nonzero(ranked.candidate.mask)) * float(resolution_m) * float(resolution_m)
        feature_payload = dict(ranked.candidate.features or {})
        feature_payload["label_id"] = int(getattr(ranked.candidate, "label_id", 0) or 0)
        if ranked.suppressed_by is not None:
            feature_payload["suppressed_by"] = int(ranked.suppressed_by)
        feature_payload["lifecycle_stage"] = "rank_kept_pending" if bool(ranked.keep) else "rank_rejected"
        session.add(
            FieldDetectionCandidateModel(
                organization_id=run.organization_id,
                aoi_run_id=run_id,
                tile_diagnostic_id=tile_diagnostic_id,
                field_id=None,
                branch=str(ranked.candidate.branch or "unknown"),
                geom=from_shape(geometry, srid=4326),
                area_m2=round(float(area_m2), 2),
                score=float(ranked.candidate.score or 0.0),
                rank=int(ranked.rank),
                kept=bool(ranked.keep),
                reject_reason=(
                    None if bool(ranked.keep) else str(ranked.candidate.reject_reason or "filtered")
                ),
                features=_sanitize_json_floats(feature_payload),
                model_version=str(model_version),
            )
        )
        inserted += 1
    return int(inserted)


def _candidate_overlap_score(candidate_geom: Any, field_geom: Any) -> float:
    """Return a stable overlap score for candidate-to-field matching.

    We bias toward candidate coverage because a tile-level candidate can be a
    subset of a final merged field.
    """
    try:
        if candidate_geom is None or field_geom is None:
            return 0.0
        if candidate_geom.is_empty or field_geom.is_empty:
            return 0.0
        intersection = candidate_geom.intersection(field_geom)
        if intersection.is_empty:
            return 0.0
        inter_area = float(intersection.area)
        candidate_area = max(float(candidate_geom.area), 1e-9)
        field_area = max(float(field_geom.area), 1e-9)
        coverage = inter_area / candidate_area
        iou = inter_area / max(candidate_area + field_area - inter_area, 1e-9)
        return float(max(coverage, iou))
    except Exception:
        return 0.0


def _summarize_persisted_detection_candidates(rows: list[Any]) -> dict[str, Any]:
    branch_counts: dict[str, dict[str, int]] = {}
    reject_summary: dict[str, int] = {}
    kept_total = 0
    total = 0
    for row in rows:
        total += 1
        branch = str(getattr(row, "branch", "unknown") or "unknown")
        kept = bool(getattr(row, "kept", False))
        if kept:
            kept_total += 1
        branch_bucket = dict(branch_counts.get(branch) or {"total": 0, "kept": 0})
        branch_bucket["total"] += 1
        if kept:
            branch_bucket["kept"] += 1
        branch_counts[branch] = branch_bucket
        reject_reason = getattr(row, "reject_reason", None)
        if reject_reason:
            reject_summary[str(reject_reason)] = int(reject_summary.get(str(reject_reason)) or 0) + 1
    return {
        "candidate_branch_counts": branch_counts,
        "candidate_reject_summary": reject_summary,
        "candidates_total": int(total),
        "candidates_kept": int(kept_total),
    }


def _finalize_detection_candidates(
    *,
    session: Any,
    run_id: uuid.UUID,
    FieldModel: Any,
    FieldDetectionCandidateModel: Any,
    pre_topology_field_geojson: list[dict[str, Any]] | None = None,
    object_classifier_rejected_geojson: list[dict[str, Any]] | None = None,
    final_match_threshold: float = 0.35,
) -> dict[str, Any]:
    """Attach kept candidates to final fields and classify terminal reject reasons."""
    from geoalchemy2.shape import to_shape
    from shapely.geometry import shape as shapely_shape

    lifecycle_counts = {
        "matched_to_final": 0,
        "dropped_by_object_classifier": 0,
        "dropped_after_topology_cleanup": 0,
        "dropped_after_merge": 0,
    }

    final_fields = list(
        session.query(FieldModel)
        .filter(FieldModel.aoi_run_id == run_id)
        .all()
    )
    final_field_geoms: list[tuple[uuid.UUID, Any]] = []
    for field in final_fields:
        try:
            final_field_geoms.append((field.id, to_shape(field.geom)))
        except Exception:
            continue

    pre_topology_geoms = []
    for geometry_geojson in list(pre_topology_field_geojson or []):
        try:
            pre_topology_geoms.append(shapely_shape(geometry_geojson))
        except Exception:
            continue

    rejected_classifier_geoms = []
    for geometry_geojson in list(object_classifier_rejected_geojson or []):
        try:
            rejected_classifier_geoms.append(shapely_shape(geometry_geojson))
        except Exception:
            continue

    candidate_rows = list(
        session.query(FieldDetectionCandidateModel)
        .filter(FieldDetectionCandidateModel.aoi_run_id == run_id)
        .all()
    )

    for row in candidate_rows:
        feature_payload = dict(getattr(row, "features", {}) or {})
        if not bool(getattr(row, "kept", False)):
            feature_payload.setdefault("lifecycle_stage", "rank_rejected")
            row.features = _sanitize_json_floats(feature_payload)
            continue
        try:
            candidate_geom = to_shape(row.geom)
        except Exception:
            candidate_geom = None

        best_field_id = None
        best_field_score = 0.0
        for field_id, field_geom in final_field_geoms:
            overlap_score = _candidate_overlap_score(candidate_geom, field_geom)
            if overlap_score > best_field_score:
                best_field_score = overlap_score
                best_field_id = field_id

        if best_field_id is not None and best_field_score >= float(final_match_threshold):
            row.field_id = best_field_id
            row.kept = True
            row.reject_reason = None
            feature_payload["lifecycle_stage"] = "final_field"
            feature_payload["final_match_score"] = round(float(best_field_score), 4)
            lifecycle_counts["matched_to_final"] += 1
            row.features = _sanitize_json_floats(feature_payload)
            continue

        row.field_id = None
        row.kept = False
        classifier_score = max(
            [_candidate_overlap_score(candidate_geom, geom) for geom in rejected_classifier_geoms] or [0.0]
        )
        pre_topology_score = max(
            [_candidate_overlap_score(candidate_geom, geom) for geom in pre_topology_geoms] or [0.0]
        )
        if classifier_score >= float(final_match_threshold):
            row.reject_reason = "dropped_by_object_classifier"
            feature_payload["lifecycle_stage"] = "dropped_by_object_classifier"
            lifecycle_counts["dropped_by_object_classifier"] += 1
        elif pre_topology_score >= float(final_match_threshold):
            row.reject_reason = "dropped_after_topology_cleanup"
            feature_payload["lifecycle_stage"] = "dropped_after_topology_cleanup"
            lifecycle_counts["dropped_after_topology_cleanup"] += 1
        else:
            row.reject_reason = "dropped_after_merge"
            feature_payload["lifecycle_stage"] = "dropped_after_merge"
            lifecycle_counts["dropped_after_merge"] += 1
        feature_payload["final_match_score"] = round(float(best_field_score), 4)
        feature_payload["pre_topology_match_score"] = round(float(pre_topology_score), 4)
        feature_payload["classifier_reject_match_score"] = round(float(classifier_score), 4)
        row.features = _sanitize_json_floats(feature_payload)

    summary = _summarize_persisted_detection_candidates(candidate_rows)
    summary["candidate_lifecycle"] = lifecycle_counts
    return summary


def _resolve_auto_detect_version(settings) -> int:
    """Clamp the requested autodetect version to the supported range."""
    try:
        version = int(getattr(settings, "AUTO_DETECT_VERSION", 1))
    except Exception:
        version = 1
    return max(1, min(4, version))


def _resolve_seed_mode(raw_seed_mode: str | None) -> str:
    """Map request seed mode to internal modes (auto|grid|custom)."""
    mode = str(raw_seed_mode or "auto").lower()
    if mode in {"auto", "grid", "custom"}:
        return mode
    if mode in {"distance"}:
        return "auto"
    if mode in {"edges"}:
        return "grid"
    return "auto"


def _resolve_aoi_polygon(aoi_params: dict) -> Any:
    from processing.fields.tiling import bbox_to_polygon, point_radius_to_polygon, polygon_coords_to_polygon

    aoi_type = str(aoi_params.get("type", "point_radius"))
    if aoi_type == "bbox":
        bbox = aoi_params.get("bbox")
        if bbox is None:
            raise ValueError("aoi.bbox is required when aoi.type='bbox'")
        return bbox_to_polygon(bbox)
    if aoi_type == "polygon":
        polygon_coords = aoi_params.get("polygon")
        if polygon_coords is None:
            raise ValueError("aoi.polygon is required when aoi.type='polygon'")
        return polygon_coords_to_polygon(polygon_coords)

    lat = float(aoi_params.get("lat"))
    lon = float(aoi_params.get("lon"))
    radius_km = float(aoi_params.get("radius_km"))
    return point_radius_to_polygon(lat, lon, radius_km)


def _estimate_vpd_kpa(temperature_c: float | None, humidity_pct: float | None) -> float | None:
    if temperature_c is None or humidity_pct is None:
        return None
    try:
        temp = float(temperature_c)
        rh = max(1.0, min(100.0, float(humidity_pct)))
    except Exception:
        return None
    saturation = 0.6108 * math.exp((17.27 * temp) / (temp + 237.3))
    return max(0.0, float(saturation * (1.0 - rh / 100.0)))


def _wind_vector_from_speed_direction(
    speed_m_s: float | None,
    direction_deg: float | None,
) -> tuple[float | None, float | None]:
    if speed_m_s is None or direction_deg is None:
        return None, None
    try:
        speed = float(speed_m_s)
        direction = math.radians(float(direction_deg))
    except Exception:
        return None, None
    # Meteorological direction is where the wind comes from.
    u = -speed * math.sin(direction)
    v = -speed * math.cos(direction)
    return float(u), float(v)


def _fetch_openmeteo_snapshot(lat: float, lon: float) -> dict[str, float | None]:
    """Забрать погодный срез для визуальных слоёв. Ошибки не роняют пайплайн."""
    try:
        import httpx
    except Exception:
        return {}

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "soil_moisture_0_to_1cm",
            ]
        ),
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            current = dict(response.json().get("current") or {})
    except Exception as exc:
        logger.warning("grid_weather_snapshot_skipped", error=str(exc), latitude=lat, longitude=lon)
        return {}

    temperature_c = current.get("temperature_2m")
    humidity_pct = current.get("relative_humidity_2m")
    wind_speed = current.get("wind_speed_10m")
    wind_direction = current.get("wind_direction_10m")
    u_wind, v_wind = _wind_vector_from_speed_direction(wind_speed, wind_direction)
    return {
        "precipitation_mm": current.get("precipitation"),
        "wind_speed_m_s": wind_speed,
        "u_wind_10m": u_wind,
        "v_wind_10m": v_wind,
        "wind_direction_deg": wind_direction,
        "soil_moist": current.get("soil_moisture_0_to_1cm"),
        "vpd_mean": _estimate_vpd_kpa(temperature_c, humidity_pct),
    }


def _get_tile_transformer(tile_crs: str):
    """Return a cached Transformer for the given tile CRS -> EPSG:4326."""
    if str(tile_crs).upper() == "EPSG:4326":
        return None
    from pyproj import Transformer
    return Transformer.from_crs(tile_crs, "EPSG:4326", always_xy=True)


def _build_grid_cell_geom_4326(tile: dict, row0: int, row1: int, col0: int, col1: int, transformer=None):
    from shapely.geometry import box

    transform = tile["transform"]
    tile_crs = tile["crs"]

    x0, y0 = transform * (col0, row0)
    x1, y1 = transform * (col1, row1)
    geom = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if str(tile_crs).upper() == "EPSG:4326":
        return geom

    if transformer is None:
        from pyproj import Transformer
        transformer = Transformer.from_crs(tile_crs, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = geom.bounds
    llx, lly = transformer.transform(minx, miny)
    urx, ury = transformer.transform(maxx, maxy)
    return box(min(llx, urx), min(lly, ury), max(llx, urx), max(lly, ury))


def _build_utm_box_geom_4326(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    tile_crs: str,
    transformer=None,
):
    from shapely.geometry import box

    if transformer is None:
        transformer = _get_tile_transformer(tile_crs)
    llx, lly = transformer.transform(minx, miny)
    urx, ury = transformer.transform(maxx, maxy)
    return box(min(llx, urx), min(lly, ury), max(llx, urx), max(lly, ury))


def _masked_mean_or_none(arr: np.ndarray, mask: np.ndarray) -> float | None:
    values = np.asarray(arr)[mask]
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _build_grid_cells_for_tile(
    *,
    tile: dict,
    labels_clean: np.ndarray,
    gdf,
    ndvi_mean: np.ndarray,
    ndwi_mean: np.ndarray,
    ndmi_mean: np.ndarray,
    bsi_mean: np.ndarray,
    weather_snapshot: dict[str, float | None],
    zoom_level: int,
    cell_px: int,
    grid_origin_utm: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    if gdf is None or getattr(gdf, "empty", True):
        return []

    rows: list[dict[str, Any]] = []
    height, width = labels_clean.shape
    step = max(8, int(cell_px))
    field_binary = (labels_clean > 0).astype(np.float32)

    # Cache transformer once per tile instead of creating per cell
    transformer = _get_tile_transformer(tile["crs"])
    pixel_size_m = abs(float(tile["transform"].a))
    cell_size_m = float(step * pixel_size_m)
    grid_origin_x, grid_origin_y = grid_origin_utm if grid_origin_utm is not None else (None, None)

    # Pre-extract weather values (avoid repeated dict lookups)
    w_precip = weather_snapshot.get("precipitation_mm")
    w_wind_speed = weather_snapshot.get("wind_speed_m_s")
    w_u_wind = weather_snapshot.get("u_wind_10m")
    w_v_wind = weather_snapshot.get("v_wind_10m")
    w_wind_dir = weather_snapshot.get("wind_direction_deg")
    w_vpd = weather_snapshot.get("vpd_mean")
    w_soil = weather_snapshot.get("soil_moist")

    for row0 in range(0, height, step):
        row1 = min(height, row0 + step)
        for col0 in range(0, width, step):
            col1 = min(width, col0 + step)

            if grid_origin_utm is None:
                cell_geom = _build_grid_cell_geom_4326(tile, row0, row1, col0, col1, transformer=transformer)
                cell_row = int(row0 // step)
                cell_col = int(col0 // step)
            else:
                x0, y0 = tile["transform"] * (col0, row0)
                x1, y1 = tile["transform"] * (col1, row1)
                cell_minx = min(x0, x1)
                cell_maxx = max(x0, x1)
                cell_miny = min(y0, y1)
                cell_maxy = max(y0, y1)
                cx = 0.5 * (cell_minx + cell_maxx)
                cy = 0.5 * (cell_miny + cell_maxy)
                cell_col = int(math.floor(((cx - float(grid_origin_x)) / max(cell_size_m, 1e-6)) + 1e-9))
                cell_row = int(math.floor(((cy - float(grid_origin_y)) / max(cell_size_m, 1e-6)) + 1e-9))
                canonical_minx = float(grid_origin_x) + (cell_col * cell_size_m)
                canonical_miny = float(grid_origin_y) + (cell_row * cell_size_m)
                canonical_maxx = canonical_minx + cell_size_m
                canonical_maxy = canonical_miny + cell_size_m
                cell_geom = _build_utm_box_geom_4326(
                    canonical_minx,
                    canonical_miny,
                    canonical_maxx,
                    canonical_maxy,
                    tile_crs=tile["crs"],
                    transformer=transformer,
                )
            if cell_geom.is_empty:
                continue

            cell_block = field_binary[row0:row1, col0:col1]
            cell_area = cell_block.size
            field_coverage = float(cell_block.sum() / cell_area) if cell_area > 0 else 0.0

            # Inline masked_mean to avoid repeated isfinite recomputation
            ndvi_slice = ndvi_mean[row0:row1, col0:col1]
            ndwi_slice = ndwi_mean[row0:row1, col0:col1]
            ndmi_slice = ndmi_mean[row0:row1, col0:col1]
            bsi_slice = bsi_mean[row0:row1, col0:col1]

            rows.append(
                {
                    "geometry": cell_geom,
                    "zoom_level": int(zoom_level),
                    "row": cell_row,
                    "col": cell_col,
                    "field_coverage": round(field_coverage, 4),
                    "ndvi_mean": _masked_mean_or_none(ndvi_slice, np.isfinite(ndvi_slice)),
                    "ndwi_mean": _masked_mean_or_none(ndwi_slice, np.isfinite(ndwi_slice)),
                    "ndmi_mean": _masked_mean_or_none(ndmi_slice, np.isfinite(ndmi_slice)),
                    "bsi_mean": _masked_mean_or_none(bsi_slice, np.isfinite(bsi_slice)),
                    "precipitation_mm": w_precip,
                    "wind_speed_m_s": w_wind_speed,
                    "u_wind_10m": w_u_wind,
                    "v_wind_10m": w_v_wind,
                    "wind_direction_deg": w_wind_dir,
                    "gdd_sum": None,
                    "vpd_mean": w_vpd,
                    "soil_moist": w_soil,
                }
            )
    return rows


def _aggregate_grid_cell_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    numeric_fields = (
        "field_coverage",
        "ndvi_mean",
        "ndwi_mean",
        "ndmi_mean",
        "bsi_mean",
        "precipitation_mm",
        "wind_speed_m_s",
        "u_wind_10m",
        "v_wind_10m",
        "wind_direction_deg",
        "gdd_sum",
        "vpd_mean",
        "soil_moist",
    )

    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["zoom_level"]), int(row["row"]), int(row["col"]))
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                "geometry": row["geometry"],
                "zoom_level": int(row["zoom_level"]),
                "row": int(row["row"]),
                "col": int(row["col"]),
                "_values": {field: [] for field in numeric_fields},
            }
            grouped[key] = bucket
        for field in numeric_fields:
            value = row.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                bucket["_values"][field].append(float(value))

    aggregated: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        item = {
            "geometry": bucket["geometry"],
            "zoom_level": bucket["zoom_level"],
            "row": bucket["row"],
            "col": bucket["col"],
        }
        for field in numeric_fields:
            values = bucket["_values"][field]
            item[field] = round(float(np.mean(values)), 6) if values else None
        aggregated.append(item)
    return aggregated


def _iter_grid_zoom_levels(base_cell_px: int) -> list[tuple[int, int]]:
    base = max(8, int(base_cell_px))
    return [
        (0, base * 4),
        (1, base * 2),
        (2, base),
    ]


def _build_feature_stack_v4(
    *,
    edge_composite: np.ndarray,
    max_ndvi: np.ndarray,
    mean_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    ndwi_mean: np.ndarray,
    bsi_mean: np.ndarray,
    scl_valid_fraction: np.ndarray,
    rgb_r: np.ndarray,
    rgb_g: np.ndarray,
    rgb_b: np.ndarray,
    feature_channels: tuple[str, ...] | None = None,
    s1_vv_mean: np.ndarray | None = None,
    s1_vh_mean: np.ndarray | None = None,
    ndvi_entropy: np.ndarray | None = None,
    mndwi_max: np.ndarray | None = None,
    ndmi_mean: np.ndarray | None = None,
    ndwi_median: np.ndarray | None = None,
    green_median: np.ndarray | None = None,
    swir_median: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper for extracted feature stack assembly."""
    from processing.fields.feature_stack import build_feature_stack_v4

    return build_feature_stack_v4(
        edge_composite=edge_composite,
        max_ndvi=max_ndvi,
        mean_ndvi=mean_ndvi,
        ndvi_std=ndvi_std,
        ndwi_mean=ndwi_mean,
        bsi_mean=bsi_mean,
        scl_valid_fraction=scl_valid_fraction,
        rgb_r=rgb_r,
        rgb_g=rgb_g,
        rgb_b=rgb_b,
        feature_channels=feature_channels,
        s1_vv_mean=s1_vv_mean,
        s1_vh_mean=s1_vh_mean,
        ndvi_entropy=ndvi_entropy,
        mndwi_max=mndwi_max,
        ndmi_mean=ndmi_mean,
        ndwi_median=ndwi_median,
        green_median=green_median,
        swir_median=swir_median,
    )


def _tile_seed_points_from_lonlat(
    *,
    seed_points_lonlat: list[list[float]] | None,
    tile_transform,
    tile_crs: str,
) -> list[tuple[int, int]]:
    """Convert request seed points [lon,lat] to tile pixel coordinates (row,col)."""
    if not seed_points_lonlat:
        return []
    try:
        from pyproj import Transformer
        from rasterio.transform import rowcol
    except Exception:
        return []

    to_tile_crs = Transformer.from_crs("EPSG:4326", tile_crs, always_xy=True)
    out: list[tuple[int, int]] = []
    for point in seed_points_lonlat:
        if len(point) != 2:
            continue
        lon, lat = float(point[0]), float(point[1])
        x, y = to_tile_crs.transform(lon, lat)
        row, col = rowcol(tile_transform, x, y, op=round)
        out.append((int(row), int(col)))
    return out


def _save_debug_tile_dump(
    run_id_str: str,
    tile_id: str,
    payload: dict[str, np.ndarray],
) -> str:
    debug_dir = Path(__file__).resolve().parents[1] / "debug_runs" / run_id_str
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / f"{tile_id}.npz"
    np.savez_compressed(output_path, **payload)
    return str(output_path)


def _save_debug_vector_layer(
    run_id_str: str,
    tile_id: str,
    layer_name: str,
    gdf,
) -> str | None:
    """Persist a debug GeoPackage layer when geopandas is available."""
    if gdf is None or getattr(gdf, "empty", True):
        return None
    try:
        import geopandas as gpd  # noqa: F401
    except Exception:
        return None

    debug_dir = Path(__file__).resolve().parents[1] / "debug_runs" / run_id_str
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / f"{tile_id}_{layer_name}.gpkg"
    try:
        gdf.to_file(output_path, driver="GPKG")
    except Exception:
        logger.warning(
            "debug_vector_save_failed",
            tile_id=tile_id,
            layer=layer_name,
        )
        return None
    return str(output_path)


def _default_database_url_sync() -> str:
    return os.getenv("DATABASE_URL_SYNC") or "postgresql+psycopg://agromap:agromap_dev_password@localhost:5432/agromap"


def _mark_run_failed_best_effort(
    *,
    run_id: uuid.UUID,
    error: Exception | str,
    failure_stage: str,
) -> None:
    """Best-effort failure persistence for very early task crashes."""
    error_msg = str(error)[:1000]
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from storage.db import AoiRun

        engine = create_engine(_default_database_url_sync(), echo=False)
        try:
            with Session(engine) as session:
                run = session.get(AoiRun, run_id)
                if run is None:
                    return
                params = dict(run.params or {})
                runtime_meta = dict(params.get("runtime") or {})
                runtime_meta["error"] = error_msg
                runtime_meta["failure_stage"] = str(failure_stage or "unknown")
                params["runtime"] = _sanitize_json_floats(runtime_meta)
                run.params = params
                run.status = "failed"
                run.error_msg = error_msg
                run.progress = int(max(int(run.progress or 0), 0))
                session.commit()
        finally:
            engine.dispose()
    except Exception:
        logger.warning(
            "autodetect_mark_failed_best_effort_failed",
            aoi_run_id=str(run_id),
            failure_stage=failure_stage,
            exc_info=True,
        )


@celery.task(name="tasks.autodetect.run_autodetect", bind=True, max_retries=2)
def run_autodetect(self, run_id_str: str, use_sam: bool | None = None) -> dict:
    """Execute the full field detection pipeline.

    This runs synchronously inside a Celery worker.
    """
    configure_logging("INFO")
    run_id = uuid.UUID(run_id_str)

    logger.info("autodetect_start", aoi_run_id=run_id_str)
    ACTIVE_RUNS.inc()
    t_start = time.time()
    latency_breakdown: dict[str, float] = {
        "tiling_s": 0.0,
        "tiles_total_s": 0.0,
        "merge_tiles_s": 0.0,
        "db_insert_s": 0.0,
        "topology_cleanup_s": 0.0,
    }
    failure_stage = "config_validation"
    engine = None
    sentinel_client = None

    try:
        base_settings = get_settings()
        failure_stage = "db_engine_init"
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from storage.db import (
            ActiveLearningCandidate,
            AoiRun,
            Field,
            FieldDetectionCandidate,
            GridCell,
            LabelReview,
            LabelTask,
            LabelVersion,
            TileDiagnostic,
        )

        engine = create_engine(
            base_settings.DATABASE_URL_SYNC,
            echo=False,
        )

        with Session(engine) as session:
            failure_stage = "run_lookup"
            run = session.get(AoiRun, run_id)
            if run is None:
                logger.error("run_not_found", aoi_run_id=run_id_str)
                raise RuntimeError(f"Run not found: {run_id_str}")

            # Guard against infinite re-queuing caused by acks_late + reject_on_worker_lost:
            # each attempt increments a counter stored in run.params; fail hard after 3 attempts.
            _bootstrap_params = dict(run.params or {})
            _attempt = int((_bootstrap_params.get("runtime") or {}).get("worker_attempt", 0)) + 1
            _MAX_WORKER_ATTEMPTS = 3
            if _attempt > _MAX_WORKER_ATTEMPTS:
                logger.error(
                    "autodetect_max_attempts_exceeded",
                    aoi_run_id=run_id_str,
                    attempt=_attempt,
                    max_attempts=_MAX_WORKER_ATTEMPTS,
                )
                run.status = "failed"
                run.error_msg = f"Детект завершился с ошибкой после {_attempt - 1} попыток (worker был убит). Попробуйте уменьшить область или снизить нагрузку."
                session.commit()
                return {"status": "error", "message": "Max worker attempts exceeded"}
            _bootstrap_runtime = dict((_bootstrap_params.get("runtime") or {}))
            _bootstrap_runtime["worker_attempt"] = _attempt
            _bootstrap_params["runtime"] = _bootstrap_runtime
            run.params = _bootstrap_params

            failure_stage = "run_bootstrap"
            run.status = "running"
            run.progress = int(max(int(run.progress or 0), 5))
            session.commit()

            params = dict(run.params or {})
            settings = _apply_runtime_config(base_settings, params.get("config"))
            detect_preset = str((params.get("config") or {}).get("preset") or "standard").strip().lower()
            detect_pipeline_profile = _resolve_detect_pipeline_profile(settings, detect_preset)
            preset_runtime_overrides: dict[str, Any] = {}
            if detect_preset == "fast":
                preset_runtime_overrides = {
                    "SOUTH_COMPONENT_BRIDGE_ENABLED": False,
                    "SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS": 0,
                    "POST_MERGE_MAX_COMPONENTS": min(
                        int(getattr(settings, "POST_MERGE_MAX_COMPONENTS", 2000)),
                        int(getattr(settings, "POST_MERGE_MAX_COMPONENTS_FAST", 512)),
                    ),
                    "ENABLE_CANDIDATE_RANKER": False,
                    "FEATURE_ML_PRIMARY": False,
                }
            elif detect_preset == "standard":
                preset_runtime_overrides = {
                    "SOUTH_COMPONENT_BRIDGE_ENABLED": True,
                    "SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS": min(int(getattr(settings, "SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS", 400)), 240),
                    "POST_MERGE_MAX_COMPONENTS": min(int(getattr(settings, "POST_MERGE_MAX_COMPONENTS", 2000)), 1200),
                }
            elif detect_preset == "quality":
                preset_runtime_overrides = {
                    "SOUTH_COMPONENT_BRIDGE_ENABLED": True,
                    "SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS": min(int(getattr(settings, "SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS", 400)), 320),
                }
            if preset_runtime_overrides:
                settings = settings.model_copy(update=preset_runtime_overrides)
            detect_pipeline_profile = _resolve_detect_pipeline_profile(settings, detect_preset)
            use_sam_override = (
                bool(use_sam) if use_sam is not None else params.get("use_sam")
            )
            if use_sam_override is not None:
                use_sam_flag = bool(use_sam_override)
                settings = settings.model_copy(
                    update={
                        "FRAMEWORK_SAM_ENABLED": use_sam_flag,
                        "FEATURE_SAM2_PRIMARY": use_sam_flag,
                    }
                )
            auto_detect_version = _resolve_auto_detect_version(settings)
            aoi_params = dict(params.get("aoi", {}))
            debug_run = bool(params.get("debug"))
            aoi_params.setdefault("type", "point_radius")
            aoi_params.setdefault("lat", settings.DEFAULT_CENTER_LAT)
            aoi_params.setdefault("lon", settings.DEFAULT_CENTER_LON)
            aoi_params.setdefault("radius_km", settings.DEFAULT_RADIUS_KM)
            aoi_poly = _resolve_aoi_polygon(aoi_params)
            aoi_lat = float(getattr(aoi_poly.centroid, "y", aoi_params.get("lat", settings.DEFAULT_CENTER_LAT)))
            aoi_lon = float(getattr(aoi_poly.centroid, "x", aoi_params.get("lon", settings.DEFAULT_CENTER_LON)))
            region_band = resolve_region_band(aoi_lat, settings)
            region_boundary_profile = resolve_region_boundary_profile(aoi_lat, settings)
            season_start_mmdd, season_end_mmdd = get_adaptive_season_window(aoi_lat, settings)

            resolution_m = float(params.get("resolution_m", settings.DEFAULT_RESOLUTION_M))
            px_area_m2 = get_px_area_m2(resolution_m)
            settings = settings.model_copy(update={"POST_PX_AREA_M2": int(round(px_area_m2))})

            max_cloud_pct = params.get("max_cloud_pct", settings.MAX_CLOUD_PCT)
            min_field_area_ha = params.get("min_field_area_ha", settings.MIN_FIELD_AREA_HA)
            target_slices = max(
                1,
                int(
                    params.get(
                        "target_dates",
                        getattr(settings, "DATE_SELECTION_TARGET_DATES", settings.S2_TEMPORAL_SLICES),
                    )
                ),
            )
            seed_mode = _resolve_seed_mode(params.get("seed_mode"))
            seed_points_lonlat = params.get("seed_points")
            credentials_missing = not settings.SH_CLIENT_ID or not settings.SH_CLIENT_SECRET
            using_synthetic_data = credentials_missing and settings.ALLOW_SYNTHETIC_DATA and debug_run
            north_s1_required = detect_preset in {"standard", "quality"} and region_band == "north"
            preset_tta_mode = "none"
            multi_scale_scales: tuple[float, ...] = (1.0,)
            if detect_preset == "standard":
                preset_tta_mode = str(getattr(settings, "ML_TTA_STANDARD_MODE", "flip2"))
                if bool(getattr(settings, "ML_MULTI_SCALE_STANDARD", False)):
                    aux_scales = tuple(float(v) for v in getattr(settings, "ML_MULTI_SCALE_AUX_SCALES", (0.75,)))
                    multi_scale_scales = (1.0,) + tuple(v for v in aux_scales if v > 0 and abs(v - 1.0) > 1e-6)
            elif detect_preset == "quality":
                preset_tta_mode = str(getattr(settings, "ML_TTA_QUALITY_MODE", "rotate4"))
                if bool(getattr(settings, "ML_MULTI_SCALE_QUALITY", True)):
                    aux_scales = tuple(float(v) for v in getattr(settings, "ML_MULTI_SCALE_AUX_SCALES", (0.75,)))
                    multi_scale_scales = (1.0,) + tuple(v for v in aux_scales if v > 0 and abs(v - 1.0) > 1e-6)
            s1_enabled_for_run = bool(
                detect_preset == "quality"
                or north_s1_required
                or (detect_preset == "standard" and bool(getattr(settings, "FEATURE_S1_FUSION", False)))
            )
            if detect_preset == "fast":
                s1_enabled_for_run = False
            settings = settings.model_copy(
                update={
                    "FEATURE_S1_FUSION": bool(s1_enabled_for_run),
                    "S1_ENABLED": bool(s1_enabled_for_run),
                }
            )

            runtime_meta = {
                "using_synthetic_data": using_synthetic_data,
                "debug_run": debug_run,
                "preset": detect_preset,
                "pipeline_profile": str(detect_pipeline_profile["name"]),
                "preview_only": bool(detect_pipeline_profile["preview_only"]),
                "output_mode": str(detect_pipeline_profile.get("output_mode") or "field_boundaries"),
                "operational_eligible": bool(detect_pipeline_profile.get("operational_eligible", True)),
                "max_radius_km": 40 if detect_preset == "fast" else 8 if detect_preset == "quality" else 20,
                "recommended_radius_km": 30 if detect_preset == "fast" else 8 if detect_preset == "quality" else 20,
                "enabled_stages": list(detect_pipeline_profile["enabled_stages"]),
                "auto_detect_version": auto_detect_version,
                "framework": {
                    "sam_enabled": bool(getattr(settings, "FRAMEWORK_SAM_ENABLED", True)),
                    "sam_field_det": bool(getattr(settings, "FRAMEWORK_SAM_FIELD_DET", True)),
                    "weak_worldcover": bool(getattr(settings, "FRAMEWORK_USE_WEAK_WORLDCOVER", True)),
                    "skip_watershed_for_largest": bool(
                        getattr(settings, "FRAMEWORK_SKIP_WATERSHED_FOR_LARGEST", True)
                    ),
                },
                "sentinel_credentials_missing": credentials_missing,
                "season_start": season_start_mmdd,
                "season_end": season_end_mmdd,
                "time_windows": [],
                "selected_dates": [],
                "selected_scene_signature": "",
                "low_quality_input": False,
                "date_selection_profile": str(
                    getattr(settings, "DATE_SELECTION_PROFILE", "adaptive_region")
                ),
                "mode": str(getattr(settings, "MODE", "production")).strip().lower(),
                "date_selection_region_band": region_band,
                "date_selection_low_confidence": False,
                "selected_date_confidence_mean": 0.0,
                "region_band": region_band,
                "region_boundary_profile": region_boundary_profile,
                "region_profile_reason": "latitude_adaptive",
                "region_profile_version": "v1",
                "regional_quality_target": _regional_quality_target(region_boundary_profile),
                "region_profile_actions": [],
                "warnings": [],
                "tiles": [],
                "fallback_rate_tile": 0.0,
                "quality_gate_failed": False,
                "weak_label_source": "unknown",
                "ml_primary_used": False,
                "model_backend": "unknown",
                "boundary_quality_profile": str(
                    getattr(settings, "BOUNDARY_QUALITY_PROFILE", "quality_first")
                ),
                "hydro_boundary_profile": str(
                    getattr(settings, "HYDRO_BOUNDARY_PROFILE", "water_aware")
                ),
                "sam_runtime_mode": (
                    "disabled"
                    if str(getattr(settings, "SAM_RUNTIME_POLICY", "safe_optional")).strip().lower()
                    == "disabled"
                    else "fallback_non_sam"
                ),
                "sam_failure_reason": None,
                "sam_memory_budget_mb": float(getattr(settings, "SAM_MAX_EST_MEMORY_MB", 2200)),
                "progress_stage": "fetch",
                "stage_progress_pct": 0.0,
                "tile_progress_pct": 0.0,
                "water_edge_risk_detected": False,
                "road_drift_risk_detected": False,
                "n_valid_scenes": 0,
                "edge_signal_p90": 0.0,
                "ml_quality_score": 0.0,
                "fusion_profile": "none",
                "area_change_post_smooth": 0.0,
                "lambda_edge": float(settings.WATERSHED_LAMBDA),
                "lambda_edge_candidates": [float(v) for v in settings.WATERSHED_LAMBDA_CANDIDATES],
                "watershed_min_distance": int(settings.WATERSHED_MIN_DISTANCE),
                "seed_mode": seed_mode,
                "pixel_area_m2": float(px_area_m2),
                "ab_use_sam": bool(use_sam_override) if use_sam_override is not None else None,
                "tta_mode": preset_tta_mode,
                "multi_scale_scales": [float(v) for v in multi_scale_scales],
                "s1_planned": bool(s1_enabled_for_run),
                "sentinel_account_used": "primary",
                "sentinel_failover_level": 0,
                "sentinel_band_contract": "unknown",
                "latency_breakdown_s": dict(latency_breakdown),
                "failure_stage": None,
                "candidate_branch_counts": {},
                "candidate_reject_summary": {},
                "candidates_total": 0,
                "candidates_kept": 0,
            }
            if detect_preset == "fast":
                runtime_meta["warnings"].append("south_bridge disabled for fast preset")
                runtime_meta["warnings"].append("fast preset uses preview-only simplified pipeline")
            elif detect_preset in {"standard", "quality"}:
                runtime_meta["warnings"].append(
                    f"south_bridge capped at {int(getattr(settings, 'SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS', 0))} components"
                )
            logger.info(
                "autodetect_runtime_mode",
                aoi_run_id=run_id_str,
                using_synthetic_data=using_synthetic_data,
                preset=detect_preset,
                debug_run=debug_run,
                sentinel_credentials_missing=credentials_missing,
            )
            # Restore progress floor from prior worker attempt (Celery re-queue after OOM)
            # Without this, runtime_meta["progress_pct"] starts at 0 and max() can go backward.
            _prior_runtime = dict((run.params or {}).get("runtime") or {})
            _prior_pct = float(_prior_runtime.get("progress_pct") or 0.0)
            if _prior_pct > 0.0:
                runtime_meta["progress_pct"] = _prior_pct

            _persist_runtime_meta(run, runtime_meta)
            session.commit()

            if credentials_missing and not using_synthetic_data:
                raise RuntimeError("Sentinel Hub credentials are required; synthetic mode is for tests only")

            time_windows = _build_time_windows(
                run.time_start,
                run.time_end,
                target_slices=target_slices,
                season_start_mmdd=season_start_mmdd,
                season_end_mmdd=season_end_mmdd,
            )
            if not time_windows:
                raise RuntimeError("No Sentinel-2 time windows remain after applying the configured season")
            runtime_meta["time_windows"] = [
                {"time_from": time_from, "time_to": time_to}
                for time_from, time_to in time_windows
            ]
            _persist_runtime_meta(run, runtime_meta)
            session.commit()

            # Step 1: Build AOI and tiles
            failure_stage = "tiling"
            t_step = time.time()
            from processing.fields.tiling import make_tiles

            weather_snapshot = _fetch_openmeteo_snapshot(aoi_lat, aoi_lon)
            grid_cell_rows: list[dict[str, Any]] = []
            total_grid_cells_inserted = 0
            # Prefer per-preset tile_size_px stored by preflight; fall back to settings.TILE_SIZE_PX
            preflight_tile_size_px = int(
                (params.get("config") or {}).get("preflight", {}).get("tile_size_px")
                or settings.TILE_SIZE_PX
            )
            tiles = make_tiles(
                aoi_poly,
                tile_size_m=preflight_tile_size_px * resolution_m,
                overlap_m=settings.TILE_OVERLAP_M,
                resolution_m=resolution_m,
            )
            grid_origin_utm = (
                min(float(tile["bbox_utm"][0]) for tile in tiles),
                min(float(tile["bbox_utm"][1]) for tile in tiles),
            ) if tiles else None
            runtime_meta["grid_origin_utm"] = (
                {"x": round(float(grid_origin_utm[0]), 3), "y": round(float(grid_origin_utm[1]), 3)}
                if grid_origin_utm is not None
                else None
            )

            tiling_time = time.time() - t_step
            STEP_DURATION.labels(step="tiling").observe(tiling_time)
            latency_breakdown["tiling_s"] = round(float(tiling_time), 3)
            logger.info("tiling_done", n_tiles=len(tiles), aoi_run_id=run_id_str)

            # WorldCover exclusion prior
            wc_prior = None
            if settings.USE_WORLDCOVER_PRIOR:
                from processing.priors.worldcover import WorldCoverPrior
                wc_prior = WorldCoverPrior(
                    year=settings.WORLDCOVER_YEAR,
                    cache_dir=Path(settings.PRIORS_CACHE_DIR),
                    exclude_classes=settings.WC_EXCLUDE_CLASSES,
                )
                logger.info("worldcover_prior_enabled", year=settings.WORLDCOVER_YEAR)

            run.progress = int(max(int(run.progress or 0), 10))
            session.commit()

            # Step 2-9: Process each tile
            failure_stage = "tile_processing"
            _event_loop = None  # Reusable event loop for async calls
            if not using_synthetic_data:
                from providers.sentinelhub.client import SentinelHubClient

                sentinel_client = SentinelHubClient()
                _event_loop = asyncio.new_event_loop()

            from processing.fields.indices import compute_all_indices
            from processing.fields.composite import (
                build_valid_mask_from_scl,
                compute_phenometrics,
                select_dates_adaptive,
                select_dates_by_coverage,
                build_median_composite,
            )
            from processing.fields.phenoclassify import (
                WATER,
                CROP,
                FOREST,
                PhenoThresholds,
                classify_land_cover,
                compute_hydro_masks,
            )
            from processing.fields.edge_composite import (
                compute_edge_stats,
            )
            from processing.fields.temporal_composite import build_multiyear_composite
            from processing.fields.segmentation import watershed_segment
            from processing.fields.obia_filter import filter_segments, filter_segments_preview
            from processing.fields.postprocess import run_postprocess
            from processing.fields.vectorize import (
                merge_tile_polygons,
                polygonize_labels,
                summarize_polygon_areas,
            )

            ml_inferencer = None
            ml_primary_requested = bool(getattr(settings, "FEATURE_ML_PRIMARY", True))
            if ml_primary_requested:
                try:
                    from processing.fields.ml_inference import FieldBoundaryInferencer

                    use_onnx_requested = bool(getattr(settings, "ML_USE_ONNX", True))
                    requested_model_path = Path(
                        str(getattr(settings, "ML_MODEL_PATH", "models/boundary_unet_v1.onnx"))
                    )
                    norm_path = Path(
                        str(
                            getattr(
                                settings,
                                "ML_MODEL_NORM_STATS_PATH",
                                "models/boundary_unet_v1.norm.json",
                            )
                        )
                    )

                    backend_root = Path(__file__).resolve().parents[1]
                    project_root = backend_root.parent

                    def _resolve_local_path(path_value: Path) -> Path:
                        if path_value.is_absolute():
                            return path_value
                        local_candidate = backend_root / path_value
                        if local_candidate.exists():
                            return local_candidate
                        project_candidate = project_root / path_value
                        if project_candidate.exists():
                            return project_candidate
                        return local_candidate

                    model_candidates: list[Path] = []
                    if use_onnx_requested:
                        if requested_model_path.suffix.lower() == ".onnx":
                            model_candidates.append(requested_model_path)
                            model_candidates.append(requested_model_path.with_suffix(".pth"))
                        else:
                            model_candidates.append(requested_model_path.with_suffix(".onnx"))
                            model_candidates.append(requested_model_path)
                    else:
                        if requested_model_path.suffix.lower() == ".pth":
                            model_candidates.append(requested_model_path)
                            model_candidates.append(requested_model_path.with_suffix(".onnx"))
                        else:
                            model_candidates.append(requested_model_path.with_suffix(".pth"))
                            model_candidates.append(requested_model_path)

                    discover_dirs: list[Path] = []
                    requested_parent = _resolve_local_path(requested_model_path).parent
                    for candidate_dir in (requested_parent, backend_root / "models", backend_root):
                        if candidate_dir.exists() and candidate_dir not in discover_dirs:
                            discover_dirs.append(candidate_dir)
                    preferred_suffixes = [".onnx", ".pth"] if use_onnx_requested else [".pth", ".onnx"]
                    for candidate_dir in discover_dirs:
                        for suffix in preferred_suffixes:
                            for discovered in sorted(candidate_dir.glob(f"boundary_unet*{suffix}")):
                                model_candidates.append(discovered)

                    resolved_candidates: list[Path] = []
                    for candidate in model_candidates:
                        resolved = _resolve_local_path(candidate)
                        if resolved not in resolved_candidates:
                            resolved_candidates.append(resolved)
                    for candidate in resolved_candidates:
                        if (
                            candidate.suffix.lower() == ".onnx"
                            and not candidate.exists()
                            and candidate.with_suffix(".onnx.data").exists()
                        ):
                            logger.warning(
                                "ml_model_graph_missing_but_external_data_found",
                                expected_onnx=str(candidate),
                                external_data=str(candidate.with_suffix(".onnx.data")),
                            )
                    existing_model = next((candidate for candidate in resolved_candidates if candidate.exists()), None)
                    model_path = existing_model or resolved_candidates[0]

                    if not norm_path.is_absolute():
                        local_norm = Path(__file__).resolve().parents[1] / norm_path
                        norm_path = local_norm if local_norm.exists() else norm_path

                    ml_inferencer = FieldBoundaryInferencer(
                        str(model_path),
                        norm_stats_path=str(norm_path),
                        device=str(getattr(settings, "ML_INFERENCE_DEVICE", "auto")),
                        use_onnx=bool(getattr(settings, "ML_USE_ONNX", True)),
                        feature_profile=str(getattr(settings, "ML_FEATURE_PROFILE", "v2_16ch")),
                    )
                    runtime_meta["ml_primary"] = {
                        "enabled": True,
                        "model_path": str(model_path),
                        "model_candidates": [str(path) for path in resolved_candidates],
                        "model_backend": ml_inferencer.backend,
                        "feature_profile": str(getattr(ml_inferencer, "feature_profile", "unknown")),
                        "channels": list(getattr(ml_inferencer, "feature_channels", [])),
                        "norm_stats_path": str(norm_path),
                        "ml_use_onnx_requested": use_onnx_requested,
                        "score_threshold": float(getattr(settings, "ML_SCORE_THRESHOLD", 0.35)),
                    }
                    runtime_meta["model_backend"] = ml_inferencer.backend
                    runtime_meta["weak_label_source"] = (
                        str(ml_inferencer.metadata.get("weak_label_source") or "unknown")
                        if isinstance(getattr(ml_inferencer, "metadata", None), dict)
                        else "unknown"
                    )
                    logger.info(
                        "ml_primary_enabled",
                        model_path=str(model_path),
                        backend=ml_inferencer.backend,
                        ml_use_onnx_requested=use_onnx_requested,
                        score_threshold=float(getattr(settings, "ML_SCORE_THRESHOLD", 0.35)),
                    )
                except Exception as exc:
                    runtime_meta["ml_primary"] = {
                        "enabled": False,
                        "error": str(exc),
                    }
                    runtime_meta["model_backend"] = "unknown"
                    runtime_meta["weak_label_source"] = "unknown"
                    logger.warning("ml_primary_disabled", error=str(exc))
            else:
                runtime_meta["ml_primary"] = {"enabled": False}
                runtime_meta["model_backend"] = "unknown"
                runtime_meta["weak_label_source"] = "unknown"

            unet_model = None
            predict_edge_map_fn = None
            if bool(getattr(settings, "FEATURE_UNET_EDGE", True)):
                try:
                    from processing.fields.unet_edge import UNetEdgeDetector, predict_edge_map

                    unet_model = UNetEdgeDetector.load_pretrained(
                        str(getattr(settings, "UNET_EDGE_MODEL", "")),
                        device=str(getattr(settings, "UNET_DEVICE", "cpu")),
                    )
                    predict_edge_map_fn = predict_edge_map
                except Exception as exc:
                    logger.warning("unet_edge_preload_failed", error=str(exc))

            sam_primary_segmentor = None
            sam_masks_to_label = None
            if bool(getattr(settings, "FEATURE_SAM2_PRIMARY", True)) and auto_detect_version >= 3:
                try:
                    from processing.fields.sam_primary import (
                        SAM2FieldSegmentor,
                        masks_to_label_raster,
                    )

                    sam_primary_segmentor = SAM2FieldSegmentor(
                        checkpoint_path=str(getattr(settings, "SAM2_CHECKPOINT", "")),
                        device=str(getattr(settings, "UNET_DEVICE", "cpu")),
                    )
                    sam_masks_to_label = masks_to_label_raster
                except Exception as exc:
                    logger.warning("sam2_primary_preload_failed", error=str(exc))

            # Adaptive phenological thresholds based on AOI centroid latitude
            from core.config import get_adaptive_pheno_thresholds

            aoi_centroid_lat = float(aoi_lat)
            pheno_overrides = get_adaptive_pheno_thresholds(aoi_centroid_lat, settings)
            runtime_meta["date_selection_region_band"] = region_band

            thresholds = PhenoThresholds(
                ndwi_water=pheno_overrides.get("PHENO_NDWI_WATER", settings.PHENO_NDWI_WATER),
                mndwi_water=pheno_overrides.get("PHENO_MNDWI_WATER", settings.PHENO_MNDWI_WATER),
                bsi_built=pheno_overrides.get("PHENO_BSI_BUILT", settings.PHENO_BSI_BUILT),
                std_built=pheno_overrides.get("PHENO_STD_BUILT", settings.PHENO_STD_BUILT),
                ndvi_forest_min=pheno_overrides.get("PHENO_NDVI_FOREST_MIN", settings.PHENO_NDVI_FOREST_MIN),
                delta_forest=pheno_overrides.get("PHENO_DELTA_FOREST", settings.PHENO_DELTA_FOREST),
                ndvi_grass_mean=pheno_overrides.get("PHENO_NDVI_GRASS_MEAN", settings.PHENO_NDVI_GRASS_MEAN),
                delta_grass=pheno_overrides.get("PHENO_DELTA_GRASS", settings.PHENO_DELTA_GRASS),
                ndvi_crop_max=pheno_overrides.get("PHENO_NDVI_CROP_MAX", settings.PHENO_NDVI_CROP_MAX),
                ndvi_crop_min=pheno_overrides.get("PHENO_NDVI_CROP_MIN", settings.PHENO_NDVI_CROP_MIN),
                delta_crop=pheno_overrides.get("PHENO_DELTA_CROP", settings.PHENO_DELTA_CROP),
                msi_crop=pheno_overrides.get("PHENO_MSI_CROP", settings.PHENO_MSI_CROP),
                n_valid_min=settings.PHENO_N_VALID_MIN,
            )
            if pheno_overrides:
                logger.info(
                    "adaptive_pheno_thresholds",
                    lat=aoi_centroid_lat,
                    overrides=pheno_overrides,
                    aoi_run_id=run_id_str,
                )

            all_gdfs = []
            selected_date_tokens: set[str] = set()

            for i, tile in enumerate(tiles):
                tile_t0 = time.time()
                logger.info("tile_start", tile_id=tile["tile_id"], aoi_run_id=run_id_str)

                h, w = tile["shape"]
                n_dates = target_slices
                tile_runtime = {
                    "tile_id": tile["tile_id"],
                    "bbox": list(tile["bbox_4326"]) if tile.get("bbox_4326") is not None else None,
                    "pipeline_profile": str(detect_pipeline_profile["name"]),
                    "preview_only": bool(detect_pipeline_profile["preview_only"]),
                    "output_mode": str(detect_pipeline_profile.get("output_mode") or "field_boundaries"),
                    "operational_eligible": bool(detect_pipeline_profile.get("operational_eligible", True)),
                    "enabled_stages": list(detect_pipeline_profile["enabled_stages"]),
                    "scene_windows": [],
                    "selected_dates": [],
                    "selected_scene_signature": "",
                    "low_quality_input": False,
                    "low_confidence_reason": None,
                    "debug_artifact": None,
                    "ml_primary_used": False,
                    "ml_quality_score": 0.0,
                    "tta_consensus": None,
                    "boundary_uncertainty": None,
                    "geometry_confidence": None,
                    "tta_extent_disagreement": None,
                    "tta_boundary_disagreement": None,
                    "uncertainty_source": "tta_unavailable",
                    "fusion_profile": "none",
                    "fallback_rate_tile": 0.0,
                    "quality_gate_failed": False,
                    "weak_label_source": runtime_meta.get("weak_label_source"),
                    "model_backend": runtime_meta.get("model_backend"),
                    "geometry_refine_profile": str(
                        getattr(settings, "GEOMETRY_REFINE_PROFILE", "balanced")
                    ),
                    "ml_extent_threshold": None,
                    "contour_shrink_ratio": 1.0,
                    "centroid_shift_m": 0.0,
                    "road_barrier_retry_used": False,
                    "road_snap_reject_used": False,
                    "hydro_rescue_used": False,
                    "open_water_pixels": 0,
                    "seasonal_wet_pixels": 0,
                    "riparian_soft_pixels": 0,
                    "riparian_hard_pixels": 0,
                    "water_edge_overlap_ratio": 0.0,
                    "road_edge_overlap_ratio": 0.0,
                    "boundary_shift_to_road_ratio": 0.0,
                    "region_profile_applied": region_boundary_profile,
                    "region_profile_actions": [],
                    "split_risk_score": 0.0,
                    "shrink_risk_score": 0.0,
                    "components_after_clean": 0,
                    "components_after_grow": 0,
                    "components_after_gap_close": 0,
                    "components_after_infill": 0,
                    "components_after_merge": 0,
                    "components_after_watershed": 0,
                    "watershed_applied": False,
                    "watershed_skipped_reason": None,
                    "watershed_rollback_reason": None,
                    "split_score_p50": 0.0,
                    "split_score_p90": 0.0,
                    "pre_vector_area_px": 0,
                    "post_vector_area_m2": 0.0,
                    "post_smooth_area_m2": 0.0,
                    "area_change_post_smooth": 0.0,
                    "vectorize_area_ratio": 1.0,
                    "smooth_area_ratio": 1.0,
                    "n_valid_scenes": 0,
                    "edge_signal_p90": 0.0,
                    "sam_tile_mode": "skipped",
                    "sam_runtime_mode": "fallback_non_sam",
                    "sam_peak_mem_estimate_mb": 0.0,
                    "sam_elapsed_s": 0.0,
                    "sam_polygons_before_filter": 0,
                    "sam_polygons_after_filter": 0,
                    "postprocess_stage_pixels": {},
                    "geometry_diagnostics": {},
                    "processing_profile": "normal",
                    "recovery_second_pass_used": False,
                    "candidate_branch_counts": {},
                    "candidate_reject_summary": {},
                    "candidates_total": 0,
                    "candidates_kept": 0,
                }
                result = None
                bands = None
                scl = None
                valid_mask = None
                indices = None
                valid_sel = None
                ndvi_sel = None
                ndwi_sel = None
                mndwi_sel = None
                edge_bands = None
                temporal_composite = None
                edge_composite = None
                ndvi_std = None
                feature_stack = None
                extent_prob = None
                boundary_prob = None
                distance_prob = None
                labels = None
                labels_clean = None
                sam_input = None
                sam_raw = None
                sam_filtered = None
                ranked_candidates: list[Any] = []
                candidate_branches: list[Any] = []
                processing_profile = {"name": "normal", "enable_second_pass": False, "allow_degraded_output": False}
                processing_settings = settings
                ndre_sel = None
                ndre_post = None

                if using_synthetic_data:
                    t_count = len(time_windows)
                    bands = {
                        "B2": np.random.uniform(0.02, 0.08, (t_count, h, w)).astype(np.float32),
                        "B3": np.random.uniform(0.03, 0.10, (t_count, h, w)).astype(np.float32),
                        "B4": np.random.uniform(0.02, 0.09, (t_count, h, w)).astype(np.float32),
                        "B5": np.random.uniform(0.03, 0.12, (t_count, h, w)).astype(np.float32),
                        "B6": np.random.uniform(0.04, 0.15, (t_count, h, w)).astype(np.float32),
                        "B7": np.random.uniform(0.05, 0.18, (t_count, h, w)).astype(np.float32),
                        "B8": np.random.uniform(0.15, 0.50, (t_count, h, w)).astype(np.float32),
                        "B8A": np.random.uniform(0.14, 0.46, (t_count, h, w)).astype(np.float32),
                        "B11": np.random.uniform(0.08, 0.25, (t_count, h, w)).astype(np.float32),
                        "B12": np.random.uniform(0.05, 0.20, (t_count, h, w)).astype(np.float32),
                    }
                    scl = np.full((t_count, h, w), 4, dtype=np.uint8)
                    for t in range(t_count):
                        field_factor = 0.3 + 0.5 * np.sin(np.pi * t / t_count)
                        for fy in range(0, h, 200):
                            for fx in range(0, w, 250):
                                fh = np.random.randint(80, 160)
                                fw = np.random.randint(100, 200)
                                ey, ex = min(fy + fh, h), min(fx + fw, w)
                                bands["B8"][t, fy:ey, fx:ex] *= (1.0 + field_factor)
                                bands["B4"][t, fy:ey, fx:ex] *= (1.0 - 0.3 * field_factor)
                else:
                    logger.info(
                        "sentinel_time_windows",
                        tile_id=tile["tile_id"],
                        n_windows=len(time_windows),
                        date_from=time_windows[0][0],
                        date_to=time_windows[-1][1],
                        aoi_run_id=run_id_str,
                    )
                    fetch_progress_start = 0.02
                    fetch_progress_end = 0.15

                    def _fetch_progress_callback(
                        completed_windows: int,
                        total_windows: int,
                        time_from: str,
                        time_to: str,
                    ) -> None:
                        safe_total_windows = max(int(total_windows), 1)
                        safe_completed = min(max(int(completed_windows), 0), safe_total_windows)
                        fraction = fetch_progress_start + (
                            (fetch_progress_end - fetch_progress_start)
                            * (safe_completed / safe_total_windows)
                        )
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "fetch",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=fraction,
                            detail=f"windows {safe_completed}/{safe_total_windows} · {time_from} → {time_to}",
                            force_commit=True,
                        )

                    use_rededge_evalscript = bool(getattr(settings, "USE_REDEDGE_EVALSCRIPT", True))
                    result = _event_loop.run_until_complete(
                        sentinel_client.fetch_multitemporal_harmonized(
                            tile["bbox_4326"],
                            time_windows,
                            w,
                            h,
                            max_cloud_pct=max_cloud_pct,
                            progress_callback=_fetch_progress_callback,
                            prefer_v4=use_rededge_evalscript,
                        )
                    )
                    tile_runtime["sentinel_account_used"] = getattr(sentinel_client, "last_account_alias", "primary")
                    tile_runtime["sentinel_failover_level"] = int(getattr(sentinel_client, "last_failover_level", 0))
                    tile_runtime["sentinel_band_contract"] = (
                        "harmonized_v4" if use_rededge_evalscript else "harmonized_raw"
                    )
                    runtime_meta["sentinel_account_used"] = tile_runtime["sentinel_account_used"]
                    runtime_meta["sentinel_failover_level"] = tile_runtime["sentinel_failover_level"]
                    runtime_meta["sentinel_band_contract"] = tile_runtime["sentinel_band_contract"]
                    required_bands = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
                    missing = [band for band in required_bands if band not in result]
                    if missing:
                        raise RuntimeError(
                            f"Sentinel response missing required bands for tile {tile['tile_id']}: {missing}"
                        )
                    bands = {
                        band: np.asarray(result[band], dtype=np.float32)
                        for band in required_bands
                    }
                    if "SCL" in result:
                        scl = np.asarray(result["SCL"], dtype=np.uint8)
                    else:
                        fallback_t = int(next(iter(bands.values())).shape[0])
                        scl = np.full((fallback_t, h, w), 4, dtype=np.uint8)
                    t_count = scl.shape[0]
                    if t_count == 0:
                        raise RuntimeError(f"No Sentinel scenes available for tile {tile['tile_id']}")
                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "fetch",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.15,
                    detail=f"windows {t_count}/{max(len(time_windows), 1)}",
                )
                valid_mask = build_valid_mask_from_scl(scl)
                indices = compute_all_indices(bands)
                if str(getattr(settings, "DATE_SELECTION_PROFILE", "adaptive_region")).strip().lower() == "adaptive_region":
                    selected, selection_meta = select_dates_adaptive(
                        valid_mask,
                        indices,
                        time_windows,
                        aoi_centroid_lat,
                        n_dates,
                        int(getattr(settings, "DATE_SELECTION_MIN_GOOD_DATES", settings.S2_MIN_GOOD_DATES)),
                        settings,
                        return_metadata=True,
                    )
                else:
                    selected, selection_meta = select_dates_by_coverage(
                        valid_mask,
                        min_valid_pct=float(
                            getattr(settings, "DATE_SELECTION_MIN_VALID_PCT", 0.5)
                        ),
                        n_dates=n_dates,
                        min_good_dates=int(
                            getattr(settings, "DATE_SELECTION_MIN_GOOD_DATES", settings.S2_MIN_GOOD_DATES)
                        ),
                        return_metadata=True,
                    )

                scene_coverages = valid_mask.reshape(t_count, -1).mean(axis=1)
                selected_idx = {int(idx) for idx in selected.tolist()}
                tile_runtime["selected_date_confidence_mean"] = round(
                    float(selection_meta.get("selected_date_confidence_mean") or 0.0),
                    4,
                )
                runtime_meta["selected_date_confidence_mean"] = tile_runtime["selected_date_confidence_mean"]
                runtime_meta["date_selection_region_band"] = str(
                    selection_meta.get("region_band") or runtime_meta.get("date_selection_region_band") or "central"
                )
                runtime_meta["date_selection_low_confidence"] = bool(
                    runtime_meta.get("date_selection_low_confidence")
                    or bool(selection_meta.get("low_quality_input"))
                )
                if bool(selection_meta["low_quality_input"]):
                    tile_runtime["low_quality_input"] = True
                    runtime_meta["low_quality_input"] = True
                    warning = (
                        f"Tile {tile['tile_id']} has only "
                        f"{selection_meta['good_date_count']} good dates after coverage filtering"
                    )
                    runtime_meta["warnings"].append(warning)
                    logger.warning(
                        "low_quality_sentinel_input",
                        tile_id=tile["tile_id"],
                        good_date_count=int(selection_meta["good_date_count"]),
                        selected_date_count=int(selection_meta["selected_date_count"]),
                    )

                for scene_idx, (time_from, time_to) in enumerate(time_windows):
                    is_selected = scene_idx in selected_idx
                    tile_runtime["scene_windows"].append(
                        {
                            "time_from": time_from,
                            "time_to": time_to,
                            "p_valid": round(float(scene_coverages[scene_idx]), 4),
                            "score_total": round(
                                float((selection_meta.get("score_total") or [0.0] * t_count)[scene_idx]),
                                6,
                            ),
                            "score_components": (
                                (selection_meta.get("score_components") or [{} for _ in range(t_count)])[scene_idx]
                            ),
                            "selected_reason": (
                                (selection_meta.get("selected_reason") or ["not_selected"] * t_count)[scene_idx]
                            ),
                            "selected": is_selected,
                        }
                    )
                    logger.info(
                        "sentinel_scene_coverage",
                        tile_id=tile["tile_id"],
                        scene_index=scene_idx,
                        time_from=time_from,
                        time_to=time_to,
                        p_valid=round(float(scene_coverages[scene_idx]), 4),
                        selected=is_selected,
                    )
                    if is_selected:
                        selected_token = time_from[:10]
                        selected_date_tokens.add(selected_token)
                        tile_runtime["selected_dates"].append(selected_token)
                scene_signature_payload = "|".join(
                    f"{time_windows[idx][0]}::{time_windows[idx][1]}"
                    for idx in sorted(selected_idx)
                )
                tile_runtime["selected_scene_signature"] = hashlib.sha1(
                    scene_signature_payload.encode("utf-8")
                ).hexdigest()[:16]
                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "date_selection",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.28,
                )

                valid_sel = valid_mask[selected]
                ndvi_sel = indices["NDVI"][selected]
                ndre_sel = indices["NDRE"][selected] if "NDRE" in indices else None
                ndwi_sel = indices["NDWI"][selected]
                mndwi_sel = indices["MNDWI"][selected]
                bsi_sel = indices["BSI"][selected]
                msi_sel = indices["MSI"][selected]
                scl_water_mask = np.any(scl == 6, axis=0)
                # Free full temporal stacks - we only need selected slices from now on
                del valid_mask, indices

                pheno = compute_phenometrics(ndvi_sel, valid_sel)
                ndwi_med = build_median_composite(ndwi_sel, valid_sel)
                if ndre_sel is not None:
                    ndre_post = build_median_composite(ndre_sel, valid_sel)
                _ndwi_masked = np.where(valid_sel, ndwi_sel, np.nan)
                ndwi_mean = nanmean_safe(_ndwi_masked, axis=0, fill_value=np.nan)
                del _ndwi_masked
                np.nan_to_num(ndwi_mean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                ndwi_mean = ndwi_mean.astype(np.float32, copy=False)
                del ndwi_sel  # Free after composites built
                _mndwi_masked = np.where(valid_sel, mndwi_sel, np.nan)
                mndwi_max = nanmax_safe(_mndwi_masked, axis=0, fill_value=np.nan)
                del _mndwi_masked
                np.nan_to_num(mndwi_max, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                mndwi_max = mndwi_max.astype(np.float32, copy=False)
                del mndwi_sel  # Free after use
                bsi_med = build_median_composite(bsi_sel, valid_sel)
                del bsi_sel
                msi_med = build_median_composite(msi_sel, valid_sel)
                del msi_sel
                nir_med = build_median_composite(bands["B8"][selected], valid_sel)
                swir_med = build_median_composite(bands["B11"][selected], valid_sel)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndmi_mean = np.where(
                        (nir_med + swir_med) > 1e-6,
                        (nir_med - swir_med) / (nir_med + swir_med),
                        0.0,
                    )
                valid_count = valid_sel.sum(axis=0)
                scl_sel = scl[selected]
                del scl  # Free full SCL stack
                _scl_masked = np.where(valid_sel, scl_sel, np.nan)
                scl_median = nanmedian_safe(_scl_masked, axis=0, fill_value=np.nan)
                del _scl_masked, scl_sel  # No longer needed
                np.nan_to_num(scl_median, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                scl_median = scl_median.astype(np.float32, copy=False)

                low_obs_pct = float((valid_count < settings.TILE_MIN_OBS_COUNT).mean())
                if low_obs_pct > settings.TILE_MAX_LOW_OBS_PCT:
                    tile_runtime["low_quality_input"] = True
                    tile_runtime["low_confidence_reason"] = "low_observation_fraction"
                    runtime_meta["low_quality_input"] = True
                    runtime_meta["warnings"].append(
                        f"Tile {tile['tile_id']} has elevated low-observation share: "
                        f"{round(low_obs_pct, 3)} > {float(settings.TILE_MAX_LOW_OBS_PCT):.3f}"
                    )
                    logger.warning(
                        "tile_low_observation_recovery_candidate",
                        tile_id=tile["tile_id"],
                        low_obs_pct=round(low_obs_pct, 3),
                        threshold=float(settings.TILE_MAX_LOW_OBS_PCT),
                    )
                if _should_skip_low_observation_tile(low_obs_pct, settings):
                    logger.warning(
                        "skipping_extreme_low_observation_tile",
                        tile_id=tile["tile_id"],
                        low_obs_pct=round(low_obs_pct, 3),
                        threshold=max(0.85, float(settings.TILE_MAX_LOW_OBS_PCT) + 0.25),
                    )
                    tile_runtime["quality_gate_failed"] = True
                    tile_runtime["low_confidence_reason"] = "extreme_low_observation"
                    TILES_PROCESSED.inc()
                    runtime_meta["tiles"].append(tile_runtime)
                    _persist_runtime_meta(run, runtime_meta)
                    _set_tile_progress(
                        run,
                        runtime_meta,
                        session,
                        "tile_done",
                        tile_index=i,
                        tile_count=len(tiles),
                        phase_fraction=1.0,
                    )
                    result = bands = scl = valid_mask = indices = valid_sel = ndvi_sel = None
                    ndwi_sel = mndwi_sel = edge_bands = temporal_composite = edge_composite = None
                    ndvi_std = feature_stack = extent_prob = boundary_prob = distance_prob = None
                    labels = labels_clean = sam_input = sam_raw = sam_filtered = None
                    _run_tile_gc(settings)
                    continue

                edge_bands = {k: bands[k][selected] for k in ["B2", "B3", "B4", "B8"]}
                edge_bands["ndvi"] = ndvi_sel
                # Pre-compute medians from bands before freeing the temporal stack
                _b4_sel = bands["B4"][selected]
                _b2_sel = bands["B2"][selected]
                _b3_sel = bands["B3"][selected]
                rgb_g_med = build_median_composite(_b4_sel, valid_sel)
                rgb_b_med = build_median_composite(_b2_sel, valid_sel)
                green_med = build_median_composite(_b3_sel, valid_sel)
                del _b4_sel, _b2_sel, _b3_sel
                # Free raw temporal bands - we no longer need them
                del bands
                result = None

                temporal_composite = build_multiyear_composite(
                    ndvi_stack=ndvi_sel,
                    valid_mask=valid_sel,
                    edge_bands=edge_bands,
                    cfg=settings,
                )
                edge_composite = temporal_composite["edge_composite"]
                ndvi_std = temporal_composite["ndvi_std"]
                n_valid_scenes = int(temporal_composite.get("n_valid_scenes", int(valid_sel.shape[0])))
                tile_runtime["n_valid_scenes"] = n_valid_scenes
                runtime_meta["n_valid_scenes"] = max(int(runtime_meta.get("n_valid_scenes") or 0), n_valid_scenes)

                # --- V4: Tile quality assessment ---
                try:
                    from processing.fields.quality_controller import assess_tile_quality
                    qc_report = assess_tile_quality(
                        valid_mask=valid_sel,
                        edge_composite=edge_composite,
                        ndvi_stack=ndvi_sel,
                        scl_stack=None,  # already freed
                        cfg=settings,
                    )
                    tile_runtime["qc_mode"] = qc_report.mode.value
                    tile_runtime["qc_coverage"] = round(qc_report.coverage_fraction, 4)
                    tile_runtime["qc_edge_p90"] = round(qc_report.edge_strength_p90, 4)
                    tile_runtime["qc_ndvi_std"] = round(qc_report.ndvi_temporal_std_mean, 4)
                    tile_runtime["edge_signal_p90"] = round(qc_report.edge_strength_p90, 4)
                    tile_runtime["qc_reasons"] = qc_report.reasons
                except Exception as exc:
                    logger.warning("tile_qc_skipped", tile_id=tile["tile_id"], error=str(exc))
                    tile_runtime["qc_mode"] = "normal"

                processing_profile = _resolve_processing_profile(tile_runtime.get("qc_mode"), settings)
                # Merge pipeline-level overrides (preset) with per-tile QC overrides.
                # Pipeline overrides are applied first, QC overrides on top (higher priority).
                merged_overrides = {
                    **dict(detect_pipeline_profile.get("config_overrides") or {}),
                    **dict(processing_profile.get("config_overrides") or {}),
                }
                processing_settings = _apply_runtime_config(settings, merged_overrides)
                tile_runtime["processing_profile"] = str(processing_profile["name"])
                profile_counts = dict(runtime_meta.get("processing_profile_counts") or {})
                profile_counts[tile_runtime["processing_profile"]] = int(
                    profile_counts.get(tile_runtime["processing_profile"]) or 0
                ) + 1
                runtime_meta["processing_profile_counts"] = profile_counts
                if tile_runtime["processing_profile"] == "skip_tile":
                    tile_runtime["quality_gate_failed"] = True
                    tile_runtime["low_confidence_reason"] = "qc_skip_tile"
                    logger.warning(
                        "tile_skipped_by_qc_profile",
                        tile_id=tile["tile_id"],
                        qc_mode=tile_runtime.get("qc_mode"),
                        reasons=tile_runtime.get("qc_reasons"),
                    )
                    TILES_PROCESSED.inc()
                    runtime_meta["tiles"].append(tile_runtime)
                    _persist_runtime_meta(run, runtime_meta)
                    _set_tile_progress(
                        run,
                        runtime_meta,
                        session,
                        "tile_done",
                        tile_index=i,
                        tile_count=len(tiles),
                        phase_fraction=1.0,
                    )
                    result = bands = scl = valid_mask = indices = valid_sel = ndvi_sel = None
                    ndwi_sel = mndwi_sel = edge_bands = temporal_composite = edge_composite = None
                    ndvi_std = feature_stack = extent_prob = boundary_prob = distance_prob = None
                    labels = labels_clean = sam_input = sam_raw = sam_filtered = None
                    _run_tile_gc(settings)
                    continue

                min_valid_scenes = int(getattr(settings, "MIN_VALID_SCENES_FOR_BOUNDARY", 3))
                if n_valid_scenes < min_valid_scenes:
                    tile_runtime["low_quality_input"] = True
                    tile_runtime["low_confidence_reason"] = "insufficient_valid_scenes"
                    runtime_meta["low_quality_input"] = True
                    runtime_meta["date_selection_low_confidence"] = True
                    warning = (
                        f"Tile {tile['tile_id']} skipped: n_valid_scenes={n_valid_scenes} < "
                        f"min_required={min_valid_scenes}"
                    )
                    runtime_meta["warnings"].append(warning)
                    logger.warning(
                        "tile_low_confidence_insufficient_valid_scenes",
                        tile_id=tile["tile_id"],
                        n_valid_scenes=n_valid_scenes,
                        min_required=min_valid_scenes,
                    )
                    if not (
                        bool(processing_profile.get("enable_second_pass"))
                        or bool(processing_profile.get("allow_degraded_output"))
                    ):
                        tile_runtime["quality_gate_failed"] = True
                        runtime_meta["quality_gate_failed"] = True
                        TILES_PROCESSED.inc()
                        runtime_meta["tiles"].append(tile_runtime)
                        _persist_runtime_meta(run, runtime_meta)
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "tile_done",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=1.0,
                        )
                        result = bands = scl = valid_mask = indices = valid_sel = ndvi_sel = None
                        ndwi_sel = mndwi_sel = edge_bands = temporal_composite = edge_composite = None
                        ndvi_std = feature_stack = extent_prob = boundary_prob = distance_prob = None
                        labels = labels_clean = sam_input = sam_raw = sam_filtered = None
                        _run_tile_gc(settings)
                        continue
                selected_windows = [time_windows[int(idx)] for idx in selected.tolist()]
                s1_features = None
                if (
                    bool(getattr(settings, "FEATURE_S1_FUSION", False))
                    and bool(getattr(settings, "S1_ENABLED", False))
                    and selected_windows
                ):
                    try:
                        from processing.fields.s1_preprocess import preprocess_s1
                        from providers.s1_client import SentinelHubS1Client

                        s1_client = SentinelHubS1Client()
                        s1_time_from = selected_windows[0][0]
                        s1_time_to = selected_windows[-1][1]
                        with S1_FETCH_TIME.time():
                            s1_data = (_event_loop or asyncio.new_event_loop()).run_until_complete(
                                s1_client.fetch_tile(
                                    tile["bbox_4326"],
                                    s1_time_from,
                                    s1_time_to,
                                    tile["shape"][1],
                                    tile["shape"][0],
                                )
                            )
                        s1_features = preprocess_s1(s1_data["VV"], s1_data["VH"], settings)
                        tile_runtime["s1_enabled"] = True
                        tile_runtime["s1_account_used"] = getattr(s1_client._s2_client, "last_account_alias", "primary")
                    except Exception as exc:
                        logger.warning(
                            "s1_fusion_skipped",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )
                        tile_runtime["s1_enabled"] = False

                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "temporal_composite",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.42,
                )

                classes = classify_land_cover(
                    pheno,
                    ndwi_med,
                    bsi_med,
                    msi_med,
                    valid_count,
                    thresholds,
                    mndwi_max=mndwi_max,
                    scl_water_mask=scl_water_mask,
                    ndwi_mean=ndwi_mean,
                )
                hydro_masks = compute_hydro_masks(
                    ndwi_med,
                    valid_count,
                    thresholds,
                    mndwi_max=mndwi_max,
                    scl_water_mask=scl_water_mask,
                    ndwi_mean=ndwi_mean,
                    hydro_profile=str(getattr(settings, "HYDRO_BOUNDARY_PROFILE", "water_aware")),
                    open_water_ndwi=float(
                        getattr(settings, "HYDRO_OPEN_WATER_NDWI", thresholds.ndwi_water)
                    ),
                    open_water_mndwi=float(
                        getattr(settings, "HYDRO_OPEN_WATER_MNDWI", thresholds.mndwi_water)
                    ),
                    seasonal_wet_ndwi=float(
                        getattr(settings, "HYDRO_SEASONAL_WET_NDWI", thresholds.ndwi_water * 0.7)
                    ),
                    seasonal_wet_mndwi=float(
                        getattr(settings, "HYDRO_SEASONAL_WET_MNDWI", thresholds.mndwi_water * 0.5)
                    ),
                    riparian_buffer_px=int(getattr(settings, "HYDRO_RIPARIAN_BUFFER_PX", 2)),
                )
                water_mask = hydro_masks["open_water_mask"]
                candidate_mask = classes == CROP
                ndvi_post = temporal_composite["max_ndvi"]
                ndvi_mean_post = temporal_composite["mean_ndvi"]
                del temporal_composite  # Free composite dict, arrays are referenced by local vars
                del edge_bands  # No longer needed after composites

                tile_lc_fractions = None
                exclusion_mask = None
                worldcover_grid = None
                if wc_prior is not None:
                    worldcover_grid = wc_prior.load_worldcover_grid(
                        tile["bbox_4326"], tile["transform"], tile["shape"], tile["crs"],
                    )
                    exclusion_mask = wc_prior.build_exclusion_mask(
                        tile["bbox_4326"], tile["transform"], tile["shape"], tile["crs"],
                    )
                    use_weak_worldcover = bool(
                        getattr(settings, "FRAMEWORK_USE_WEAK_WORLDCOVER", True)
                        and settings.USE_WEAK_WORLDCOVER_BARRIER
                    )
                    if use_weak_worldcover:
                        logger.info(
                            "worldcover_weak_prior_loaded",
                            tile_id=tile["tile_id"],
                            priority_pixels=int(np.count_nonzero(exclusion_mask)),
                        )
                    else:
                        pre_wc = int(candidate_mask.sum())
                        candidate_mask = candidate_mask & ~exclusion_mask
                        post_wc = int(candidate_mask.sum())
                        logger.info(
                            "worldcover_filter",
                            tile_id=tile["tile_id"],
                            candidates_before=pre_wc,
                            candidates_after=post_wc,
                            excluded_pixels=int(exclusion_mask.sum()),
                        )
                    tile_lc_fractions = wc_prior.build_landcover_fractions(
                        tile["bbox_4326"], tile["transform"], tile["shape"], tile["crs"],
                    )

                postprocess_debug = None
                postprocess_progress = {
                    "road_barrier_start": 0.49,
                    "road_barrier_done": 0.51,
                    "boundary_fill_done": 0.54,
                    "clean_done": 0.57,
                    "grow_done": 0.59,
                    "gap_close_done": 0.60,
                    "infill_done": 0.603,
                    "bridge_done": 0.606,
                    "merge_done": 0.61,
                    "watershed_done": 0.62,
                    "finalize_done": 0.63,
                }

                def _postprocess_progress_callback(checkpoint: str) -> None:
                    checkpoint_text = str(checkpoint)
                    if checkpoint_text.startswith("boundary_merge_scan:"):
                        _, completed_raw, total_raw = checkpoint_text.split(":", 2)
                        completed = max(0, int(completed_raw))
                        total = max(1, int(total_raw))
                        ratio = min(completed, total) / total
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "candidate_postprocess",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.515 + (0.023 * ratio),
                            detail=f"boundary merge {min(completed, total)}/{total}",
                        )
                        return
                    if checkpoint_text.startswith("gap_close_scan:"):
                        _, completed_raw, total_raw = checkpoint_text.split(":", 2)
                        completed = max(0, int(completed_raw))
                        total = max(1, int(total_raw))
                        ratio = min(completed, total) / total
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "candidate_postprocess",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.592 + (0.007 * ratio),
                            detail=f"gap close {min(completed, total)}/{total}",
                        )
                        return
                    if checkpoint_text.startswith("infill_scan:"):
                        _, completed_raw, total_raw = checkpoint_text.split(":", 2)
                        completed = max(0, int(completed_raw))
                        total = max(1, int(total_raw))
                        ratio = min(completed, total) / total
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "candidate_postprocess",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.6005 + (0.002 * ratio),
                            detail=f"infill {min(completed, total)}/{total}",
                        )
                        return
                    if checkpoint_text.startswith("bridge_scan:"):
                        _, completed_raw, total_raw = checkpoint_text.split(":", 2)
                        completed = max(0, int(completed_raw))
                        total = max(1, int(total_raw))
                        ratio = min(completed, total) / total
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "candidate_postprocess",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.6035 + (0.002 * ratio),
                            detail=f"bridge {min(completed, total)}/{total}",
                        )
                        return
                    if checkpoint_text.startswith("merge_scan:"):
                        _, completed_raw, total_raw = checkpoint_text.split(":", 2)
                        completed = max(0, int(completed_raw))
                        total = max(1, int(total_raw))
                        ratio = min(completed, total) / total
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "candidate_postprocess",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.6065 + (0.003 * ratio),
                            detail=f"merge {min(completed, total)}/{total}",
                        )
                        return
                    phase_fraction = postprocess_progress.get(str(checkpoint))
                    if phase_fraction is None:
                        return
                    detail = checkpoint_text.replace("_done", "").replace("_", " ")
                    _set_tile_progress(
                        run,
                        runtime_meta,
                        session,
                        "candidate_postprocess",
                        tile_index=i,
                        tile_count=len(tiles),
                        phase_fraction=phase_fraction,
                        detail=detail,
                    )

                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "candidate_postprocess",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.48,
                    detail="postprocess start",
                )
                if debug_run:
                    candidate_mask, postprocess_debug = run_postprocess(
                        candidate_mask=candidate_mask,
                        water_mask=water_mask,
                        classes=classes,
                        ndvi=ndvi_post,
                        ndwi=ndwi_med,
                        cfg=processing_settings,
                        open_water_mask=hydro_masks["open_water_mask"],
                        seasonal_wet_mask=hydro_masks["seasonal_wet_mask"],
                        riparian_soft_mask=hydro_masks["riparian_soft_mask"],
                        riparian_hard_mask=hydro_masks["riparian_hard_mask"],
                        nir=nir_med,
                        swir=swir_med,
                        edge_composite=edge_composite,
                        ndvi_std=ndvi_std,
                        worldcover_mask=worldcover_grid,
                        bbox=tile["bbox_4326"],
                        tile_transform=tile["transform"],
                        out_shape=tile["shape"],
                        crs_epsg=tile["crs"],
                        region_profile=region_boundary_profile,
                        progress_callback=_postprocess_progress_callback,
                        return_debug_steps=True,
                        return_candidate_masks=True,
                    )
                else:
                    candidate_mask, postprocess_debug = run_postprocess(
                        candidate_mask=candidate_mask,
                        water_mask=water_mask,
                        classes=classes,
                        ndvi=ndvi_post,
                        ndwi=ndwi_med,
                        cfg=processing_settings,
                        open_water_mask=hydro_masks["open_water_mask"],
                        seasonal_wet_mask=hydro_masks["seasonal_wet_mask"],
                        riparian_soft_mask=hydro_masks["riparian_soft_mask"],
                        riparian_hard_mask=hydro_masks["riparian_hard_mask"],
                        nir=nir_med,
                        swir=swir_med,
                        edge_composite=edge_composite,
                        ndvi_std=ndvi_std,
                        worldcover_mask=worldcover_grid,
                        bbox=tile["bbox_4326"],
                        tile_transform=tile["transform"],
                        out_shape=tile["shape"],
                        crs_epsg=tile["crs"],
                        region_profile=region_boundary_profile,
                        progress_callback=_postprocess_progress_callback,
                        return_debug_stats=True,
                        return_candidate_masks=True,
                    )
                candidate_masks_payload = {}
                road_mask_for_ranking = None
                if isinstance(postprocess_debug, dict):
                    postprocess_stats = postprocess_debug.get("stats", {})
                    candidate_masks_payload = dict(postprocess_debug.get("candidate_masks") or {})
                    tile_runtime["postprocess_steps"] = postprocess_stats
                    tile_runtime["postprocess_stage_pixels"] = _extract_postprocess_stage_pixels(
                        postprocess_stats
                    )
                    debug_masks_payload = dict(postprocess_debug.get("masks") or {})
                    if "step_01_road_mask" in debug_masks_payload:
                        road_mask_for_ranking = np.asarray(
                            debug_masks_payload["step_01_road_mask"],
                            dtype=bool,
                        )
                    summary_stats = (
                        postprocess_stats.get("summary", {})
                        if isinstance(postprocess_stats, dict)
                        else {}
                    )
                    tile_runtime["road_barrier_retry_used"] = bool(
                        summary_stats.get("road_barrier_retry_used")
                    )
                    for summary_key in (
                        "open_water_pixels",
                        "seasonal_wet_pixels",
                        "riparian_soft_pixels",
                        "riparian_hard_pixels",
                        "water_edge_overlap_ratio",
                        "road_edge_overlap_ratio",
                        "boundary_shift_to_road_ratio",
                        "hydro_rescue_used",
                        "road_snap_reject_used",
                        "components_after_clean",
                        "components_after_grow",
                        "components_after_gap_close",
                        "components_after_infill",
                        "components_after_merge",
                        "components_after_watershed",
                        "split_risk_score",
                        "shrink_risk_score",
                        "region_profile_applied",
                    ):
                        if summary_key in summary_stats:
                            tile_runtime[summary_key] = summary_stats.get(summary_key)
                    if "region_profile_actions" in summary_stats:
                        merged_actions = list(tile_runtime.get("region_profile_actions") or [])
                        merged_actions.extend(list(summary_stats.get("region_profile_actions") or []))
                        tile_runtime["region_profile_actions"] = merged_actions
                    runtime_meta["water_edge_risk_detected"] = bool(
                        runtime_meta.get("water_edge_risk_detected")
                        or float(summary_stats.get("water_edge_overlap_ratio") or 0.0) > 0.05
                    )
                    runtime_meta["road_drift_risk_detected"] = bool(
                        runtime_meta.get("road_drift_risk_detected")
                        or float(summary_stats.get("road_edge_overlap_ratio") or 0.0)
                        > float(getattr(settings, "ROAD_SNAP_REJECT_MAX_OVERLAP_RATIO", 0.08))
                    )
                tc_growth_amplitude = np.zeros_like(ndvi_post, dtype=np.float32)
                tc_has_growth_peak = np.zeros_like(ndvi_post, dtype=np.float32)
                tc_ndvi_entropy = np.zeros_like(ndvi_post, dtype=np.float32)
                # Temporal coherence filter: remove pixels without crop-like temporal pattern
                try:
                    from processing.fields.temporal_coherence import compute_temporal_coherence

                    tc_metrics = compute_temporal_coherence(ndvi_sel, valid_sel)
                    tc_growth_amplitude = np.asarray(
                        tc_metrics.get("growth_amplitude"),
                        dtype=np.float32,
                    )
                    tc_has_growth_peak = np.asarray(
                        tc_metrics.get("has_growth_peak"),
                        dtype=np.float32,
                    )
                    tc_ndvi_entropy = np.asarray(
                        tc_metrics.get("ndvi_temporal_entropy"),
                        dtype=np.float32,
                    )
                    tc_mask, tc_diag = _build_temporal_coherence_mask(
                        growth_amplitude=tc_growth_amplitude,
                        has_growth_peak=tc_has_growth_peak,
                        ndvi_entropy=tc_ndvi_entropy,
                        candidate_masks_payload=candidate_masks_payload,
                        processing_profile=processing_profile,
                        cfg=processing_settings,
                    )
                    tile_runtime["temporal_coherence_relaxed"] = bool(
                        tc_diag.get("relaxed", False)
                    )
                    tile_runtime["temporal_coherence_growth_min"] = float(
                        tc_diag.get("growth_amplitude_min", 0.20)
                    )
                    tile_runtime["temporal_coherence_entropy_max"] = float(
                        tc_diag.get("entropy_max", 2.5)
                    )
                    tile_runtime["temporal_coherence_boundary_keep_pixels"] = int(
                        tc_diag.get("boundary_keep_pixels", 0)
                    )
                    pre_tc = int(np.count_nonzero(candidate_mask))
                    candidate_mask &= tc_mask
                    post_tc = int(np.count_nonzero(candidate_mask))
                    if pre_tc != post_tc:
                        logger.info(
                            "temporal_coherence_filter",
                            tile_id=tile["tile_id"],
                            removed_pixels=pre_tc - post_tc,
                            remaining_pixels=post_tc,
                        )
                except Exception as tc_exc:
                    logger.warning(
                        "temporal_coherence_skipped",
                        tile_id=tile["tile_id"],
                        error=str(tc_exc),
                    )
                valid_scene_total = int(valid_sel.shape[0]) if valid_sel is not None else 1
                # Free temporal stacks — no longer needed after coherence computation
                del ndvi_sel, valid_sel
                gc.collect()

                if exclusion_mask is not None and not (
                    bool(getattr(settings, "FRAMEWORK_USE_WEAK_WORLDCOVER", True))
                    and settings.USE_WEAK_WORLDCOVER_BARRIER
                ):
                    candidate_mask &= ~exclusion_mask
                    if postprocess_debug is not None:
                        from scipy.ndimage import label as nd_label

                        final_step = "step_12_after_worldcover_reapply"
                        _, final_components = nd_label(candidate_mask.astype(bool))
                        postprocess_debug.setdefault("masks", {})[final_step] = (
                            candidate_mask.astype(np.uint8)
                        )
                        postprocess_debug.setdefault("stats", {})[final_step] = {
                            "pixels": int(np.count_nonzero(candidate_mask)),
                            "components": int(final_components),
                            "coverage_ratio": float(
                                np.count_nonzero(candidate_mask) / candidate_mask.size
                            ) if candidate_mask.size else 0.0,
                        }
                candidate_pct = float(candidate_mask.mean() * 100.0)
                tile_runtime["candidate_pct"] = round(candidate_pct, 2)
                logger.info(
                    "candidate_mask_stats",
                    tile_id=tile["tile_id"],
                    t_count=int(t_count),
                    candidate_pct=round(candidate_pct, 2),
                    ndvi_delta_mean=float(np.nanmean(pheno["ndvi_delta"])),
                    ndvi_delta_p90=float(np.nanpercentile(pheno["ndvi_delta"], 90)),
                    water_pct=round(float(water_mask.mean() * 100.0), 2),
                )
                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "candidate_postprocess",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.55,
                )
                ml_progress_callback = _make_tile_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="model_inference",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_map={
                        "ml_scale": (0.556, 0.018, "ml scale"),
                        "ml_tta": (0.576, 0.016, "ml tta"),
                        "ml_patch_infer": (0.594, 0.016, "ml patch infer"),
                        "ml_patches": (0.612, 0.010, "ml patches"),
                        "ml_blend": (0.624, 0.010, "ml blend"),
                    },
                    default_phase=(0.556, 0.078, "ml inference"),
                )
                segmentation_progress_callback = _make_tile_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="segmentation",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_map={
                        "surface": (0.722, 0.006, "surface"),
                        "markers": (0.728, 0.006, "markers"),
                        "watershed": (0.734, 0.006, "watershed"),
                        "pair_scan": (0.740, 0.008, "split pairs"),
                        "pair_merge": (0.748, 0.008, "pair merge"),
                        "zero_fill": (0.756, 0.006, "zero fill"),
                    },
                    default_phase=(0.722, 0.040, "segmentation"),
                )
                obia_progress_callback = _make_tile_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="segmentation",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_map={
                        "props": (0.762, 0.014, "segment props"),
                        "filter": (0.776, 0.014, "segment filter"),
                    },
                    default_phase=(0.762, 0.028, "obia"),
                )
                vectorize_progress_callback = _make_tile_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="boundary_refine",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_map={
                        "polygonize": (0.792, 0.030, "polygonize"),
                    },
                    default_phase=(0.792, 0.030, "vectorize"),
                )
                snake_progress_callback = _make_tile_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="boundary_refine",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_map={
                        "snake_fields": (0.842, 0.028, "snake refine"),
                    },
                    default_phase=(0.842, 0.028, "boundary refine"),
                )
                object_feature_progress_callback = _make_tile_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="object_classifier",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_map={
                        "object_zonal": (0.882, 0.014, "tile features"),
                    },
                    default_phase=(0.882, 0.014, "object features"),
                )

                if not candidate_mask.any():
                    if bool(processing_profile.get("enable_second_pass")) or bool(
                        processing_profile.get("allow_degraded_output")
                    ):
                        recovery_seed_masks = []
                        for key in (
                            "final_candidate_mask",
                            "boundary_field_mask",
                            "crop_soft_mask",
                            "field_candidate",
                            "legacy_seed_mask",
                        ):
                            seed_mask = candidate_masks_payload.get(key)
                            if seed_mask is None:
                                continue
                            recovery_seed_masks.append(np.asarray(seed_mask, dtype=bool))
                        if recovery_seed_masks:
                            recovery_seed = np.logical_or.reduce(recovery_seed_masks)
                            if np.any(recovery_seed):
                                candidate_mask = recovery_seed
                                tile_runtime["recovery_seed_used"] = True
                                tile_runtime["recovery_seed_pixels"] = int(np.count_nonzero(recovery_seed))
                                logger.info(
                                    "candidate_mask_recovered_from_branch_union",
                                    tile_id=tile["tile_id"],
                                    processing_profile=tile_runtime.get("processing_profile"),
                                    recovery_pixels=tile_runtime["recovery_seed_pixels"],
                                )
                    if debug_run:
                        tile_runtime["debug_artifact"] = _save_debug_tile_dump(
                            run_id_str,
                            tile["tile_id"],
                            {
                                "candidate_mask": candidate_mask.astype(np.uint8),
                                "max_ndvi": ndvi_post.astype(np.float32),
                                "mean_ndvi": ndvi_mean_post.astype(np.float32),
                                "ndvi_std": ndvi_std.astype(np.float32),
                                "edge_composite": edge_composite.astype(np.float32),
                                "labels": np.zeros_like(candidate_mask, dtype=np.int32),
                                "labels_clean": np.zeros_like(candidate_mask, dtype=np.int32),
                                "water_mask": water_mask.astype(np.uint8),
                                "worldcover_mask": (
                                    worldcover_grid.astype(np.uint8)
                                    if worldcover_grid is not None
                                    else np.zeros_like(candidate_mask, dtype=np.uint8)
                                ),
                                "osm_mask": np.zeros_like(candidate_mask, dtype=np.uint8),
                                **(
                                    {
                                        name: np.asarray(mask)
                                        for name, mask in postprocess_debug.get("masks", {}).items()
                                    }
                                    if postprocess_debug is not None
                                    else {}
                                ),
                            },
                        )
                if not candidate_mask.any():
                    runtime_meta["selected_dates"] = sorted(selected_date_tokens)
                    runtime_meta["tiles"].append(tile_runtime)
                    _persist_runtime_meta(run, runtime_meta)
                    logger.info("no_candidates", tile_id=tile["tile_id"])
                    TILES_PROCESSED.inc()
                    _set_tile_progress(
                        run,
                        runtime_meta,
                        session,
                        "tile_done",
                        tile_index=i,
                        tile_count=len(tiles),
                        phase_fraction=1.0,
                    )
                    result = bands = scl = valid_mask = indices = valid_sel = ndvi_sel = None
                    ndwi_sel = mndwi_sel = edge_bands = temporal_composite = edge_composite = None
                    ndvi_std = feature_stack = extent_prob = boundary_prob = distance_prob = None
                    labels = labels_clean = sam_input = sam_raw = sam_filtered = None
                    _run_tile_gc(settings)
                    continue

                edge_stats = compute_edge_stats(edge_composite)
                tile_runtime["edge_stats"] = edge_stats
                edge_signal_p90 = float(np.nanpercentile(edge_composite, 90)) if edge_composite.size else 0.0
                tile_runtime["edge_signal_p90"] = round(edge_signal_p90, 4)
                runtime_meta["edge_signal_p90"] = max(
                    float(runtime_meta.get("edge_signal_p90") or 0.0),
                    edge_signal_p90,
                )
                logger.info(
                    "edge_composite_stats",
                    tile_id=tile["tile_id"],
                    edge_min=round(float(edge_stats["min"]), 4),
                    edge_max=round(float(edge_stats["max"]), 4),
                    edge_mean=round(float(edge_stats["mean"]), 4),
                    histogram=edge_stats["histogram"],
                )
                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "model_inference",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.64,
                )
                preset_candidate_limit = int(detect_pipeline_profile["max_candidates_per_tile"])
                simplified_component_cap = int(detect_pipeline_profile["post_merge_max_components"])
                use_simplified_tile_path = bool(detect_pipeline_profile["preview_only"])
                simplified_path_reason = "preview_only_fast_path" if use_simplified_tile_path else None
                memory_guard_triggered, current_rss_mb, stage_memory_limit_mb = _should_downgrade_for_memory(
                    settings,
                    detect_preset,
                    stage="tile_heavy_pipeline",
                )
                tile_runtime["rss_before_heavy_pipeline_mb"] = round(float(current_rss_mb), 2)
                tile_runtime["memory_soft_limit_mb"] = round(float(stage_memory_limit_mb), 2)
                if memory_guard_triggered and not use_simplified_tile_path:
                    use_simplified_tile_path = True
                    simplified_path_reason = f"memory_guard_{detect_preset}"
                    warning = (
                        f"Tile {tile['tile_id']} downgraded to simplified path "
                        f"(rss {current_rss_mb:.0f} MB / limit {stage_memory_limit_mb:.0f} MB)"
                    )
                    runtime_meta["warnings"].append(warning)
                    tile_runtime.setdefault("warnings", []).append(warning)
                    logger.warning(
                        "tile_simplified_path_enabled",
                        tile_id=tile["tile_id"],
                        preset=detect_preset,
                        reason=simplified_path_reason,
                        rss_mb=round(float(current_rss_mb), 2),
                        limit_mb=round(float(stage_memory_limit_mb), 2),
                    )
                if float(edge_stats["max"]) < processing_settings.EDGE_WEAK_THRESHOLD:
                    warning = f"Tile {tile['tile_id']} edge composite is too weak"
                    runtime_meta["warnings"].append(warning)
                    logger.warning(
                        "weak_edge_composite",
                        tile_id=tile["tile_id"],
                        edge_max=round(float(edge_stats["max"]), 4),
                        threshold=processing_settings.EDGE_WEAK_THRESHOLD,
                    )
                    if processing_settings.SKIP_WEAK_EDGE_TILES:
                        logger.warning("skipping_weak_edge_tile", tile_id=tile["tile_id"])
                        TILES_PROCESSED.inc()
                        runtime_meta["tiles"].append(tile_runtime)
                        _persist_runtime_meta(run, runtime_meta)
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "tile_done",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=1.0,
                        )
                        result = bands = scl = valid_mask = indices = valid_sel = ndvi_sel = None
                        ndwi_sel = mndwi_sel = edge_bands = temporal_composite = edge_composite = None
                        ndvi_std = feature_stack = extent_prob = boundary_prob = distance_prob = None
                        labels = labels_clean = sam_input = sam_raw = sam_filtered = None
                        _run_tile_gc(settings)
                        continue

                edge_prob = edge_composite.astype(np.float32, copy=False)
                boundary_prob = None
                ml_distance_map = None
                ml_primary_used = False
                ml_score = 0.0
                ml_threshold = float(getattr(settings, "ML_SCORE_THRESHOLD", 0.35))
                pre_ml_candidate_mask = candidate_mask.copy()
                tile_runtime["fusion_profile"] = f"{region_boundary_profile}:pre_ml_only"
                if (not use_simplified_tile_path) and ml_inferencer is not None:
                    try:
                        scl_valid_fraction = _safe_valid_fraction(valid_count, valid_scene_total)
                        # Match training: rgb_r=NIR(B8), rgb_g=Red(B4), rgb_b=Blue(B2)
                        rgb_r = nir_med  # B8 NIR median (already computed)
                        rgb_g = rgb_g_med  # Pre-computed above
                        rgb_b = rgb_b_med  # Pre-computed above
                        ndwi_med_val = ndwi_med  # Already computed above
                        # NDMI already computed above as ndmi_mean
                        ndmi_mean_val = ndmi_mean.astype(np.float32, copy=False)
                        feature_stack = _build_feature_stack_v4(
                            edge_composite=edge_composite,
                            max_ndvi=ndvi_post,
                            mean_ndvi=ndvi_mean_post,
                            ndvi_std=ndvi_std,
                            ndwi_mean=ndwi_mean,
                            bsi_mean=bsi_med,
                            scl_valid_fraction=scl_valid_fraction,
                            rgb_r=rgb_r,
                            rgb_g=rgb_g,
                            rgb_b=rgb_b,
                            s1_vv_mean=(
                                s1_features["VV_edge"] if s1_features is not None else None
                            ),
                            s1_vh_mean=(
                                s1_features["VHVV_ratio"] if s1_features is not None else None
                            ),
                            ndvi_entropy=tc_ndvi_entropy,
                            mndwi_max=mndwi_max,
                            ndmi_mean=ndmi_mean_val,
                            ndwi_median=ndwi_med_val,
                            green_median=green_med,
                            swir_median=swir_med,
                            feature_channels=tuple(getattr(ml_inferencer, "feature_channels", ())) or None,
                        )
                        ml_pred = ml_inferencer.predict(
                            feature_stack,
                            tile_size=int(getattr(settings, "ML_TILE_SIZE", 512)),
                            overlap=int(getattr(settings, "ML_OVERLAP", 128)),
                            tta_mode=str(runtime_meta.get("tta_mode") or "none"),
                            scales=tuple(float(v) for v in runtime_meta.get("multi_scale_scales") or (1.0,)),
                            progress_callback=ml_progress_callback,
                        )
                        extent_prob = np.asarray(ml_pred["extent"], dtype=np.float32)
                        boundary_prob = np.asarray(ml_pred["boundary"], dtype=np.float32)
                        distance_prob = np.asarray(ml_pred["distance"], dtype=np.float32)
                        ml_score = float(ml_pred.get("score", 0.0))
                        tta_consensus = ml_pred.get("tta_consensus")
                        boundary_uncertainty = ml_pred.get("boundary_uncertainty")
                        geometry_confidence = ml_pred.get("geometry_confidence")
                        tta_extent_disagreement = ml_pred.get("tta_extent_disagreement")
                        tta_boundary_disagreement = ml_pred.get("tta_boundary_disagreement")
                        uncertainty_source = str(ml_pred.get("uncertainty_source") or "tta_unavailable").strip()
                        del ml_pred, feature_stack
                        tile_runtime["ml_quality_score"] = round(ml_score, 4)
                        if isinstance(tta_consensus, (int, float)):
                            tile_runtime["tta_consensus"] = round(float(tta_consensus), 4)
                        if isinstance(boundary_uncertainty, (int, float)):
                            tile_runtime["boundary_uncertainty"] = round(float(boundary_uncertainty), 4)
                        if isinstance(geometry_confidence, (int, float)):
                            tile_runtime["geometry_confidence"] = round(float(geometry_confidence), 4)
                        if isinstance(tta_extent_disagreement, (int, float)):
                            tile_runtime["tta_extent_disagreement"] = round(float(tta_extent_disagreement), 4)
                        if isinstance(tta_boundary_disagreement, (int, float)):
                            tile_runtime["tta_boundary_disagreement"] = round(float(tta_boundary_disagreement), 4)
                        tile_runtime["uncertainty_source"] = uncertainty_source or "tta_unavailable"
                        runtime_meta["ml_quality_score"] = max(
                            float(runtime_meta.get("ml_quality_score") or 0.0),
                            ml_score,
                        )
                        if isinstance(tile_runtime.get("tta_consensus"), (int, float)):
                            runtime_meta["tta_consensus"] = max(
                                float(runtime_meta.get("tta_consensus") or 0.0),
                                float(tile_runtime["tta_consensus"]),
                            )
                        if isinstance(tile_runtime.get("geometry_confidence"), (int, float)):
                            runtime_meta["geometry_confidence"] = max(
                                float(runtime_meta.get("geometry_confidence") or 0.0),
                                float(tile_runtime["geometry_confidence"]),
                            )
                        regional_extent_threshold = _resolve_regional_extent_threshold(
                            processing_settings,
                            region_boundary_profile,
                        )
                        ml_seed_mask, ml_seed_debug = _boundary_guided_ml_seed(
                            extent_prob=extent_prob,
                            boundary_prob=boundary_prob,
                            ndvi=ndvi_post,
                            ndvi_std=ndvi_std,
                            cfg=processing_settings,
                            extent_threshold_override=regional_extent_threshold,
                            dilation_px_override=_resolve_regional_dilation_px(
                                processing_settings,
                                region_boundary_profile,
                            ),
                        )
                        tile_runtime["ml_extent_threshold"] = ml_seed_debug["ml_extent_threshold"]

                        tile_runtime["ml_score"] = round(ml_score, 4)
                        tile_runtime["ml_extent_coverage_pct"] = round(
                            float(ml_seed_mask.mean() * 100.0),
                            2,
                        )
                        stage_pixels = dict(tile_runtime.get("postprocess_stage_pixels") or {})
                        stage_pixels["after_ml_seed"] = int(
                            ml_seed_debug["seed_pixels_after_dilation"]
                        )
                        tile_runtime["postprocess_stage_pixels"] = stage_pixels
                        tile_runtime["boundary_dilation_added_pixels"] = int(
                            ml_seed_debug["boundary_dilation_added_pixels"]
                        )

                        if ml_score >= ml_threshold and np.any(ml_seed_mask):
                            candidate_mask, ml_fusion_actions = _fuse_ml_primary_candidate(
                                ml_seed_mask,
                                pre_ml_candidate_mask,
                                region_boundary_profile,
                            )
                            edge_prob = boundary_prob
                            ml_distance_map = distance_prob
                            ml_primary_used = True
                            tile_runtime["ml_primary_used"] = True
                            if ml_fusion_actions:
                                tile_runtime["region_profile_actions"] = list(
                                    tile_runtime.get("region_profile_actions") or []
                                ) + ml_fusion_actions
                                tile_runtime["fusion_profile"] = (
                                    f"{region_boundary_profile}:{','.join(sorted(set(ml_fusion_actions)))}"
                                )
                            else:
                                tile_runtime["fusion_profile"] = f"{region_boundary_profile}:ml_seed_only"
                            tile_runtime["ml_fused_coverage_pct"] = round(
                                float(candidate_mask.mean() * 100.0),
                                2,
                            )
                        else:
                            tile_runtime["ml_primary_used"] = False
                            tile_runtime["fusion_profile"] = f"{region_boundary_profile}:ml_low_score_fallback"
                            if not bool(getattr(settings, "ML_FALLBACK_ON_LOW_SCORE", True)):
                                candidate_mask = ml_seed_mask
                                edge_prob = boundary_prob
                                ml_distance_map = distance_prob
                                ml_primary_used = True
                    except Exception as exc:
                        tile_runtime["ml_primary_used"] = False
                        tile_runtime["fusion_profile"] = f"{region_boundary_profile}:ml_error"
                        tile_runtime["ml_error"] = str(exc)
                        logger.warning(
                            "ml_primary_inference_failed",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )

                if (
                    (not use_simplified_tile_path)
                    and (not ml_primary_used)
                    and unet_model is not None
                    and predict_edge_map_fn is not None
                ):
                    try:
                        with UNET_INFERENCE_TIME.time():
                            edge_prob = predict_edge_map_fn(
                                unet_model,
                                edge_composite,
                                ndvi_post,
                                ndvi_std,
                                scl_median,
                                tile["transform"],
                                device=str(getattr(settings, "UNET_DEVICE", "cpu")),
                                vv_edge=s1_features["VV_edge"] if s1_features is not None else None,
                                vhvv_ratio=s1_features["VHVV_ratio"] if s1_features is not None else None,
                                threshold=float(getattr(settings, "UNET_EDGE_THRESHOLD", 0.5)),
                            )
                            if tile_runtime.get("fusion_profile") == f"{region_boundary_profile}:pre_ml_only":
                                tile_runtime["fusion_profile"] = f"{region_boundary_profile}:unet_fallback"
                    except Exception as exc:
                        logger.warning(
                            "unet_fallback_owt",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )

                if ml_primary_requested:
                    tile_runtime["fallback_rate_tile"] = 0.0 if ml_primary_used else 1.0

                from processing.fields.owt import oriented_watershed

                owt_edge = (
                    edge_prob
                    if bool(getattr(settings, "FEATURE_UNET_EDGE", True)) or ml_primary_used
                    else oriented_watershed(edge_composite, ndvi_post, cfg=processing_settings)
                )
                if (
                    not ml_primary_used
                    and unet_model is None
                    and tile_runtime.get("fusion_profile") == f"{region_boundary_profile}:pre_ml_only"
                ):
                    tile_runtime["fusion_profile"] = f"{region_boundary_profile}:owt_fallback"

                candidate_branches: list[Any] = []
                boundary_branch_mask = candidate_masks_payload.get("boundary_field_mask")
                boundary_branch_pixels = 0
                if (not use_simplified_tile_path) and boundary_branch_mask is not None:
                    boundary_branch_mask = np.asarray(boundary_branch_mask, dtype=bool)
                    boundary_branch_pixels = int(np.count_nonzero(boundary_branch_mask))
                    candidate_branches.extend(
                        _connected_component_candidates(
                            boundary_branch_mask,
                            "boundary_first",
                            max_components=preset_candidate_limit,
                        )
                    )
                crop_branch_mask = candidate_masks_payload.get("crop_soft_mask")
                crop_branch_pixels = 0
                if (not use_simplified_tile_path) and crop_branch_mask is not None:
                    crop_branch_mask = np.asarray(crop_branch_mask, dtype=bool)
                    crop_branch_pixels = int(np.count_nonzero(crop_branch_mask))
                    candidate_branches.extend(
                        _connected_component_candidates(
                            crop_branch_mask,
                            "crop_region",
                            max_components=preset_candidate_limit,
                        )
                    )
                if (
                    bool(processing_profile.get("force_boundary_union"))
                    and boundary_branch_mask is not None
                    and np.any(boundary_branch_mask)
                ):
                    candidate_mask |= boundary_branch_mask
                    tile_runtime["boundary_first_bias_applied"] = True
                    tile_runtime["candidate_source_mode"] = "boundary_first_biased"
                    tile_runtime["boundary_first_pixels"] = int(boundary_branch_pixels)
                    tile_runtime["crop_soft_pixels"] = int(crop_branch_pixels)
                if (not use_simplified_tile_path) and (not candidate_branches) and np.any(candidate_mask):
                    candidate_branches.extend(
                        _connected_component_candidates(
                            candidate_mask,
                            "crop_region",
                            max_components=preset_candidate_limit,
                        )
                    )

                if bool(processing_profile.get("enable_second_pass")):
                    recovery_edge_source = boundary_prob if boundary_prob is not None else owt_edge
                    missed_mask, missed_diag = _build_recovery_missed_mask(
                        candidate_masks_payload=candidate_masks_payload,
                        candidate_mask=candidate_mask,
                        processing_profile=processing_profile,
                        edge_source=recovery_edge_source if recovery_edge_source is not None else edge_composite,
                    )
                    tile_runtime["recovery_missed_zone_pixels"] = int(np.count_nonzero(missed_mask))
                    tile_runtime["recovery_edge_seed_pixels"] = int(missed_diag.get("edge_seed_pixels") or 0)
                    tile_runtime["recovery_guide_edge_halo_pixels"] = int(
                        missed_diag.get("guide_edge_halo_pixels") or 0
                    )
                    tile_runtime["recovery_boundary_halo_pixels"] = int(
                        missed_diag.get("boundary_halo_pixels") or 0
                    )
                    if missed_diag.get("edge_seed_threshold") is not None:
                        tile_runtime["recovery_edge_seed_threshold"] = float(
                            missed_diag["edge_seed_threshold"]
                        )
                    if np.any(missed_mask):
                        recovery_mask, recovery_debug = run_postprocess(
                            candidate_mask=missed_mask,
                            water_mask=water_mask,
                            classes=classes,
                            ndvi=ndvi_post,
                            ndwi=ndwi_med,
                            cfg=processing_settings,
                            open_water_mask=hydro_masks["open_water_mask"],
                            seasonal_wet_mask=hydro_masks["seasonal_wet_mask"],
                            riparian_soft_mask=hydro_masks["riparian_soft_mask"],
                            riparian_hard_mask=hydro_masks["riparian_hard_mask"],
                            nir=nir_med,
                            swir=swir_med,
                            edge_composite=edge_composite,
                            boundary_prob=boundary_prob if boundary_prob is not None else edge_prob,
                            ndvi_std=ndvi_std,
                            worldcover_mask=worldcover_grid,
                            bbox=tile["bbox_4326"],
                            tile_transform=tile["transform"],
                            out_shape=tile["shape"],
                            crs_epsg=tile["crs"],
                            region_profile=region_boundary_profile,
                            return_debug_stats=True,
                            return_candidate_masks=True,
                        )
                        if np.any(recovery_mask):
                            candidate_mask |= recovery_mask.astype(bool)
                            if not use_simplified_tile_path:
                                candidate_branches.extend(
                                    _connected_component_candidates(
                                        np.asarray(recovery_mask, dtype=bool),
                                        "recovery_second_pass",
                                        max_components=preset_candidate_limit,
                                    )
                                )
                            tile_runtime["recovery_second_pass_used"] = True
                            tile_runtime["recovery_second_pass_pixels"] = int(
                                np.count_nonzero(recovery_mask)
                            )
                            if isinstance(recovery_debug, dict):
                                recovery_stats = dict(recovery_debug.get("stats") or {})
                                if recovery_stats:
                                    tile_runtime["recovery_second_pass_stats"] = recovery_stats

                if candidate_branches:
                    if len(candidate_branches) > preset_candidate_limit:
                        if detect_preset in {"fast", "standard"}:
                            use_simplified_tile_path = True
                            simplified_path_reason = "candidate_overflow_fast_like_tile_path"
                            tile_runtime.setdefault("warnings", []).append(
                                f"candidate overflow: simplified path {len(candidate_branches)}/{preset_candidate_limit}"
                            )
                            runtime_meta["warnings"].append(
                                f"Tile {tile['tile_id']} switched to simplified path because candidate count reached {len(candidate_branches)}"
                            )
                            candidate_branches = []
                        else:
                            candidate_branches.sort(
                                key=lambda c: int(np.count_nonzero(c.mask)) if c.mask is not None else 0,
                                reverse=True,
                            )
                            n_dropped = len(candidate_branches) - preset_candidate_limit
                            candidate_branches = candidate_branches[:preset_candidate_limit]
                            logger.warning(
                                "candidate_cap_applied",
                                tile_id=tile["tile_id"],
                                kept=preset_candidate_limit,
                                dropped=n_dropped,
                                aoi_run_id=run_id_str,
                            )
                            tile_runtime.setdefault("warnings", []).append(
                                f"candidate_cap: kept {preset_candidate_limit}/{preset_candidate_limit + n_dropped} candidates"
                            )
                    if (not use_simplified_tile_path) and bool(
                        detect_pipeline_profile.get("enable_candidate_ranker")
                    ):
                        from processing.fields.candidate_ranker import (
                            compute_branch_agreement,
                            compute_candidate_features,
                            rank_and_suppress,
                            score_candidates_rule_based,
                        )
                        candidate_total = max(len(candidate_branches), 1)
                        candidate_progress_last_emit = 0.0
                        candidate_progress_steps = {
                            "features": (0.645, 0.020, "candidate features"),
                            "branch_agreement": (0.665, 0.020, "branch agreement"),
                            "rank_prepare": (0.685, 0.005, "candidate rank prep"),
                            "suppress": (0.690, 0.020, "candidate suppress"),
                            "done": (0.710, 0.000, "candidate rank"),
                        }

                        def _candidate_progress_callback(
                            completed: int,
                            total: int,
                            stage_name: str,
                            *,
                            force: bool = False,
                        ) -> None:
                            nonlocal candidate_progress_last_emit
                            safe_total = max(int(total), 1)
                            safe_completed = min(max(int(completed), 0), safe_total)
                            phase_start, phase_span, detail_prefix = candidate_progress_steps.get(
                                str(stage_name),
                                (0.690, 0.020, "candidate stage"),
                            )
                            now = time.time()
                            if not force and safe_completed < safe_total and (
                                now - candidate_progress_last_emit
                            ) < 1.0:
                                return
                            candidate_progress_last_emit = now
                            ratio = safe_completed / safe_total
                            phase_fraction = phase_start + (phase_span * ratio)
                            _set_tile_progress(
                                run,
                                runtime_meta,
                                session,
                                "candidate_postprocess",
                                tile_index=i,
                                tile_count=len(tiles),
                                phase_fraction=phase_fraction,
                                detail=f"{detail_prefix} {safe_completed}/{safe_total}",
                                stage_progress_pct=ratio * 100.0,
                                force_commit=force,
                            )

                        _candidate_progress_callback(0, candidate_total, "features", force=True)

                        for idx, candidate in enumerate(candidate_branches, start=1):
                            compute_candidate_features(
                                candidate,
                                edge_composite=edge_composite,
                                ndvi=ndvi_post,
                                ndvi_std=ndvi_std,
                                ndre=ndre_post,
                                boundary_prob=boundary_prob if boundary_prob is not None else edge_prob,
                                worldcover=worldcover_grid,
                                road_mask=road_mask_for_ranking,
                                tile_quality_score=_candidate_tile_quality_score(tile_runtime),
                                selected_dates_count=n_valid_scenes,
                            )
                            _candidate_progress_callback(idx, candidate_total, "features")
                        _candidate_progress_callback(
                            candidate_total,
                            candidate_total,
                            "features",
                            force=True,
                        )
                        compute_branch_agreement(
                            candidate_branches,
                            progress_callback=_candidate_progress_callback,
                        )
                        score_candidates_rule_based(candidate_branches)
                        _apply_branch_score_bias(candidate_branches, processing_profile)
                        _candidate_progress_callback(0, candidate_total, "rank_prepare", force=True)
                        ranked_candidates = rank_and_suppress(
                            candidate_branches,
                            cfg=processing_settings,
                            progress_callback=_candidate_progress_callback,
                        )
                        _candidate_progress_callback(
                            candidate_total,
                            candidate_total,
                            "done",
                            force=True,
                        )
                        candidate_summary = _summarize_ranked_candidates(ranked_candidates)
                        tile_runtime["candidate_branch_counts"] = candidate_summary["candidate_branch_counts"]
                        tile_runtime["candidate_reject_summary"] = candidate_summary["candidate_reject_summary"]
                        tile_runtime["candidates_total"] = int(candidate_summary["candidates_total"])
                        tile_runtime["candidates_kept"] = int(candidate_summary["candidates_kept"])
                        _accumulate_candidate_summary(runtime_meta, candidate_summary)

                        kept_masks = [
                            np.asarray(ranked.candidate.mask, dtype=bool)
                            for ranked in ranked_candidates
                            if bool(ranked.keep)
                        ]
                        if kept_masks:
                            kept_union = np.logical_or.reduce(kept_masks)
                            if (
                                tile_runtime.get("processing_profile") != "normal"
                                or not np.any(candidate_mask)
                            ):
                                candidate_mask = kept_union
                            else:
                                candidate_mask |= kept_union

                primary_sam_used = False
                labels = np.zeros_like(candidate_mask, dtype=np.int32)
                if (
                    (not use_simplified_tile_path)
                    and sam_primary_segmentor is not None
                    and sam_masks_to_label is not None
                ):
                    try:
                        with SAM2_INFERENCE_TIME.time():
                            sam_masks, sam_scores = sam_primary_segmentor.segment_fields(
                                edge_prob,
                                ndvi_post,
                                ndvi_std,
                                tile["transform"],
                                settings,
                                candidate_mask=candidate_mask,
                            )
                        labels = sam_masks_to_label(sam_masks, candidate_mask.shape)
                        if bool(getattr(settings, "FEATURE_SNIC_REFINE", False)) and labels.max() > 0:
                            from processing.fields.snic_merge import snic_merge_fields

                            labels = snic_merge_fields(labels, ndvi_post, ndvi_std, settings)
                        primary_sam_used = int(labels.max()) > 0
                        tile_runtime["sam_primary_masks"] = int(len(sam_masks))
                        tile_runtime["sam_primary_scores"] = (
                            [round(float(s), 4) for s in np.asarray(sam_scores).tolist()[:10]]
                            if len(sam_scores) > 0
                            else []
                        )
                    except Exception as exc:
                        logger.warning(
                            "sam2_primary_fallback",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )

                if use_simplified_tile_path:
                    labels, simple_diag = _simple_candidate_select(
                        candidate_mask,
                        max_components=simplified_component_cap,
                    )
                    tile_runtime["watershed_applied"] = False
                    tile_runtime["watershed_skipped_reason"] = simplified_path_reason
                    candidate_summary = {
                        "candidate_branch_counts": {
                            "preview": {
                                "total": int(simple_diag.get("component_count") or 0),
                                "kept": int(simple_diag.get("kept_components") or 0),
                            }
                        },
                        "candidate_reject_summary": (
                            {"preview_pruned": int(simple_diag.get("dropped_components") or 0)}
                            if int(simple_diag.get("dropped_components") or 0) > 0
                            else {}
                        ),
                        "candidates_total": int(simple_diag.get("component_count") or 0),
                        "candidates_kept": int(simple_diag.get("kept_components") or 0),
                    }
                    tile_runtime["candidate_branch_counts"] = candidate_summary["candidate_branch_counts"]
                    tile_runtime["candidate_reject_summary"] = candidate_summary["candidate_reject_summary"]
                    tile_runtime["candidates_total"] = int(candidate_summary["candidates_total"])
                    tile_runtime["candidates_kept"] = int(candidate_summary["candidates_kept"])
                    _accumulate_candidate_summary(runtime_meta, candidate_summary)
                    _set_tile_progress(
                        run,
                        runtime_meta,
                        session,
                        "segmentation",
                        tile_index=i,
                        tile_count=len(tiles),
                        phase_fraction=0.74,
                        detail=(
                            f"preview segments {int(simple_diag.get('kept_components') or 0)}/"
                            f"{int(simple_diag.get('component_count') or 0)}"
                        ),
                    )
                elif not primary_sam_used:
                    tile_custom_seed_points = (
                        _tile_seed_points_from_lonlat(
                            seed_points_lonlat=seed_points_lonlat,
                            tile_transform=tile["transform"],
                            tile_crs=tile["crs"],
                        )
                        if seed_mode == "custom"
                        else None
                    )
                    labels = watershed_segment(
                        edge_prob,
                        candidate_mask,
                        osm_mask=None,
                        lambda_edge=processing_settings.WATERSHED_LAMBDA,
                        min_distance=processing_settings.WATERSHED_MIN_DISTANCE,
                        seed_mode=seed_mode,
                        custom_seed_points=tile_custom_seed_points,
                        grid_step=max(8, int(processing_settings.WATERSHED_MIN_DISTANCE * 2)),
                        precomputed_distance=ml_distance_map,
                        ndvi=ndvi_post,
                        ndvi_std=ndvi_std,
                        boundary_prob=boundary_prob if boundary_prob is not None else edge_prob,
                        owt_edge=owt_edge,
                        cfg=processing_settings,
                        return_diagnostics=True,
                        progress_callback=segmentation_progress_callback,
                    )
                    if isinstance(labels, tuple):
                        labels, segmentation_debug = labels
                    else:
                        segmentation_debug = {}
                    tile_runtime["watershed_applied"] = bool(
                        (segmentation_debug or {}).get("watershed_applied")
                    )
                    tile_runtime["watershed_skipped_reason"] = (segmentation_debug or {}).get(
                        "watershed_skipped_reason"
                    )
                    tile_runtime["watershed_rollback_reason"] = (segmentation_debug or {}).get(
                        "watershed_rollback_reason"
                    )
                    tile_runtime["split_score_p50"] = float(
                        (segmentation_debug or {}).get("split_score_p50") or 0.0
                    )
                    tile_runtime["split_score_p90"] = float(
                        (segmentation_debug or {}).get("split_score_p90") or 0.0
                    )
                    if segmentation_debug:
                        tile_runtime["components_after_watershed"] = int(
                            (segmentation_debug or {}).get("components_after_watershed") or 0
                        )
                        tile_runtime.setdefault("geometry_diagnostics", {})
                        tile_runtime["geometry_diagnostics"] = {
                            **dict(tile_runtime.get("geometry_diagnostics") or {}),
                            **{
                                key: value
                                for key, value in segmentation_debug.items()
                                if key
                                in {
                                    "watershed_applied",
                                    "watershed_skipped_reason",
                                    "watershed_rollback_reason",
                                    "split_score_p50",
                                    "split_score_p90",
                                    "components_before_watershed",
                                    "components_after_watershed",
                                }
                            },
                        }

                if use_simplified_tile_path:
                    labels_clean = filter_segments_preview(
                        labels,
                        {**pheno, "ndwi": ndwi_med},
                        min_area_m2=min_field_area_ha * 10000,
                        min_ndvi_delta=max(
                            0.02,
                            float(getattr(processing_settings, "OBIA_MIN_NDVI_DELTA", 0.15)) * 0.2,
                        ),
                        max_mean_ndwi=(
                            float(getattr(processing_settings, "OBIA_MAX_MEAN_NDWI", 0.2)) + 0.12
                            if getattr(processing_settings, "OBIA_MAX_MEAN_NDWI", None) is not None
                            else None
                        ),
                        pixel_size_m=resolution_m,
                        lc_fractions=tile_lc_fractions,
                        max_tree_frac=0.85 if tile_lc_fractions else None,
                        max_water_frac=0.45 if tile_lc_fractions else None,
                        progress_callback=obia_progress_callback,
                    )
                else:
                    labels_clean = filter_segments(
                        labels,
                        {**pheno, "ndwi": ndwi_med},
                        min_area_m2=min_field_area_ha * 10000,
                        max_shape_index=processing_settings.OBIA_MAX_SHAPE_INDEX,
                        min_ndvi_delta=processing_settings.OBIA_MIN_NDVI_DELTA,
                        max_mean_ndwi=processing_settings.OBIA_MAX_MEAN_NDWI,
                        pixel_size_m=resolution_m,
                        lc_fractions=tile_lc_fractions,
                        min_cropland_frac=processing_settings.WC_MIN_CROPLAND_FRAC if tile_lc_fractions else None,
                        max_noncrop_frac=processing_settings.WC_MAX_NONCROP_FRAC if tile_lc_fractions else None,
                        shape_index_ideal=processing_settings.OBIA_SHAPE_INDEX_IDEAL,
                        shape_index_hard_max=processing_settings.OBIA_SHAPE_INDEX_HARD_MAX,
                        shape_ndvi_delta_override=processing_settings.OBIA_SHAPE_NDVI_DELTA_OVERRIDE,
                        max_hole_frac=processing_settings.OBIA_MAX_HOLE_FRAC,
                        max_internal_tree_frac=processing_settings.OBIA_MAX_INTERNAL_TREE_FRAC if tile_lc_fractions else None,
                        max_internal_water_frac=processing_settings.OBIA_MAX_INTERNAL_WATER_FRAC if tile_lc_fractions else None,
                        relax_labels=(
                            {
                                int(lbl)
                                for lbl in np.unique(labels)
                                if lbl > 0
                                and boundary_prob is not None
                                and float(np.mean(boundary_prob[labels == lbl]))
                                >= float(getattr(processing_settings, "OBIA_RELAX_MIN_BOUNDARY_CONF", 0.65))
                            }
                            if bool(getattr(processing_settings, "OBIA_RELAX_IF_ML_CONFIDENT", True))
                            and ml_primary_used
                            and boundary_prob is not None
                            else None
                        ),
                        relax_shape_multiplier=float(
                            getattr(processing_settings, "OBIA_RELAX_SHAPE_MULTIPLIER", 1.35)
                        ),
                        relax_hole_multiplier=float(
                            getattr(processing_settings, "OBIA_RELAX_HOLE_MULTIPLIER", 1.50)
                        ),
                        relax_tree_multiplier=float(
                            getattr(processing_settings, "OBIA_RELAX_TREE_MULTIPLIER", 1.50)
                        ),
                        progress_callback=obia_progress_callback,
                    )

                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "segmentation",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.72,
                )
                pre_vector_area_px = int(np.count_nonzero(labels_clean > 0))
                tile_runtime["pre_vector_area_px"] = pre_vector_area_px
                vectorize_tol = _resolve_vectorize_simplify_tol(settings, region_boundary_profile)
                gdf = polygonize_labels(
                    labels_clean,
                    tile["transform"],
                    tile["crs"],
                    min_area_ha=min_field_area_ha,
                    simplify_tol_m=vectorize_tol,
                    progress_callback=vectorize_progress_callback,
                )
                post_vector_area_m2 = float(gdf["area_m2"].sum()) if (not gdf.empty and "area_m2" in gdf.columns) else 0.0
                tile_runtime["post_vector_area_m2"] = round(post_vector_area_m2, 2)
                tile_runtime["post_smooth_area_m2"] = round(post_vector_area_m2, 2)
                pixel_area_m2 = float(resolution_m) * float(resolution_m)
                vectorize_area_ratio = post_vector_area_m2 / max(pre_vector_area_px * pixel_area_m2, 1e-6)
                tile_runtime["vectorize_area_ratio"] = round(float(vectorize_area_ratio), 4)
                tile_runtime["smooth_area_ratio"] = round(float(vectorize_area_ratio), 4)
                if (
                    region_boundary_profile == "north_boundary"
                    and vectorize_tol > 0.0
                    and pre_vector_area_px > 0
                    and vectorize_area_ratio < float(getattr(settings, "NORTH_STAGE_ROLLBACK_MIN_AREA_RATIO", 0.95))
                ):
                    gdf_retry = polygonize_labels(
                        labels_clean,
                        tile["transform"],
                        tile["crs"],
                        min_area_ha=min_field_area_ha,
                        simplify_tol_m=0.0,
                        progress_callback=vectorize_progress_callback,
                    )
                    retry_area_m2 = (
                        float(gdf_retry["area_m2"].sum())
                        if (not gdf_retry.empty and "area_m2" in gdf_retry.columns)
                        else 0.0
                    )
                    retry_ratio = retry_area_m2 / max(pre_vector_area_px * pixel_area_m2, 1e-6)
                    if retry_ratio >= vectorize_area_ratio:
                        gdf = gdf_retry
                        post_vector_area_m2 = retry_area_m2
                        vectorize_area_ratio = retry_ratio
                        tile_runtime["post_vector_area_m2"] = round(post_vector_area_m2, 2)
                        tile_runtime["post_smooth_area_m2"] = round(post_vector_area_m2, 2)
                        tile_runtime["vectorize_area_ratio"] = round(float(vectorize_area_ratio), 4)
                        tile_runtime["smooth_area_ratio"] = round(float(vectorize_area_ratio), 4)
                        actions = list(tile_runtime.get("region_profile_actions") or [])
                        actions.append("north_vectorize_rollback")
                        tile_runtime["region_profile_actions"] = actions
                if (
                    detect_pipeline_profile.get("enable_snake_refine")
                    and settings.SNAKE_REFINE_ENABLED
                    and not gdf.empty
                ):
                    try:
                        from processing.fields.active_contour_refine import refine_all_fields

                        gdf_local = gdf.to_crs(tile["crs"])
                        gdf_before_refine = gdf_local.copy()
                        gdf_local, snake_diag = refine_all_fields(
                            gdf_local,
                            owt_edge,
                            tile["transform"],
                            settings,
                            return_diagnostics=True,
                            progress_callback=snake_progress_callback,
                        )
                        geom_diag = _summarize_geometry_diagnostics(
                            gdf_before_refine,
                            gdf_local,
                            snake_diag,
                        )
                        tile_runtime["geometry_diagnostics"] = geom_diag
                        tile_runtime["contour_shrink_ratio"] = round(
                            float(geom_diag.get("contour_shrink_ratio", 1.0)),
                            4,
                        )
                        tile_runtime["centroid_shift_m"] = round(
                            float(geom_diag.get("centroid_shift_m", 0.0)),
                            4,
                        )
                        gdf = gdf_local.to_crs("EPSG:4326")
                    except Exception as exc:
                        logger.warning(
                            "snake_refine_skipped",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )
                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "boundary_refine",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.84,
                )
                traditional_gdf = gdf.copy()
                traditional_field_count = int(len(traditional_gdf))

                framework_sam = bool(getattr(settings, "FRAMEWORK_SAM_ENABLED", True))
                sam_runtime_policy = str(
                    getattr(settings, "SAM_RUNTIME_POLICY", "safe_optional")
                ).strip().lower()
                sam_requested = bool(
                    detect_pipeline_profile.get("enable_sam")
                    and
                    sam_runtime_policy != "disabled"
                    and framework_sam
                    and not primary_sam_used
                    and not use_simplified_tile_path
                    and auto_detect_version >= 2
                    and (
                        getattr(settings, "FRAMEWORK_SAM_FIELD_DET", True)
                        or getattr(settings, "SAM_FIELD_DET", False)
                        or settings.SAM_ENABLED
                    )
                )
                if sam_requested:
                    try:
                        from processing.fields.merge_hybrid import merge_sam_with_traditional
                        from processing.fields.sam_field_boundary import (
                            build_sam_composite,
                            run_sam_with_crop_boxes,
                            run_sam_sequential,
                        )
                        from processing.fields.sam_filter import filter_sam_polygons

                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "sam_refine",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.90,
                        )
                        sam_output_dir = (
                            Path(__file__).resolve().parents[1]
                            / settings.SAM_OUTPUT_DIR.format(run_id=run_id_str)
                            / str(tile["tile_id"])
                        )
                        sam_input = build_sam_composite(
                            ndvi_post,
                            edge_prob,
                            ndvi_mean_post,
                        )
                        candidate_coverage_pct = float(np.count_nonzero(candidate_mask) / max(candidate_mask.size, 1) * 100.0)
                        component_count = max(1, traditional_field_count)
                        sam_allowed, sam_skip_reason, sam_peak_mem_mb = _sam_preflight_budget(
                            composite_uint8=sam_input,
                            candidate_mask=candidate_mask,
                            candidate_coverage_pct=candidate_coverage_pct,
                            component_count=component_count,
                            cfg=settings,
                        )
                        tile_runtime["sam_peak_mem_estimate_mb"] = round(float(sam_peak_mem_mb), 2)
                        tile_runtime["sam_tile_mode"] = "skipped"
                        runtime_meta["sam_memory_budget_mb"] = float(
                            getattr(settings, "SAM_MAX_EST_MEMORY_MB", 2200)
                        )
                        sam_raw = traditional_gdf.iloc[0:0].copy()

                        if traditional_gdf.empty:
                            sam_allowed = False
                            sam_skip_reason = "no_traditional_candidates"

                        if not sam_allowed:
                            runtime_meta["sam_runtime_mode"] = "skipped_budget"
                            tile_runtime["sam_runtime_mode"] = "skipped_budget"
                            runtime_meta["sam_failure_reason"] = sam_skip_reason
                            logger.info(
                                "sam_budget_skip",
                                tile_id=tile["tile_id"],
                                reason=sam_skip_reason,
                                estimated_mem_mb=round(float(sam_peak_mem_mb), 2),
                                candidate_coverage_pct=round(candidate_coverage_pct, 2),
                                component_count=component_count,
                            )
                        else:
                            _set_tile_progress(
                                run,
                                runtime_meta,
                                session,
                                "sam_refine",
                                tile_index=i,
                                tile_count=len(tiles),
                                phase_fraction=0.93,
                            )
                            sam_t0 = time.time()
                            if bool(getattr(settings, "SAM_USE_CROP_BOXES", True)):
                                tile_runtime["sam_tile_mode"] = "crop_boxes"
                                sam_raw = run_sam_with_crop_boxes(
                                    sam_input,
                                    traditional_gdf,
                                    tile["transform"],
                                    tile["crs"],
                                    settings,
                                    sam_output_dir,
                                )
                            else:
                                tile_runtime["sam_tile_mode"] = "full"
                                sam_raw = run_sam_sequential(
                                    sam_input,
                                    tile["transform"],
                                    tile["crs"],
                                    settings,
                                    sam_output_dir,
                                )
                            tile_runtime["sam_elapsed_s"] = round(float(time.time() - sam_t0), 2)
                            tile_runtime["sam_polygons_before_filter"] = int(len(sam_raw))
                            runtime_meta["sam_runtime_mode"] = "executed"
                            tile_runtime["sam_runtime_mode"] = "executed"
                            runtime_meta["sam_failure_reason"] = None

                        sam_filtered = filter_sam_polygons(
                            sam_raw,
                            max_ndvi=ndvi_post,
                            ndvi_std=ndvi_std,
                            water_mask=water_mask,
                            forest_mask=(classes == FOREST),
                            worldcover_mask=(
                                worldcover_grid
                                if worldcover_grid is not None
                                else exclusion_mask
                                if exclusion_mask is not None
                                else np.zeros_like(candidate_mask, dtype=bool)
                            ),
                            transform=tile["transform"],
                            cfg=settings,
                        )
                        tile_runtime["sam_polygons_after_filter"] = int(len(sam_filtered))
                        if not sam_filtered.empty:
                            sam_filtered = sam_filtered.to_crs("EPSG:4326")
                        if auto_detect_version == 2:
                            gdf = merge_sam_with_traditional(
                                traditional_gdf,
                                sam_filtered,
                                settings,
                                ndvi_mask=candidate_mask,
                            )
                        elif auto_detect_version >= 3:
                            if sam_filtered.empty:
                                logger.warning(
                                    "sam_full_mode_fallback_to_traditional",
                                    tile_id=tile["tile_id"],
                                )
                                gdf = traditional_gdf
                            else:
                                gdf = merge_sam_with_traditional(
                                    traditional_gdf.iloc[0:0].copy(),
                                    sam_filtered,
                                    settings,
                                    ndvi_mask=candidate_mask,
                                )
                        else:
                            gdf = traditional_gdf
                        tile_runtime["sam_fields"] = int(len(sam_filtered))
                        tile_runtime["hybrid_fields"] = int(len(gdf))
                        tile_runtime["sam_mode"] = (
                            tile_runtime["sam_tile_mode"] if auto_detect_version >= 3 else "mixed"
                        )
                        logger.info(
                            "hybrid_merge_done",
                            tile_id=tile["tile_id"],
                            sam_primary=bool(getattr(settings, "SAM_FIELD_DET", False)),
                            traditional_fields=traditional_field_count,
                            sam_fields=int(len(sam_filtered)),
                            merged_fields=int(len(gdf)),
                        )
                        if debug_run:
                            raw_gpkg = _save_debug_vector_layer(
                                run_id_str,
                                tile["tile_id"],
                                "sam_raw",
                                sam_raw.to_crs("EPSG:4326") if not sam_raw.empty else sam_raw,
                            )
                            filtered_gpkg = _save_debug_vector_layer(
                                run_id_str,
                                tile["tile_id"],
                                "sam_filtered",
                                sam_filtered,
                            )
                            if raw_gpkg:
                                tile_runtime["sam_raw_gpkg"] = raw_gpkg
                            if filtered_gpkg:
                                tile_runtime["sam_filtered_gpkg"] = filtered_gpkg
                    except Exception as exc:
                        logger.warning(
                            "sam_pipeline_skipped",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )
                        runtime_meta["sam_runtime_mode"] = "skipped_error"
                        tile_runtime["sam_runtime_mode"] = "skipped_error"
                        runtime_meta["sam_failure_reason"] = str(exc)[:240]
                        gdf = traditional_gdf
                else:
                    runtime_meta["sam_runtime_mode"] = (
                        "disabled"
                        if sam_runtime_policy == "disabled"
                        else "fallback_non_sam"
                    )
                    tile_runtime["sam_runtime_mode"] = runtime_meta["sam_runtime_mode"]
                    gdf = traditional_gdf

                if debug_run:
                    traditional_gpkg = _save_debug_vector_layer(
                        run_id_str,
                        tile["tile_id"],
                        "traditional_fields",
                        traditional_gdf,
                    )
                    final_gpkg = _save_debug_vector_layer(
                        run_id_str,
                        tile["tile_id"],
                        "final_fields",
                        gdf,
                    )
                    if traditional_gpkg:
                        tile_runtime["traditional_gpkg"] = traditional_gpkg
                    if final_gpkg:
                        tile_runtime["final_gpkg"] = final_gpkg
                    if settings.DEBUG_COMPARE_VERSIONS:
                        v3_gpkg = _save_debug_vector_layer(
                            run_id_str,
                            tile["tile_id"],
                            "v3_fields",
                            traditional_gdf,
                        )
                        v4_gpkg = _save_debug_vector_layer(
                            run_id_str,
                            tile["tile_id"],
                            "v4_fields",
                            gdf,
                        )
                        if v3_gpkg:
                            tile_runtime["v3_gpkg"] = v3_gpkg
                        if v4_gpkg:
                            tile_runtime["v4_gpkg"] = v4_gpkg

                if (
                    detect_pipeline_profile.get("enable_object_classifier")
                    and settings.USE_OBJECT_CLASSIFIER
                    and not gdf.empty
                ):
                    _set_tile_progress(
                        run,
                        runtime_meta,
                        session,
                        "object_classifier",
                        tile_index=i,
                        tile_count=len(tiles),
                        phase_fraction=0.88,
                        detail="tile features",
                    )
                    try:
                        from processing.fields.object_classifier import compute_object_features
                        from processing.priors.worldcover import CROPLAND_CLASS
                        from scipy.ndimage import distance_transform_edt

                        scl_valid_fraction_map = _safe_valid_fraction(valid_count, valid_scene_total)
                        road_distance_m = np.full(tile["shape"], 1000.0, dtype=np.float32)
                        if isinstance(postprocess_debug, dict):
                            debug_masks = postprocess_debug.get("masks", {})
                            road_mask = None
                            for road_key in ("step_01_road_mask", "road_mask"):
                                if road_key in debug_masks:
                                    road_mask = np.asarray(debug_masks[road_key], dtype=np.uint8) > 0
                                    break
                            if road_mask is not None and np.any(road_mask):
                                road_distance_m = (
                                    distance_transform_edt(~road_mask).astype(np.float32)
                                    * float(resolution_m)
                                )

                        raster_features = {
                            "ndvi_mean": ndvi_mean_post.astype(np.float32, copy=False),
                            "ndvi_max": ndvi_post.astype(np.float32, copy=False),
                            "ndvi_delta": pheno["ndvi_delta"].astype(np.float32, copy=False),
                            "ndwi_mean": ndwi_mean.astype(np.float32, copy=False),
                            "msi_mean": msi_med.astype(np.float32, copy=False),
                            "bsi_mean": bsi_med.astype(np.float32, copy=False),
                            "ndvi_variance": np.square(ndvi_std.astype(np.float32, copy=False)),
                            "growth_amplitude": tc_growth_amplitude.astype(np.float32, copy=False),
                            "has_growth_peak": tc_has_growth_peak.astype(np.float32, copy=False),
                            "ndvi_entropy": tc_ndvi_entropy.astype(np.float32, copy=False),
                            "distance_to_road_m": road_distance_m,
                            "scl_valid_fraction_mean": scl_valid_fraction_map.astype(
                                np.float32,
                                copy=False,
                            ),
                            "ndvi_std": ndvi_std.astype(np.float32, copy=False),
                            "ndvistd_mean": ndvi_std.astype(np.float32, copy=False),
                            "edge_mean": edge_composite.astype(np.float32, copy=False),
                            "edge_max": edge_composite.astype(np.float32, copy=False),
                        }
                        worldcover_crop_mask = None
                        if worldcover_grid is not None:
                            worldcover_crop_mask = worldcover_grid == CROPLAND_CLASS

                        gdf = compute_object_features(
                            gdf.to_crs(tile["crs"]),
                            raster_data=raster_features,
                            worldcover_mask=worldcover_crop_mask,
                            transform=tile["transform"],
                            progress_callback=object_feature_progress_callback,
                        ).to_crs("EPSG:4326")
                    except Exception as exc:
                        logger.warning(
                            "object_classifier_feature_prep_skipped",
                            tile_id=tile["tile_id"],
                            error=str(exc),
                        )

                if not gdf.empty:
                    grid_levels = list(_iter_grid_zoom_levels(int(getattr(settings, "GRID_LAYER_CELL_PX", 64))))
                    total_grid_levels = max(len(grid_levels), 1)
                    tile_grid_rows: list[dict[str, Any]] = []
                    for zoom_pos, (zoom_level, cell_px) in enumerate(grid_levels, start=1):
                        _set_tile_progress(
                            run,
                            runtime_meta,
                            session,
                            "tile_finalize",
                            tile_index=i,
                            tile_count=len(tiles),
                            phase_fraction=0.90 + (0.05 * (zoom_pos - 1) / total_grid_levels),
                            detail=f"grid zoom {zoom_pos}/{total_grid_levels}",
                        )
                        tile_grid_rows.extend(
                            _build_grid_cells_for_tile(
                                tile=tile,
                                labels_clean=labels_clean,
                                gdf=gdf,
                                ndvi_mean=ndvi_mean_post,
                                ndwi_mean=ndwi_mean,
                                ndmi_mean=ndmi_mean.astype(np.float32, copy=False),
                                bsi_mean=bsi_med,
                                weather_snapshot=weather_snapshot,
                                zoom_level=zoom_level,
                                cell_px=cell_px,
                                grid_origin_utm=grid_origin_utm,
                            )
                        )
                    # Flush grid cells per tile to avoid accumulating all in memory
                    grid_cell_rows.extend(tile_grid_rows)
                    total_grid_cells_inserted += len(tile_grid_rows)
                    del tile_grid_rows
                    all_gdfs.append(gdf)
                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "tile_finalize",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=0.96,
                    detail="tile artifacts",
                )

                tile_runtime["n_fields"] = int(len(gdf))
                tile_runtime["labels_raw"] = int(labels.max())
                tile_runtime["labels_clean"] = int(labels_clean.max())

                if debug_run:
                    tile_runtime["debug_artifact"] = _save_debug_tile_dump(
                        run_id_str,
                        tile["tile_id"],
                        {
                            "candidate_mask": candidate_mask.astype(np.uint8),
                            "max_ndvi": ndvi_post.astype(np.float32),
                            "mean_ndvi": ndvi_mean_post.astype(np.float32),
                            "ndvi_std": ndvi_std.astype(np.float32),
                            "edge_composite": edge_composite.astype(np.float32),
                            "edge_prob": edge_prob.astype(np.float32),
                            "ml_distance": (
                                ml_distance_map.astype(np.float32)
                                if ml_distance_map is not None
                                else np.zeros_like(candidate_mask, dtype=np.float32)
                            ),
                            "owt_edge": owt_edge.astype(np.float32),
                            "scl_median": scl_median.astype(np.float32),
                            "labels": labels.astype(np.int32),
                            "labels_clean": labels_clean.astype(np.int32),
                            "water_mask": water_mask.astype(np.uint8),
                            "worldcover_mask": (
                                worldcover_grid.astype(np.uint8)
                                if worldcover_grid is not None
                                else np.zeros_like(candidate_mask, dtype=np.uint8)
                            ),
                            "osm_mask": np.zeros_like(candidate_mask, dtype=np.uint8),
                            **(
                                {
                                    "s1_vv_edge": s1_features["VV_edge"].astype(np.float32),
                                    "s1_vhvv_ratio": s1_features["VHVV_ratio"].astype(np.float32),
                                }
                                if s1_features is not None
                                else {}
                            ),
                            **(
                                {
                                    name: np.asarray(mask)
                                    for name, mask in postprocess_debug.get("masks", {}).items()
                                }
                                if postprocess_debug is not None
                                else {}
                            ),
                        },
                    )

                TILES_PROCESSED.inc()
                tile_time = time.time() - tile_t0
                latency_breakdown["tiles_total_s"] += float(tile_time)
                tile_runtime["time_s"] = round(tile_time, 2)
                runtime_meta["selected_dates"] = sorted(selected_date_tokens)
                runtime_meta["tiles"].append(tile_runtime)
                _persist_runtime_meta(run, runtime_meta)

                # --- V4: Store tile diagnostic record ---
                try:
                    diag = TileDiagnostic(
                        organization_id=run.organization_id,
                        aoi_run_id=run_id,
                        tile_index=i,
                        quality_mode=str(tile_runtime.get("qc_mode", "normal")),
                        coverage_fraction=tile_runtime.get("qc_coverage"),
                        valid_scene_count=tile_runtime.get("n_valid_scenes", 0),
                        edge_strength_mean=None,
                        edge_strength_p90=tile_runtime.get("qc_edge_p90"),
                        ndvi_temporal_std=tile_runtime.get("qc_ndvi_std"),
                        cloud_interference=None,
                        selected_dates=list(tile_runtime.get("selected_dates", [])),
                        runtime_flags={
                            "low_quality_input": tile_runtime.get("low_quality_input", False),
                            "quality_gate_failed": tile_runtime.get("quality_gate_failed", False),
                            "ml_primary_used": tile_runtime.get("ml_primary_used", False),
                            "fusion_profile": tile_runtime.get("fusion_profile", "none"),
                            "processing_profile": tile_runtime.get("processing_profile", "normal"),
                            "qc_reasons": tile_runtime.get("qc_reasons", []),
                            "candidate_branch_counts": tile_runtime.get("candidate_branch_counts", {}),
                        },
                        artifact_refs={
                            "debug_artifact": tile_runtime.get("debug_artifact"),
                            "traditional_gpkg": tile_runtime.get("traditional_gpkg"),
                            "sam_raw_gpkg": tile_runtime.get("sam_raw_gpkg"),
                            "sam_filtered_gpkg": tile_runtime.get("sam_filtered_gpkg"),
                        },
                        candidates_total=int(tile_runtime.get("candidates_total", 0)),
                        candidates_kept=int(tile_runtime.get("candidates_kept", 0)),
                        processing_time_s=tile_time,
                    )
                    session.add(diag)
                    session.flush()
                    _persist_detection_candidates(
                        session=session,
                        run=run,
                        run_id=run_id,
                        tile_diagnostic_id=int(diag.id),
                        ranked_candidates=ranked_candidates,
                        tile_transform=tile["transform"],
                        tile_crs=tile["crs"],
                        resolution_m=resolution_m,
                        model_version=_resolve_detection_model_version(
                            processing_settings,
                            auto_detect_version,
                        ),
                        FieldDetectionCandidateModel=FieldDetectionCandidate,
                    )
                except Exception as exc:
                    logger.warning("tile_diagnostic_insert_skipped", tile_id=tile["tile_id"], error=str(exc))
                logger.info(
                    "tile_done",
                    tile_id=tile["tile_id"],
                    n_fields=len(gdf),
                    time_s=round(tile_time, 2),
                )

                _set_tile_progress(
                    run,
                    runtime_meta,
                    session,
                    "tile_done",
                    tile_index=i,
                    tile_count=len(tiles),
                    phase_fraction=1.0,
                    force_commit=True,
                )
                # Aggressive cleanup: free all tile-scoped arrays
                result = bands = scl = valid_mask = indices = valid_sel = ndvi_sel = None
                ndwi_sel = mndwi_sel = edge_bands = temporal_composite = edge_composite = None
                ndvi_std = feature_stack = extent_prob = boundary_prob = distance_prob = None
                labels = labels_clean = sam_input = sam_raw = sam_filtered = None
                ranked_candidates = []
                candidate_branches = []
                rgb_g_med = rgb_b_med = green_med = None
                _run_tile_gc(settings)

            # Step 10: Merge tiles
            failure_stage = "merge_tiles"
            t_step = time.time()
            polygons_before_merge = int(sum(len(gdf) for gdf in all_gdfs))
            _set_post_progress(run, runtime_meta, session, "merge", phase_fraction=0.12)
            logger.info(
                "merge_start",
                aoi_run_id=run_id_str,
                polygons_before_merge=polygons_before_merge,
                tile_count=len(all_gdfs),
            )
            if all_gdfs:
                merge_progress_callback = _make_post_stage_progress_callback(
                    run=run,
                    runtime_meta=runtime_meta,
                    session=session,
                    stage="merge",
                    phase_map={
                        "merge_groups": (0.14, 0.18, "merge groups"),
                    },
                    default_phase=(0.14, 0.18, "merge"),
                )
                merged = merge_tile_polygons(
                    all_gdfs,
                    overlap_m=settings.TILE_OVERLAP_M,
                    min_iou=settings.MERGE_TILE_MIN_IOU,
                    only_in_overlap=settings.MERGE_TILE_ONLY_IN_OVERLAP,
                    progress_callback=merge_progress_callback,
                )
            else:
                import geopandas as gpd

                merged = gpd.GeoDataFrame(
                    columns=["label", "geometry", "area_m2", "perimeter_m"],
                    geometry="geometry",
                    crs="EPSG:4326",
                )

            if (
                detect_pipeline_profile.get("enable_post_merge_smooth")
                and settings.POST_MERGE_SMOOTH
                and not merged.empty
            ):
                try:
                    from processing.fields.boundary_smooth import smooth_all_fields

                    merged_before_smooth = merged.copy()
                    before_area_m2 = (
                        float(merged_before_smooth["area_m2"].sum())
                        if "area_m2" in merged_before_smooth.columns
                        else 0.0
                    )
                    merged = smooth_all_fields(
                        merged,
                        settings,
                        region_profile=region_boundary_profile,
                        progress_callback=_make_post_stage_progress_callback(
                            run=run,
                            runtime_meta=runtime_meta,
                            session=session,
                            stage="merge",
                            phase_map={
                                "smooth_fields": (0.33, 0.12, "smooth fields"),
                            },
                            default_phase=(0.33, 0.12, "smooth"),
                        ),
                    )
                    after_area_m2 = (
                        float(merged["area_m2"].sum())
                        if (not merged.empty and "area_m2" in merged.columns)
                        else 0.0
                    )
                    runtime_meta["area_change_post_smooth"] = round(after_area_m2 - before_area_m2, 2)
                    if (
                        region_boundary_profile == "north_boundary"
                        and not merged_before_smooth.empty
                    ):
                        smooth_ratio = after_area_m2 / max(before_area_m2, 1e-6)
                        runtime_meta["post_smooth_area_m2"] = round(after_area_m2, 2)
                        runtime_meta["smooth_area_ratio"] = round(float(smooth_ratio), 4)
                        if smooth_ratio < float(getattr(settings, "NORTH_STAGE_ROLLBACK_MIN_AREA_RATIO", 0.95)):
                            merged = merged_before_smooth
                            runtime_meta["post_smooth_area_m2"] = round(before_area_m2, 2)
                            runtime_meta["smooth_area_ratio"] = 1.0
                            actions = list(runtime_meta.get("region_profile_actions") or [])
                            actions.append("north_smooth_rollback")
                            runtime_meta["region_profile_actions"] = actions
                except Exception as exc:
                    logger.warning("merge_smoothing_skipped", error=str(exc))

            merge_time = time.time() - t_step
            STEP_DURATION.labels(step="merge_tiles").observe(merge_time)
            latency_breakdown["merge_tiles_s"] = round(float(merge_time), 3)
            merge_area_stats = summarize_polygon_areas(merged)
            runtime_meta["merge"] = {
                "polygons_before_merge": polygons_before_merge,
                "polygons_after_merge": int(len(merged)),
                "area_quantiles_m2": merge_area_stats,
            }
            _persist_runtime_meta(run, runtime_meta)
            logger.info(
                "merge_done",
                n_fields=len(merged),
                aoi_run_id=run_id_str,
                polygons_before_merge=polygons_before_merge,
                polygons_after_merge=len(merged),
                area_p50_m2=round(merge_area_stats["p50"], 2),
                area_p90_m2=round(merge_area_stats["p90"], 2),
                area_p99_m2=round(merge_area_stats["p99"], 2),
            )
            _set_post_progress(run, runtime_meta, session, "merge", phase_fraction=0.40)

            # Step 10b: ML object classifier filter
            failure_stage = "object_classifier"
            object_classifier_rejected_geojson: list[dict[str, Any]] = []
            _set_post_progress(run, runtime_meta, session, "object_classifier", phase_fraction=0.52)
            if detect_pipeline_profile.get("enable_object_classifier") and settings.USE_OBJECT_CLASSIFIER:
                try:
                    from processing.fields.object_classifier import ObjectClassifier

                    backend_root = Path(__file__).resolve().parents[1]
                    requested_clf_path = Path(str(settings.OBJECT_CLASSIFIER_PATH))
                    clf_candidates: list[Path] = []
                    if requested_clf_path.is_absolute():
                        clf_candidates.append(requested_clf_path)
                    else:
                        clf_candidates.extend(
                            [
                                backend_root / requested_clf_path,
                                backend_root.parent / requested_clf_path,
                                requested_clf_path,
                            ]
                        )
                    clf_candidates.extend(
                        [
                            backend_root / "models" / "object_classifier.pkl",
                            backend_root / "object_classifier.pkl",
                        ]
                    )
                    unique_candidates: list[Path] = []
                    for candidate in clf_candidates:
                        if candidate not in unique_candidates:
                            unique_candidates.append(candidate)
                    clf_path = next((p for p in unique_candidates if p.exists()), unique_candidates[0])
                    if clf_path != requested_clf_path:
                        logger.info(
                            "object_classifier_path_fallback",
                            requested_path=str(requested_clf_path),
                            resolved_path=str(clf_path),
                            candidates=[str(p) for p in unique_candidates],
                        )
                    if clf_path.exists():
                        clf = ObjectClassifier.load(clf_path)
                        scores = clf.predict_proba(merged)
                        merged["field_score"] = scores
                        n_before = len(merged)
                        rejected = merged[merged["field_score"] < settings.OBJECT_MIN_SCORE]
                        object_classifier_rejected_geojson = [
                            geom.__geo_interface__
                            for geom in rejected.geometry
                            if geom is not None and not geom.is_empty
                        ]
                        for _, rej_row in rejected.iterrows():
                            logger.info(
                                "ml_classifier_rejected",
                                aoi_run_id=run_id_str,
                                field_score=round(float(rej_row["field_score"]), 4),
                                area_m2=round(float(rej_row.get("area_m2", 0.0)), 2),
                            )
                        merged = merged[merged["field_score"] >= settings.OBJECT_MIN_SCORE].copy()
                        logger.info(
                            "object_classifier_filter",
                            aoi_run_id=run_id_str,
                            before=n_before,
                            after=len(merged),
                            threshold=settings.OBJECT_MIN_SCORE,
                        )
                    else:
                        logger.warning("object_classifier_not_found", path=str(clf_path))
                except Exception as exc:
                    logger.warning("object_classifier_skipped", error=str(exc))
            _set_post_progress(run, runtime_meta, session, "object_classifier", phase_fraction=0.66)

            # Step 11: Insert into DB
            failure_stage = "db_insert"
            t_step = time.time()
            _set_post_progress(run, runtime_meta, session, "db_insert", phase_fraction=0.78)
            from geoalchemy2.shape import from_shape
            from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

            session.query(GridCell).filter(GridCell.aoi_run_id == run_id).delete(synchronize_session=False)
            inserted_fields: list[tuple[Field, dict[str, Any], dict[str, Any]]] = []
            if grid_cell_rows:
                raw_grid_cell_count = len(grid_cell_rows)
                grid_cell_rows = _aggregate_grid_cell_rows(grid_cell_rows)
                runtime_meta["grid_cells_raw"] = int(raw_grid_cell_count)
                runtime_meta["grid_cells_deduped"] = int(len(grid_cell_rows))
                BULK_CHUNK = 5000
                grid_mappings = [
                    {
                        "organization_id": run.organization_id,
                        "aoi_run_id": run_id,
                        "geom": from_shape(r["geometry"], srid=4326),
                        "zoom_level": int(r["zoom_level"]),
                        "row": int(r["row"]),
                        "col": int(r["col"]),
                        "field_coverage": r.get("field_coverage"),
                        "ndvi_mean": r.get("ndvi_mean"),
                        "ndwi_mean": r.get("ndwi_mean"),
                        "ndmi_mean": r.get("ndmi_mean"),
                        "bsi_mean": r.get("bsi_mean"),
                        "precipitation_mm": r.get("precipitation_mm"),
                        "wind_speed_m_s": r.get("wind_speed_m_s"),
                        "u_wind_10m": r.get("u_wind_10m"),
                        "v_wind_10m": r.get("v_wind_10m"),
                        "wind_direction_deg": r.get("wind_direction_deg"),
                        "gdd_sum": r.get("gdd_sum"),
                        "vpd_mean": r.get("vpd_mean"),
                        "soil_moist": r.get("soil_moist"),
                    }
                    for r in grid_cell_rows
                ]
                for chunk_start in range(0, len(grid_mappings), BULK_CHUNK):
                    session.bulk_insert_mappings(
                        GridCell,
                        grid_mappings[chunk_start : chunk_start + BULK_CHUNK],
                    )
                del grid_mappings  # Free memory after insert
            _grid_cell_count = len(grid_cell_rows)
            del grid_cell_rows  # Free grid cell data
            gc.collect()

            for _, row in merged.iterrows():
                geom = row.geometry
                if geom.geom_type == "Polygon":
                    geom = ShapelyMultiPolygon([geom])
                field = Field(
                    id=uuid.uuid4(),
                    organization_id=run.organization_id,
                    aoi_run_id=run_id,
                    geom=from_shape(geom, srid=4326),
                    area_m2=row.get("area_m2", 0.0),
                    perimeter_m=row.get("perimeter_m", 0.0),
                    quality_score=row.get("field_score"),
                    source=str(detect_pipeline_profile.get("field_source") or "autodetect"),
                )
                session.add(field)
                # Use __geo_interface__ directly - it's already a serializable dict
                inserted_fields.append((field, dict(row), geom.__geo_interface__))

            session.flush()
            if (
                run.organization_id is not None
                and inserted_fields
                and detect_pipeline_profile.get("enable_active_learning")
            ):
                region_rarity = 0.85 if region_boundary_profile in {"north_boundary", "south_recall"} else 0.35
                error_mode_quota = 0.8 if (
                    runtime_meta.get("quality_gate_failed")
                    or runtime_meta.get("water_edge_risk_detected")
                    or runtime_meta.get("road_drift_risk_detected")
                    or runtime_meta.get("date_selection_low_confidence")
                ) else 0.25
                disagreement_score = float(max(0.0, min(1.0, runtime_meta.get("fallback_rate_tile") or 0.0)))
                candidate_specs: list[dict[str, Any]] = []
                for field, field_row, geometry_geojson in inserted_fields:
                    quality_score = field_row.get("field_score")
                    if isinstance(quality_score, (int, float)):
                        uncertainty = float(max(0.0, min(1.0, 1.0 - float(quality_score))))
                    else:
                        uncertainty = 0.5
                    priority = (
                        0.45 * uncertainty
                        + 0.25 * disagreement_score
                        + 0.15 * region_rarity
                        + 0.15 * error_mode_quota
                    )
                    candidate_payload = {
                        "region_band": region_band,
                        "region_boundary_profile": region_boundary_profile,
                        "runtime_warning_flags": {
                            "quality_gate_failed": bool(runtime_meta.get("quality_gate_failed")),
                            "water_edge_risk_detected": bool(runtime_meta.get("water_edge_risk_detected")),
                            "road_drift_risk_detected": bool(runtime_meta.get("road_drift_risk_detected")),
                            "date_selection_low_confidence": bool(runtime_meta.get("date_selection_low_confidence")),
                        },
                        "field_score": quality_score,
                        "source": "autodetect_active_learning",
                    }
                    session.add(
                        ActiveLearningCandidate(
                            organization_id=run.organization_id,
                            aoi_run_id=run.id,
                            field_id=field.id,
                            priority_score=priority,
                            uncertainty_score=uncertainty,
                            rule_ml_disagreement=disagreement_score,
                            region_rarity=region_rarity,
                            error_mode_quota=error_mode_quota,
                            candidate_payload=candidate_payload,
                            status="queued",
                        )
                    )
                    candidate_specs.append(
                        {
                            "field": field,
                            "priority": priority,
                            "uncertainty": uncertainty,
                            "geometry_geojson": geometry_geojson,
                            "payload": candidate_payload,
                        }
                    )

                for spec in sorted(candidate_specs, key=lambda item: item["priority"], reverse=True)[:25]:
                    if spec["priority"] < 0.55:
                        continue
                    task = LabelTask(
                        organization_id=run.organization_id,
                        aoi_run_id=run.id,
                        field_id=spec["field"].id,
                        created_by_user_id=run.created_by_user_id,
                        title=f"Review autodetect field {spec['field'].id}",
                        status="queued",
                        source="active_learning",
                        queue_name=region_band,
                        priority_score=spec["priority"],
                        task_payload={
                            **spec["payload"],
                            "field_id": str(spec["field"].id),
                            "uncertainty_score": spec["uncertainty"],
                        },
                    )
                    session.add(task)
                    session.flush()
                    checksum = hashlib.sha256(
                        json.dumps(spec["geometry_geojson"], sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    version = LabelVersion(
                        organization_id=run.organization_id,
                        label_task_id=task.id,
                        created_by_user_id=run.created_by_user_id,
                        version_no=1,
                        geometry_geojson=spec["geometry_geojson"],
                        quality_tier="model_seed",
                        source=str(detect_pipeline_profile.get("field_source") or "autodetect"),
                        checksum=checksum,
                        notes="Autogenerated active learning seed from autodetect output",
                    )
                    session.add(version)
                    session.flush()
                    session.add(
                        LabelReview(
                            organization_id=run.organization_id,
                            label_task_id=task.id,
                            label_version_id=version.id,
                            decision="pending",
                        )
                    )
            runtime_meta["grid_cells"] = int(_grid_cell_count)
            runtime_meta["active_learning_candidates"] = int(
                len(inserted_fields) if detect_pipeline_profile.get("enable_active_learning") else 0
            )
            db_insert_time = time.time() - t_step
            STEP_DURATION.labels(step="db_insert").observe(db_insert_time)
            latency_breakdown["db_insert_s"] = round(float(db_insert_time), 3)
            _set_post_progress(run, runtime_meta, session, "db_insert", phase_fraction=0.90)

            # Step 12: Topology cleanup (if PostGIS available)
            failure_stage = "topology_cleanup"
            topology_t0 = time.time()
            _set_post_progress(run, runtime_meta, session, "topology", phase_fraction=0.96)
            try:
                from processing.fields.topology import get_topology_cleanup_sql
                topology_tol_deg = float(
                    float(getattr(settings, "TOPOLOGY_SIMPLIFY_TOL_M", 1.0)) / 111_320.0
                )
                topology_simplify_enabled = bool(getattr(settings, "TOPOLOGY_SIMPLIFY_ENABLED", False))
                if region_boundary_profile == "north_boundary":
                    topology_simplify_enabled = bool(
                        getattr(settings, "NORTH_TOPOLOGY_SIMPLIFY_ENABLED", False)
                    )
                for sql in get_topology_cleanup_sql(
                    simplify_enabled=topology_simplify_enabled,
                    simplify_tol_deg=topology_tol_deg,
                ):
                    session.execute(
                        __import__("sqlalchemy").text(sql),
                        {"run_id": str(run_id)},
                    )
                session.commit()
                logger.info("topology_cleanup_done", aoi_run_id=run_id_str)
            except Exception as e:
                logger.warning("topology_cleanup_skipped", error=str(e))
            finally:
                latency_breakdown["topology_cleanup_s"] = round(float(time.time() - topology_t0), 3)

            candidate_lifecycle_summary = _finalize_detection_candidates(
                session=session,
                run_id=run_id,
                FieldModel=Field,
                FieldDetectionCandidateModel=FieldDetectionCandidate,
                pre_topology_field_geojson=[geometry_geojson for _, _, geometry_geojson in inserted_fields],
                object_classifier_rejected_geojson=object_classifier_rejected_geojson,
            )
            runtime_meta["candidate_branch_counts"] = dict(
                candidate_lifecycle_summary.get("candidate_branch_counts") or {}
            )
            runtime_meta["candidate_reject_summary"] = dict(
                candidate_lifecycle_summary.get("candidate_reject_summary") or {}
            )
            runtime_meta["candidates_total"] = int(
                candidate_lifecycle_summary.get("candidates_total") or 0
            )
            runtime_meta["candidates_kept"] = int(
                candidate_lifecycle_summary.get("candidates_kept") or 0
            )
            runtime_meta["candidate_lifecycle"] = dict(
                candidate_lifecycle_summary.get("candidate_lifecycle") or {}
            )

            failure_stage = "finalize"
            runtime_meta["field_count"] = int(len(merged))
            runtime_meta = _enrich_runtime_meta(
                runtime_meta,
                latency_breakdown=latency_breakdown,
                t_start=t_start,
            )
            runtime_meta["total_time_s"] = round(time.time() - t_start, 2)
            _persist_runtime_meta(run, runtime_meta)
            run.status = "done"
            run.progress = 100
            session.commit()

            total_time = time.time() - t_start
            DETECT_DURATION.labels(aoi_run_id=run_id_str).observe(total_time)
            logger.info("autodetect_done", aoi_run_id=run_id_str,
                        total_time_s=round(total_time, 2), n_fields=len(merged))

            return {"status": "done", "n_fields": len(merged), "time_s": round(total_time, 2)}

    except Exception as e:
        logger.error(
            "autodetect_error",
            aoi_run_id=run_id_str,
            error=str(e),
            failure_stage=failure_stage,
            traceback=traceback.format_exc(),
        )
        _mark_run_failed_best_effort(
            run_id=run_id,
            error=e,
            failure_stage=failure_stage,
        )
        return {
            "status": "error",
            "message": str(e),
            "failure_stage": failure_stage,
        }

    finally:
        _evt_loop = locals().get('_event_loop')
        if sentinel_client is not None:
            try:
                close_loop = _evt_loop if _evt_loop is not None else asyncio.new_event_loop()
                try:
                    close_loop.run_until_complete(sentinel_client.close())
                finally:
                    close_loop.close()
            except Exception:
                logger.warning("sentinel_client_close_failed", exc_info=True)
        elif _evt_loop is not None:
            try:
                _evt_loop.close()
            except Exception:
                pass
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                logger.warning("autodetect_engine_dispose_failed", exc_info=True)
        ACTIVE_RUNS.dec()
