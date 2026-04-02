"""Shared constants and helper functions for fields/detect/runs/debug APIs.

Extracted from the monolithic api/fields.py to improve maintainability.
Each sub-module (detect.py, runs.py, debug.py) imports from here.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from fastapi import HTTPException
from geoalchemy2.shape import to_shape
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import DetectRequest
from core.config import get_settings
from core.region import resolve_region_band, resolve_region_boundary_profile
from core.config import get_adaptive_season_window
from processing.fields.tiling import (
    bbox_to_polygon,
    make_tiles,
    point_radius_to_polygon,
    polygon_coords_to_polygon,
)
from services.temporal_analytics_service import GEOMETRY_FOUNDATION
from services.trust_service import describe_detect_launch
from storage.db import AoiRun
from storage.fields_repo import FieldsRepository

_settings = get_settings()

# ---------------------------------------------------------------------------
# Detection preset configs
# ---------------------------------------------------------------------------

DETECT_PRESET_CONFIGS = {
    "fast": {
        "resolution_m": 10,
        "target_dates": 4,
        "use_sam": False,
        "min_field_area_ha": 0.5,
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
        "tile_size_px": 896,
        "max_tiles": 36,
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
        "tile_size_px": 768,
        "max_tiles": 24,
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

HARD_TILE_LIMIT = 72
HARD_COMPLEXITY_LIMIT = 1_450.0
HARD_RAM_LIMIT_MB = 6_500

# ---------------------------------------------------------------------------
# Stage labels
# ---------------------------------------------------------------------------

STAGE_LABELS = {
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

# ---------------------------------------------------------------------------
# AOI helpers
# ---------------------------------------------------------------------------


def resolve_aoi(req: DetectRequest):
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


def infer_detect_preset(req: DetectRequest, *, use_sam: bool) -> str:
    raw = str((req.config or {}).get("preset") or "").strip().lower()
    if raw in DETECT_PRESET_CONFIGS:
        return raw
    if use_sam or req.target_dates >= 9 or req.min_field_area_ha <= 0.11:
        return "quality"
    if req.target_dates <= 4 and req.min_field_area_ha >= 0.5:
        return "fast"
    return "standard"


def estimate_runtime_class(complexity_score: float) -> str:
    if complexity_score <= 24:
        return "short"
    if complexity_score <= 60:
        return "medium"
    if complexity_score <= 120:
        return "long"
    return "extreme"


# ---------------------------------------------------------------------------
# Run payload helpers
# ---------------------------------------------------------------------------

import re as _re

_DETAIL_PROGRESS_RE = _re.compile(
    r"^(?P<label>[a-z ]+?)\s+(?P<current>\d+)/(?P<total>\d+)(?:\s+[·-]\s+(?P<extra>.+))?$",
    _re.IGNORECASE,
)


def runtime_with_stale_flag(run) -> dict | None:
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


def effective_run_status(run, runtime: dict | None) -> str:
    if runtime and runtime.get("stale_running") and run.status == "running":
        return "stale"
    return str(run.status)


def aggregate_runtime_mode(runtime: dict | None, key: str) -> str | None:
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


def candidate_summary(runtime: dict | None) -> dict[str, Any]:
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


def stage_code(runtime: dict | None, status: str) -> str | None:
    stage = str((runtime or {}).get("progress_stage") or status or "").strip().lower()
    return stage or None


def stage_label(runtime: dict | None, status: str) -> str | None:
    s_code = stage_code(runtime, status)
    if not s_code:
        return None
    return STAGE_LABELS.get(s_code, s_code.replace("_", " "))


def updated_at_iso(run, runtime: dict | None) -> str | None:
    heartbeat = (runtime or {}).get("last_heartbeat_ts")
    if isinstance(heartbeat, str) and heartbeat.strip():
        return heartbeat
    if run.created_at is None:
        return None
    created_at = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    return created_at.isoformat()


def elapsed_s(run, runtime: dict | None) -> int | None:
    ua = updated_at_iso(run, runtime)
    if ua is None or run.created_at is None:
        return None
    try:
        updated_dt = datetime.fromisoformat(ua.replace("Z", "+00:00"))
    except ValueError:
        return None
    started_dt = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    return max(0, int(round((updated_dt - started_dt).total_seconds())))


def progress_pct(run, runtime: dict | None) -> float:
    runtime_value = (runtime or {}).get("progress_pct")
    if isinstance(runtime_value, (int, float)):
        return round(float(runtime_value), 2)
    return round(float(run.progress or 0), 2)


def runtime_progress_metric(runtime: dict | None, key: str) -> float | None:
    v = (runtime or {}).get(key)
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return None


def runtime_int_metric(runtime: dict | None, key: str) -> int | None:
    v = (runtime or {}).get(key)
    if isinstance(v, (int, float)):
        return max(0, int(round(float(v))))
    return None


def estimated_remaining_s(run, runtime: dict | None, prog_pct: float, status: str) -> int | None:
    if status == "done":
        return 0
    if status not in {"running", "stale"} or prog_pct <= 0 or prog_pct >= 100 or run.created_at is None:
        return None
    ua = updated_at_iso(run, runtime)
    if ua is None:
        return None
    try:
        updated_dt = datetime.fromisoformat(ua.replace("Z", "+00:00"))
    except ValueError:
        return None
    started_dt = run.created_at if run.created_at.tzinfo is not None else run.created_at.replace(tzinfo=timezone.utc)
    elapsed = max((updated_dt - started_dt).total_seconds(), 1.0)
    return max(0, int(round(elapsed * (100 - prog_pct) / prog_pct)))


def geometry_summary(runtime: dict | None) -> dict:
    runtime = dict(runtime or {})
    tiles = [tile for tile in list(runtime.get("tiles") or []) if isinstance(tile, dict)]

    def _agg_num(key: str, *, as_int: bool = False):
        raw = runtime.get(key)
        if isinstance(raw, (int, float)):
            v = float(raw)
            return int(round(v)) if as_int else round(v, 3)
        values = [float(t[key]) for t in tiles if isinstance(t.get(key), (int, float))]
        if not values:
            return None
        median_v = float(np.median(values))
        return int(round(median_v)) if as_int else round(median_v, 3)

    def _agg_reason(key: str) -> str | None:
        direct = runtime.get(key)
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        reasons = []
        for t in tiles:
            r = t.get(key)
            if isinstance(r, str) and r.strip() and r.strip() not in reasons:
                reasons.append(r.strip())
        return "; ".join(reasons[:3]) if reasons else None

    direct_applied = runtime.get("watershed_applied")
    if isinstance(direct_applied, bool):
        watershed_applied = direct_applied
    else:
        applied_values = [bool(t.get("watershed_applied")) for t in tiles if "watershed_applied" in t]
        watershed_applied = any(applied_values) if applied_values else None

    return {
        "head_count": int(GEOMETRY_FOUNDATION.get("head_count") or 3),
        "heads": list(GEOMETRY_FOUNDATION.get("heads") or []),
        "tta_standard": str(GEOMETRY_FOUNDATION.get("tta_standard") or "flip2"),
        "tta_quality": str(GEOMETRY_FOUNDATION.get("tta_quality") or "rotate4"),
        "retrain_description": str(GEOMETRY_FOUNDATION.get("retrain_description") or ""),
        "geometry_confidence": _agg_num("geometry_confidence"),
        "tta_consensus": _agg_num("tta_consensus"),
        "boundary_uncertainty": _agg_num("boundary_uncertainty"),
        "tta_extent_disagreement": _agg_num("tta_extent_disagreement"),
        "tta_boundary_disagreement": _agg_num("tta_boundary_disagreement"),
        "uncertainty_source": _agg_reason("uncertainty_source"),
        "watershed_applied": watershed_applied,
        "watershed_skipped_reason": _agg_reason("watershed_skipped_reason"),
        "watershed_rollback_reason": _agg_reason("watershed_rollback_reason"),
        "components_after_grow": _agg_num("components_after_grow", as_int=True),
        "components_after_gap_close": _agg_num("components_after_gap_close", as_int=True),
        "components_after_infill": _agg_num("components_after_infill", as_int=True),
        "components_after_merge": _agg_num("components_after_merge", as_int=True),
        "components_after_watershed": _agg_num("components_after_watershed", as_int=True),
        "split_score_p50": _agg_num("split_score_p50"),
        "split_score_p90": _agg_num("split_score_p90"),
        "tiles_summarized": len(tiles),
    }
