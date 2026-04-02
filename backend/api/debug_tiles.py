"""Debug tiles, debug layers, and detection candidates API.

Routes: /fields/runs/{run_id}/debug/*, /fields/runs/{run_id}/candidates
Extracted from the monolithic api/fields.py for maintainability.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import DetectionCandidateInfo, DetectionCandidatesResponse
from storage.db import FieldDetectionCandidate, TileDiagnostic, get_db
from storage.fields_repo import FieldsRepository
from datetime import timezone

router = APIRouter(prefix="/fields", tags=["fields"])

# ---------------------------------------------------------------------------
# Debug layer metadata
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.lstrip("#")
    if len(normalized) != 6:
        return (255, 255, 255)
    return tuple(int(normalized[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _normalize_debug_array(
    layer_name: str, raw: np.ndarray
) -> tuple[np.ndarray, dict[str, float | None]]:
    array = np.asarray(raw)
    meta: dict[str, float | None] = {"min": None, "max": None}
    if array.ndim != 2:
        raise HTTPException(status_code=422, detail=f"debug layer '{layer_name}' is not 2D")
    if layer_name in {"boundary_prob", "owt_edge"}:
        work = np.asarray(array, dtype=np.float32)
        finite = work[np.isfinite(work)]
        if finite.size == 0:
            return np.zeros_like(work, dtype=np.float32), meta
        min_v = float(np.nanpercentile(finite, 5))
        max_v = float(np.nanpercentile(finite, 95))
        if max_v <= min_v:
            max_v = min_v + 1e-6
        normalized = np.clip((work - min_v) / (max_v - min_v), 0.0, 1.0)
        meta["min"] = round(min_v, 4)
        meta["max"] = round(max_v, 4)
        return normalized.astype(np.float32, copy=False), meta
    normalized = np.asarray(array > 0, dtype=np.float32)
    meta["min"] = 0.0
    meta["max"] = 1.0
    return normalized, meta


def _colorize_debug_array(layer_name: str, normalized: np.ndarray) -> np.ndarray:
    style = _DEBUG_LAYER_STYLE.get(layer_name) or {}
    if layer_name == "boundary_prob":
        r = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        g = np.clip((normalized ** 1.6) * 120.0, 0, 255).astype(np.uint8)
        b = np.clip((1.0 - normalized) * 90.0, 0, 255).astype(np.uint8)
        a = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        return np.stack([r, g, b, a], axis=-1)
    if layer_name == "owt_edge":
        cyan = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        white = np.clip((normalized ** 0.75) * 255.0, 0, 255).astype(np.uint8)
        a = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        return np.stack([white, cyan, cyan, a], axis=-1)
    r, g, b = _hex_to_rgb(str(style.get("color") or "#ffffff"))
    a = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    out = np.zeros((*normalized.shape, 4), dtype=np.uint8)
    out[..., 0] = r
    out[..., 1] = g
    out[..., 2] = b
    out[..., 3] = a
    return out


def _encode_rgba_png(image: np.ndarray) -> str:
    try:
        from PIL import Image  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"PNG encoder unavailable: {exc}") from exc
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _runtime_debug_tiles(run) -> list[dict]:
    runtime = dict((run.params or {}).get("runtime") or {})
    tiles = runtime.get("tiles") if isinstance(runtime, dict) else None
    return list(tiles or [])


def _runtime_tile_maps(run) -> tuple[dict[int, str], dict[str, int]]:
    runtime = dict((run.params or {}).get("runtime") or {})
    tiles = [tile for tile in list(runtime.get("tiles") or []) if isinstance(tile, dict)]
    index_to_id: dict[int, str] = {}
    id_to_index: dict[str, int] = {}
    for index, tile in enumerate(tiles):
        tile_id = str(tile.get("tile_id") or f"tile-{index}")
        index_to_id[index] = tile_id
        id_to_index[tile_id] = index
    return index_to_id, id_to_index


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


def _detection_candidate_payload(
    candidate, diagnostic, *, tile_id: str | None = None
) -> dict[str, Any]:
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


async def _load_detection_candidates(
    db: AsyncSession,
    *,
    organization_id,
    run_id,
    limit: int,
    kept: bool | None = None,
    branch: str | None = None,
    tile_index: int | None = None,
):
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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
    return {"run_id": str(run.id), "status": str(run.status), "tiles": tiles}


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
    return {"run_id": str(run.id), **_tile_debug_payload(tile)}


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
                    candidate, diagnostic,
                    tile_id=index_to_id.get(int(getattr(diagnostic, "tile_index", -1)))
                    if diagnostic is not None else None,
                )
            )
            for candidate, diagnostic in rows
        ],
    )


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
                    candidate, diagnostic,
                    tile_id=index_to_id.get(tile_index),
                )
            )
            for candidate, diagnostic in rows
        ],
    )
