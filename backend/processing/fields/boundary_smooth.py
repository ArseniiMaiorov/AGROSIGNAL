"""Raster and vector cleanup helpers for field boundaries."""
from __future__ import annotations

from collections.abc import Callable

import geopandas as gpd
import numpy as np
from scipy.ndimage import binary_fill_holes, label as nd_label
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from skimage.morphology import (
    closing,
    disk,
)

from utils.raster import remove_small_components

ProgressCallback = Callable[[str, int, int], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int,
) -> None:
    if progress_callback is None:
        return
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    progress_callback(str(stage), safe_completed, safe_total)


def _chaikin_smooth_coords(coords: list, iterations: int = 2) -> list:
    """Chaikin corner-cutting for smooth polygon edges."""
    if len(coords) < 4:
        return coords
    for _ in range(iterations):
        new_coords = []
        n = len(coords) - 1
        for i in range(n):
            p0 = coords[i]
            p1 = coords[(i + 1) % n]
            new_coords.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_coords.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        new_coords.append(new_coords[0])
        coords = new_coords
    return coords


def _chaikin_smooth_polygon(poly) -> BaseGeometry:
    """Apply Chaikin smoothing to a Polygon."""
    if not hasattr(poly, "exterior") or poly.is_empty:
        return poly
    ext = _chaikin_smooth_coords(list(poly.exterior.coords), iterations=2)
    holes = [_chaikin_smooth_coords(list(h.coords), iterations=2) for h in poly.interiors]
    try:
        result = Polygon(ext, holes)
        return result if result.is_valid and not result.is_empty else result.buffer(0)
    except Exception:
        return poly


def clean_raster_mask(
    mask: np.ndarray,
    cfg,
    hard_exclusion_mask: np.ndarray | None = None,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Remove spikes, close small gaps, and trim undersized fragments.

    Args:
        mask: boolean candidate mask.
        cfg: settings object.
        hard_exclusion_mask: optional barrier mask (forest/water/built-up).
            When provided, small holes that overlap with barriers are NOT filled.
    """
    cleaned = mask.astype(bool, copy=True)
    holes_skipped_due_to_forbidden = np.zeros_like(cleaned, dtype=bool)
    cleaned = closing(cleaned, disk(max(1, int(cfg.POST_MORPH_CLOSE_RADIUS))))
    if hard_exclusion_mask is not None:
        forbidden_filled = cleaned & hard_exclusion_mask
        if np.any(forbidden_filled):
            holes_skipped_due_to_forbidden |= forbidden_filled
        cleaned &= ~hard_exclusion_mask

    min_px = max(1, int(cfg.POST_MIN_FIELD_AREA_HA * 10_000 / cfg.POST_PX_AREA_M2))
    cleaned = remove_small_components(cleaned, min_px)

    # Barrier-aware hole fill instead of blind remove_small_holes
    max_hole_px = max(1, int(0.2 * 10_000 / cfg.POST_PX_AREA_M2))
    labeled_holes, n_holes = nd_label(~cleaned)
    # Identify border-touching labels (background, not real holes)
    border_labels = set()
    if n_holes > 0:
        border_labels.update(labeled_holes[0, :].ravel())
        border_labels.update(labeled_holes[-1, :].ravel())
        border_labels.update(labeled_holes[:, 0].ravel())
        border_labels.update(labeled_holes[:, -1].ravel())
    for hole_id in range(1, n_holes + 1):
        if hole_id in border_labels:
            continue
        hole = labeled_holes == hole_id
        hole_px = int(np.count_nonzero(hole))
        if hole_px > max_hole_px:
            continue
        # Skip holes that overlap with barriers (forest/water/built-up)
        if hard_exclusion_mask is not None and np.any(hard_exclusion_mask[hole]):
            holes_skipped_due_to_forbidden |= hole
            continue
        cleaned[hole] = True

    if hard_exclusion_mask is not None:
        cleaned &= ~hard_exclusion_mask

    cleaned = cleaned.astype(bool)
    if return_debug:
        return cleaned, {
            "holes_skipped_due_to_forbidden": holes_skipped_due_to_forbidden,
        }
    return cleaned


def close_boundary_gaps(
    mask: np.ndarray,
    edge_composite: np.ndarray,
    cfg,
    hard_exclusion_mask: np.ndarray | None = None,
    return_debug: bool = False,
    region_profile: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, dict[str, np.ndarray]]:
    """Fill internal holes when the edge evidence inside the hole is weak.

    Args:
        mask: boolean candidate mask.
        edge_composite: edge strength raster.
        cfg: settings object.
        hard_exclusion_mask: optional barrier mask. Holes that contain any
            barrier pixels (forest/water/built-up) are never filled.
    """
    if edge_composite.shape != mask.shape:
        raise ValueError("edge_composite must match mask shape")

    labeled, n_labels = nd_label(mask.astype(bool))
    if n_labels <= 0:
        return mask.astype(bool, copy=True), 0

    result = mask.astype(bool, copy=True)
    added = 0
    threshold = float(cfg.POST_GAP_EDGE_THRESHOLD)
    holes_skipped_due_to_forbidden = np.zeros_like(result, dtype=bool)

    token = str(region_profile or "").strip().lower()
    if token == "south_recall":
        max_gap_ha = float(getattr(cfg, "SOUTH_GAP_CLOSE_MAX_HA", 1.5))
    elif token == "north_boundary":
        max_gap_ha = float(getattr(cfg, "NORTH_GAP_CLOSE_MAX_HA", 0.5))
    else:
        max_gap_ha = float(getattr(cfg, "POST_GAP_CLOSE_MAX_HA", 1.0))
    max_gap_px = max(1, int(max_gap_ha * 10_000 / cfg.POST_PX_AREA_M2))
    heartbeat_every = max(1, min(64, n_labels // 12 if n_labels > 12 else 1))

    for component_id in range(1, n_labels + 1):
        component = labeled == component_id
        filled = binary_fill_holes(component)
        holes = filled & ~component
        try:
            if not np.any(holes):
                continue
            # Skip holes exceeding maximum gap size.
            hole_px = int(np.count_nonzero(holes))
            if hole_px > max_gap_px:
                continue
            # Skip holes that contain barrier pixels (forest/water/built-up)
            if hard_exclusion_mask is not None and np.any(hard_exclusion_mask[holes]):
                holes_skipped_due_to_forbidden |= holes
                continue
            mean_edge = float(edge_composite[holes].mean()) if np.any(holes) else 1.0
            # Fill holes where boundary evidence is WEAK (low mean edge).
            # Also allow filling if hole is very small (<50 pixels) regardless
            # of edge strength — tiny gaps between detected regions are usually
            # artifacts of over-segmentation rather than real boundaries.
            small_hole = hole_px < 50
            if mean_edge < threshold or small_hole:
                result |= holes
                added += int(np.count_nonzero(holes))
        finally:
            if progress_callback is not None and (
                component_id == n_labels
                or component_id == 1
                or component_id % heartbeat_every == 0
            ):
                try:
                    progress_callback(int(component_id), int(n_labels))
                except Exception:
                    pass

    if return_debug:
        return result, added, {
            "holes_skipped_due_to_forbidden": holes_skipped_due_to_forbidden,
        }
    return result, added


def smooth_field_polygon(geom: BaseGeometry, cfg, region_profile: str | None = None) -> BaseGeometry:
    """Smooth a polygon with simplify + Chaikin corner-cutting + light buffer."""
    if geom is None or geom.is_empty:
        return geom

    token = str(region_profile or "").strip().lower()
    if token == "north_boundary":
        tolerance = max(0.0, float(getattr(cfg, "NORTH_BOUNDARY_SMOOTH_SIMPLIFY_TOL_M", 0.5)))
        buf = max(0.0, float(getattr(cfg, "NORTH_POST_BUFFER_SMOOTH_M", 0.0)))
        min_area_ratio = float(getattr(cfg, "NORTH_STAGE_ROLLBACK_MIN_AREA_RATIO", 0.95))
    else:
        tolerance = max(
            0.0,
            float(getattr(cfg, "BOUNDARY_SMOOTH_SIMPLIFY_TOL_M", getattr(cfg, "POST_SIMPLIFY_TOLERANCE", 0.0))),
        )
        buf = max(0.0, float(cfg.POST_BUFFER_SMOOTH_M))
        min_area_ratio = 0.92
    original_area = float(getattr(geom, "area", 0.0))

    smoothed = geom.simplify(tolerance=tolerance, preserve_topology=True) if tolerance > 0.0 else geom

    has_holes = False
    geom_type = getattr(smoothed, "geom_type", "")
    if geom_type == "Polygon":
        has_holes = len(getattr(smoothed, "interiors", ())) > 0
    elif geom_type == "MultiPolygon":
        has_holes = any(len(getattr(part, "interiors", ())) > 0 for part in smoothed.geoms)

    # Apply Chaikin corner-cutting for natural-looking boundaries
    # Skip for polygons with holes to preserve interior ring geometry
    if not has_holes:
        if geom_type == "Polygon":
            smoothed = _chaikin_smooth_polygon(smoothed)
        elif geom_type == "MultiPolygon":
            from shapely.geometry import MultiPolygon
            parts = [_chaikin_smooth_polygon(p) for p in smoothed.geoms]
            smoothed = MultiPolygon([p for p in parts if p.is_valid and not p.is_empty])

    # Light symmetric buffer smoothing (only if compact, no holes, small buffer)
    is_compact = bool(
        original_area <= 0.0
        or float(getattr(smoothed, "length", 0.0)) / max(np.sqrt(max(original_area, 1e-6)), 1e-6) < 25.0
    )
    if buf > 0 and not has_holes and is_compact:
        buffered = smoothed.buffer(buf).buffer(-buf)
        if not buffered.is_empty and buffered.is_valid:
            buffered_area = float(getattr(buffered, "area", 0.0))
            area_ratio = buffered_area / max(original_area, 1e-6)
            if min_area_ratio <= area_ratio <= 1.08:
                smoothed = buffered

    if not smoothed.is_valid:
        smoothed = smoothed.buffer(0)
    return smoothed


def smooth_all_fields(
    gdf: gpd.GeoDataFrame,
    cfg,
    region_profile: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> gpd.GeoDataFrame:
    """Smooth all geometries in a GeoDataFrame and drop undersized outputs."""
    if gdf.empty:
        return gdf.copy()

    original_crs = gdf.crs
    work = gdf.copy()
    if work.crs is not None and work.crs.is_geographic:
        work = work.to_crs("EPSG:3857")

    total_rows = max(len(work), 1)
    _emit_progress(progress_callback, "smooth_fields", 0, total_rows)
    smoothed_geometries = []
    for row_pos, geom in enumerate(work["geometry"], start=1):
        smoothed_geometries.append(
            smooth_field_polygon(geom, cfg, region_profile=region_profile)
        )
        if row_pos % 8 == 0 or row_pos == total_rows:
            _emit_progress(progress_callback, "smooth_fields", row_pos, total_rows)
    work["geometry"] = smoothed_geometries
    min_area_m2 = float(cfg.POST_MIN_FIELD_AREA_HA) * 10_000.0
    work = work[
        work.geometry.notna()
        & (~work.geometry.is_empty)
        & (work.geometry.area >= min_area_m2)
    ].copy()
    work["area_m2"] = work.geometry.area
    work["perimeter_m"] = work.geometry.length

    if original_crs is not None and work.crs != original_crs:
        work = work.to_crs(original_crs)
    return work
