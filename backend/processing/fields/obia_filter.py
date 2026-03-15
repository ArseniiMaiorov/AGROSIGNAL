"""Object-Based Image Analysis (OBIA) filtering of segments."""
import math
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

COASTLINE_ARTIFACT_PERIMETER_M = 3000.0
COASTLINE_ARTIFACT_MAX_AREA_M2 = 100000.0
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


def compute_segment_properties(
    labels: np.ndarray,
    pheno_metrics: dict[str, np.ndarray],
    pixel_size_m: float = 10.0,
    lc_fractions: dict[str, np.ndarray] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    """Compute properties for each labeled segment.

    Args:
        labels: (H, W) int label array.
        pheno_metrics: dict with ndvi_delta key at minimum.
        pixel_size_m: pixel size in meters.
        lc_fractions: optional dict with cropland_frac, shrubland_frac, wetland_frac,
            water_frac, tree_frac arrays.

    Returns:
        List of dicts with: label, area_m2, perimeter_m, shape_index, mean_ndvi_delta, mean_ndwi,
        and optionally cropland_frac, shrubland_frac, wetland_frac, water_frac, tree_frac.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    ndvi_delta = pheno_metrics.get("ndvi_delta", np.zeros_like(labels, dtype=float))
    ndwi = pheno_metrics.get("ndwi")

    properties = []
    total_labels = max(int(unique_labels.size), 1)
    _emit_progress(progress_callback, "props", 0, total_labels)
    if unique_labels.size == 0:
        _emit_progress(progress_callback, "props", 1, 1)
        return properties
    for idx, lbl in enumerate(unique_labels, start=1):
        mask = labels == lbl
        area_px = mask.sum()
        area_m2 = area_px * pixel_size_m * pixel_size_m

        eroded = ndimage.binary_erosion(mask)
        perimeter_px = mask.sum() - eroded.sum()
        perimeter_m = perimeter_px * pixel_size_m

        if area_m2 > 0 and perimeter_m > 0:
            shape_index = perimeter_m / (2 * math.sqrt(math.pi * area_m2))
        else:
            shape_index = 999.0

        mean_delta = float(np.nanmean(ndvi_delta[mask])) if mask.any() else 0.0
        mean_ndwi = float(np.nanmean(ndwi[mask])) if mask.any() and ndwi is not None else 0.0

        # Hole fraction: ratio of internal holes to mask area
        filled = binary_fill_holes(mask)
        holes = filled & ~mask
        hole_frac = float(holes.sum() / area_px) if area_px > 0 else 0.0

        props = {
            "label": int(lbl),
            "area_m2": area_m2,
            "perimeter_m": perimeter_m,
            "shape_index": shape_index,
            "mean_ndvi_delta": mean_delta,
            "mean_ndwi": mean_ndwi,
            "hole_frac": hole_frac,
        }

        if lc_fractions is not None and mask.any():
            for key in ("cropland_frac", "shrubland_frac", "wetland_frac", "water_frac", "tree_frac"):
                arr = lc_fractions.get(key)
                props[key] = float(np.nanmean(arr[mask])) if arr is not None else 0.0
        elif lc_fractions is not None:
            for key in ("cropland_frac", "shrubland_frac", "wetland_frac", "water_frac", "tree_frac"):
                props[key] = 0.0

        properties.append(props)
        if idx % 32 == 0 or idx == total_labels:
            _emit_progress(progress_callback, "props", idx, total_labels)

    return properties


def _check_shape_twostep(
    shape_index: float,
    mean_ndvi_delta: float,
    shape_ideal: float | None,
    shape_hard_max: float | None,
    shape_ndvi_override: float | None,
    max_shape_index: float,
) -> bool:
    """Two-step shape filter. Returns True if segment passes."""
    if shape_ideal is not None and shape_hard_max is not None and shape_ndvi_override is not None:
        if shape_index < shape_ideal:
            return True
        if shape_index <= shape_hard_max:
            return mean_ndvi_delta > shape_ndvi_override
        return False
    # Fallback: simple threshold
    return shape_index <= max_shape_index


def filter_segments(
    labels: np.ndarray,
    pheno_metrics: dict[str, np.ndarray],
    min_area_m2: float = 3000,
    max_shape_index: float = 3.0,
    min_ndvi_delta: float = 0.15,
    max_mean_ndwi: float | None = None,
    pixel_size_m: float = 10.0,
    lc_fractions: dict[str, np.ndarray] | None = None,
    min_cropland_frac: float | None = None,
    max_noncrop_frac: float | None = None,
    shape_index_ideal: float | None = None,
    shape_index_hard_max: float | None = None,
    shape_ndvi_delta_override: float | None = None,
    coastline_artifact_perimeter_m: float = COASTLINE_ARTIFACT_PERIMETER_M,
    coastline_artifact_max_area_m2: float = COASTLINE_ARTIFACT_MAX_AREA_M2,
    max_hole_frac: float | None = None,
    max_internal_tree_frac: float | None = None,
    max_internal_water_frac: float | None = None,
    relax_labels: set[int] | None = None,
    relax_shape_multiplier: float = 1.0,
    relax_hole_multiplier: float = 1.0,
    relax_tree_multiplier: float = 1.0,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Filter segments by area, shape, land cover, and phenological criteria.

    Args:
        labels: (H, W) int label array.
        pheno_metrics: dict with ndvi_delta key.
        min_area_m2: minimum area in square meters (0.3 ha = 3000 m2).
        max_shape_index: maximum shape index (fallback for simple filter).
        min_ndvi_delta: minimum mean NDVI delta for segment.
        max_mean_ndwi: optional maximum allowed segment mean NDWI.
        pixel_size_m: pixel size in meters.
        lc_fractions: optional LC fraction arrays for land cover filter.
        min_cropland_frac: min cropland fraction to keep segment (LC filter).
        max_noncrop_frac: max combined shrubland+wetland+water+tree fraction (LC filter).
        shape_index_ideal: two-step shape filter ideal threshold.
        shape_index_hard_max: two-step shape filter hard max.
        shape_ndvi_delta_override: NDVI delta override for mid-range shapes.
        coastline_artifact_perimeter_m: reject small-area segments above this perimeter.
        coastline_artifact_max_area_m2: max area for the coastline-artifact rejection.

    Returns:
        (H, W) filtered label array (rejected segments set to 0).
    """
    props = compute_segment_properties(
        labels,
        pheno_metrics,
        pixel_size_m,
        lc_fractions,
        progress_callback=progress_callback,
    )

    valid_labels = set()
    total_props = max(len(props), 1)
    _emit_progress(progress_callback, "filter", 0, total_props)
    if not props:
        _emit_progress(progress_callback, "filter", 1, 1)
        return labels.copy()
    for idx, p in enumerate(props, start=1):
        relax_this = bool(relax_labels and p["label"] in relax_labels)
        shape_limit = float(max_shape_index) * (
            float(relax_shape_multiplier) if relax_this else 1.0
        )
        hole_limit = (
            float(max_hole_frac) * (float(relax_hole_multiplier) if relax_this else 1.0)
            if max_hole_frac is not None
            else None
        )
        tree_limit = (
            float(max_internal_tree_frac) * (float(relax_tree_multiplier) if relax_this else 1.0)
            if max_internal_tree_frac is not None
            else None
        )
        if p["area_m2"] < min_area_m2:
            continue
        if p["mean_ndvi_delta"] < min_ndvi_delta:
            continue
        if max_mean_ndwi is not None and p["mean_ndwi"] > max_mean_ndwi:
            continue
        if (
            p["perimeter_m"] > coastline_artifact_perimeter_m
            and p["area_m2"] < coastline_artifact_max_area_m2
        ):
            continue

        if not _check_shape_twostep(
            p["shape_index"], p["mean_ndvi_delta"],
            shape_ideal=shape_index_ideal,
            shape_hard_max=shape_index_hard_max,
            shape_ndvi_override=shape_ndvi_delta_override,
            max_shape_index=shape_limit,
        ):
            continue

        # Hole fraction filter
        if hole_limit is not None and p.get("hole_frac", 0.0) > hole_limit:
            continue

        # Internal tree fraction filter
        if tree_limit is not None and p.get("tree_frac", 0.0) > tree_limit:
            continue

        # Internal water fraction filter
        if max_internal_water_frac is not None and p.get("water_frac", 0.0) > max_internal_water_frac:
            continue

        # Land cover filter
        if min_cropland_frac is not None and max_noncrop_frac is not None:
            crop_f = p.get("cropland_frac", 1.0)
            noncrop_f = (
                p.get("shrubland_frac", 0.0)
                + p.get("wetland_frac", 0.0)
                + p.get("water_frac", 0.0)
                + p.get("tree_frac", 0.0)
            )
            if crop_f < min_cropland_frac and noncrop_f > max_noncrop_frac:
                continue

        valid_labels.add(p["label"])
        if idx % 32 == 0 or idx == total_props:
            _emit_progress(progress_callback, "filter", idx, total_props)

    filtered = labels.copy()
    for lbl in np.unique(labels):
        if lbl > 0 and lbl not in valid_labels:
            filtered[filtered == lbl] = 0

    return filtered


def filter_segments_preview(
    labels: np.ndarray,
    pheno_metrics: dict[str, np.ndarray],
    *,
    min_area_m2: float = 5000.0,
    min_ndvi_delta: float = 0.02,
    max_mean_ndwi: float | None = None,
    pixel_size_m: float = 10.0,
    lc_fractions: dict[str, np.ndarray] | None = None,
    max_tree_frac: float | None = None,
    max_water_frac: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Relaxed preview-only filtering for coarse agricultural contours.

    This path intentionally avoids strict shape/hole filters used for operational
    field boundaries. It keeps broader agricultural masses so fast mode can return
    preview contours over larger AOIs without collapsing to zero results.
    """
    props = compute_segment_properties(
        labels,
        pheno_metrics,
        pixel_size_m,
        lc_fractions,
        progress_callback=progress_callback,
    )

    valid_labels = set()
    total_props = max(len(props), 1)
    _emit_progress(progress_callback, "preview_filter", 0, total_props)
    if not props:
        _emit_progress(progress_callback, "preview_filter", 1, 1)
        return labels.copy()

    for idx, p in enumerate(props, start=1):
        if p["area_m2"] < float(min_area_m2):
            continue
        if p["mean_ndvi_delta"] < float(min_ndvi_delta):
            continue
        if max_mean_ndwi is not None and p["mean_ndwi"] > float(max_mean_ndwi):
            continue
        if max_tree_frac is not None and p.get("tree_frac", 0.0) > float(max_tree_frac):
            continue
        if max_water_frac is not None and p.get("water_frac", 0.0) > float(max_water_frac):
            continue
        valid_labels.add(p["label"])
        if idx % 32 == 0 or idx == total_props:
            _emit_progress(progress_callback, "preview_filter", idx, total_props)

    filtered = labels.copy()
    for lbl in np.unique(labels):
        if lbl > 0 and lbl not in valid_labels:
            filtered[filtered == lbl] = 0

    return filtered
