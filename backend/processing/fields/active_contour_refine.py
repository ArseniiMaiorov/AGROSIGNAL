"""Snake-based contour refinement for vector field boundaries."""
from __future__ import annotations

from typing import Callable

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from skimage.filters import gaussian
from skimage.segmentation import active_contour

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


def _largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        try:
            return max(geom.geoms, key=lambda g: g.area)
        except Exception:
            return None
    return None


def _geom_metrics(geom) -> dict[str, float]:
    if geom is None or geom.is_empty:
        return {
            "area": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "bbox_iou_self": 1.0,
            "vertex_count": 0.0,
        }
    centroid = geom.centroid
    return {
        "area": float(geom.area),
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
        "bbox_iou_self": 1.0,
        "vertex_count": float(len(getattr(geom.exterior, "coords", ()))),
    }


def _bbox_iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    ax1, ay1, ax2, ay2 = a.bounds
    bx1, by1, bx2, by2 = b.bounds
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = max(area_a + area_b - inter, 1e-6)
    return float(inter / union)


def refine_field_with_snake(
    polygon,
    edge_image: np.ndarray,
    transform,
    cfg,
    alpha: float = 0.015,
    beta: float = 10.0,
    w_edge: float = -1.0,
    max_px_dist: float = 15.0,
) -> tuple[Polygon | None, dict[str, float | bool | str]]:
    """Refine one polygon boundary by pulling it toward nearby strong edges."""
    polygon = _largest_polygon(polygon)
    diagnostics: dict[str, float | bool | str] = {
        "applied": False,
        "rejected_reason": "",
        "area_ratio": 1.0,
        "centroid_shift_m": 0.0,
        "bbox_overlap": 1.0,
        "edge_confidence": 0.0,
        "vertex_count_delta": 0.0,
    }
    if polygon is None or not polygon.is_valid:
        diagnostics["rejected_reason"] = "invalid_input"
        return polygon, diagnostics

    mode = str(getattr(cfg, "SNAKE_REFINE_MODE", "guarded")).strip().lower()
    if not bool(getattr(cfg, "SNAKE_REFINE_ENABLED", True)) or mode == "off":
        diagnostics["rejected_reason"] = "disabled"
        return polygon, diagnostics

    original_metrics = _geom_metrics(polygon)
    alpha = float(getattr(cfg, "SNAKE_ALPHA", alpha))
    beta = float(getattr(cfg, "SNAKE_BETA", beta))
    w_edge = float(getattr(cfg, "SNAKE_W_EDGE", w_edge))
    max_px_dist = float(getattr(cfg, "SNAKE_MAX_PX_DIST", max_px_dist))
    max_centroid_shift_m = float(getattr(cfg, "SNAKE_MAX_CENTROID_SHIFT_M", 6.0))
    min_area_ratio = float(getattr(cfg, "SNAKE_MIN_AREA_RATIO", 0.90))
    max_area_ratio = float(getattr(cfg, "SNAKE_MAX_AREA_RATIO", 1.12))

    coords = np.asarray(polygon.exterior.coords, dtype=float)
    if coords.shape[0] < 4:
        diagnostics["rejected_reason"] = "too_few_vertices"
        return polygon, diagnostics
    if mode == "guarded":
        if len(coords) > 300:
            diagnostics["rejected_reason"] = "too_many_vertices"
            return polygon, diagnostics
        if len(getattr(polygon, "interiors", ())) > 0:
            diagnostics["rejected_reason"] = "has_holes"
            return polygon, diagnostics
        minx, miny, maxx, maxy = polygon.bounds
        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)
        aspect = max(width / height, height / width)
        if aspect > 5.0:
            diagnostics["rejected_reason"] = "elongated_shape"
            return polygon, diagnostics

    snake_init = []
    inv_transform = ~transform
    for x, y in coords:
        col, row = inv_transform * (x, y)
        snake_init.append((float(row), float(col)))
    snake_init = np.asarray(snake_init, dtype=float)

    h, w = edge_image.shape
    sample_rows = np.clip(np.round(snake_init[:, 0]).astype(int), 0, h - 1)
    sample_cols = np.clip(np.round(snake_init[:, 1]).astype(int), 0, w - 1)
    edge_confidence = float(np.mean(edge_image[sample_rows, sample_cols])) if snake_init.size else 0.0
    diagnostics["edge_confidence"] = edge_confidence
    if mode == "guarded" and edge_confidence < 0.05:
        diagnostics["rejected_reason"] = "weak_edge_confidence"
        return polygon, diagnostics

    image_smooth = gaussian(edge_image.astype(np.float32), sigma=1.0, preserve_range=True)
    try:
        snake_result = active_contour(
            image_smooth,
            snake_init,
            alpha=alpha,
            beta=beta,
            w_edge=w_edge,
            w_line=0.0,
            max_num_iter=100,
            boundary_condition="periodic",
        )
    except Exception:
        diagnostics["rejected_reason"] = "solver_failed"
        return polygon, diagnostics

    delta = np.linalg.norm(snake_result - snake_init, axis=1)
    too_far = delta > max_px_dist
    snake_result[too_far] = snake_init[too_far]

    geo_coords = []
    for row, col in snake_result:
        row_f = float(np.clip(row, 0.0, max(0.0, h - 1)))
        col_f = float(np.clip(col, 0.0, max(0.0, w - 1)))
        x, y = transform * (col_f + 0.5, row_f + 0.5)
        geo_coords.append((x, y))

    if len(geo_coords) < 4:
        diagnostics["rejected_reason"] = "degenerate_output"
        return polygon, diagnostics
    if geo_coords[0] != geo_coords[-1]:
        geo_coords.append(geo_coords[0])

    refined = Polygon(geo_coords)
    if not refined.is_valid:
        refined = refined.buffer(0)
    if refined.is_empty or refined.area < (polygon.area * 0.5):
        diagnostics["rejected_reason"] = "collapsed_geometry"
        return polygon, diagnostics

    refined_metrics = _geom_metrics(refined)
    area_ratio = refined_metrics["area"] / max(original_metrics["area"], 1e-6)
    centroid_shift = float(
        np.hypot(
            refined_metrics["centroid_x"] - original_metrics["centroid_x"],
            refined_metrics["centroid_y"] - original_metrics["centroid_y"],
        )
    )
    bbox_overlap = _bbox_iou(polygon, refined)
    diagnostics.update(
        {
            "area_ratio": float(area_ratio),
            "centroid_shift_m": centroid_shift,
            "bbox_overlap": bbox_overlap,
            "vertex_count_delta": float(
                refined_metrics["vertex_count"] - original_metrics["vertex_count"]
            ),
        }
    )
    if mode == "guarded":
        if area_ratio < min_area_ratio:
            diagnostics["rejected_reason"] = "area_shrink"
            return polygon, diagnostics
        if area_ratio > max_area_ratio:
            diagnostics["rejected_reason"] = "area_growth"
            return polygon, diagnostics
        if centroid_shift > max_centroid_shift_m:
            diagnostics["rejected_reason"] = "centroid_shift"
            return polygon, diagnostics
        if bbox_overlap < 0.60:
            diagnostics["rejected_reason"] = "bbox_overlap"
            return polygon, diagnostics

    diagnostics["applied"] = True
    return refined, diagnostics


def refine_all_fields(
    gdf,
    edge_image: np.ndarray,
    transform,
    cfg,
    *,
    return_diagnostics: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    """Apply snake refinement to field polygons when enabled."""
    if gdf is None or gdf.empty or not bool(getattr(cfg, "SNAKE_REFINE_ENABLED", False)):
        return (gdf, []) if return_diagnostics else gdf

    work = gdf.copy()
    min_area = float(getattr(cfg, "POST_MIN_FIELD_AREA_HA", 0.5)) * 10_000.0
    mode = str(getattr(cfg, "SNAKE_REFINE_MODE", "guarded")).strip().lower()
    diagnostics: list[dict[str, float | bool | str | int]] = []

    metric_areas = None
    if getattr(work, "crs", None) is not None and not work.crs.is_geographic:
        metric_areas = work.geometry.area.to_numpy(dtype=float)
    elif getattr(work, "crs", None) is not None:
        metric_areas = work.to_crs("EPSG:3857").geometry.area.to_numpy(dtype=float)
    else:
        metric_areas = np.zeros(len(work), dtype=float)

    total_rows = max(len(work), 1)
    _emit_progress(progress_callback, "snake_fields", 0, total_rows)
    for pos, (idx, row) in enumerate(work.iterrows()):
        geom = row.geometry
        poly = _largest_polygon(geom)
        if poly is None:
            _emit_progress(progress_callback, "snake_fields", pos + 1, total_rows)
            continue
        if metric_areas[pos] < min_area:
            diagnostics.append({"row": int(pos), "applied": False, "rejected_reason": "below_min_area"})
            _emit_progress(progress_callback, "snake_fields", pos + 1, total_rows)
            continue
        if mode == "guarded" and len(poly.exterior.coords) > 300:
            diagnostics.append({"row": int(pos), "applied": False, "rejected_reason": "too_many_vertices"})
            _emit_progress(progress_callback, "snake_fields", pos + 1, total_rows)
            continue

        refined, diag = refine_field_with_snake(poly, edge_image, transform, cfg)
        diagnostics.append({"row": int(pos), **diag})
        if refined is None or refined.is_empty:
            _emit_progress(progress_callback, "snake_fields", pos + 1, total_rows)
            continue
        if geom.geom_type == "MultiPolygon":
            work.at[idx, "geometry"] = MultiPolygon([refined])
        else:
            work.at[idx, "geometry"] = refined
        _emit_progress(progress_callback, "snake_fields", pos + 1, total_rows)

    if return_diagnostics:
        return work, diagnostics
    return work
