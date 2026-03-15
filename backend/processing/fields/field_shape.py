"""Optional alpha-shape helpers for complex field outlines."""
from __future__ import annotations

import numpy as np
import rasterio
from shapely.geometry import MultiPoint, Polygon

try:  # pragma: no cover - optional dependency
    import alphashape
except Exception:  # pragma: no cover
    alphashape = None


def pixels_to_points(mask: np.ndarray, transform) -> np.ndarray:
    """Convert True pixels in a mask to map-space point coordinates."""
    rows, cols = np.where(mask.astype(bool))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.column_stack([xs, ys]).astype(np.float64, copy=False)


def best_alpha(points: np.ndarray, target_coverage: float = 0.90) -> float:
    """Choose the first alpha whose area covers enough of the convex hull."""
    if len(points) < 3:
        return 0.002
    convex_area = MultiPoint(points).convex_hull.area
    if convex_area <= 0:
        return 0.002
    if alphashape is None:
        return 0.002

    for alpha in np.arange(0.001, 0.020, 0.001):
        shape = alphashape.alphashape(points, alpha)
        if shape.is_empty:
            break
        if (shape.area / convex_area) >= float(target_coverage):
            return float(alpha)
    return 0.002


def field_alpha_shape(
    mask: np.ndarray,
    transform,
    cfg,
) -> Polygon:
    """Build an alpha shape for one field mask, with safe convex-hull fallback."""
    points = pixels_to_points(mask, transform)
    if len(points) == 0:
        return Polygon()

    if len(points) < int(cfg.ALPHA_MIN_FIELD_PX) or alphashape is None:
        return MultiPoint(points).convex_hull

    points_ds = points[:: max(1, int(cfg.ALPHA_SHAPE_DOWNSAMPLE))]
    alpha = cfg.ALPHA_SHAPE_ALPHA or best_alpha(points_ds, cfg.ALPHA_SHAPE_COVERAGE)
    shape = alphashape.alphashape(points_ds, alpha)
    if shape.is_empty or not shape.is_valid:
        return MultiPoint(points_ds).convex_hull
    return shape
