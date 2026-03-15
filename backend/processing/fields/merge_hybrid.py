"""Hybrid merge between traditional and SAM-derived field polygons."""
from __future__ import annotations

import geopandas as gpd
import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


def compute_iou(geom_a, geom_b) -> float:
    """Compute IoU for two geometries."""
    try:
        inter = geom_a.intersection(geom_b).area
        union = geom_a.union(geom_b).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _with_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    work = gdf.copy()
    metric = work.to_crs("EPSG:3857")
    work["area_m2"] = metric.geometry.area
    work["perimeter_m"] = metric.geometry.length
    if "label" not in work:
        work["label"] = np.arange(1, len(work) + 1, dtype=int)
    return work


def merge_sam_with_traditional(
    traditional_gdf: gpd.GeoDataFrame,
    sam_gdf: gpd.GeoDataFrame,
    cfg,
    ndvi_mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Use SAM polygons as primary detections, traditional polygons as soft support."""
    if sam_gdf.empty:
        logger.warning("hybrid_merge_sam_empty")
        return _with_metrics(traditional_gdf)
    if traditional_gdf.empty:
        logger.warning("hybrid_merge_traditional_empty")
        return _with_metrics(sam_gdf)

    traditional = traditional_gdf.copy()
    sam = sam_gdf.copy()
    if sam.crs != traditional.crs:
        sam = sam.to_crs(traditional.crs)

    merged_polys = []
    matched_traditional_indices: set[int] = set()

    for s_idx, s_row in sam.iterrows():
        s_geom = s_row.geometry
        if s_geom is None or s_geom.is_empty:
            continue

        best_iou = 0.0
        best_traditional_idx = None
        best_traditional_geom = None
        for t_idx, t_row in traditional.iterrows():
            t_geom = t_row.geometry
            if t_geom is None or t_geom.is_empty or not s_geom.intersects(t_geom):
                continue
            iou = compute_iou(s_geom, t_geom)
            if iou > best_iou:
                best_iou = iou
                best_traditional_idx = t_idx
                best_traditional_geom = t_geom

        merged_geom = s_geom
        if best_traditional_geom is None:
            merged_polys.append(merged_geom)
            continue

        if best_iou >= float(cfg.HYBRID_MERGE_MIN_IOU):
            union_geom = s_geom.union(best_traditional_geom)
            if union_geom.is_valid and not union_geom.is_empty:
                merged_geom = union_geom
                matched_traditional_indices.add(int(best_traditional_idx))

        merged_polys.append(merged_geom)

    independent = [
        row.geometry
        for t_idx, row in traditional.iterrows()
        if int(t_idx) not in matched_traditional_indices
        and row.geometry is not None
        and not row.geometry.is_empty
        and not any(row.geometry.intersects(existing) for existing in merged_polys)
    ]
    result = gpd.GeoDataFrame(geometry=merged_polys + independent, crs=traditional.crs)
    logger.info(
        "hybrid_merge_stats",
        traditional=len(traditional),
        sam=len(sam),
        matched=len(matched_traditional_indices),
        independent_traditional=len(independent),
        sam_primary=True,
    )
    return _with_metrics(result)
