"""Raster-to-vector conversion of labeled segments."""
from collections import deque
from typing import Callable

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from core.logging import get_logger

logger = get_logger(__name__)
MAX_SAFE_MERGE_POLYGONS = 4000
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
    try:
        progress_callback(str(stage), safe_completed, safe_total)
    except Exception as exc:
        logger.warning(
            "vectorize_progress_callback_failed",
            stage=str(stage),
            completed=safe_completed,
            total=safe_total,
            error=str(exc),
        )


def _chaikin_smooth_coords(coords: list, iterations: int = 2) -> list:
    """Chaikin corner-cutting algorithm for smooth polygon boundaries.

    Each iteration replaces each segment AB with two points at 1/4 and 3/4,
    producing progressively smoother curves while staying close to the original.
    """
    if len(coords) < 4:
        return coords
    for _ in range(iterations):
        new_coords = []
        n = len(coords) - 1  # last == first for closed rings
        for i in range(n):
            p0 = coords[i]
            p1 = coords[(i + 1) % n]
            new_coords.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_coords.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        new_coords.append(new_coords[0])  # close ring
        coords = new_coords
    return coords


def _chaikin_smooth_polygon(poly, iterations: int = 3):
    """Apply Chaikin smoothing to a Polygon (exterior + holes)."""
    if not hasattr(poly, "exterior") or poly.is_empty:
        return poly
    ext = _chaikin_smooth_coords(list(poly.exterior.coords), iterations)
    holes = []
    for interior in poly.interiors:
        holes.append(_chaikin_smooth_coords(list(interior.coords), iterations))
    try:
        smoothed = Polygon(ext, holes)
        if smoothed.is_valid and not smoothed.is_empty:
            return smoothed
        return smoothed.buffer(0)
    except Exception:
        return poly


def _estimate_utm_epsg(lon: float, lat: float = 0.0) -> int:
    """Estimate UTM EPSG code from longitude and latitude."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone
MAX_AGG_COLUMNS = {"ndvi_max", "edge_max"}


def _normalized_geom_signature(geom) -> str:
    normalized = geom.normalize()
    return normalized.wkb_hex


def _aggregate_numeric_value(column: str, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if column in MAX_AGG_COLUMNS:
        return float(np.nanmax(finite))
    return float(np.nanmean(finite))


def summarize_polygon_areas(gdf: gpd.GeoDataFrame) -> dict[str, float]:
    """Return area quantiles in square meters for logging."""
    if gdf.empty:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}

    areas = gdf["area_m2"].to_numpy(dtype=float)
    return {
        "p50": float(np.quantile(areas, 0.5)),
        "p90": float(np.quantile(areas, 0.9)),
        "p99": float(np.quantile(areas, 0.99)),
    }


def polygonize_labels(
    labels: np.ndarray,
    transform: Affine,
    src_crs: str,
    min_area_ha: float = 0.3,
    simplify_tol_m: float = 5.0,
    progress_callback: ProgressCallback | None = None,
) -> gpd.GeoDataFrame:
    """Convert labeled raster to polygon GeoDataFrame.

    Args:
        labels: (H, W) int label array.
        transform: rasterio Affine transform for the raster.
        src_crs: CRS string (e.g., 'EPSG:32636') of the raster.
        min_area_ha: minimum field area in hectares.
        simplify_tol_m: simplification tolerance in meters.

    Returns:
        GeoDataFrame with columns: label, geometry, area_m2, perimeter_m
        in EPSG:4326.
    """
    min_area_m2 = min_area_ha * 10000
    simplify_tol_m = max(0.0, float(simplify_tol_m))

    mask = labels > 0
    if not mask.any():
        return gpd.GeoDataFrame(
            columns=["label", "geometry", "area_m2", "perimeter_m"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    polygons = []
    label_ids = []
    total_labels = max(int(np.count_nonzero(np.unique(labels) > 0)), 1)
    processed_labels = 0
    _emit_progress(progress_callback, "polygonize", 0, total_labels)

    for geom_dict, value in shapes(
        labels.astype(np.int32),
        mask=mask,
        transform=transform,
        connectivity=8,
    ):
        if value == 0:
            continue
        processed_labels += 1
        poly = shape(geom_dict)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area >= min_area_m2:
            if simplify_tol_m > 0.0:
                simplified = poly.simplify(simplify_tol_m, preserve_topology=True)
                # Chaikin corner-cutting for smooth natural-looking boundaries
                if simplified.geom_type == "Polygon":
                    simplified = _chaikin_smooth_polygon(simplified, iterations=3)
                elif simplified.geom_type == "MultiPolygon":
                    parts = [_chaikin_smooth_polygon(p, iterations=3) for p in simplified.geoms]
                    simplified = MultiPolygon([p for p in parts if p.is_valid and not p.is_empty])
            else:
                simplified = poly
            if simplified.is_valid and not simplified.is_empty:
                polygons.append(simplified)
                label_ids.append(int(value))
        if processed_labels % 16 == 0 or processed_labels >= total_labels:
            _emit_progress(
                progress_callback,
                "polygonize",
                min(processed_labels, total_labels),
                total_labels,
            )

    if not polygons:
        return gpd.GeoDataFrame(
            columns=["label", "geometry", "area_m2", "perimeter_m"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(
        {"label": label_ids},
        geometry=polygons,
        crs=src_crs,
    )

    gdf["area_m2"] = gdf.geometry.area
    gdf["perimeter_m"] = gdf.geometry.length

    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()

    gdf = gdf.to_crs("EPSG:4326")

    final_geoms = []
    for geom in gdf.geometry:
        if geom.geom_type == "Polygon":
            geom = MultiPolygon([geom])
        final_geoms.append(geom)
    gdf["geometry"] = final_geoms

    return gdf.reset_index(drop=True)


def merge_tile_polygons(
    gdfs: list[gpd.GeoDataFrame],
    overlap_m: float = 500,
    min_iou: float | None = None,
    only_in_overlap: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> gpd.GeoDataFrame:
    """Merge polygons from overlapping tiles.

    Polygons that touch or overlap in the overlap zone are unioned.

    Args:
        gdfs: list of GeoDataFrames from different tiles.
        overlap_m: overlap buffer for matching (not used for actual buffering).

    Returns:
        Merged GeoDataFrame in EPSG:4326.
    """
    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["label", "geometry", "area_m2", "perimeter_m"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    combined = gpd.pd.concat(gdfs, ignore_index=True)
    if combined.empty:
        return combined

    combined = combined.to_crs("EPSG:4326")
    combined = combined.reset_index(drop=True)
    aggregated_columns = [
        col
        for col in combined.select_dtypes(include=[np.number]).columns
        if col != "label"
    ]

    if len(combined) > MAX_SAFE_MERGE_POLYGONS:
        logger.warning(
            "merge_tile_polygons_skip_overlap_union",
            polygon_count=len(combined),
            max_safe=MAX_SAFE_MERGE_POLYGONS,
        )
        result = combined.copy()
    else:
        sindex = combined.sindex
        geometries = combined.geometry.tolist()
        labels = combined["label"].fillna(0).astype(int).tolist()

        remaining = set(range(len(combined)))
        merged_geoms = []
        merged_rows = []
        total_groups = max(len(combined), 1)
        groups_done = 0
        _emit_progress(progress_callback, "merge_groups", 0, total_groups)

        while remaining:
            root = remaining.pop()
            group = [root]
            queue = deque([root])

            while queue:
                current = queue.popleft()
                current_geom = geometries[current]
                candidates = list(sindex.intersection(current_geom.bounds))
                for candidate in candidates:
                    if candidate == current or candidate not in remaining:
                        continue
                    if not current_geom.intersects(geometries[candidate]):
                        continue
                    # IOU gate: only merge if intersection / min(area1, area2) >= threshold
                    if min_iou is not None:
                        try:
                            inter_area = current_geom.intersection(geometries[candidate]).area
                            min_area = min(current_geom.area, geometries[candidate].area)
                            if min_area > 0 and (inter_area / min_area) < min_iou:
                                continue
                        except Exception:
                            continue
                    remaining.remove(candidate)
                    queue.append(candidate)
                    group.append(candidate)

            if len(group) == 1:
                merged = geometries[root]
            else:
                merged = unary_union([geometries[idx] for idx in group])

            if merged.geom_type == "Polygon":
                merged = MultiPolygon([merged])

            merged_geoms.append(merged)
            group_frame = combined.iloc[group]
            row = {"label": labels[root]}
            for column in aggregated_columns:
                values = group_frame[column].to_numpy(dtype=float, copy=False)
                row[column] = _aggregate_numeric_value(column, values)
            merged_rows.append(row)
            groups_done += 1
            if groups_done % 16 == 0 or groups_done == total_groups:
                _emit_progress(progress_callback, "merge_groups", groups_done, total_groups)

        result = gpd.GeoDataFrame(merged_rows, geometry=merged_geoms, crs="EPSG:4326")

    _union = result.geometry.union_all() if hasattr(result.geometry, "union_all") else result.geometry.unary_union
    centroid = _union.centroid
    utm_epsg = _estimate_utm_epsg(centroid.x, centroid.y)
    utm_result = result.to_crs(epsg=utm_epsg)
    result["area_m2"] = utm_result.geometry.area
    result["perimeter_m"] = utm_result.geometry.length
    result = result.sort_values("area_m2", ascending=False).reset_index(drop=True)
    signatures = result.geometry.apply(_normalized_geom_signature)
    result = result.loc[~signatures.duplicated()].copy()

    return result.reset_index(drop=True)
