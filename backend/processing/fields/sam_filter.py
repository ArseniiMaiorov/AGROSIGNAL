"""Filter SAM polygons using semantic and phenological constraints."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from core.logging import get_logger
from processing.priors.worldcover import SHRUBLAND_CLASS, TREE_CLASS, WATER_CLASS, WETLAND_CLASS

logger = get_logger(__name__)


def sample_raster_for_polygon(geom, raster: np.ndarray, transform) -> np.ndarray:
    """Sample raster values under a polygon as a flat array."""
    mask = rasterize(
        [(geom, 1)],
        out_shape=raster.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    return raster[mask]


def _worldcover_nonfield_mask(worldcover_mask: np.ndarray | None) -> np.ndarray | None:
    """Convert raw WorldCover classes to a boolean non-field mask."""
    if worldcover_mask is None:
        return None
    if worldcover_mask.dtype == bool:
        return worldcover_mask.astype(bool, copy=False)
    return np.isin(
        worldcover_mask,
        (TREE_CLASS, SHRUBLAND_CLASS, WATER_CLASS, WETLAND_CLASS),
    )


def filter_sam_polygons(
    sam_gdf: gpd.GeoDataFrame,
    max_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    water_mask: np.ndarray,
    forest_mask: np.ndarray,
    worldcover_mask: np.ndarray | None,
    transform,
    cfg,
) -> gpd.GeoDataFrame:
    """Keep only SAM polygons that behave like crop fields."""
    if sam_gdf.empty:
        return sam_gdf.copy()

    valid = []
    rejected = {"area": 0, "forest": 0, "water": 0, "worldcover": 0, "ndvi": 0, "std": 0}
    min_area_m2 = float(cfg.SAM_MIN_MASK_REGION_AREA) * float(cfg.POST_PX_AREA_M2)
    worldcover_nonfield = _worldcover_nonfield_mask(worldcover_mask)

    for _, row in sam_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or not geom.is_valid:
            continue
        if geom.area < min_area_m2:
            rejected["area"] += 1
            continue

        forest_ratio = float(np.nanmean(sample_raster_for_polygon(geom, forest_mask, transform))) \
            if np.any(forest_mask) else 0.0
        if forest_ratio > float(cfg.HYBRID_SAM_MAX_FOREST_RATIO):
            rejected["forest"] += 1
            continue

        water_ratio = float(np.nanmean(sample_raster_for_polygon(geom, water_mask, transform))) \
            if np.any(water_mask) else 0.0
        if water_ratio > float(cfg.HYBRID_SAM_MAX_WATER_RATIO):
            rejected["water"] += 1
            continue

        worldcover_ratio = float(
            np.nanmean(sample_raster_for_polygon(geom, worldcover_nonfield, transform))
        ) if worldcover_nonfield is not None and np.any(worldcover_nonfield) else 0.0
        if worldcover_ratio > 0.6:
            rejected["worldcover"] += 1
            continue

        mean_max_ndvi = float(np.nanmean(sample_raster_for_polygon(geom, max_ndvi, transform)))
        if mean_max_ndvi < float(cfg.PHENO_FIELD_MAX_NDVI_MIN):
            rejected["ndvi"] += 1
            continue

        mean_std = float(np.nanmean(sample_raster_for_polygon(geom, ndvi_std, transform)))
        if mean_std < float(cfg.PHENO_FIELD_NDVI_STD_MIN):
            rejected["std"] += 1
            continue

        valid.append(geom)

    logger.info(
        "sam_filter_stats",
        total=len(sam_gdf),
        valid=len(valid),
        rejected=rejected,
    )
    return gpd.GeoDataFrame(geometry=valid, crs=sam_gdf.crs)
