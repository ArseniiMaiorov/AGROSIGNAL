"""Optional SAM-based field-boundary workflow."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
except Exception:  # pragma: no cover
    rasterio = None
    shapes = None
    shape = None


def build_sam_input_composite(
    max_ndvi: np.ndarray,
    edge_composite: np.ndarray,
    mean_ndvi: np.ndarray,
) -> np.ndarray:
    """Build a normalized 3-channel composite for SAM."""
    if not (max_ndvi.shape == edge_composite.shape == mean_ndvi.shape):
        raise ValueError("max_ndvi, edge_composite, and mean_ndvi must share the same shape")

    def _norm(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=np.float32)
        lo, hi = np.nanpercentile(arr[finite], 2), np.nanpercentile(arr[finite], 98)
        return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

    rgb = np.stack([
        _norm(max_ndvi),
        _norm(edge_composite),
        _norm(mean_ndvi),
    ], axis=-1)
    return (rgb * 255).astype(np.uint8)


def build_sam_composite(
    max_ndvi: np.ndarray,
    edge_composite: np.ndarray,
    mean_ndvi: np.ndarray,
) -> np.ndarray:
    """Alias kept for the versioned integration contract."""
    return build_sam_input_composite(max_ndvi, edge_composite, mean_ndvi)


def _empty_gdf(crs_epsg: int | str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=f"EPSG:{crs_epsg}")


def run_sam_segmentation(
    composite_uint8: np.ndarray,
    transform,
    crs_epsg: int | str,
    cfg,
    output_dir: Path,
) -> gpd.GeoDataFrame:
    """Run automatic SAM mask generation when optional deps are present."""
    try:  # pragma: no cover - optional heavy dependency
        import samgeo
    except Exception as exc:  # pragma: no cover
        logger.warning("sam_disabled_dependency_missing", error=str(exc))
        return _empty_gdf(crs_epsg)

    if rasterio is None:
        logger.warning("sam_disabled_rasterio_unavailable")
        return _empty_gdf(crs_epsg)

    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = output_dir / "sam_input.tif"
    output_path = output_dir / "sam_output.tif"
    vector_path = output_dir / "sam_polygons.gpkg"

    height, width = composite_uint8.shape[:2]
    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
    ) as dst:
        for band_idx in range(3):
            dst.write(composite_uint8[..., band_idx], band_idx + 1)

    sam = samgeo.SamGeo(
        model_type=cfg.SAM_MODEL_TYPE,
        checkpoint=str(cfg.SAM_CHECKPOINT_PATH),
        automatic=True,
    )
    sam.generate(
        source=str(tiff_path),
        output=str(output_path),
        foreground=True,
        unique=True,
        points_per_side=int(cfg.SAM_POINTS_PER_SIDE),
    )
    sam.tiff_to_vector(tiff_path=str(output_path), output=str(vector_path))

    if not vector_path.exists():
        return _empty_gdf(crs_epsg)
    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(f"EPSG:{crs_epsg}")
    elif str(gdf.crs) != f"EPSG:{crs_epsg}":
        gdf = gdf.to_crs(f"EPSG:{crs_epsg}")
    return gdf


def run_sam_sequential(
    composite_uint8: np.ndarray,
    transform,
    crs_epsg: int | str,
    cfg,
    output_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Run SAM automatic segmentation as the primary field-boundary detector."""
    if output_dir is None:
        output_dir = Path(".sam_tmp")
    return run_sam_segmentation(
        composite_uint8,
        transform,
        crs_epsg,
        cfg,
        output_dir,
    )


def run_sam_with_crop_boxes(
    composite_uint8: np.ndarray,
    traditional_crop_gdf: gpd.GeoDataFrame,
    transform,
    crs_epsg: int | str,
    cfg,
    output_dir: Path,
) -> gpd.GeoDataFrame:
    """Prompt SAM with traditional crop boxes; fallback to traditional polygons."""
    if traditional_crop_gdf.empty:
        return _empty_gdf(crs_epsg)

    try:  # pragma: no cover - optional heavy dependency
        import samgeo
    except Exception:
        fallback = traditional_crop_gdf.copy()
        if fallback.crs is None:
            fallback = fallback.set_crs(f"EPSG:{crs_epsg}")
        elif str(fallback.crs) != f"EPSG:{crs_epsg}":
            fallback = fallback.to_crs(f"EPSG:{crs_epsg}")
        return fallback[["geometry"]].copy()

    if rasterio is None or shapes is None or shape is None:
        fallback = traditional_crop_gdf.copy()
        if fallback.crs is None:
            fallback = fallback.set_crs(f"EPSG:{crs_epsg}")
        elif str(fallback.crs) != f"EPSG:{crs_epsg}":
            fallback = fallback.to_crs(f"EPSG:{crs_epsg}")
        return fallback[["geometry"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = output_dir / "sam_input.tif"
    height, width = composite_uint8.shape[:2]
    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
    ) as dst:
        for band_idx in range(3):
            dst.write(composite_uint8[..., band_idx], band_idx + 1)

    sam = samgeo.SamGeo(
        model_type=cfg.SAM_MODEL_TYPE,
        checkpoint=str(cfg.SAM_CHECKPOINT_PATH),
        automatic=False,
    )
    sam.set_image(str(tiff_path))

    crop_gdf = traditional_crop_gdf
    if crop_gdf.crs is None:
        crop_gdf = crop_gdf.set_crs(f"EPSG:{crs_epsg}")
    elif str(crop_gdf.crs) != f"EPSG:{crs_epsg}":
        crop_gdf = crop_gdf.to_crs(f"EPSG:{crs_epsg}")

    all_polygons = []
    inverse = ~transform
    padding_px = max(0, int(getattr(cfg, "SAM_CROP_BOX_PADDING_PX", 0)))
    for _, row in crop_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        col0, row1 = inverse * (minx, miny)
        col1, row0 = inverse * (maxx, maxy)
        left = max(0, int(np.floor(min(col0, col1))) - padding_px)
        right = min(width - 1, int(np.ceil(max(col0, col1))) + padding_px)
        top = max(0, int(np.floor(min(row0, row1))) - padding_px)
        bottom = min(height - 1, int(np.ceil(max(row0, row1))) + padding_px)
        if right <= left or bottom <= top:
            all_polygons.append(geom)
            continue
        box = [left, top, right, bottom]
        try:
            masks = sam.predict(
                boxes=[box],
                point_coords=None,
                point_labels=None,
                multimask_output=False,
            )
        except Exception:
            all_polygons.append(geom)
            continue

        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            for geom_dict, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                if value == 1:
                    all_polygons.append(shape(geom_dict))

    if not all_polygons:
        return _empty_gdf(crs_epsg)
    return gpd.GeoDataFrame(geometry=all_polygons, crs=f"EPSG:{crs_epsg}")
