"""Маски ESA WorldCover v200."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from core.logging import get_logger

logger = get_logger(__name__)

# ESA WorldCover v200 (2021) — public COG tiles on S3.
# Tile naming: ESA_WorldCover_10m_2021_v200_{N|S}{lat}_{E|W}{lon}_Map.tif
_S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
_VERSION = "v200"
_TILE_DEG = 3  # each tile covers 3x3 degrees

CROPLAND_CLASS = 40
SHRUBLAND_CLASS = 20
WATER_CLASS = 80
WETLAND_CLASS = 90
TREE_CLASS = 10


def _tile_name(lat_tile: int, lon_tile: int, year: int) -> str:
    """Build the COG filename for a given 3x3-degree tile origin."""
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    return (
        f"ESA_WorldCover_10m_{year}_{_VERSION}_"
        f"{ns}{abs(lat_tile):02d}{ew}{abs(lon_tile):03d}_Map.tif"
    )


def _tile_url(lat_tile: int, lon_tile: int, year: int) -> str:
    name = _tile_name(lat_tile, lon_tile, year)
    return f"{_S3_BASE}/{_VERSION}/{year}/map/{name}"


def _intersecting_tiles(bbox_wgs84: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Return (lat_origin, lon_origin) pairs of 3x3-degree tiles covering the bbox."""
    minx, miny, maxx, maxy = bbox_wgs84
    tiles = []
    lat = int(np.floor(miny / _TILE_DEG)) * _TILE_DEG
    while lat < maxy:
        lon = int(np.floor(minx / _TILE_DEG)) * _TILE_DEG
        while lon < maxx:
            tiles.append((lat, lon))
            lon += _TILE_DEG
        lat += _TILE_DEG
    return tiles


def _cache_key(
    mode: str,
    year: int,
    bbox: tuple,
    tile_transform: Affine,
    tile_shape: tuple[int, int],
    dst_crs: str,
) -> str:
    raw = f"{mode}_{year}_{bbox}_{tuple(tile_transform)[:6]}_{tile_shape}_{dst_crs}"
    return hashlib.md5(raw.encode()).hexdigest()


def _mask_for_classes(dst_data: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    """Построить bool-маску для набора классов WorldCover."""
    if not classes:
        return np.zeros_like(dst_data, dtype=bool)
    return np.isin(dst_data, np.asarray(classes, dtype=dst_data.dtype))


DEFAULT_EXCLUDE_CLASSES = (TREE_CLASS, SHRUBLAND_CLASS, WATER_CLASS, WETLAND_CLASS)


def load_worldcover_prior(
    wc_mask: np.ndarray,
    pheno_masks: dict[str, np.ndarray],
    cfg,
) -> np.ndarray:
    """Build a weak WorldCover prior guided by NDVI phenology.

    WorldCover is treated as a correcting layer only: it contributes only where
    WorldCover marks forest/wetland classes and the temporal NDVI signal also
    behaves like forest. This avoids cutting crop edges from hard raster vetoes.
    """
    if "is_forest" not in pheno_masks:
        raise ValueError("pheno_masks must include is_forest")
    if wc_mask.shape != pheno_masks["is_forest"].shape:
        raise ValueError("wc_mask must match phenology mask shape")

    # ESA WorldCover v200 uses 10=Tree cover, 90=Herbaceous wetland.
    wc_priority = (wc_mask == TREE_CLASS) | (wc_mask == WETLAND_CLASS)
    return pheno_masks["is_forest"] & wc_priority


class WorldCoverPrior:
    """Build binary prior masks from ESA WorldCover COG tiles."""

    def __init__(
        self,
        year: int = 2021,
        cache_dir: Path | None = None,
        exclude_classes: tuple[int, ...] = DEFAULT_EXCLUDE_CLASSES,
    ) -> None:
        self.year = year
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.exclude_classes = tuple(int(class_id) for class_id in exclude_classes)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_worldcover_grid(
        self,
        aoi_wgs84: tuple[float, float, float, float],
        tile_transform: Affine,
        tile_shape: tuple[int, int],
        dst_crs: str | CRS,
    ) -> np.ndarray:
        """Load WorldCover classes reprojected to the target raster grid."""
        dst_crs = CRS.from_user_input(dst_crs)
        h, w = tile_shape

        cache_path = None
        if self.cache_dir:
            key = _cache_key(
                "worldcover_grid",
                self.year,
                aoi_wgs84,
                tile_transform,
                tile_shape,
                str(dst_crs),
            )
            cache_path = self.cache_dir / f"wc_grid_{key}.npy"
            if cache_path.exists():
                logger.debug("worldcover_cache_hit", key=key)
                return np.load(cache_path)

        wc_tiles = _intersecting_tiles(aoi_wgs84)
        if not wc_tiles:
            logger.warning("worldcover_no_tiles", bbox=aoi_wgs84)
            return np.zeros((h, w), dtype=np.uint8)

        dst_data = np.zeros((h, w), dtype=np.uint8)

        for lat_tile, lon_tile in wc_tiles:
            url = _tile_url(lat_tile, lon_tile, self.year)
            vsicurl = f"/vsicurl/{url}"
            try:
                with rasterio.open(vsicurl) as src:
                    tmp = np.zeros((h, w), dtype=np.uint8)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=tmp,
                        dst_transform=tile_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
                    # Overlay: keep non-zero values (later tiles overwrite)
                    mask = tmp > 0
                    dst_data[mask] = tmp[mask]
            except Exception as exc:
                logger.warning("worldcover_tile_read_failed", url=url, error=str(exc), exc_info=True)

        if cache_path is not None:
            np.save(cache_path, dst_data)

        return dst_data

    def load_worldcover_grid(
        self,
        aoi_wgs84: tuple[float, float, float, float],
        tile_transform: Affine,
        tile_shape: tuple[int, int],
        dst_crs: str | CRS,
    ) -> np.ndarray:
        """Return raw WorldCover classes aligned to the target raster grid."""
        return self._load_worldcover_grid(aoi_wgs84, tile_transform, tile_shape, dst_crs)

    def build_cropland_mask(
        self,
        aoi_wgs84: tuple[float, float, float, float],
        tile_transform: Affine,
        tile_shape: tuple[int, int],
        dst_crs: str | CRS,
    ) -> np.ndarray:
        """Return a legacy boolean (H, W) cropland mask aligned to the target grid."""
        dst_data = self._load_worldcover_grid(aoi_wgs84, tile_transform, tile_shape, dst_crs)
        if not np.any(dst_data):
            return np.ones(tile_shape, dtype=bool)
        return _mask_for_classes(dst_data, (CROPLAND_CLASS,))

    def build_exclusion_mask(
        self,
        aoi_wgs84: tuple[float, float, float, float],
        tile_transform: Affine,
        tile_shape: tuple[int, int],
        dst_crs: str | CRS,
    ) -> np.ndarray:
        """Return True where WorldCover marks sure non-field classes.

        Excludes water, wetlands, and tree cover regardless of the pheno classifier.
        """
        dst_data = self._load_worldcover_grid(aoi_wgs84, tile_transform, tile_shape, dst_crs)
        return _mask_for_classes(dst_data, self.exclude_classes)

    def build_landcover_fractions(
        self,
        aoi_wgs84: tuple[float, float, float, float],
        tile_transform: Affine,
        tile_shape: tuple[int, int],
        dst_crs: str | CRS,
    ) -> dict[str, np.ndarray]:
        """Return per-class fraction maps smoothed with a 5x5 window.

        Args:
            aoi_wgs84: (minx, miny, maxx, maxy) bounding box in EPSG:4326.
            tile_transform: Affine transform of the destination raster.
            tile_shape: (height, width) of the destination raster.
            dst_crs: CRS of the destination raster.

        Returns:
            dict with keys cropland_frac, shrubland_frac, wetland_frac, water_frac, tree_frac.
            Each value is a float32 (H, W) array with values 0..1.
        """
        from scipy.ndimage import uniform_filter

        h, w = tile_shape
        dst_data = self._load_worldcover_grid(aoi_wgs84, tile_transform, tile_shape, dst_crs)
        if not np.any(dst_data):
            logger.warning("worldcover_no_tiles_fractions", bbox=aoi_wgs84)
            return {
                "cropland_frac": np.zeros((h, w), dtype=np.float32),
                "shrubland_frac": np.zeros((h, w), dtype=np.float32),
                "wetland_frac": np.zeros((h, w), dtype=np.float32),
                "water_frac": np.zeros((h, w), dtype=np.float32),
                "tree_frac": np.zeros((h, w), dtype=np.float32),
            }

        class_map = {
            "cropland_frac": CROPLAND_CLASS,
            "shrubland_frac": SHRUBLAND_CLASS,
            "wetland_frac": WETLAND_CLASS,
            "water_frac": WATER_CLASS,
            "tree_frac": TREE_CLASS,
        }
        fractions: dict[str, np.ndarray] = {}
        for key, cls_val in class_map.items():
            binary = (dst_data == cls_val).astype(np.float32)
            fractions[key] = uniform_filter(binary, size=5).astype(np.float32)

        return fractions
