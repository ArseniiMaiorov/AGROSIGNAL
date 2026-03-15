"""Optional OSM-backed road rasterization for merge barriers."""
from __future__ import annotations

from hashlib import md5
from pathlib import Path
import pickle

import numpy as np
from rasterio.features import rasterize

from core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised only when optional dependency is installed
    import osmnx as ox
except Exception:  # pragma: no cover - keep runtime optional
    ox = None


def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    token = ",".join(f"{float(v):.6f}" for v in bbox)
    return md5(token.encode("utf-8")).hexdigest()[:12]


def _normalize_epsg(crs_epsg: int | str) -> int:
    if isinstance(crs_epsg, int):
        return crs_epsg
    if isinstance(crs_epsg, str) and crs_epsg.upper().startswith("EPSG:"):
        return int(crs_epsg.split(":", 1)[1])
    return int(crs_epsg)


def _empty_cached(cache_path: Path, out_shape: tuple[int, int]) -> np.ndarray:
    empty = np.zeros(out_shape, dtype=bool)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(empty, f)
    return empty


def _buffer_for_highway(value, cfg) -> int:
    if cfg is None:
        return 8
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item in cfg.ROAD_OSM_BUFFER_MAP:
                return int(cfg.ROAD_OSM_BUFFER_MAP[item])
        value = value[0] if value else None
    if isinstance(value, str):
        return int(cfg.ROAD_OSM_BUFFER_MAP.get(value, cfg.ROAD_OSM_BUFFER_DEFAULT_M))
    return int(cfg.ROAD_OSM_BUFFER_DEFAULT_M)


def fetch_road_mask(
    bbox: tuple[float, float, float, float],
    transform,
    out_shape: tuple[int, int],
    crs_epsg: int | str,
    cache_dir: Path = Path(".cache/roads"),
    cfg=None,
) -> np.ndarray:
    """Fetch roads from OSM and rasterize them into the tile grid.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:4326.
        transform: raster transform of the output tile.
        out_shape: raster shape (H, W).
        crs_epsg: projected EPSG used by the tile.
        cache_dir: on-disk cache to avoid repeated Overpass hits.
        cfg: settings object.
    """
    out_shape = (int(out_shape[0]), int(out_shape[1]))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    epsg = _normalize_epsg(crs_epsg)
    cache_path = cache_dir / f"road_{_bbox_hash(tuple(map(float, bbox)))}_{epsg}.pkl"

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            cached = np.asarray(cached, dtype=bool)
            if cached.shape == out_shape:
                return cached
        except Exception:
            logger.warning("road_osm_cache_invalid", cache_path=str(cache_path))

    if cfg is not None and not bool(getattr(cfg, "ROAD_OSM_ENABLED", True)):
        logger.debug("road_osm_disabled")
        return _empty_cached(cache_path, out_shape)

    if ox is None:
        logger.debug("road_osm_dependency_missing", cache_path=str(cache_path))
        return _empty_cached(cache_path, out_shape)

    west, south, east, north = map(float, bbox)
    tags = {"highway": list(getattr(cfg, "ROAD_OSM_TAGS", ()))} if cfg else {"highway": True}

    # Bound Overpass wait time and optionally disable long rate-limit sleeps.
    if cfg is not None:
        try:
            ox.settings.requests_timeout = max(
                1,
                int(getattr(cfg, "ROAD_OSM_REQUEST_TIMEOUT_S", 45)),
            )
            ox.settings.overpass_rate_limit = bool(
                getattr(cfg, "ROAD_OSM_OVERPASS_RATE_LIMIT", False)
            )
        except Exception as exc:
            logger.debug("road_osm_settings_apply_failed", error=str(exc))

    try:  # pragma: no cover - depends on network and Overpass availability
        try:
            roads = ox.features_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                tags=tags,
            )
        except TypeError:
            # osmnx >= 2.0 expects bbox=(left, bottom, right, top)
            roads = ox.features_from_bbox(
                bbox=(west, south, east, north),
                tags=tags,
            )
    except Exception as exc:  # pragma: no cover - external dependency path
        logger.warning("road_osm_fetch_failed", error=str(exc), bbox=list(bbox))
        return _empty_cached(cache_path, out_shape)

    if roads.empty or "geometry" not in roads:
        return _empty_cached(cache_path, out_shape)

    lines = roads[roads.geometry.notna()].copy()
    if lines.empty:
        return _empty_cached(cache_path, out_shape)

    lines = lines[lines.geometry.type.isin(["LineString", "MultiLineString"])]
    if lines.empty:
        return _empty_cached(cache_path, out_shape)

    try:  # pragma: no cover - depends on optional dependency stack
        lines = lines.to_crs(epsg=epsg)
    except Exception as exc:
        logger.warning("road_osm_reproject_failed", error=str(exc), epsg=epsg)
        return _empty_cached(cache_path, out_shape)

    if "highway" in lines:
        highway_values = lines["highway"]
    else:
        highway_values = np.full(len(lines), None, dtype=object)
    lines["buf_m"] = [
        _buffer_for_highway(value, cfg)
        for value in highway_values
    ]
    lines["geometry"] = lines.apply(lambda row: row.geometry.buffer(float(row["buf_m"])), axis=1)

    shapes = [(geom, 1) for geom in lines.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return _empty_cached(cache_path, out_shape)

    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    with cache_path.open("wb") as f:
        pickle.dump(mask, f)
    return mask
