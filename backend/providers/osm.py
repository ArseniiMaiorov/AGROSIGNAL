"""OpenStreetMap data provider for exclusion masks."""
import hashlib
import json
import math
import time

import httpx
import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as shapely_transform, unary_union

from core.logging import get_logger

logger = get_logger(__name__)

OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


async def _overpass_post(query: str, *, timeout: float = 90) -> httpx.Response | None:
    """Try each Overpass mirror until one succeeds (status 200)."""
    last_resp: httpx.Response | None = None
    for url in OVERPASS_URLS:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, data={"data": query}, timeout=timeout)
            if resp.status_code == 200:
                return resp
            logger.warning("overpass_mirror_fail", url=url, status=resp.status_code)
            last_resp = resp
        except Exception as exc:
            logger.warning("overpass_mirror_error", url=url, error=str(exc)[:120])
    return last_resp

EXCLUDE_QUERY_TEMPLATE = """
[out:json][timeout:60];
(
  way["natural"="water"]({bbox});
  relation["natural"="water"]({bbox});
  way["waterway"]({bbox});
  way["natural"="wood"]({bbox});
  relation["natural"="wood"]({bbox});
  way["landuse"="forest"]({bbox});
  relation["landuse"="forest"]({bbox});
  way["building"]({bbox});
  way["landuse"~"residential|industrial|commercial|retail|cemetery"]({bbox});
  relation["landuse"~"residential|industrial|commercial|retail|cemetery"]({bbox});
  way["highway"]({bbox});
);
out geom;
"""

def _build_farmland_query(bbox: str, landuse_tags: tuple[str, ...]) -> str:
    tags = [t.strip() for t in landuse_tags if str(t).strip()]
    if not tags:
        tags = ["farmland", "farm", "meadow"]
    tag_pattern = "|".join(tags)
    return f"""
[out:json][timeout:60];
(
  way["landuse"~"{tag_pattern}"]({bbox});
  relation["landuse"~"{tag_pattern}"]({bbox});
);
out geom;
"""

class _TTLCache:
    """Simple in-memory cache with TTL expiration and max size."""
    __slots__ = ("_store", "_maxsize", "_ttl")

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 3600):
        self._store: dict[str, tuple[float, list]] = {}
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def get(self, key: str) -> list | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def put(self, key: str, value: list) -> None:
        if len(self._store) >= self._maxsize:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.monotonic(), value)


_osm_exclusion_cache = _TTLCache(maxsize=256, ttl_seconds=3600)
_osm_farmland_cache = _TTLCache(maxsize=256, ttl_seconds=3600)


def _bbox_str(bbox_4326: tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bbox_4326
    return f"{miny},{minx},{maxy},{maxx}"


def _utm_epsg_for_bbox(bbox_4326: tuple[float, float, float, float]) -> int:
    minx, miny, maxx, maxy = bbox_4326
    center_lon = 0.5 * (minx + maxx)
    center_lat = 0.5 * (miny + maxy)
    zone = int((center_lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    return (32600 if center_lat >= 0 else 32700) + zone


def _extract_geometry_dict(element: dict, *, polygon_only: bool) -> dict | None:
    if element.get("type") not in {"way", "relation"}:
        return None
    if "geometry" not in element:
        return None
    coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
    if len(coords) < 2:
        return None

    is_closed = len(coords) >= 4 and coords[0] == coords[-1]
    if is_closed:
        return {"type": "Polygon", "coordinates": [coords]}

    if polygon_only:
        if len(coords) < 3:
            return None
        closed = coords + [coords[0]]
        return {"type": "Polygon", "coordinates": [closed]}

    return {"type": "LineString", "coordinates": coords}


def _compactness(area: float, perimeter: float) -> float:
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


async def fetch_osm_exclusion_geometries(
    bbox_4326: tuple[float, float, float, float],
) -> list[dict]:
    """Fetch OSM geometries for exclusion mask.

    Args:
        bbox_4326: (minx, miny, maxx, maxy) in EPSG:4326.

    Returns:
        List of GeoJSON-like geometry dicts.
    """
    cache_key = hashlib.md5(json.dumps({"bbox": bbox_4326, "mode": "exclude"}).encode()).hexdigest()
    cached = _osm_exclusion_cache.get(cache_key)
    if cached is not None:
        return cached

    query = EXCLUDE_QUERY_TEMPLATE.replace("{bbox}", _bbox_str(bbox_4326))

    resp = await _overpass_post(query, timeout=90)

    if resp is None or resp.status_code != 200:
        logger.error("osm_fetch_error", status=getattr(resp, "status_code", None))
        return []

    data = resp.json()
    geometries: list[dict] = []
    for element in data.get("elements", []):
        geom = _extract_geometry_dict(element, polygon_only=False)
        if geom is not None:
            geometries.append(geom)

    _osm_exclusion_cache.put(cache_key, geometries)
    logger.info("osm_fetched", bbox=bbox_4326, n_geometries=len(geometries))
    return geometries


async def fetch_osm_farmland_geometries(
    bbox_4326: tuple[float, float, float, float],
    *,
    min_area_ha: float = 1.0,
    max_area_ha: float = 500.0,
    min_compactness: float = 0.1,
    landuse_tags: tuple[str, ...] | None = None,
) -> list[dict]:
    """Fetch OSM farmland polygons filtered by area and compactness."""
    selected_tags = tuple(landuse_tags or ("farmland", "farm", "meadow"))
    cache_payload = {
        "bbox": bbox_4326,
        "mode": "farmland",
        "landuse_tags": selected_tags,
        "min_area_ha": float(min_area_ha),
        "max_area_ha": float(max_area_ha),
        "min_compactness": float(min_compactness),
    }
    cache_key = hashlib.md5(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
    cached = _osm_farmland_cache.get(cache_key)
    if cached is not None:
        return cached

    query = _build_farmland_query(_bbox_str(bbox_4326), selected_tags)

    resp = await _overpass_post(query, timeout=90)

    if resp is None or resp.status_code != 200:
        logger.error("osm_farmland_fetch_error", status=getattr(resp, "status_code", None))
        return []

    data = resp.json()
    epsg = _utm_epsg_for_bbox(bbox_4326)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    geometries: list[dict] = []
    for element in data.get("elements", []):
        geom_dict = _extract_geometry_dict(element, polygon_only=True)
        if geom_dict is None:
            continue
        try:
            geom_wgs84 = shape(geom_dict)
            geom_utm = shapely_transform(lambda x, y, z=None: to_utm.transform(x, y), geom_wgs84)
            if geom_utm.is_empty or not geom_utm.is_valid:
                continue
            if isinstance(geom_utm, MultiPolygon):
                area_m2 = float(sum(poly.area for poly in geom_utm.geoms))
                perimeter_m = float(sum(poly.length for poly in geom_utm.geoms))
            else:
                area_m2 = float(geom_utm.area)
                perimeter_m = float(geom_utm.length)

            area_ha = area_m2 / 10_000.0
            compactness = _compactness(area_m2, perimeter_m)
            if area_ha < float(min_area_ha) or area_ha > float(max_area_ha):
                continue
            if compactness < float(min_compactness):
                continue
            geometries.append(geom_dict)
        except Exception:
            continue

    _osm_farmland_cache.put(cache_key, geometries)
    logger.info(
        "osm_farmland_fetched",
        bbox=bbox_4326,
        n_geometries=len(geometries),
        min_area_ha=float(min_area_ha),
        max_area_ha=float(max_area_ha),
        min_compactness=float(min_compactness),
        landuse_tags=selected_tags,
    )
    return geometries


def build_osm_exclusion_mask(
    geometries: list[dict],
    transform: Affine,
    shape_hw: tuple[int, int],
    crs: str,
    road_buffer_m: float = 15.0,
) -> np.ndarray:
    """Build raster exclusion mask from OSM geometries.

    Args:
        geometries: list of GeoJSON geometry dicts in EPSG:4326.
        transform: rasterio Affine transform (in UTM CRS).
        shape_hw: (height, width) of output raster.
        crs: target CRS string (e.g., 'EPSG:32636').
        road_buffer_m: buffer in meters for linear features.

    Returns:
        (H, W) bool array — True where to exclude.
    """
    if not geometries:
        return np.zeros(shape_hw, dtype=bool)

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    utm_geoms = []
    for g in geometries:
        try:
            geom = shape(g)
            geom_utm = shapely_transform(
                lambda x, y, z=None: to_utm.transform(x, y),
                geom,
            )
            if geom_utm.geom_type in ("LineString", "MultiLineString"):
                geom_utm = geom_utm.buffer(road_buffer_m)
            elif geom_utm.geom_type == "Point":
                geom_utm = geom_utm.buffer(road_buffer_m)
            if not geom_utm.is_empty and geom_utm.is_valid:
                utm_geoms.append(geom_utm)
        except Exception:
            continue

    if not utm_geoms:
        return np.zeros(shape_hw, dtype=bool)

    mask = rasterize(
        [(g, 1) for g in utm_geoms],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask.astype(bool)


def build_osm_farmland_mask(
    geometries: list[dict],
    transform: Affine,
    shape_hw: tuple[int, int],
    crs: str,
) -> np.ndarray:
    """Rasterize OSM farmland polygons to the target grid."""
    if not geometries:
        return np.zeros(shape_hw, dtype=bool)

    to_target = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    target_geoms = []
    for g in geometries:
        try:
            geom = shape(g)
            geom_target = shapely_transform(
                lambda x, y, z=None: to_target.transform(x, y),
                geom,
            )
            if geom_target.is_empty or not geom_target.is_valid:
                continue
            if isinstance(geom_target, Polygon):
                target_geoms.append(geom_target)
            elif isinstance(geom_target, MultiPolygon):
                target_geoms.extend(list(geom_target.geoms))
        except Exception:
            continue

    if not target_geoms:
        return np.zeros(shape_hw, dtype=bool)

    mask = rasterize(
        [(g, 1) for g in target_geoms],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)
