"""AOI tiling and coordinate utilities."""
import math

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform


def get_utm_epsg(lon: float, lat: float) -> int:
    """Compute UTM zone EPSG code for a given lon/lat.

    Args:
        lon: longitude in degrees.
        lat: latitude in degrees.

    Returns:
        EPSG code (e.g., 32636 for zone 36N).
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def point_radius_to_polygon(
    lat: float, lon: float, radius_km: float
) -> Polygon:
    """Create a circular polygon from point + radius.

    Projects to UTM, buffers, projects back to 4326.

    Args:
        lat: center latitude.
        lon: center longitude.
        radius_km: radius in kilometers.

    Returns:
        Polygon in EPSG:4326.
    """
    utm_epsg = get_utm_epsg(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_4326 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(lon, lat)
    circle_utm = Polygon(
        [
            (cx + radius_km * 1000 * math.cos(a), cy + radius_km * 1000 * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, 64)
        ]
    )

    circle_4326 = shapely_transform(
        lambda x, y, z=None: to_4326.transform(x, y),
        circle_utm,
    )
    return circle_4326


def bbox_to_polygon(bbox: list[float] | tuple[float, float, float, float]) -> Polygon:
    """Build Polygon from bbox [min_lon, min_lat, max_lon, max_lat] in EPSG:4326."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly 4 values: [min_lon, min_lat, max_lon, max_lat]")
    minx, miny, maxx, maxy = map(float, bbox)
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox must satisfy min < max for both axes")
    return box(minx, miny, maxx, maxy)


def polygon_coords_to_polygon(coords: list[list[float]] | tuple[tuple[float, float], ...]) -> Polygon:
    """Build Polygon from WGS84 coordinate sequence [[lon, lat], ...]."""
    if len(coords) < 3:
        raise ValueError("polygon must contain at least 3 coordinate pairs")
    ring = [(float(lon), float(lat)) for lon, lat in coords]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("invalid polygon coordinates")
    return poly


def make_tiles(
    aoi_4326: Polygon,
    tile_size_m: float = 20480,
    overlap_m: float = 500,
    resolution_m: float = 10,
) -> list[dict]:
    """Split AOI into tiles for processing.

    Args:
        aoi_4326: AOI polygon in EPSG:4326.
        tile_size_m: tile size in meters.
        overlap_m: overlap between tiles in meters.
        resolution_m: pixel resolution in meters.

    Returns:
        List of tile dicts with: bbox_4326, transform, shape, crs, tile_id.
    """
    centroid = aoi_4326.centroid
    utm_epsg = get_utm_epsg(centroid.x, centroid.y)
    crs = f"EPSG:{utm_epsg}"

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_4326 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    aoi_utm = shapely_transform(
        lambda x, y, z=None: to_utm.transform(x, y),
        aoi_4326,
    )

    minx, miny, maxx, maxy = aoi_utm.bounds
    step = tile_size_m - overlap_m
    if tile_size_m <= 0 or resolution_m <= 0:
        raise ValueError("tile_size_m and resolution_m must be > 0")
    if step <= 0:
        raise ValueError("overlap_m must be smaller than tile_size_m")

    tiles = []
    tile_id = 0
    y = miny
    row = 0
    while y < maxy:
        x = minx
        col = 0
        while x < maxx:
            tx_min = x - overlap_m / 2
            ty_min = y - overlap_m / 2
            tx_max = x + tile_size_m + overlap_m / 2
            ty_max = y + tile_size_m + overlap_m / 2

            tile_box_utm = box(tx_min, ty_min, tx_max, ty_max)
            if not tile_box_utm.intersects(aoi_utm):
                x += step
                col += 1
                continue

            width_px = int((tx_max - tx_min) / resolution_m)
            height_px = int((ty_max - ty_min) / resolution_m)
            tile_transform = from_bounds(tx_min, ty_min, tx_max, ty_max, width_px, height_px)

            tile_4326 = shapely_transform(
                lambda xx, yy, z=None: to_4326.transform(xx, yy),
                tile_box_utm,
            )
            bbox_4326 = tile_4326.bounds

            tiles.append({
                "tile_id": tile_id,
                "bbox_4326": bbox_4326,
                "bbox_utm": (tx_min, ty_min, tx_max, ty_max),
                "transform": tile_transform,
                "shape": (height_px, width_px),
                "crs": crs,
                "row": row,
                "col": col,
            })
            tile_id += 1
            x += step
            col += 1
        y += step
        row += 1

    return tiles
