"""Tests for AOI tiling."""
import pytest
from shapely.geometry import Point

from processing.fields.tiling import (
    bbox_to_polygon,
    get_utm_epsg,
    make_tiles,
    point_radius_to_polygon,
    polygon_coords_to_polygon,
)


class TestGetUtmEpsg:
    def test_luga_region(self):
        epsg = get_utm_epsg(29.89, 58.69)
        assert epsg == 32635  # zone 35N (lon 29.89 => zone 35)

    def test_southern_hemisphere(self):
        epsg = get_utm_epsg(29.89, -33.0)
        assert epsg == 32735

    def test_zone_boundaries(self):
        assert get_utm_epsg(3.0, 50.0) == 32631   # zone 1
        assert get_utm_epsg(33.0, 50.0) == 32636   # zone 36


class TestPointRadiusToPolygon:
    def test_creates_polygon(self):
        poly = point_radius_to_polygon(58.689077, 29.892103, 15.0)
        assert poly.is_valid
        assert not poly.is_empty
        assert poly.geom_type == "Polygon"

    def test_contains_center(self):
        poly = point_radius_to_polygon(58.689077, 29.892103, 15.0)
        center = Point(29.892103, 58.689077)
        assert poly.contains(center)

    def test_approximate_area(self):
        poly = point_radius_to_polygon(58.689077, 29.892103, 15.0)
        # Approximate area in degrees, rough check
        bounds = poly.bounds
        lon_span = bounds[2] - bounds[0]
        lat_span = bounds[3] - bounds[1]
        # 15km radius => ~30km diameter => ~0.27 deg lat, larger lon at this latitude
        assert 0.2 < lat_span < 0.4
        assert lon_span > lat_span  # at 59N, longitude degrees are shorter

    def test_small_radius(self):
        poly = point_radius_to_polygon(58.689077, 29.892103, 1.0)
        assert poly.is_valid


class TestMakeTiles:
    def test_basic_tiling(self):
        aoi = point_radius_to_polygon(58.689077, 29.892103, 15.0)
        tiles = make_tiles(aoi, tile_size_m=20480, overlap_m=500, resolution_m=10)
        assert len(tiles) > 0

        for t in tiles:
            assert "bbox_4326" in t
            assert "transform" in t
            assert "shape" in t
            assert "crs" in t
            assert "tile_id" in t
            assert len(t["bbox_4326"]) == 4

    def test_small_aoi_single_tile(self):
        aoi = point_radius_to_polygon(58.689077, 29.892103, 1.0)
        tiles = make_tiles(aoi, tile_size_m=20480, overlap_m=500)
        assert len(tiles) >= 1

    def test_tile_shapes_valid(self):
        aoi = point_radius_to_polygon(58.689077, 29.892103, 5.0)
        tiles = make_tiles(aoi, tile_size_m=10240, overlap_m=500, resolution_m=10)
        for t in tiles:
            h, w = t["shape"]
            assert h > 0
            assert w > 0

    def test_rejects_non_positive_step(self):
        aoi = point_radius_to_polygon(58.689077, 29.892103, 1.0)
        with pytest.raises(ValueError):
            make_tiles(aoi, tile_size_m=1000, overlap_m=1000, resolution_m=10)


class TestAoiConverters:
    def test_bbox_to_polygon(self):
        poly = bbox_to_polygon([29.0, 58.0, 30.0, 59.0])
        assert poly.is_valid
        assert poly.geom_type == "Polygon"
        assert poly.bounds == pytest.approx((29.0, 58.0, 30.0, 59.0))

    def test_bbox_to_polygon_rejects_invalid(self):
        with pytest.raises(ValueError):
            bbox_to_polygon([29.0, 58.0, 29.0, 59.0])

    def test_polygon_coords_to_polygon(self):
        poly = polygon_coords_to_polygon(
            [[29.0, 58.0], [30.0, 58.0], [30.0, 59.0], [29.0, 58.0]]
        )
        assert poly.is_valid
        assert poly.geom_type == "Polygon"

    def test_polygon_coords_to_polygon_rejects_invalid(self):
        with pytest.raises(ValueError):
            polygon_coords_to_polygon([[29.0, 58.0], [30.0, 58.0]])
