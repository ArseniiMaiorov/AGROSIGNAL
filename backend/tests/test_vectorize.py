"""Tests for vectorization (polygonize)."""
import numpy as np
import pytest
from rasterio.transform import from_bounds

from processing.fields.vectorize import (
    merge_tile_polygons,
    polygonize_labels,
    summarize_polygon_areas,
)


class TestPolygonizeLabels:
    def test_basic_polygonize(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:50, 10:50] = 1  # 40x40 px = 160000 m2 at 10m
        labels[60:90, 60:90] = 2  # 30x30 px = 90000 m2 at 10m

        transform = from_bounds(300000, 6500000, 301000, 6501000, 100, 100)
        gdf = polygonize_labels(labels, transform, "EPSG:32636", min_area_ha=0.3, simplify_tol_m=5.0)

        assert len(gdf) == 2
        assert "area_m2" in gdf.columns
        assert "perimeter_m" in gdf.columns
        assert gdf.crs.to_epsg() == 4326

    def test_filters_small(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:15, 10:15] = 1  # 5x5 px = 2500 m2 at 10m < 3000 (0.3ha)
        labels[30:48, 30:48] = 2  # 18x18 px = 32400 m2

        transform = from_bounds(300000, 6500000, 300500, 6500500, 50, 50)
        gdf = polygonize_labels(labels, transform, "EPSG:32636", min_area_ha=0.3)

        assert len(gdf) == 1  # only the large one

    def test_empty_labels(self):
        labels = np.zeros((30, 30), dtype=np.int32)
        transform = from_bounds(300000, 6500000, 300300, 6500300, 30, 30)
        gdf = polygonize_labels(labels, transform, "EPSG:32636")
        assert len(gdf) == 0

    def test_geometry_is_multipolygon(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[5:45, 5:45] = 1
        transform = from_bounds(300000, 6500000, 300500, 6500500, 50, 50)
        gdf = polygonize_labels(labels, transform, "EPSG:32636", min_area_ha=0.0)
        for geom in gdf.geometry:
            assert geom.geom_type == "MultiPolygon"

    def test_simplify_zero_keeps_polygon_without_forced_clip(self):
        labels = np.zeros((40, 40), dtype=np.int32)
        labels[8:32, 8:32] = 1
        transform = from_bounds(300000, 6500000, 300400, 6500400, 40, 40)

        gdf = polygonize_labels(
            labels,
            transform,
            "EPSG:32636",
            min_area_ha=0.0,
            simplify_tol_m=0.0,
        )

        assert len(gdf) == 1
        assert gdf.iloc[0]["area_m2"] == pytest.approx(24 * 24 * 100.0)

    def test_polygonize_emits_progress_callback(self):
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[5:20, 5:20] = 1
        labels[25:45, 25:45] = 2
        transform = from_bounds(300000, 6500000, 300600, 6500600, 60, 60)
        events: list[tuple[str, int, int]] = []

        gdf = polygonize_labels(
            labels,
            transform,
            "EPSG:32636",
            min_area_ha=0.0,
            progress_callback=lambda stage, done, total: events.append((stage, done, total)),
        )

        assert len(gdf) == 2
        assert events
        assert events[0][0] == "polygonize"


class TestMergeTilePolygons:
    def test_merge_empty(self):
        gdf = merge_tile_polygons([])
        assert len(gdf) == 0

    def test_merge_single(self):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box
        poly = MultiPolygon([box(29.0, 58.0, 29.1, 58.1)])
        gdf = gpd.GeoDataFrame(
            {"label": [1], "area_m2": [50000.0], "perimeter_m": [900.0]},
            geometry=[poly], crs="EPSG:4326",
        )
        merged = merge_tile_polygons([gdf])
        assert len(merged) == 1

    def test_deduplicates_identical_polygons(self):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box

        poly = MultiPolygon([box(29.0, 58.0, 29.1, 58.1)])
        gdf1 = gpd.GeoDataFrame(
            {"label": [1], "area_m2": [50000.0], "perimeter_m": [900.0]},
            geometry=[poly], crs="EPSG:4326",
        )
        gdf2 = gpd.GeoDataFrame(
            {"label": [2], "area_m2": [49999.0], "perimeter_m": [900.0]},
            geometry=[poly], crs="EPSG:4326",
        )
        merged = merge_tile_polygons([gdf1, gdf2])
        assert len(merged) == 1

    def test_preserves_numeric_ml_columns(self):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box

        poly = MultiPolygon([box(29.0, 58.0, 29.1, 58.1)])
        gdf1 = gpd.GeoDataFrame(
            {
                "label": [1],
                "area_m2": [50000.0],
                "perimeter_m": [900.0],
                "ndvi_mean": [0.2],
                "edge_max": [0.4],
            },
            geometry=[poly],
            crs="EPSG:4326",
        )
        gdf2 = gpd.GeoDataFrame(
            {
                "label": [2],
                "area_m2": [50000.0],
                "perimeter_m": [900.0],
                "ndvi_mean": [0.6],
                "edge_max": [0.9],
            },
            geometry=[poly],
            crs="EPSG:4326",
        )

        merged = merge_tile_polygons([gdf1, gdf2])

        assert len(merged) == 1
        assert merged.at[0, "ndvi_mean"] == pytest.approx(0.4)
        assert merged.at[0, "edge_max"] == pytest.approx(0.9)

    def test_merges_transitively_connected_polygons(self):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box

        gdf = gpd.GeoDataFrame(
            {"label": [1, 2, 3], "area_m2": [1.0, 1.0, 1.0], "perimeter_m": [1.0, 1.0, 1.0]},
            geometry=[
                MultiPolygon([box(29.00, 58.00, 29.02, 58.02)]),
                MultiPolygon([box(29.015, 58.00, 29.035, 58.02)]),
                MultiPolygon([box(29.03, 58.00, 29.05, 58.02)]),
            ],
            crs="EPSG:4326",
        )
        merged = merge_tile_polygons([gdf])
        assert len(merged) == 1

    def test_area_summary(self):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box

        gdf = gpd.GeoDataFrame(
            {"label": [1, 2], "area_m2": [100.0, 400.0], "perimeter_m": [40.0, 80.0]},
            geometry=[
                MultiPolygon([box(29.0, 58.0, 29.01, 58.01)]),
                MultiPolygon([box(29.1, 58.1, 29.12, 58.12)]),
            ],
            crs="EPSG:4326",
        )
        stats = summarize_polygon_areas(gdf)
        assert stats["p50"] == pytest.approx(250.0)

    def test_merge_emits_progress_callback(self):
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box

        gdf = gpd.GeoDataFrame(
            {"label": [1, 2], "area_m2": [1.0, 1.0], "perimeter_m": [1.0, 1.0]},
            geometry=[
                MultiPolygon([box(29.00, 58.00, 29.02, 58.02)]),
                MultiPolygon([box(29.015, 58.00, 29.035, 58.02)]),
            ],
            crs="EPSG:4326",
        )
        events: list[tuple[str, int, int]] = []

        merge_tile_polygons(
            [gdf],
            progress_callback=lambda stage, done, total: events.append((stage, done, total)),
        )

        assert events
        assert events[0][0] == "merge_groups"
