"""Tests for hybrid traditional + SAM polygon merge."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon

from processing.fields.merge_hybrid import merge_sam_with_traditional


class DummyCfg:
    HYBRID_MERGE_MIN_IOU = 0.20


def test_merge_sam_with_traditional_unions_overlapping_polygons():
    traditional = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs="EPSG:4326",
    )
    sam = gpd.GeoDataFrame(
        geometry=[Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])],
        crs="EPSG:4326",
    )

    merged = merge_sam_with_traditional(traditional, sam, DummyCfg())

    assert len(merged) == 1
    assert merged.iloc[0].geometry.area > traditional.iloc[0].geometry.area
    assert merged.iloc[0].geometry.area >= sam.iloc[0].geometry.area
    assert "area_m2" in merged.columns


def test_merge_sam_with_traditional_keeps_independent_traditional_fallback():
    traditional = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
        crs="EPSG:4326",
    )
    sam = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1.2, 0), (1.2, 1.2), (0, 1.2)])],
        crs="EPSG:4326",
    )

    merged = merge_sam_with_traditional(traditional, sam, DummyCfg())

    assert len(merged) == 2
