"""Tests for boundary-first region filling."""
from __future__ import annotations

import numpy as np

from processing.fields.boundary_fill import (
    boundary_to_regions,
    build_boundary_probability_map,
    filter_regions_by_phenology,
)


class DummyCfg:
    PHENO_FIELD_MAX_NDVI_MIN = 0.45
    PHENO_FIELD_NDVI_STD_MIN = 0.15


def test_boundary_to_regions_extracts_interior_hole_as_region():
    boundary = np.zeros((20, 20), dtype=np.float32)
    boundary[4:16, 4] = 1.0
    boundary[4:16, 15] = 1.0
    boundary[4, 4:16] = 1.0
    boundary[15, 4:16] = 1.0

    labeled, n = boundary_to_regions(boundary, min_region_px=10, boundary_thresh=0.5)

    assert n >= 1
    assert labeled[10, 10] > 0


def test_filter_regions_by_phenology_keeps_field_like_region():
    labeled = np.zeros((10, 10), dtype=np.int32)
    labeled[2:8, 2:8] = 1
    max_ndvi = np.zeros((10, 10), dtype=np.float32)
    max_ndvi[2:8, 2:8] = 0.6
    ndvi_std = np.zeros((10, 10), dtype=np.float32)
    ndvi_std[2:8, 2:8] = 0.2
    barrier = np.zeros((10, 10), dtype=bool)

    mask = filter_regions_by_phenology(labeled, max_ndvi, ndvi_std, barrier, DummyCfg())

    assert mask[4:6, 4:6].all()


def test_build_boundary_probability_map_uses_hard_barriers():
    owt = np.full((6, 6), 0.2, dtype=np.float32)
    road = np.zeros((6, 6), dtype=bool)
    road[2, :] = True
    water = np.zeros((6, 6), dtype=bool)
    forest = np.zeros((6, 6), dtype=bool)

    boundary = build_boundary_probability_map(owt, road, water, forest, DummyCfg())

    assert np.allclose(boundary[2, :], 1.0)
