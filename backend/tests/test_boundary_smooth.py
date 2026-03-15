"""Tests for barrier-aware raster cleanup and hole-safe vector smoothing."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from processing.fields.boundary_smooth import (
    clean_raster_mask,
    close_boundary_gaps,
    smooth_field_polygon,
)


class DummyCfg:
    POST_MORPH_CLOSE_RADIUS = 1
    POST_MIN_FIELD_AREA_HA = 0.05
    POST_PX_AREA_M2 = 100
    POST_GAP_EDGE_THRESHOLD = 0.15
    POST_SIMPLIFY_TOLERANCE = 0.0
    POST_BUFFER_SMOOTH_M = 4.0
    NORTH_GAP_CLOSE_MAX_HA = 0.35
    NORTH_BOUNDARY_SMOOTH_SIMPLIFY_TOL_M = 0.5
    NORTH_POST_BUFFER_SMOOTH_M = 0.0
    NORTH_STAGE_ROLLBACK_MIN_AREA_RATIO = 0.95


def test_clean_raster_mask_does_not_fill_forbidden_hole():
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 3:13] = True
    mask[7:9, 7:9] = False

    forbidden = np.zeros_like(mask, dtype=bool)
    forbidden[7:9, 7:9] = True

    cleaned, debug = clean_raster_mask(
        mask,
        DummyCfg(),
        hard_exclusion_mask=forbidden,
        return_debug=True,
    )

    assert not cleaned[7:9, 7:9].any()
    assert debug["holes_skipped_due_to_forbidden"][7:9, 7:9].all()


def test_close_boundary_gaps_does_not_fill_forbidden_hole_even_with_weak_edge():
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 3:13] = True
    mask[7:9, 7:9] = False

    edge = np.zeros_like(mask, dtype=np.float32)
    forbidden = np.zeros_like(mask, dtype=bool)
    forbidden[7:9, 7:9] = True

    closed, added, debug = close_boundary_gaps(
        mask,
        edge,
        DummyCfg(),
        hard_exclusion_mask=forbidden,
        return_debug=True,
    )

    assert added == 0
    assert not closed[7:9, 7:9].any()
    assert debug["holes_skipped_due_to_forbidden"][7:9, 7:9].all()


def test_smooth_field_polygon_preserves_holes_by_skipping_buffer():
    outer = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    hole = [(35, 35), (65, 35), (65, 65), (35, 65), (35, 35)]
    geom = Polygon(outer, [hole])

    smoothed = smooth_field_polygon(geom, DummyCfg())

    assert smoothed.is_valid
    assert smoothed.geom_type == "Polygon"
    assert len(smoothed.interiors) == 1
    assert smoothed.area == pytest.approx(geom.area)


def test_close_boundary_gaps_respects_north_specific_smaller_gap_limit():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:17, 10:17] = False  # 49 px => 0.49 ha, fillable by default but not in north

    edge = np.zeros_like(mask, dtype=np.float32)

    closed_default, added_default = close_boundary_gaps(mask, edge, DummyCfg())
    closed_north, added_north = close_boundary_gaps(
        mask,
        edge,
        DummyCfg(),
        region_profile="north_boundary",
    )

    assert added_default > 0
    assert closed_default[10:17, 10:17].all()
    assert added_north == 0
    assert not closed_north[10:17, 10:17].any()


def test_close_boundary_gaps_emits_progress_for_large_component_scan():
    mask = np.zeros((32, 32), dtype=bool)
    for offset in range(0, 24, 4):
        mask[offset : offset + 2, offset : offset + 2] = True
    edge = np.zeros_like(mask, dtype=np.float32)
    events: list[tuple[int, int]] = []

    close_boundary_gaps(
        mask,
        edge,
        DummyCfg(),
        progress_callback=lambda done, total: events.append((done, total)),
    )

    assert events
    assert events[0][0] >= 1
    assert events[-1][0] == events[-1][1]
