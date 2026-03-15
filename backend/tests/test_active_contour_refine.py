from __future__ import annotations

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from processing.fields.active_contour_refine import refine_field_with_snake


class _CfgOff:
    SNAKE_REFINE_ENABLED = True
    SNAKE_REFINE_MODE = "off"


class _CfgGuarded:
    SNAKE_REFINE_ENABLED = True
    SNAKE_REFINE_MODE = "guarded"
    SNAKE_MAX_PX_DIST = 15.0
    SNAKE_MAX_CENTROID_SHIFT_M = 6.0
    SNAKE_MIN_AREA_RATIO = 0.9
    SNAKE_MAX_AREA_RATIO = 1.12


def test_refine_field_with_snake_returns_original_when_mode_off():
    polygon = Polygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
    edge = np.zeros((64, 64), dtype=np.float32)
    transform = from_origin(0, 64, 1, 1)

    refined, diagnostics = refine_field_with_snake(polygon, edge, transform, _CfgOff())

    assert refined.equals(polygon)
    assert diagnostics["applied"] is False
    assert diagnostics["rejected_reason"] == "disabled"


def test_refine_field_with_snake_guarded_skips_holes():
    polygon = Polygon(
        [(10, 10), (40, 10), (40, 40), (10, 40), (10, 10)],
        holes=[[(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)]],
    )
    edge = np.ones((64, 64), dtype=np.float32)
    transform = from_origin(0, 64, 1, 1)

    refined, diagnostics = refine_field_with_snake(polygon, edge, transform, _CfgGuarded())

    assert refined.equals(polygon)
    assert diagnostics["applied"] is False
    assert diagnostics["rejected_reason"] == "has_holes"
