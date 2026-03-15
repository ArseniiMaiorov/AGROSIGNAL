"""Tests for the multi-year temporal composite scene-loader path."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from processing.fields.temporal_composite import build_multiyear_composite


class DummyCfg:
    EDGE_ALPHA = 0.7
    EDGE_CANNY_SIGMA = 1.2
    EDGE_COVERAGE_THRESHOLD = 0.30
    TEMPORAL_YEARS_BACK = 0
    TEMPORAL_BEST_N_SCENES = 8
    TEMPORAL_SCL_INVALID = (8,)
    TEMPORAL_NDVI_VALID_MIN = 0.05


def test_multiyear_composite_skips_fully_cloudy_scenes():
    ndvi = np.random.uniform(0.2, 0.6, size=(32, 32)).astype(np.float32)
    scl = np.full((32, 32), 8, dtype=np.uint8)
    mock_scene = {
        "ndvi": ndvi,
        "scl": scl,
        "meta": {"cloud_cover": 100},
    }

    result = build_multiyear_composite(
        scene_loader=lambda *_: [mock_scene],
        date_from=datetime(2024, 5, 1),
        date_to=datetime(2024, 5, 30),
        cfg=DummyCfg(),
    )

    assert result["n_valid_scenes"] == 0
    assert result["max_ndvi"].shape == (32, 32)
    assert np.count_nonzero(result["max_ndvi"]) == 0
