"""Tests for multi-temporal stack aggregation."""
from __future__ import annotations

import numpy as np

from processing.fields.temporal_stack import build_temporal_stack


class DummyCfg:
    EDGE_ALPHA = 0.7
    EDGE_CANNY_SIGMA = 1.2
    EDGE_COVERAGE_THRESHOLD = 0.30


def test_build_temporal_stack_aggregates_valid_pixels():
    ndvi_stack = np.array(
        [
            [[0.1, 0.5], [0.2, 0.3]],
            [[0.8, 0.4], [0.1, 0.2]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [[True, True], [True, False]],
            [[True, True], [False, True]],
        ],
        dtype=bool,
    )

    bands = {
        "B2": np.zeros_like(ndvi_stack),
        "B3": np.zeros_like(ndvi_stack),
        "B4": np.zeros_like(ndvi_stack),
        "B8": np.ones_like(ndvi_stack),
        "ndvi": ndvi_stack,
    }

    stack = build_temporal_stack(
        ndvi_stack=ndvi_stack,
        valid_mask=valid_mask,
        edge_bands=bands,
        cfg=DummyCfg(),
    )

    assert stack["max_ndvi"].shape == (2, 2)
    assert stack["mean_ndvi"].shape == (2, 2)
    assert stack["ndvi_std"].shape == (2, 2)
    assert stack["edge_composite"].shape == (2, 2)
    assert stack["max_ndvi"][0, 0] == np.float32(0.8)
    assert stack["mean_ndvi"][0, 1] == np.float32(0.45)
    assert stack["mean_ndvi"][1, 0] == np.float32(0.2)
