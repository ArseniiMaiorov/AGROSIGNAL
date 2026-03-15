"""Tests for v3 SNIC/SLIC-style label refinement."""
from __future__ import annotations

import numpy as np

from processing.fields.snic_merge import snic_merge_fields


class DummyCfg:
    SNIC_REFINE_ENABLED = False
    SNIC_N_SEGMENTS = 100
    SNIC_COMPACTNESS = 0.01
    SNIC_MERGE_NDVI_THRESH = 0.05


def test_snic_merge_fields_merges_adjacent_regions_with_similar_ndvi():
    labels = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )
    maxndvi = np.array(
        [
            [0.50, 0.50, 0.53, 0.53],
            [0.50, 0.50, 0.53, 0.53],
            [0.50, 0.50, 0.53, 0.53],
            [0.50, 0.50, 0.53, 0.53],
        ],
        dtype=np.float32,
    )
    ndvistd = np.full((4, 4), 0.2, dtype=np.float32)

    merged = snic_merge_fields(labels, maxndvi, ndvistd, DummyCfg())

    assert set(np.unique(merged)) == {1}


def test_snic_merge_fields_keeps_distinct_regions_when_ndvi_diff_is_large():
    labels = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )
    maxndvi = np.array(
        [
            [0.40, 0.40, 0.70, 0.70],
            [0.40, 0.40, 0.70, 0.70],
            [0.40, 0.40, 0.70, 0.70],
            [0.40, 0.40, 0.70, 0.70],
        ],
        dtype=np.float32,
    )
    ndvistd = np.full((4, 4), 0.2, dtype=np.float32)

    merged = snic_merge_fields(labels, maxndvi, ndvistd, DummyCfg())

    assert set(np.unique(merged)) == {1, 2}
