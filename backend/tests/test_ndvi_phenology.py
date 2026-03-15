"""Tests for NDVI phenology priors."""
from __future__ import annotations

import numpy as np

from processing.fields.ndvi_phenology import compute_field_candidate, compute_phenology_masks


class DummyCfg:
    PHENO_FIELD_NDVI_STD_MIN = 0.15
    PHENO_GRASS_NDVI_STD_MAX = 0.12
    PHENO_FOREST_NDVI_STD_MAX = 0.05
    PHENO_FIELD_MAX_NDVI_MIN = 0.45
    PHENO_FIELD_MAX_NDVI_MAX = 0.62
    PHENO_GRASS_MAX_NDVI_MAX = 0.55
    PHENO_FOREST_MAX_NDVI_MIN = 0.65


def test_compute_phenology_masks_separates_field_grass_forest():
    max_ndvi = np.array([[0.72, 0.51, 0.18]], dtype=np.float32)
    ndvi_std = np.array([[0.03, 0.20, 0.02]], dtype=np.float32)

    masks = compute_phenology_masks(max_ndvi, ndvi_std, DummyCfg())

    assert masks["is_forest"][0, 0]
    assert masks["is_field"][0, 1]
    assert masks["is_bare"][0, 2]
    assert not masks["is_grass"][0, 1]


def test_compute_field_candidate_uses_relaxed_ndvi_window_and_barriers():
    max_ndvi = np.array([[0.41, 0.52, 0.76]], dtype=np.float32)
    ndvi_std = np.array([[0.18, 0.16, 0.20]], dtype=np.float32)
    barrier_mask = np.array([[False, True, False]], dtype=bool)

    candidate = compute_field_candidate(max_ndvi, ndvi_std, barrier_mask, DummyCfg())

    assert candidate[0, 0]
    assert not candidate[0, 1]
    assert not candidate[0, 2]
