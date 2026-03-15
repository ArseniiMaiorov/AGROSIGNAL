"""Thin wrapper around NDVI phenology masks for use as a prior."""
from __future__ import annotations

import numpy as np

from processing.fields.ndvi_phenology import compute_phenology_masks


def build_phenology_prior(
    max_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    cfg,
) -> dict[str, np.ndarray]:
    """Return reusable prior masks derived from temporal NDVI statistics."""
    return compute_phenology_masks(max_ndvi=max_ndvi, ndvi_std=ndvi_std, cfg=cfg)
