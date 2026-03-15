"""Phenology-based priors derived from max NDVI and NDVI variability."""
from __future__ import annotations

import numpy as np


def compute_phenology_masks(
    max_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    cfg,
) -> dict[str, np.ndarray]:
    """Classify stable vs. seasonal vegetation using temporal NDVI features."""
    if max_ndvi.shape != ndvi_std.shape:
        raise ValueError("max_ndvi and ndvi_std must share the same shape")

    finite = np.isfinite(max_ndvi) & np.isfinite(ndvi_std)

    # Coniferous forest: high NDVI, very low variability
    is_forest_conifer = (
        finite
        & (max_ndvi > float(cfg.PHENO_FOREST_MAX_NDVI_MIN))
        & (ndvi_std < float(cfg.PHENO_FOREST_NDVI_STD_MAX))
    )
    # Deciduous forest: high NDVI with moderate seasonal variability
    is_forest_deciduous = (
        finite
        & (max_ndvi >= float(getattr(cfg, "PHENO_FOREST_DECID_MAX_NDVI_MIN", 0.60)))
        & (ndvi_std >= float(getattr(cfg, "PHENO_FOREST_DECID_NDVI_STD_MIN", 0.07)))
        & (ndvi_std <= float(getattr(cfg, "PHENO_FOREST_DECID_NDVI_STD_MAX", 0.18)))
    )
    is_forest = is_forest_conifer | is_forest_deciduous

    is_grass = (
        finite
        & (max_ndvi < float(cfg.PHENO_GRASS_MAX_NDVI_MAX))
        & (ndvi_std < float(cfg.PHENO_GRASS_NDVI_STD_MAX))
        & ~is_forest
    )

    is_field = (
        finite
        & (max_ndvi > float(cfg.PHENO_FIELD_MAX_NDVI_MIN))
        & (ndvi_std > float(cfg.PHENO_FIELD_NDVI_STD_MIN))
        & ~is_forest
        & ~is_grass
    )

    is_bare = finite & (max_ndvi < 0.22) & (ndvi_std < 0.04)

    return {
        "is_field": is_field,
        "is_grass": is_grass,
        "is_forest": is_forest,
        "is_bare": is_bare,
    }


def compute_field_candidate(
    max_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
) -> np.ndarray:
    """Build an NDVI-driven field candidate mask before class-based clipping."""
    if max_ndvi.shape != ndvi_std.shape:
        raise ValueError("max_ndvi and ndvi_std must share the same shape")
    if barrier_mask.shape != max_ndvi.shape:
        raise ValueError("barrier_mask must match max_ndvi shape")

    finite = np.isfinite(max_ndvi) & np.isfinite(ndvi_std)
    offset = float(getattr(cfg, "PHENO_FIELD_CANDIDATE_NDVI_OFFSET", 0.05))
    max_ndvi_min = float(cfg.PHENO_FIELD_MAX_NDVI_MIN) - offset
    max_ndvi_max = float(getattr(cfg, "PHENO_FIELD_MAX_NDVI_MAX", 0.62)) + offset
    ndvi_std_min = float(cfg.PHENO_FIELD_NDVI_STD_MIN)

    return (
        finite
        & (max_ndvi > max_ndvi_min)
        & (max_ndvi < max_ndvi_max)
        & (ndvi_std > ndvi_std_min)
        & ~barrier_mask.astype(bool, copy=False)
    )
