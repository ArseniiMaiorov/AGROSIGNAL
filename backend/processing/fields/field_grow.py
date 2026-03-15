"""Seeded region growing for extending crop cores into border grass pixels."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from processing.fields.phenoclassify import GRASS


def seeded_grow_into_field(
    field_mask: np.ndarray,
    ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
    growable_mask: np.ndarray | None = None,
    region_profile: str | None = None,
    boundary_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Expand field seeds into nearby pixels that still look field-like in NDVI space."""
    if ndvi.shape != field_mask.shape:
        raise ValueError("ndvi must match field_mask shape")
    if ndvi_std.shape != field_mask.shape:
        raise ValueError("ndvi_std must match field_mask shape")
    if barrier_mask.shape != field_mask.shape:
        raise ValueError("barrier_mask must match field_mask shape")
    if boundary_prob is not None and boundary_prob.shape != field_mask.shape:
        raise ValueError("boundary_prob must match field_mask shape")

    token = str(region_profile or "").strip().lower()
    if token == "south_recall":
        ndvi_lo = float(cfg.PHENO_FIELD_MAX_NDVI_MIN) - float(
            getattr(cfg, "SOUTH_POST_GROW_NDVI_RELAX", 0.15)
        )
        ndvi_hi = float(getattr(cfg, "PHENO_FIELD_MAX_NDVI_MAX", cfg.PHENO_NDVI_CROP_MAX)) + 0.15
        ndvi_std_min = max(0.0, float(cfg.PHENO_FIELD_NDVI_STD_MIN) - 0.02)
        max_iters = max(0, int(getattr(cfg, "SOUTH_POST_GROW_MAX_ITERS", cfg.POST_GROW_MAX_ITERS)))
    elif token == "north_boundary":
        ndvi_lo = float(cfg.PHENO_FIELD_MAX_NDVI_MIN) - 0.08
        ndvi_hi = float(getattr(cfg, "PHENO_FIELD_MAX_NDVI_MAX", cfg.PHENO_NDVI_CROP_MAX)) + 0.08
        ndvi_std_min = float(cfg.PHENO_FIELD_NDVI_STD_MIN)
        max_iters = max(0, int(getattr(cfg, "NORTH_POST_GROW_MAX_ITERS", cfg.POST_GROW_MAX_ITERS)))
    else:
        ndvi_lo = float(cfg.PHENO_FIELD_MAX_NDVI_MIN) - 0.10
        ndvi_hi = float(getattr(cfg, "PHENO_FIELD_MAX_NDVI_MAX", cfg.PHENO_NDVI_CROP_MAX)) + 0.10
        ndvi_std_min = float(cfg.PHENO_FIELD_NDVI_STD_MIN)
        max_iters = max(0, int(cfg.POST_GROW_MAX_ITERS))

    growable = (
        np.isfinite(ndvi)
        & np.isfinite(ndvi_std)
        & (ndvi > -0.05)  # reject Sentinel nodata sentinel (-0.1)
        & (ndvi >= ndvi_lo)
        & (ndvi <= ndvi_hi)
        & (ndvi_std >= ndvi_std_min)
        & (~barrier_mask)
    )
    if growable_mask is not None:
        if growable_mask.shape != field_mask.shape:
            raise ValueError("growable_mask must match field_mask shape")
        growable &= growable_mask.astype(bool, copy=False)
    bp_threshold = float(getattr(cfg, "POST_GROW_BOUNDARY_STOP_THRESHOLD", 0.30))
    if boundary_prob is not None:
        bp_arr = np.asarray(boundary_prob, dtype=np.float32)
        growable &= bp_arr <= bp_threshold

    grown = field_mask.astype(bool, copy=True)
    added = 0
    structure = np.ones((3, 3), dtype=bool)
    for iteration in range(max_iters):
        expanded = binary_dilation(grown, structure=structure)
        new_pixels = expanded & growable & ~grown
        # Progressive tightening: each iteration reduces the allowed boundary prob
        if boundary_prob is not None and iteration > 0:
            iter_threshold = bp_threshold * max(0.5, 1.0 - iteration * 0.08)
            new_pixels &= bp_arr <= iter_threshold
        new_count = int(np.count_nonzero(new_pixels))
        if new_count == 0:
            break
        grown |= new_pixels
        added += new_count

    grown &= ~barrier_mask
    return grown, added


def seeded_grow_into_grass(
    candidate_mask: np.ndarray,
    classes: np.ndarray,
    ndvi: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
    growable_mask: np.ndarray | None = None,
    region_profile: str | None = None,
    boundary_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Expand crop seeds into adjacent grass pixels using relaxed NDVI bounds."""
    token = str(region_profile or "").strip().lower()
    if token == "south_recall":
        grow_relax = float(getattr(cfg, "SOUTH_POST_GROW_NDVI_RELAX", cfg.POST_GROW_NDVI_RELAX))
        max_iters = max(0, int(getattr(cfg, "SOUTH_POST_GROW_MAX_ITERS", cfg.POST_GROW_MAX_ITERS)))
    elif token == "north_boundary":
        grow_relax = min(float(cfg.POST_GROW_NDVI_RELAX), 0.05)
        max_iters = max(0, int(getattr(cfg, "NORTH_POST_GROW_MAX_ITERS", cfg.POST_GROW_MAX_ITERS)))
    else:
        grow_relax = float(cfg.POST_GROW_NDVI_RELAX)
        max_iters = max(0, int(cfg.POST_GROW_MAX_ITERS))

    ndvi_lo = float(cfg.PHENO_NDVI_CROP_MIN) - grow_relax
    ndvi_hi = float(cfg.PHENO_NDVI_CROP_MAX) + grow_relax

    growable = (
        (classes == GRASS)
        & np.isfinite(ndvi)
        & (ndvi >= ndvi_lo)
        & (ndvi <= ndvi_hi)
        & (~barrier_mask)
    )
    bp_threshold = float(getattr(cfg, "POST_GROW_BOUNDARY_STOP_THRESHOLD", 0.30))
    bp_arr = None
    if boundary_prob is not None:
        if boundary_prob.shape != candidate_mask.shape:
            raise ValueError("boundary_prob must match candidate_mask shape")
        bp_arr = np.asarray(boundary_prob, dtype=np.float32)
        growable &= bp_arr <= bp_threshold
    if growable_mask is not None:
        if growable_mask.shape != candidate_mask.shape:
            raise ValueError("growable_mask must match candidate_mask shape")
        growable &= growable_mask.astype(bool, copy=False)

    grown = candidate_mask.astype(bool, copy=True)
    added = 0
    structure = np.ones((3, 3), dtype=bool)
    for iteration in range(max_iters):
        expanded = binary_dilation(grown, structure=structure)
        new_pixels = expanded & growable & ~grown
        if bp_arr is not None and iteration > 0:
            iter_threshold = bp_threshold * max(0.5, 1.0 - iteration * 0.08)
            new_pixels &= bp_arr <= iter_threshold
        new_count = int(np.count_nonzero(new_pixels))
        if new_count == 0:
            break
        grown |= new_pixels
        added += new_count

    grown &= ~barrier_mask
    return grown, added
