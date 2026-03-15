"""ML seed refinement and fusion helpers."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def boundary_guided_ml_seed(
    *,
    extent_prob: np.ndarray,
    boundary_prob: np.ndarray,
    ndvi: np.ndarray,
    ndvi_std: np.ndarray | None,
    cfg,
    extent_threshold_override: float | None = None,
    dilation_px_override: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build a conservative ML seed and optionally recover a safe boundary ring."""
    if extent_threshold_override is not None:
        threshold = float(extent_threshold_override)
    else:
        threshold = float(
            getattr(cfg, "ML_EXTENT_BIN_THRESHOLD", 0.42)
            if bool(getattr(cfg, "ML_EXTENT_CALIBRATION_ENABLED", True))
            else 0.5
        )
    seed = np.asarray(extent_prob, dtype=np.float32) > threshold
    if not np.any(seed):
        return seed, {
            "ml_extent_threshold": threshold,
            "seed_pixels_before_dilation": 0,
            "seed_pixels_after_dilation": 0,
            "boundary_dilation_added_pixels": 0,
        }

    if dilation_px_override is not None:
        dilation_px = max(0, int(dilation_px_override))
    else:
        dilation_px = max(0, int(getattr(cfg, "POST_BOUNDARY_DILATION_PX", 1)))
    max_dilation_px = max(dilation_px, int(getattr(cfg, "POST_BOUNDARY_DILATION_MAX_PX", 3)))
    large_field_px = max(
        1,
        int(
            float(getattr(cfg, "POST_LARGE_FIELD_RESCUE_MIN_AREA_HA", 2.0))
            * 10_000.0
            / max(float(getattr(cfg, "POST_PX_AREA_M2", 100)), 1.0)
        ),
    )
    if (
        bool(getattr(cfg, "POST_LARGE_FIELD_RESCUE_ENABLED", True))
        and int(np.count_nonzero(seed)) >= large_field_px
    ):
        dilation_px = min(max_dilation_px, max(1, dilation_px + 1))
    else:
        dilation_px = min(max_dilation_px, dilation_px)

    grown = seed.copy()
    if dilation_px > 0:
        dilated = binary_dilation(seed, iterations=dilation_px)
        ring = dilated & ~seed
        allowed = np.isfinite(ndvi)
        if ndvi_std is not None:
            ndvi_lo = float(getattr(cfg, "PHENO_FIELD_MAX_NDVI_MIN", 0.45)) - 0.12
            ndvi_hi = float(
                getattr(cfg, "PHENO_FIELD_MAX_NDVI_MAX", getattr(cfg, "PHENO_NDVI_CROP_MAX", 0.62))
            ) + 0.08
            std_lo = max(0.0, float(getattr(cfg, "PHENO_FIELD_NDVI_STD_MIN", 0.15)) - 0.05)
            allowed &= np.isfinite(ndvi_std)
            allowed &= (ndvi >= ndvi_lo) & (ndvi <= ndvi_hi) & (ndvi_std >= std_lo)
        # Use 70th percentile as the gate: only allow dilation into pixels
        # where boundary probability is BELOW this threshold (i.e., not edges).
        # 90th was too strict — it blocked dilation into legitimate field pixels
        # near boundaries, causing under-recovery and fragmented fields.
        # 70th keeps high-confidence boundaries as barriers while allowing
        # recovery of field interior pixels that have moderate boundary signal.
        boundary_gate = float(
            np.clip(
                np.nanpercentile(np.asarray(boundary_prob, dtype=np.float32), 70),
                0.25,
                0.50,
            )
        )
        allowed &= np.asarray(boundary_prob, dtype=np.float32) <= boundary_gate
        grown |= (ring & allowed)

    return grown.astype(bool), {
        "ml_extent_threshold": threshold,
        "seed_pixels_before_dilation": int(np.count_nonzero(seed)),
        "seed_pixels_after_dilation": int(np.count_nonzero(grown)),
        "boundary_dilation_added_pixels": int(np.count_nonzero(grown & ~seed)),
    }


def fuse_ml_primary_candidate(
    ml_seed_mask: np.ndarray,
    pre_ml_candidate_mask: np.ndarray,
    region_boundary_profile: str | None,
) -> tuple[np.ndarray, list[str]]:
    """Fuse ML seed with pre-ML support to avoid losing validated candidates."""
    ml_seed = np.asarray(ml_seed_mask, dtype=bool)
    pre_ml = np.asarray(pre_ml_candidate_mask, dtype=bool)
    region_token = str(region_boundary_profile or "").strip().lower()
    if not np.any(ml_seed):
        return ml_seed, []
    if not np.any(pre_ml):
        return ml_seed, []

    if region_token == "south_recall":
        fused = ml_seed | pre_ml
        if np.count_nonzero(fused) > np.count_nonzero(ml_seed):
            return fused, ["south_ml_seed_union"]
        return ml_seed, []

    if region_token == "north_boundary":
        local_support = pre_ml & binary_dilation(ml_seed, iterations=2)
        fused = ml_seed | local_support
        if np.count_nonzero(pre_ml) > (np.count_nonzero(ml_seed) * 1.05):
            fused = fused | pre_ml
        if np.count_nonzero(fused) > np.count_nonzero(ml_seed):
            return fused, ["north_ml_seed_union"]
        return ml_seed, []

    local_support = pre_ml & binary_dilation(ml_seed, iterations=2)
    fused = ml_seed | local_support
    if np.count_nonzero(pre_ml) > (np.count_nonzero(ml_seed) * 1.08):
        fused = fused | pre_ml
    if np.count_nonzero(fused) > np.count_nonzero(ml_seed):
        return fused, ["balanced_ml_seed_union"]
    return ml_seed, []
