"""Internal hole filling for large, near-convex crop regions."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.ndimage import binary_fill_holes, label as nd_label
from skimage.morphology import convex_hull_image


def infill_field_holes(
    candidate_mask: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
    allow_infill_mask: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Fill safe holes inside large field components while respecting barriers.

    Args:
        candidate_mask: boolean field mask.
        barrier_mask: hard exclusion mask (roads, water, etc.).
        cfg: settings object.
        allow_infill_mask: optional mask where True = safe to infill.
            When provided, holes are only filled if the fraction of
            allow-pixels inside the hole >= cfg.POST_INFILL_MIN_ALLOW_FRAC.
    """
    if candidate_mask.shape != barrier_mask.shape:
        raise ValueError("barrier_mask must match candidate_mask shape")
    if allow_infill_mask is not None and allow_infill_mask.shape != candidate_mask.shape:
        raise ValueError("allow_infill_mask must match candidate_mask shape")

    candidate_mask = candidate_mask.astype(bool, copy=False)
    barrier_mask = barrier_mask.astype(bool, copy=False)
    labeled, n_labels = nd_label(candidate_mask)
    if n_labels <= 0:
        return candidate_mask.copy(), 0

    result = candidate_mask.astype(bool, copy=True)
    added = 0
    min_px = max(1, int(cfg.POST_CONVEX_MIN_HA * 10_000 / cfg.POST_PX_AREA_M2))
    ratio_max = float(cfg.POST_CONVEX_RATIO_MAX)
    max_hole_frac = float(getattr(cfg, "POST_INFILL_MAX_HOLE_FRAC", 1.0))
    min_allow_frac = float(getattr(cfg, "POST_INFILL_MIN_ALLOW_FRAC", 0.0))
    last_emit_at = 0.0

    for component_id in range(1, n_labels + 1):
        try:
            component = labeled == component_id
            area_px = int(np.count_nonzero(component))
            if area_px < min_px:
                continue

            try:
                hull = convex_hull_image(component)
            except Exception:
                continue
            hull_px = int(np.count_nonzero(hull))
            if hull_px <= 0 or area_px <= 0:
                continue
            # ratio_max check: hull_px/area_px measures how much larger the
            # convex hull is vs the actual component.  A value of 1.0 means
            # perfect convexity.  We want to infill fields that are *roughly*
            # convex but have internal holes, so the ratio should be modest.
            if (hull_px / area_px) > ratio_max:
                continue

            filled = binary_fill_holes(component)
            hole_pixels = filled & ~component
            hole_area = int(np.count_nonzero(hole_pixels))
            if hole_area <= 0:
                continue

            # Hole fraction check: skip if hole is too large relative to field
            if area_px > 0 and (hole_area / area_px) > max_hole_frac:
                continue

            # Allow-infill check: skip if not enough allow-pixels in hole
            if allow_infill_mask is not None and hole_area > 0:
                allow_count = int(np.count_nonzero(allow_infill_mask[hole_pixels]))
                if (allow_count / hole_area) < min_allow_frac:
                    continue

            new_pixels = hole_pixels & ~barrier_mask
            new_count = int(np.count_nonzero(new_pixels))
            if new_count <= 0:
                continue

            result |= new_pixels
            added += new_count
        finally:
            if progress_callback is not None:
                now = time.monotonic()
                if component_id == 1 or component_id == n_labels or (now - last_emit_at) >= 2.0:
                    progress_callback(int(component_id), int(n_labels))
                    last_emit_at = now

    result &= ~barrier_mask
    return result, added
