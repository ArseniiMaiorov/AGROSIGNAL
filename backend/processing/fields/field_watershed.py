"""Optional raster-level watershed refinement before the main OBIA stage."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label
from skimage.feature import peak_local_max
from skimage.filters import gaussian, sobel
from skimage.morphology import disk, erosion
from skimage.segmentation import watershed


def watershed_field_segmentation(
    candidate_mask: np.ndarray,
    ndvi: np.ndarray,
    edge_composite: np.ndarray,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine a candidate mask using a conservative watershed pass.

    Markers are placed using distance-transform peaks with
    ``WATERSHED_MIN_DISTANCE`` controlling the minimum separation between
    markers.  This prevents over-segmentation inside large homogeneous
    fields where a simple erosion would produce many small cores.
    """
    candidate_mask = candidate_mask.astype(bool, copy=False)
    if not np.any(candidate_mask):
        empty = np.zeros_like(candidate_mask, dtype=np.int32)
        return candidate_mask.copy(), empty

    if ndvi.shape != candidate_mask.shape or edge_composite.shape != candidate_mask.shape:
        raise ValueError("ndvi and edge_composite must match candidate_mask shape")

    min_distance = max(2, int(getattr(cfg, "WATERSHED_MIN_DISTANCE", 14)))

    # Build markers from distance-transform peaks with min_distance control.
    # This replaces the old erosion-based approach which ignored
    # WATERSHED_MIN_DISTANCE and generated too many markers.
    dist = distance_transform_edt(candidate_mask)
    peak_coords = peak_local_max(
        dist,
        min_distance=min_distance,
        labels=candidate_mask.astype(np.int32),
        exclude_border=False,
    )
    if peak_coords.shape[0] <= 1:
        labels, _ = nd_label(candidate_mask)
        return candidate_mask.copy(), labels.astype(np.int32)

    markers = np.zeros_like(candidate_mask, dtype=np.int32)
    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = idx

    gradient = sobel(gaussian(ndvi.astype(np.float32), sigma=1.0, preserve_range=True))
    gradient = gradient + (
        edge_composite.astype(np.float32) * float(cfg.WATERSHED_GRADIENT_EDGE_W)
    )
    gmax = float(np.max(gradient))
    if gmax > 1e-6:
        gradient = gradient / gmax

    labels = watershed(
        image=gradient,
        markers=markers,
        mask=candidate_mask,
        compactness=float(cfg.WATERSHED_COMPACTNESS),
        # Keep the full field fill; later segmentation still separates objects.
        watershed_line=False,
    ).astype(np.int32)

    refined = labels > 0
    if not np.any(refined):
        labels, _ = nd_label(candidate_mask)
        return candidate_mask.copy(), labels.astype(np.int32)
    return refined, labels
