"""Optional SNIC/SLIC-style refinement for over-segmented labels."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

try:  # pragma: no cover - optional at runtime
    from skimage.segmentation import slic
except Exception:  # pragma: no cover
    slic = None


def snic_merge_fields(
    labels: np.ndarray,
    maxndvi: np.ndarray,
    ndvistd: np.ndarray,
    cfg,
) -> np.ndarray:
    """Greedily merge adjacent regions with very similar NDVI means."""
    if not (labels.shape == maxndvi.shape == ndvistd.shape):
        raise ValueError("labels, maxndvi and ndvistd must share the same shape")
    if labels.max() <= 1:
        return labels.astype(np.int32, copy=True)

    work = labels.astype(np.int32, copy=True)
    # NOTE: SLIC computation was removed — result was discarded (dead code).

    merge_thresh = float(getattr(cfg, "SNIC_MERGE_NDVI_THRESH", 0.05))
    unique_labels = np.unique(work)
    unique_labels = unique_labels[unique_labels > 0]
    # Compute means efficiently with bincount
    flat_work = work.ravel()
    flat_ndvi = maxndvi.ravel().astype(np.float64)
    max_lbl = int(flat_work.max()) + 1
    sums = np.bincount(flat_work.clip(0), weights=np.nan_to_num(flat_ndvi, nan=0.0), minlength=max_lbl)
    counts = np.bincount(flat_work.clip(0), minlength=max_lbl).astype(np.float64)
    counts[counts == 0] = 1.0
    means_arr = sums / counts

    for lbl in unique_labels:
        if not np.any(work == lbl):
            continue  # label was already merged away
        mask = work == lbl
        border = binary_dilation(mask, iterations=1) & ~mask
        neighbors = np.unique(work[border])
        neighbors = neighbors[neighbors > 0]
        for neighbor in neighbors:
            if lbl == neighbor:
                continue
            if not np.any(work == neighbor):
                continue  # neighbor already merged
            if abs(float(means_arr[lbl]) - float(means_arr[neighbor])) < merge_thresh:
                target = int(min(lbl, neighbor))
                source = int(max(lbl, neighbor))
                work[work == source] = target
                target_pixels = maxndvi[work == target]
                means_arr[target] = float(np.nanmean(target_pixels)) if target_pixels.size else 0.0
    return work.astype(np.int32)
