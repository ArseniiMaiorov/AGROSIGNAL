"""Boundary-first region filling helpers."""
from __future__ import annotations

from inspect import signature

import numpy as np
from scipy.ndimage import binary_dilation, label as nd_label
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

_REMOVE_SMALL_OBJECTS_SUPPORTS_MAX = "max_size" in signature(remove_small_objects).parameters


def build_boundary_probability_map(
    owt_edge: np.ndarray,
    road_mask: np.ndarray,
    water_mask: np.ndarray,
    forest_mask: np.ndarray,
    cfg,
) -> np.ndarray:
    """Combine soft OWT edges with hard barriers into one boundary map."""
    if not (
        owt_edge.shape == road_mask.shape == water_mask.shape == forest_mask.shape
    ):
        raise ValueError("all boundary inputs must share the same shape")

    boundary = np.nan_to_num(owt_edge.astype(np.float32), nan=0.0).copy()
    hard = (
        road_mask.astype(bool, copy=False)
        | water_mask.astype(bool, copy=False)
        | forest_mask.astype(bool, copy=False)
    )
    boundary[hard] = 1.0

    soft = ~hard
    if np.any(soft):
        vals = boundary[soft]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 95))
        if hi - lo > 1e-8:
            boundary[soft] = np.clip((vals - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        else:
            boundary[soft] = 0.0

    return boundary.astype(np.float32)


def boundary_to_regions(
    boundary_prob: np.ndarray,
    min_region_px: int = 50,
    boundary_thresh: float | None = None,
) -> tuple[np.ndarray, int]:
    """Turn a boundary probability map into closed interior regions."""
    if boundary_thresh is None:
        try:
            boundary_thresh = float(threshold_otsu(boundary_prob))
        except Exception:
            boundary_thresh = 0.35

    binary_boundary = boundary_prob >= float(boundary_thresh)
    closed_boundary = binary_dilation(binary_boundary, iterations=1)
    inverted = ~closed_boundary

    labeled_raw, _ = nd_label(inverted)
    border_labels = np.unique(
        np.concatenate(
            [
                labeled_raw[0, :],
                labeled_raw[-1, :],
                labeled_raw[:, 0],
                labeled_raw[:, -1],
            ]
        )
    )
    interior = (labeled_raw > 0) & ~np.isin(labeled_raw, border_labels)
    region_px = max(1, int(min_region_px))
    if _REMOVE_SMALL_OBJECTS_SUPPORTS_MAX:
        interior = remove_small_objects(interior, max_size=max(0, region_px - 1))
    else:
        interior = remove_small_objects(interior, min_size=region_px)
    labeled, n_labels = nd_label(interior)
    return labeled.astype(np.int32), int(n_labels)


def filter_regions_by_phenology(
    labeled: np.ndarray,
    max_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
) -> np.ndarray:
    """Keep only regions whose mean NDVI behavior looks like a field."""
    if not (
        labeled.shape == max_ndvi.shape == ndvi_std.shape == barrier_mask.shape
    ):
        raise ValueError("all inputs must share the same shape")

    n_labels = int(labeled.max())
    if n_labels <= 0:
        return np.zeros_like(labeled, dtype=bool)

    labels_flat = labeled.ravel().astype(np.int32, copy=False)
    active = labels_flat > 0
    if not np.any(active):
        return np.zeros_like(labeled, dtype=bool)

    label_ids = labels_flat[active]
    counts = np.bincount(label_ids, minlength=n_labels + 1).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    barrier_weights = barrier_mask.ravel()[active].astype(np.float32, copy=False)
    barrier_ratio = np.bincount(
        label_ids,
        weights=barrier_weights,
        minlength=n_labels + 1,
    ) / counts

    max_ndvi_flat = max_ndvi.ravel()
    ndvi_std_flat = ndvi_std.ravel()

    valid_ndvi = active & np.isfinite(max_ndvi_flat)
    ndvi_counts = np.bincount(labels_flat[valid_ndvi], minlength=n_labels + 1).astype(np.float32)
    ndvi_sums = np.bincount(
        labels_flat[valid_ndvi],
        weights=max_ndvi_flat[valid_ndvi].astype(np.float32, copy=False),
        minlength=n_labels + 1,
    )
    mean_max_ndvi = np.zeros(n_labels + 1, dtype=np.float32)
    np.divide(ndvi_sums, np.maximum(ndvi_counts, 1.0), out=mean_max_ndvi, where=ndvi_counts > 0)

    valid_std = active & np.isfinite(ndvi_std_flat)
    std_counts = np.bincount(labels_flat[valid_std], minlength=n_labels + 1).astype(np.float32)
    std_sums = np.bincount(
        labels_flat[valid_std],
        weights=ndvi_std_flat[valid_std].astype(np.float32, copy=False),
        minlength=n_labels + 1,
    )
    mean_std = np.zeros(n_labels + 1, dtype=np.float32)
    np.divide(std_sums, np.maximum(std_counts, 1.0), out=mean_std, where=std_counts > 0)

    keep = np.zeros(n_labels + 1, dtype=bool)
    keep[1:] = (
        (barrier_ratio[1:] <= 0.30)
        & (ndvi_counts[1:] > 0)
        & (std_counts[1:] > 0)
        & (mean_max_ndvi[1:] > float(cfg.PHENO_FIELD_MAX_NDVI_MIN))
        & (mean_std[1:] > float(cfg.PHENO_FIELD_NDVI_STD_MIN))
    )
    return keep[labeled]
