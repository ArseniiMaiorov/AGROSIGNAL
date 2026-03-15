"""Orientation-aware edge enhancement for watershed segmentation."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift
from skimage.filters import gaussian, sobel, sobel_h, sobel_v


def _normalize(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.nanpercentile(vals, 2))
    hi = float(np.nanpercentile(vals, 98))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def oriented_watershed(
    edge_composite: np.ndarray,
    ndvi: np.ndarray,
    sigma_orientation: float = 1.5,
    sigma_strength: float = 0.8,
    cfg=None,
) -> np.ndarray:
    """Enhance weak boundaries using coarse local edge orientation."""
    if edge_composite.shape != ndvi.shape:
        raise ValueError("edge_composite and ndvi must share the same shape")

    sigma_orientation = float(
        getattr(cfg, "OWT_SIGMA_ORIENTATION", sigma_orientation)
        if cfg is not None
        else sigma_orientation
    )
    sigma_strength = float(
        getattr(cfg, "OWT_SIGMA_STRENGTH", sigma_strength)
        if cfg is not None
        else sigma_strength
    )

    edge = np.nan_to_num(edge_composite.astype(np.float32), nan=0.0)
    ndvi = np.nan_to_num(ndvi.astype(np.float32), nan=0.0)

    smooth_edge = gaussian(edge, sigma=sigma_orientation, preserve_range=True)
    gx = sobel_v(smooth_edge)
    gy = sobel_h(smooth_edge)

    # Gradient is perpendicular to the boundary; rotate by 90 degrees.
    tangent = (np.arctan2(gy, gx) + (np.pi / 2.0)) % np.pi

    resp_0 = np.maximum.reduce(
        [
            edge,
            shift(edge, shift=(0, 1), order=1, mode="nearest"),
            shift(edge, shift=(0, -1), order=1, mode="nearest"),
        ]
    )
    resp_45 = np.maximum.reduce(
        [
            edge,
            shift(edge, shift=(1, 1), order=1, mode="nearest"),
            shift(edge, shift=(-1, -1), order=1, mode="nearest"),
        ]
    )
    resp_90 = np.maximum.reduce(
        [
            edge,
            shift(edge, shift=(1, 0), order=1, mode="nearest"),
            shift(edge, shift=(-1, 0), order=1, mode="nearest"),
        ]
    )
    resp_135 = np.maximum.reduce(
        [
            edge,
            shift(edge, shift=(1, -1), order=1, mode="nearest"),
            shift(edge, shift=(-1, 1), order=1, mode="nearest"),
        ]
    )
    responses = np.stack([resp_0, resp_45, resp_90, resp_135], axis=0)

    bins = np.array([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0], dtype=np.float32)
    angle_delta = np.abs(((tangent[..., None] - bins + (np.pi / 2.0)) % np.pi) - (np.pi / 2.0))
    nearest = np.argmin(angle_delta, axis=-1)
    oriented = np.take_along_axis(responses, nearest[None, ...], axis=0)[0]

    coherence = _normalize(np.sqrt(gx**2 + gy**2))
    enhanced_edge = np.maximum(edge, oriented * (0.5 + 0.5 * coherence))
    enhanced_edge = _normalize(enhanced_edge)

    ndvi_edge = sobel(gaussian(ndvi, sigma=sigma_strength, preserve_range=True))
    ndvi_edge = _normalize(ndvi_edge)

    weight_owt = float(getattr(cfg, "OWT_EDGE_WEIGHT", 0.7)) if cfg is not None else 0.7
    weight_ndvi = float(getattr(cfg, "OWT_NDVI_WEIGHT", 0.3)) if cfg is not None else 0.3
    combined = (weight_owt * enhanced_edge) + (weight_ndvi * ndvi_edge)
    return _normalize(combined)
