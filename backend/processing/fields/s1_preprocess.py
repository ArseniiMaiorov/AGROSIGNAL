"""Sentinel-1 SAR preprocessing helpers."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import scharr


def lee_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Simple Lee speckle filter."""
    mean = uniform_filter(img, size=window_size)
    sqr_mean = uniform_filter(img ** 2, size=window_size)
    variance = sqr_mean - mean ** 2
    weights = variance / (variance + mean ** 2 + 1e-10)
    return mean + weights * (img - mean)


def preprocess_s1(vv_linear: np.ndarray, vh_linear: np.ndarray, cfg=None) -> dict[str, np.ndarray]:
    """Convert raw VV/VH into edge and ratio features."""
    if vv_linear.shape != vh_linear.shape:
        raise ValueError("VV and VH must share the same shape")

    vv_db = 10.0 * np.log10(vv_linear + 1e-10)
    vh_db = 10.0 * np.log10(vh_linear + 1e-10)

    use_lee = bool(getattr(cfg, "S1_LEE_FILTER_ENABLE", True)) if cfg is not None else True
    window = int(getattr(cfg, "S1_LEE_WINDOW_SIZE", 5)) if cfg is not None else 5
    if use_lee:
        vv_proc = lee_filter(vv_db, window_size=window)
        vh_proc = lee_filter(vh_db, window_size=window)
    else:
        vv_proc = vv_db
        vh_proc = vh_db

    vv_edge = scharr(vv_proc).astype(np.float32)
    vhvv_ratio = (vh_proc / (vv_proc + 1e-6)).astype(np.float32)
    return {
        "VV_edge": vv_edge,
        "VHVV_ratio": vhvv_ratio,
    }
