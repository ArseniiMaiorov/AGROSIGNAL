"""Feature stack assembly for BoundaryUNet inference."""
from __future__ import annotations

import numpy as np


def build_feature_stack_v4(
    *,
    edge_composite: np.ndarray,
    max_ndvi: np.ndarray,
    mean_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    ndwi_mean: np.ndarray,
    bsi_mean: np.ndarray,
    scl_valid_fraction: np.ndarray,
    rgb_r: np.ndarray,
    rgb_g: np.ndarray,
    rgb_b: np.ndarray,
    feature_channels: tuple[str, ...] | None = None,
    s1_vv_mean: np.ndarray | None = None,
    s1_vh_mean: np.ndarray | None = None,
    ndvi_entropy: np.ndarray | None = None,
    mndwi_max: np.ndarray | None = None,
    ndmi_mean: np.ndarray | None = None,
    ndwi_median: np.ndarray | None = None,
    green_median: np.ndarray | None = None,
    swir_median: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble feature stack for BoundaryUNet inference with profile-aware channels."""
    if feature_channels is None:
        from processing.fields.ml_inference import FEATURE_CHANNELS

        feature_channels = tuple(FEATURE_CHANNELS)

    h, w = edge_composite.shape[:2]
    _zeros = None

    def _as_f32(arr):
        if arr is None:
            nonlocal _zeros
            if _zeros is None:
                _zeros = np.zeros((h, w), dtype=np.float32)
            return _zeros
        return np.asarray(arr, dtype=np.float32) if arr.dtype != np.float32 else arr

    source = {
        "edge_composite": _as_f32(edge_composite),
        "max_ndvi": _as_f32(max_ndvi),
        "mean_ndvi": _as_f32(mean_ndvi),
        "ndvi_std": _as_f32(ndvi_std),
        "ndwi_mean": _as_f32(ndwi_mean),
        "bsi_mean": _as_f32(bsi_mean),
        "scl_valid_fraction": np.clip(_as_f32(scl_valid_fraction), 0.0, 1.0),
        "rgb_r": np.clip(_as_f32(rgb_r), 0.0, 1.0),
        "rgb_g": np.clip(_as_f32(rgb_g), 0.0, 1.0),
        "rgb_b": np.clip(_as_f32(rgb_b), 0.0, 1.0),
        "s1_vv_mean": _as_f32(s1_vv_mean),
        "s1_vh_mean": _as_f32(s1_vh_mean),
        "ndvi_entropy": _as_f32(ndvi_entropy),
        "mndwi_max": _as_f32(mndwi_max),
        "ndmi_mean": _as_f32(ndmi_mean),
        "ndwi_median": _as_f32(ndwi_median),
        "green_median": _as_f32(green_median),
        "swir_median": _as_f32(swir_median),
    }
    n_ch = len(feature_channels)
    result = np.empty((n_ch, h, w), dtype=np.float32)
    if _zeros is None:
        _zeros = np.zeros((h, w), dtype=np.float32)
    for i, name in enumerate(feature_channels):
        result[i] = source.get(name, _zeros)
    return result
