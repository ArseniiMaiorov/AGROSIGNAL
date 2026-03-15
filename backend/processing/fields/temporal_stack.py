"""Utilities for multi-temporal raster aggregation."""
from __future__ import annotations

import numpy as np

from processing.fields.edge_composite import build_multitemporal_edge_composite


def _ensure_time_stack(arr: np.ndarray, *, name: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"{name} must be 2D or 3D, got shape={arr.shape}")


def build_temporal_stack(
    ndvi_stack: np.ndarray,
    valid_mask: np.ndarray,
    edge_bands: dict[str, np.ndarray] | None = None,
    cfg=None,
) -> dict[str, np.ndarray]:
    """Build stable per-pixel aggregates from a selected seasonal stack."""
    ndvi_stack = _ensure_time_stack(np.asarray(ndvi_stack, dtype=np.float32), name="ndvi_stack")
    valid_mask = _ensure_time_stack(np.asarray(valid_mask, dtype=bool), name="valid_mask")
    if ndvi_stack.shape != valid_mask.shape:
        raise ValueError(
            "ndvi_stack and valid_mask must match, "
            f"got {ndvi_stack.shape} vs {valid_mask.shape}"
        )

    cube = np.where(valid_mask, ndvi_stack, np.nan)
    with np.errstate(all="ignore"):
        max_ndvi = np.nanmax(cube, axis=0)
        mean_ndvi = np.nanmean(cube, axis=0)
        ndvi_std = np.nanstd(cube, axis=0)

    max_ndvi = np.where(np.isfinite(max_ndvi), max_ndvi, 0.0).astype(np.float32)
    mean_ndvi = np.where(np.isfinite(mean_ndvi), mean_ndvi, 0.0).astype(np.float32)
    ndvi_std = np.where(np.isfinite(ndvi_std), ndvi_std, 0.0).astype(np.float32)

    if edge_bands is None:
        edge_bands = {"ndvi": ndvi_stack}

    edge_alpha = float(getattr(cfg, "EDGE_ALPHA", 0.7))
    edge_sigma = float(getattr(cfg, "EDGE_CANNY_SIGMA", 1.2))
    coverage_threshold = float(getattr(cfg, "EDGE_COVERAGE_THRESHOLD", 0.30))
    binary_threshold = float(getattr(cfg, "EDGE_BINARY_THRESHOLD", 0.12))
    closing_radius = int(getattr(cfg, "EDGE_CLOSING_RADIUS", 2))
    edge_composite = build_multitemporal_edge_composite(
        edge_bands,
        valid_mask,
        alpha=edge_alpha,
        canny_sigma=edge_sigma,
        coverage_threshold=coverage_threshold,
        binary_threshold=binary_threshold,
        closing_radius=closing_radius,
    )

    return {
        "max_ndvi": max_ndvi,
        "mean_ndvi": mean_ndvi,
        "ndvi_std": ndvi_std,
        "edge_composite": edge_composite.astype(np.float32),
    }
