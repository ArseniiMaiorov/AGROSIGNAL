"""Temporal coherence metrics for field detection.

Compute growth-peak patterns and temporal entropy from NDVI time series
to distinguish croplands (clear seasonal cycle) from forests/wetlands
(stable or irregular NDVI).
"""
from __future__ import annotations

import numpy as np

from utils.nan_safe import nanmax_safe, nanmean_safe, nanmin_safe


def compute_temporal_coherence(
    ndvi_stack: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_valid_dates: int = 3,
) -> dict[str, np.ndarray]:
    """Compute temporal coherence metrics from an NDVI time series.

    Args:
        ndvi_stack: (T, H, W) NDVI values.
        valid_mask: (T, H, W) boolean mask of valid (cloud-free) pixels.
        min_valid_dates: Minimum number of valid dates per pixel to compute
            meaningful metrics.

    Returns:
        dict with:
            has_growth_peak: (H, W) bool — pixels with clear rise→peak→decline.
            ndvi_temporal_entropy: (H, W) float32 — Shannon entropy of temporal
                NDVI distribution (lower = more predictable = more likely field).
            growth_amplitude: (H, W) float32 — NDVI range (max - min) for valid
                pixels; larger for croplands.
    """
    ndvi = np.asarray(ndvi_stack, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if ndvi.ndim != 3 or valid.ndim != 3:
        raise ValueError(
            f"ndvi_stack and valid_mask must be (T,H,W), "
            f"got {ndvi.shape} and {valid.shape}"
        )

    t, h, w = ndvi.shape
    masked = np.where(valid, ndvi, np.nan)

    # Count valid observations per pixel
    n_valid = valid.sum(axis=0)  # (H, W)
    enough = n_valid >= min_valid_dates

    # Growth amplitude: max - min NDVI
    ndvi_max = nanmax_safe(masked, axis=0, fill_value=np.nan)
    ndvi_min = nanmin_safe(masked, axis=0, fill_value=np.nan)
    growth_amplitude = np.where(enough, ndvi_max - ndvi_min, 0.0).astype(np.float32)

    # Detect growth peak: rise→peak→decline pattern
    # Find argmax of NDVI per pixel, check that values before are lower
    # and values after are lower
    has_growth_peak = _detect_growth_peak(masked, valid, enough)

    # Temporal entropy of NDVI distribution
    ndvi_temporal_entropy = _compute_temporal_entropy(masked, valid, enough)

    return {
        "has_growth_peak": has_growth_peak,
        "ndvi_temporal_entropy": ndvi_temporal_entropy,
        "growth_amplitude": growth_amplitude,
    }


def _detect_growth_peak(
    masked_ndvi: np.ndarray,
    valid: np.ndarray,
    enough: np.ndarray,
) -> np.ndarray:
    """Detect pixels with a clear growth→peak→decline pattern.

    A pixel has a growth peak if there exist indices i < j < k (among valid dates)
    such that NDVI[i] < NDVI[j] and NDVI[j] > NDVI[k], with the peak NDVI[j]
    being significantly higher than both flanks.
    """
    t, h, w = masked_ndvi.shape
    has_peak = np.zeros((h, w), dtype=bool)

    if t < 3:
        return has_peak

    # Simplified check: argmax not at first or last valid date,
    # and peak > mean + 0.1
    ndvi_mean = nanmean_safe(masked_ndvi, axis=0, fill_value=np.nan)
    ndvi_max = nanmax_safe(masked_ndvi, axis=0, fill_value=np.nan)
    argmax_t = np.nanargmax(
        np.where(valid, masked_ndvi, -999.0), axis=0
    )  # (H, W)

    # Find first and last valid date per pixel
    valid_int = valid.astype(np.int32)
    first_valid = np.argmax(valid_int, axis=0)  # (H, W)
    last_valid = t - 1 - np.argmax(valid_int[::-1], axis=0)  # (H, W)

    # Peak is not at first or last valid date, and peak is substantial
    peak_not_edge = (argmax_t > first_valid) & (argmax_t < last_valid)
    peak_substantial = ndvi_max > (ndvi_mean + 0.10)

    has_peak = enough & peak_not_edge & peak_substantial

    return has_peak


def _compute_temporal_entropy(
    masked_ndvi: np.ndarray,
    valid: np.ndarray,
    enough: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute Shannon entropy of NDVI temporal distribution per pixel.

    Lower entropy means more predictable temporal pattern (typical of croplands).
    Higher entropy means irregular/random pattern (non-field areas).
    """
    t, h, w = masked_ndvi.shape
    entropy = np.zeros((h, w), dtype=np.float32)

    # Bin NDVI values [0, 1] into n_bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Clip NDVI to [0, 1] for binning
    clipped = np.clip(masked_ndvi, 0.0, 1.0)

    for yi in range(h):
        for xi in range(w):
            if not enough[yi, xi]:
                continue
            vals = clipped[:, yi, xi]
            v_mask = valid[:, yi, xi]
            v = vals[v_mask]
            if v.size < 3:
                continue
            hist, _ = np.histogram(v, bins=bin_edges)
            probs = hist.astype(np.float32) / float(v.size)
            probs = probs[probs > 0]
            entropy[yi, xi] = -np.sum(probs * np.log2(probs))

    return entropy


def temporal_coherence_mask(
    ndvi_stack: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_amplitude: float = 0.20,
    max_entropy: float = 2.5,
) -> np.ndarray:
    """Return a boolean mask of pixels likely to be crop fields based on temporal coherence.

    Pixels must have:
    - A clear growth peak pattern, OR
    - Growth amplitude >= min_amplitude AND temporal entropy <= max_entropy.
    """
    metrics = compute_temporal_coherence(ndvi_stack, valid_mask)
    has_peak = metrics["has_growth_peak"]
    amplitude = metrics["growth_amplitude"]
    entropy = metrics["ndvi_temporal_entropy"]

    return (
        has_peak | ((amplitude >= min_amplitude) & (entropy <= max_entropy))
    ).astype(bool)
