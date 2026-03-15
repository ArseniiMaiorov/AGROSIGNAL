"""Multi-temporal edge composite (Watkins approach)."""
import numpy as np
from skimage.feature import canny
from skimage.filters import scharr


def normalize_band(band_2d: np.ndarray, valid_mask_2d: np.ndarray) -> np.ndarray:
    """Normalize band to [0, 1] using min-max of valid pixels.

    Args:
        band_2d: (H, W) reflectance values.
        valid_mask_2d: (H, W) bool mask.

    Returns:
        (H, W) normalized values, 0 where invalid.
    """
    valid_vals = band_2d[valid_mask_2d]
    if len(valid_vals) == 0:
        return np.zeros_like(band_2d)

    vmin = np.nanmin(valid_vals)
    vmax = np.nanmax(valid_vals)
    rng = vmax - vmin
    if rng < 1e-10:
        return np.zeros_like(band_2d)

    normalized = (band_2d - vmin) / rng
    normalized = np.clip(normalized, 0, 1)
    normalized[~valid_mask_2d] = 0
    return normalized


def compute_canny_edges(image_2d: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Compute Canny edge map.

    Args:
        image_2d: (H, W) normalized float image.
        sigma: Gaussian smoothing sigma.

    Returns:
        (H, W) binary edge map (float 0/1).
    """
    edges = canny(image_2d, sigma=sigma)
    return edges.astype(np.float32)


def compute_scharr_edges(image_2d: np.ndarray) -> np.ndarray:
    """Compute Scharr gradient magnitude normalized to [0, 1].

    Args:
        image_2d: (H, W) normalized float image.

    Returns:
        (H, W) gradient magnitude in [0, 1].
    """
    grad = scharr(image_2d)
    gmax = grad.max()
    if gmax > 1e-10:
        grad = grad / gmax
    return grad.astype(np.float32)


def build_multitemporal_edge_composite(
    bands_stack: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    alpha: float = 0.7,
    canny_sigma: float = 1.2,
    coverage_threshold: float = 0.30,
    binary_threshold: float = 0.12,  # deprecated, kept for API compatibility
    closing_radius: int = 2,  # deprecated, kept for API compatibility
    soft_clip_percentile: float | None = 95.0,
) -> np.ndarray:
    """Build multi-temporal edge composite.

    Args:
        bands_stack: dict with keys B2, B3, B4, B8 and optional ndvi.
                     Each value is (T, H, W).
        valid_mask: (T, H, W) bool mask.
        alpha: weight for Canny vs Scharr (alpha*canny + (1-alpha)*scharr).
        canny_sigma: sigma for Canny edge detector.
        coverage_threshold: minimum valid-pixel fraction for a date to contribute.

    Returns:
        (H, W) edge composite.
    """
    _ = binary_threshold, closing_radius
    t_count = valid_mask.shape[0]
    h, w = valid_mask.shape[1], valid_mask.shape[2]

    valid_pcts = valid_mask.reshape(t_count, -1).mean(axis=1)
    eligible = valid_pcts >= coverage_threshold
    if not np.any(eligible):
        eligible = valid_pcts > 0

    total_pct = valid_pcts[eligible].sum()
    if total_pct < 1e-10:
        return np.zeros((h, w), dtype=np.float32)
    weights = np.zeros_like(valid_pcts, dtype=np.float64)
    weights[eligible] = valid_pcts[eligible] / total_pct

    composite = np.zeros((h, w), dtype=np.float64)

    band_names = [k for k in ["B2", "B3", "B4", "B8", "ndvi"] if k in bands_stack]

    for t in range(t_count):
        if not eligible[t]:
            continue
        for band_name in band_names:
            band_2d = bands_stack[band_name][t]
            mask_2d = valid_mask[t]

            normed = normalize_band(band_2d, mask_2d)
            canny_e = compute_canny_edges(normed, sigma=canny_sigma)
            scharr_e = compute_scharr_edges(normed)

            edge = alpha * canny_e + (1 - alpha) * scharr_e
            composite += weights[t] * edge

    cmax = composite.max()
    if cmax > 1e-10:
        composite = composite / cmax

    # Preserve a continuous edge gradient; avoid hard binarization that destroys
    # weak-but-informative boundary signal.
    if soft_clip_percentile is not None:
        pct = float(soft_clip_percentile)
        if 0.0 < pct < 100.0:
            positive = composite[composite > 0.0]
            if positive.size > 0:
                clip_hi = float(np.percentile(positive, pct))
                if clip_hi > 1e-10:
                    composite = np.clip(composite, 0.0, clip_hi) / clip_hi

    return np.clip(composite, 0.0, 1.0).astype(np.float32)


def compute_edge_stats(
    edge_composite: np.ndarray,
    bins: int = 10,
) -> dict[str, float | list[float] | list[int]]:
    """Summarize edge composite intensity for logging and diagnostics."""
    values = np.asarray(edge_composite, dtype=np.float32)
    finite = values[np.isfinite(values)]
    hist, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
    if finite.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "histogram": hist.astype(int).tolist(),
            "bin_edges": edges.astype(float).tolist(),
        }
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "histogram": hist.astype(int).tolist(),
        "bin_edges": edges.astype(float).tolist(),
    }
