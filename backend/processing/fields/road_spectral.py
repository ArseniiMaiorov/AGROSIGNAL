"""Spectral road detection for barrier-aware crop postprocessing."""
from __future__ import annotations

import numpy as np
from skimage.feature import canny

try:  # pragma: no cover - fallback handled below
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def build_spectral_road_mask(
    ndvi: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    cfg,
) -> np.ndarray:
    """Detect narrow roads from spectral conditions plus a linearity prior."""
    if ndvi.shape != nir.shape or ndvi.shape != swir.shape:
        raise ValueError("ndvi, nir, and swir must share the same shape")

    finite = np.isfinite(ndvi) & np.isfinite(nir) & np.isfinite(swir)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndbi = (swir - nir) / (swir + nir + 1e-6)
    ndbi = np.where(np.isfinite(ndbi), ndbi, -1.0)

    road_candidate = (
        finite
        & (ndvi < float(cfg.POST_ROAD_MAX_NDVI))
        & (nir < float(cfg.POST_ROAD_NIR_MAX))
        & (ndbi > float(cfg.POST_ROAD_NDBI_MIN))
    )

    if not np.any(road_candidate):
        return np.zeros_like(road_candidate, dtype=bool)

    edges = canny(road_candidate.astype(np.float32), sigma=1.2).astype(np.uint8) * 255
    if cv2 is None:
        try:
            from skimage.transform import probabilistic_hough_line
            from scipy.ndimage import binary_dilation
            lines = probabilistic_hough_line(
                edges > 0,
                threshold=10,
                line_length=max(5, int(getattr(cfg, "POST_ROAD_HOUGH_MIN_LEN", 15))),
                line_gap=max(1, int(getattr(cfg, "POST_ROAD_HOUGH_MAX_GAP", 5))),
            )
            if not lines:
                return np.zeros_like(road_candidate, dtype=bool)
            line_mask = np.zeros_like(road_candidate, dtype=np.uint8)
            for (x1, y1), (x2, y2) in lines:
                from skimage.draw import line as sk_line
                rr, cc = sk_line(int(y1), int(x1), int(y2), int(x2))
                valid = (rr >= 0) & (rr < line_mask.shape[0]) & (cc >= 0) & (cc < line_mask.shape[1])
                line_mask[rr[valid], cc[valid]] = 1
            buffer_px = max(1, int(getattr(cfg, "POST_ROAD_BUFFER_PX", 2)))
            line_mask = binary_dilation(line_mask > 0, iterations=buffer_px)
            return road_candidate & line_mask
        except ImportError:
            return np.zeros_like(road_candidate, dtype=bool)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(cfg.POST_ROAD_HOUGH_THRESHOLD),
        minLineLength=int(cfg.POST_ROAD_HOUGH_MIN_LEN),
        maxLineGap=int(cfg.POST_ROAD_HOUGH_MAX_GAP),
    )

    line_mask = np.zeros_like(edges, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(-1).tolist()
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)

    buffer_px = max(0, int(cfg.POST_ROAD_BUFFER_PX))
    if buffer_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * buffer_px + 1, 2 * buffer_px + 1),
        )
        line_mask = cv2.dilate(line_mask, kernel)

    return road_candidate & (line_mask > 0)
