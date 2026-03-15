"""Linear road-like artifact filtering for crop candidate masks."""
from __future__ import annotations

import numpy as np
from skimage.draw import line as draw_line
from skimage.feature import canny
from skimage.morphology import binary_dilation, disk
from skimage.transform import probabilistic_hough_line

try:
    import cv2
except Exception:  # pragma: no cover - fallback path is exercised when cv2 is absent
    cv2 = None


def _draw_segments(shape: tuple[int, int], segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    """Rasterize line segments into a boolean mask."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for (x1, y1), (x2, y2) in segments:
        rr, cc = draw_line(int(y1), int(x1), int(y2), int(x2))
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        mask[rr, cc] = True
    return mask


def build_road_mask(
    ndvi: np.ndarray,
    cfg,
) -> np.ndarray:
    """Return a boolean road mask for linear, low-NDVI artifacts."""
    if ndvi.ndim != 2:
        raise ValueError(f"ndvi must be 2D, got shape={ndvi.shape}")

    low_ndvi = np.isfinite(ndvi) & (ndvi < cfg.POST_ROAD_MAX_NDVI)
    if not np.any(low_ndvi):
        return np.zeros_like(low_ndvi, dtype=bool)

    edges = canny(
        low_ndvi.astype(np.float32),
        sigma=cfg.EDGE_CANNY_SIGMA,
    )
    if not np.any(edges):
        return np.zeros_like(low_ndvi, dtype=bool)

    radius = max(0, int(cfg.POST_ROAD_BUFFER_PX))

    if cv2 is not None:
        edges_u8 = edges.astype(np.uint8) * 255
        lines = cv2.HoughLinesP(
            edges_u8,
            rho=1,
            theta=np.pi / 180,
            threshold=int(cfg.POST_ROAD_HOUGH_THRESHOLD),
            minLineLength=int(cfg.POST_ROAD_HOUGH_MIN_LEN),
            maxLineGap=int(cfg.POST_ROAD_HOUGH_MAX_GAP),
        )

        road_lines = np.zeros_like(edges_u8, dtype=np.uint8)
        if lines is not None:
            for raw_line in lines:
                x1, y1, x2, y2 = map(int, np.asarray(raw_line).reshape(-1)[:4])
                cv2.line(road_lines, (x1, y1), (x2, y2), 255, 1)

        if radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            )
            road_lines = cv2.dilate(road_lines, kernel)

        road_lines_mask = road_lines > 0
    else:
        segments = probabilistic_hough_line(
            edges,
            threshold=int(cfg.POST_ROAD_HOUGH_THRESHOLD),
            line_length=int(cfg.POST_ROAD_HOUGH_MIN_LEN),
            line_gap=int(cfg.POST_ROAD_HOUGH_MAX_GAP),
        )
        road_lines_mask = _draw_segments(low_ndvi.shape, segments)
        if radius > 0:
            road_lines_mask = binary_dilation(road_lines_mask, disk(radius))

    return low_ndvi & road_lines_mask
