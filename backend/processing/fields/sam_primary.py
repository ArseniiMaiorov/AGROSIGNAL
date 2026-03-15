"""Primary SAM2-style raster segmentation with graceful fallbacks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as nd_label
from skimage.feature import peak_local_max

from core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


def build_sam_primary_image(
    edge_prob: np.ndarray,
    maxndvi: np.ndarray,
    ndvistd: np.ndarray,
) -> np.ndarray:
    """Compose a 3-channel uint8 image for SAM-style prompting."""
    if not (edge_prob.shape == maxndvi.shape == ndvistd.shape):
        raise ValueError("edge_prob, maxndvi and ndvistd must share the same shape")

    def _norm(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=np.uint8)
        lo = float(np.nanpercentile(arr[finite], 2))
        hi = float(np.nanpercentile(arr[finite], 98))
        out = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return (out * 255).astype(np.uint8)

    return np.stack([_norm(edge_prob), _norm(maxndvi), _norm(ndvistd)], axis=-1)


@dataclass
class SAM2FieldSegmentor:
    """Thin wrapper for optional SAM2 inference with a heuristic fallback."""

    checkpoint_path: str
    device: str = "cpu"

    def _fallback_segment(
        self,
        edge_prob: np.ndarray,
        maxndvi: np.ndarray,
        ndvistd: np.ndarray,
        cfg,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        seed_mask = candidate_mask
        if seed_mask is None:
            seed_mask = (
                (maxndvi > float(getattr(cfg, "PHENO_FIELD_MAX_NDVI_MIN", 0.45)))
                & (ndvistd > float(getattr(cfg, "PHENO_FIELD_NDVI_STD_MIN", 0.15)))
            )
        # suppress very strong edges to avoid trivial bridging
        seed_mask = seed_mask & (edge_prob < max(0.3, float(np.nanpercentile(edge_prob, 75))))
        labeled, n_labels = nd_label(seed_mask.astype(bool))
        masks = [(labeled == idx) for idx in range(1, n_labels + 1)]
        scores = np.ones(n_labels, dtype=np.float32)
        return masks, scores

    def segment_fields(
        self,
        edge_prob: np.ndarray,
        maxndvi: np.ndarray,
        ndvistd: np.ndarray,
        transform,
        cfg,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Return instance masks and confidence scores."""
        del transform  # kept for the planned production API

        image_rgb = build_sam_primary_image(edge_prob, maxndvi, ndvistd)
        try:  # pragma: no cover - optional heavy dependency
            from segment_anything_2 import Sam2Predictor, sam2_model_registry
        except Exception:
            return self._fallback_segment(edge_prob, maxndvi, ndvistd, cfg, candidate_mask)

        if torch is None:
            return self._fallback_segment(edge_prob, maxndvi, ndvistd, cfg, candidate_mask)

        try:  # pragma: no cover
            model = sam2_model_registry["vit_h"](checkpoint=self.checkpoint_path)
            model.to(self.device)
            predictor = Sam2Predictor(model)
            predictor.set_image(image_rgb)
            peaks = peak_local_max(
                edge_prob,
                min_distance=int(getattr(cfg, "SAM_POINT_SPACING", 20)),
                threshold_abs=0.3,
            )
            if peaks.size == 0:
                return self._fallback_segment(edge_prob, maxndvi, ndvistd, cfg, candidate_mask)
            point_coords = peaks[:, ::-1]
            point_labels = np.ones(len(point_coords), dtype=int)
            masks, scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            valid = scores > float(getattr(cfg, "SAM_PRED_IOU_THRESHOLD", 0.85))
            masks = np.asarray(masks)[valid]
            scores = np.asarray(scores, dtype=np.float32)[valid]
            if masks.size == 0:
                return self._fallback_segment(edge_prob, maxndvi, ndvistd, cfg, candidate_mask)
            return [np.asarray(mask, dtype=bool) for mask in masks], scores
        except Exception as exc:  # pragma: no cover
            logger.warning("sam2_primary_fallback", error=str(exc))
            return self._fallback_segment(edge_prob, maxndvi, ndvistd, cfg, candidate_mask)


def masks_to_label_raster(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Convert instance masks into a dense label raster."""
    labels = np.zeros(shape, dtype=np.int32)
    for idx, mask in enumerate(masks, start=1):
        if mask.shape != shape:
            raise ValueError("all masks must match the requested output shape")
        labels[mask.astype(bool)] = idx
    return labels
