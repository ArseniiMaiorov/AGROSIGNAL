"""Optional U-Net style edge inference with a heuristic fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.filters import scharr, threshold_otsu

from core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional heavy dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


def _normalize(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(arr[finite], 2))
    hi = float(np.nanpercentile(arr[finite], 98))
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


@dataclass
class UNetEdgeDetector:
    """Lightweight wrapper around an optional serialized edge model."""

    checkpoint_path: str | None = None
    device: str = "cpu"
    torch_model: object | None = None
    heuristic_only: bool = True

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "UNetEdgeDetector":
        """Load a TorchScript/PyTorch model when available, else use heuristic mode."""
        path = Path(checkpoint_path)
        detector = cls(
            checkpoint_path=str(path),
            device=device,
            torch_model=None,
            heuristic_only=True,
        )
        if torch is None:
            logger.info("unet_edge_dependency_missing", checkpoint=str(path))
            return detector
        if not path.exists():
            logger.info("unet_edge_model_missing", checkpoint=str(path))
            return detector

        try:  # pragma: no cover - exercised only when torch + weights exist
            model = torch.jit.load(str(path), map_location=device)
            model.eval()
            detector.torch_model = model
            detector.heuristic_only = False
        except Exception as exc:  # pragma: no cover
            logger.info("unet_edge_load_failed", checkpoint=str(path), error=str(exc))
        return detector


def build_unet_edge_input(
    edgecomp: np.ndarray,
    maxndvi: np.ndarray,
    ndvistd: np.ndarray,
    scl_median: np.ndarray,
    *,
    vv_edge: np.ndarray | None = None,
    vhvv_ratio: np.ndarray | None = None,
) -> np.ndarray:
    """Build normalized multi-channel input for U-Net style edge inference."""
    arrays = [edgecomp, maxndvi, ndvistd, scl_median]
    if vv_edge is not None:
        arrays.append(vv_edge)
    if vhvv_ratio is not None:
        arrays.append(vhvv_ratio)

    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        raise ValueError("all U-Net edge inputs must share the same shape")

    channels = [
        _normalize(edgecomp),
        _normalize(maxndvi),
        _normalize(ndvistd),
        _normalize(scl_median.astype(np.float32)),
    ]
    if vv_edge is not None:
        channels.append(_normalize(vv_edge))
    if vhvv_ratio is not None:
        channels.append(_normalize(vhvv_ratio))
    return np.stack(channels, axis=0).astype(np.float32)


def _heuristic_edge_probability(inputs: np.ndarray) -> np.ndarray:
    """Fallback edge probability map when no trained model is present."""
    edgecomp = inputs[0]
    maxndvi = inputs[1]
    ndvistd = inputs[2]
    scl = inputs[3]

    ndvi_grad = _normalize(scharr(maxndvi))
    std_grad = _normalize(scharr(ndvistd))
    scl_penalty = 1.0 - np.clip(scl, 0.0, 1.0)

    prob = 0.45 * edgecomp + 0.30 * ndvi_grad + 0.15 * std_grad + 0.10 * scl_penalty
    if inputs.shape[0] >= 5:
        prob += 0.10 * inputs[4]
    if inputs.shape[0] >= 6:
        prob += 0.05 * inputs[5]
    return _normalize(prob)


def predict_edge_map(
    model: UNetEdgeDetector | None,
    edgecomp: np.ndarray,
    maxndvi: np.ndarray,
    ndvistd: np.ndarray,
    scl_median: np.ndarray,
    transform=None,
    *,
    device: str | None = None,
    vv_edge: np.ndarray | None = None,
    vhvv_ratio: np.ndarray | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Predict an edge probability map using model inference or heuristic fallback."""
    del transform  # kept for compatibility with the planned production signature
    inputs = build_unet_edge_input(
        edgecomp,
        maxndvi,
        ndvistd,
        scl_median,
        vv_edge=vv_edge,
        vhvv_ratio=vhvv_ratio,
    )

    if model is not None and not model.heuristic_only and torch is not None and model.torch_model is not None:
        try:  # pragma: no cover - only exercised when torch + weights exist
            with torch.no_grad():
                tensor = torch.from_numpy(inputs).unsqueeze(0).to(device or model.device).float()
                pred = model.torch_model(tensor).detach().cpu().numpy().squeeze()
            prob = _normalize(np.asarray(pred, dtype=np.float32))
        except Exception as exc:  # pragma: no cover
            logger.warning("unet_edge_inference_failed", error=str(exc))
            prob = _heuristic_edge_probability(inputs)
    else:
        prob = _heuristic_edge_probability(inputs)

    edge_vals = prob[np.isfinite(prob)]
    if edge_vals.size == 0:
        return np.zeros_like(prob, dtype=np.float32)
    try:
        auto_thresh = float(threshold_otsu(edge_vals))
    except Exception:
        auto_thresh = float(threshold) if threshold is not None else 0.5
    hard_thresh = float(threshold) if threshold is not None else auto_thresh
    # Keep probabilistic contrast, but suppress very weak responses.
    prob = np.where(prob >= hard_thresh * 0.5, prob, 0.0)
    return prob.astype(np.float32)
