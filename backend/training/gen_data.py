#!/usr/bin/env python3
"""Build weakly-supervised dataset and train BoundaryUNet v4 model."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.config import get_settings
from processing.fields.ml_inference import (
    BoundaryUNet,
    FEATURE_CHANNELS,
    get_feature_channels,
    resolve_feature_profile,
)
from utils.training import compute_tile_quality_weight

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Sampler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for training. Use backend/venv/bin/python.") from exc


@dataclass
class PatchSample:
    tile_id: str
    npz_path: Path
    label_path: Path
    y0: int
    y1: int
    x0: int
    x1: int
    quality_weight: float = 1.0
    edge_valid_fraction: float = 1.0
    extent_cov: float = 0.0
    boundary_cov: float = 0.0
    edge_strength: float = 0.0


class _TileDataCache:
    """Small LRU cache for tile tensors to keep RAM bounded on 16 GB machines."""

    def __init__(
        self,
        *,
        feature_channels: tuple[str, ...],
        max_items: int,
    ) -> None:
        self.feature_channels = feature_channels
        self.max_items = max(1, int(max_items))
        self._cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def get(
        self,
        *,
        npz_path: Path,
        label_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (str(npz_path), str(label_path))
        cached = self._cache.pop(key, None)
        if cached is None:
            x, edge_valid_mask = _build_feature_stack(npz_path, feature_channels=self.feature_channels)
            extent, boundary, distance = _build_targets(label_path)
            cached = (x, extent, boundary, distance, edge_valid_mask)
        self._cache[key] = cached
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return cached


def _crop_patch(
    arr: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    patch_size: int,
    pad_mode: str,
) -> np.ndarray:
    """Crop patch and pad bottom/right when source tile is smaller than patch size."""
    cropped = arr[..., y0:y1, x0:x1] if arr.ndim == 3 else arr[y0:y1, x0:x1]
    pad_h = max(0, patch_size - cropped.shape[-2])
    pad_w = max(0, patch_size - cropped.shape[-1])
    if pad_h or pad_w:
        if arr.ndim == 3:
            pad_cfg = ((0, 0), (0, pad_h), (0, pad_w))
        else:
            pad_cfg = ((0, pad_h), (0, pad_w))
        if pad_mode == "edge":
            cropped = np.pad(cropped, pad_cfg, mode="edge")
        else:
            cropped = np.pad(cropped, pad_cfg, mode="constant")
    return np.asarray(cropped, dtype=np.float32).copy()


def _load_patch_arrays(
    sample: PatchSample,
    *,
    patch_size: int,
    tile_cache: _TileDataCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, extent, boundary, distance, _edge_valid_mask = tile_cache.get(
        npz_path=sample.npz_path,
        label_path=sample.label_path,
    )
    return (
        _crop_patch(x, y0=sample.y0, y1=sample.y1, x0=sample.x0, x1=sample.x1, patch_size=patch_size, pad_mode="edge"),
        _crop_patch(extent, y0=sample.y0, y1=sample.y1, x0=sample.x0, x1=sample.x1, patch_size=patch_size, pad_mode="constant"),
        _crop_patch(boundary, y0=sample.y0, y1=sample.y1, x0=sample.x0, x1=sample.x1, patch_size=patch_size, pad_mode="constant"),
        _crop_patch(distance, y0=sample.y0, y1=sample.y1, x0=sample.x0, x1=sample.x1, patch_size=patch_size, pad_mode="constant"),
    )


def _to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _fallback_green_median(red_median: np.ndarray, blue_median: np.ndarray) -> np.ndarray:
    """Approximate missing green reflectance from neighboring visible bands."""

    return np.clip((red_median.astype(np.float32) + blue_median.astype(np.float32)) * 0.5, 0.0, 1.0)


def _resolve_norm_stat_samples(
    train_samples_raw: list[PatchSample],
    train_samples_rebalanced: list[PatchSample],
) -> list[PatchSample]:
    """Keep normalization tied to the raw train split instead of the rebalanced loader mix."""

    return train_samples_raw or train_samples_rebalanced


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _write_progress_file(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_get(z: np.lib.npyio.NpzFile, key: str, fallback: np.ndarray) -> np.ndarray:
    return z[key].astype(np.float32) if key in z else fallback.astype(np.float32)


def _infer_tile_shape(z: np.lib.npyio.NpzFile) -> tuple[int, int]:
    """Best-effort tile shape inference from NPZ arrays."""
    for key in ("edgecomposite", "maxndvi", "meanndvi", "ndvistd"):
        if key in z:
            arr = np.asarray(z[key])
            if arr.ndim >= 2:
                return int(arr.shape[-2]), int(arr.shape[-1])
    for key in z.files:
        arr = np.asarray(z[key])
        if arr.ndim >= 2:
            return int(arr.shape[-2]), int(arr.shape[-1])
    return 256, 256


def _build_feature_stack(
    npz_path: Path,
    *,
    feature_channels: tuple[str, ...] = FEATURE_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    h, w = _infer_tile_shape(z)
    edge_raw = _safe_get(z, "edgecomposite", np.zeros((h, w), dtype=np.float32))
    edge_valid_mask = np.isfinite(edge_raw).astype(np.float32)
    edge = np.nan_to_num(edge_raw, nan=0.0, posinf=0.0, neginf=0.0)
    max_ndvi = _safe_get(z, "maxndvi", np.zeros_like(edge))
    mean_ndvi = _safe_get(z, "meanndvi", np.zeros_like(edge))
    ndvi_std = _safe_get(z, "ndvistd", np.zeros_like(edge))

    # Real spectral composites from NPZ (fallback to synthetic for old tiles)
    # Fallback to zeros for old tiles missing real spectral composites.
    # Previous synthetic formulas were physically meaningless.
    ndwi_mean = _safe_get(
        z, "ndwi_mean",
        np.zeros_like(max_ndvi),
    )
    bsi_mean = _safe_get(
        z, "bsi_mean",
        np.zeros_like(max_ndvi),
    )

    # SCL valid fraction — real per-pixel fraction if available
    if "scl_valid_fraction" in z:
        scl_valid_fraction = z["scl_valid_fraction"].astype(np.float32)
    else:
        scl_median = z["scl_median"].astype(np.float32) if "scl_median" in z else np.full_like(edge, 4.0)
        invalid_scl = np.isin(scl_median, np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.float32))
        scl_valid_fraction = (~invalid_scl).astype(np.float32)

    # Real band medians: rgb_r=NIR(B8), rgb_g=Red(B4), rgb_b=Blue(B2)
    rgb_r = _safe_get(z, "nir_median", np.clip(max_ndvi, 0.0, 1.0))
    rgb_g = _safe_get(z, "red_median", np.clip(mean_ndvi, 0.0, 1.0))
    rgb_b = _safe_get(z, "blue_median", np.clip(edge, 0.0, 1.0))

    s1_vv = np.zeros_like(edge, dtype=np.float32)
    s1_vh = np.zeros_like(edge, dtype=np.float32)

    # New channels: NDVI temporal entropy
    ndvi_entropy = _safe_get(z, "ndvi_entropy", np.zeros_like(edge))

    # Additional spectral composites from NPZ
    mndwi_max = _safe_get(z, "mndwi_max", np.zeros_like(edge))
    ndmi_mean = _safe_get(
        z, "ndmi_mean",
        np.clip((rgb_r - _safe_get(z, "swir_median", np.zeros_like(edge))) /
                np.maximum(rgb_r + _safe_get(z, "swir_median", np.zeros_like(edge)), 1e-6), -1.0, 1.0),
    )
    ndwi_median = _safe_get(z, "ndwi_median", ndwi_mean.copy())
    green_median = _safe_get(z, "green_median", _fallback_green_median(rgb_g, rgb_b))
    swir_median = _safe_get(z, "swir_median", np.zeros_like(edge))

    channel_map: dict[str, np.ndarray] = {
        "edge_composite": edge,
        "max_ndvi": max_ndvi,
        "mean_ndvi": mean_ndvi,
        "ndvi_std": ndvi_std,
        "ndwi_mean": ndwi_mean,
        "bsi_mean": bsi_mean,
        "scl_valid_fraction": scl_valid_fraction,
        "rgb_r": rgb_r,
        "rgb_g": rgb_g,
        "rgb_b": rgb_b,
        "s1_vv_mean": s1_vv,
        "s1_vh_mean": s1_vh,
        "ndvi_entropy": ndvi_entropy,
        "mndwi_max": mndwi_max,
        "ndmi_mean": ndmi_mean,
        "ndwi_median": ndwi_median,
        "green_median": green_median,
        "swir_median": swir_median,
    }
    stack = np.stack([channel_map[name] for name in feature_channels], axis=0).astype(np.float32)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    return stack, edge_valid_mask


def _build_targets(label_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.uint8)

    extent = label > 0
    if extent.any():
        inner_boundary = extent & ~binary_erosion(extent, iterations=1)
        outer_boundary = binary_dilation(extent, iterations=1) & ~extent
        boundary = binary_dilation(inner_boundary | outer_boundary, iterations=1)

        distance_inside = distance_transform_edt(extent).astype(np.float32)
        if float(distance_inside.max()) > 1e-6:
            distance_inside = distance_inside / float(distance_inside.max())
        distance_outside = distance_transform_edt(~extent).astype(np.float32)
        outer_soft_band = np.clip(1.0 - (distance_outside / 3.0), 0.0, 1.0) * (~extent)
        distance = np.maximum(distance_inside, outer_soft_band.astype(np.float32) * 0.35)
    else:
        boundary = np.zeros_like(extent, dtype=bool)
        distance = np.zeros_like(extent, dtype=np.float32)

    return (
        extent.astype(np.float32),
        boundary.astype(np.float32),
        distance.astype(np.float32),
    )


def _iter_patch_windows(h: int, w: int, patch_size: int, stride: int) -> Iterable[tuple[int, int, int, int]]:
    ys = list(range(0, max(1, h - patch_size + 1), stride))
    xs = list(range(0, max(1, w - patch_size + 1), stride))
    if ys[-1] != max(0, h - patch_size):
        ys.append(max(0, h - patch_size))
    if xs[-1] != max(0, w - patch_size):
        xs.append(max(0, w - patch_size))

    for y0 in ys:
        y1 = min(h, y0 + patch_size)
        for x0 in xs:
            x1 = min(w, x0 + patch_size)
            yield y0, y1, x0, x1


def _extract_patches(
    tile_id: str,
    x: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    label_path: Path,
    npz_path: Path,
    edge_valid_mask: np.ndarray | None,
    *,
    patch_size: int,
    stride: int,
) -> list[PatchSample]:
    _, h, w = x.shape
    scan_h = max(h, patch_size)
    scan_w = max(w, patch_size)
    patches: list[PatchSample] = []
    for y0, y1, x0, x1 in _iter_patch_windows(scan_h, scan_w, patch_size, stride):
        x_patch = _crop_patch(
            x,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            patch_size=patch_size,
            pad_mode="edge",
        )
        ex_patch = _crop_patch(
            extent,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            patch_size=patch_size,
            pad_mode="constant",
        )
        bd_patch = _crop_patch(
            boundary,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            patch_size=patch_size,
            pad_mode="constant",
        )
        ev_patch = (
            _crop_patch(
                edge_valid_mask,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                patch_size=patch_size,
                pad_mode="constant",
            )
            if edge_valid_mask is not None
            else np.ones((patch_size, patch_size), dtype=np.float32)
        )
        patches.append(
            PatchSample(
                tile_id=tile_id,
                npz_path=npz_path,
                label_path=label_path,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                edge_valid_fraction=float(np.mean(ev_patch)),
                extent_cov=float(np.mean(ex_patch > 0.5)),
                boundary_cov=float(np.mean(bd_patch > 0.5)),
                edge_strength=float(np.mean(np.abs(x_patch[0]))),
            )
        )
    return patches


def _region_key(tile_id: str) -> str:
    head, sep, tail = tile_id.rpartition("_")
    if sep and tail.isdigit():
        return head
    return tile_id


def _split_tile_ids(tile_ids: list[str], seed: int) -> dict[str, set[str]]:
    """Spatial split: all tiles from one region stay in the same subset."""
    ids = sorted(set(tile_ids))
    if not ids:
        raise ValueError("tile_ids must not be empty")

    region_to_ids: dict[str, list[str]] = {}
    for tile_id in ids:
        region_to_ids.setdefault(_region_key(tile_id), []).append(tile_id)

    regions = sorted(region_to_ids.keys())
    rnd = random.Random(seed)
    rnd.shuffle(regions)

    total_tiles = len(ids)
    target_train = max(1, int(round(0.7 * total_tiles)))
    target_val = max(1, int(round(0.15 * total_tiles))) if total_tiles >= 3 else 1
    if target_train + target_val >= total_tiles:
        target_val = max(1, total_tiles - target_train - 1)

    train_regions: list[str] = []
    val_regions: list[str] = []
    test_regions: list[str] = []
    train_count = 0
    val_count = 0
    for region in regions:
        region_count = len(region_to_ids[region])
        if train_count < target_train:
            train_regions.append(region)
            train_count += region_count
        elif val_count < target_val:
            val_regions.append(region)
            val_count += region_count
        else:
            test_regions.append(region)

    if not val_regions and train_regions:
        val_regions.append(train_regions.pop())
    if not test_regions and train_regions:
        test_regions.append(train_regions.pop())
    if not test_regions and val_regions:
        test_regions.append(val_regions[-1])

    train_ids = {tile for region in train_regions for tile in region_to_ids[region]}
    val_ids = {tile for region in val_regions for tile in region_to_ids[region]}
    test_ids = {tile for region in test_regions for tile in region_to_ids[region]}

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise RuntimeError(f"Spatial split produced overlapping ids: {sorted(overlap)}")
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            "Spatial split requires non-empty train/val/test sets. "
            f"Got train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _rebalance_train_samples(
    samples: list[PatchSample],
    *,
    seed: int,
    max_neg_pos_ratio: float = 2.0,
) -> list[PatchSample]:
    """Reduce easy negatives while keeping difficult negatives."""
    if not samples:
        return samples

    positives: list[PatchSample] = []
    difficult_negatives: list[PatchSample] = []
    easy_negatives: list[PatchSample] = []
    for sample in samples:
        if float(sample.extent_cov) >= 0.01:
            positives.append(sample)
        elif float(sample.boundary_cov) >= 0.003 or float(sample.edge_strength) >= 0.10:
            difficult_negatives.append(sample)
        else:
            easy_negatives.append(sample)

    if not positives:
        return samples

    max_negatives = int(max_neg_pos_ratio * len(positives))
    kept_negatives = list(difficult_negatives)
    remaining_budget = max(0, max_negatives - len(kept_negatives))
    if remaining_budget > 0 and easy_negatives:
        rnd = random.Random(seed)
        rnd.shuffle(easy_negatives)
        kept_negatives.extend(easy_negatives[:remaining_budget])

    balanced = positives + kept_negatives
    rnd = random.Random(seed)
    rnd.shuffle(balanced)
    return balanced


def _compute_norm_stats(
    train_samples: list[PatchSample],
    feature_channels: tuple[str, ...],
    *,
    patch_size: int,
    tile_cache_size: int = 1,
    min_channel_std: float = 1e-3,
) -> dict[str, list[float]]:
    if not train_samples:
        raise ValueError("empty train split")
    tile_cache = _TileDataCache(feature_channels=feature_channels, max_items=tile_cache_size)
    sum_x = np.zeros(len(feature_channels), dtype=np.float64)
    sum_x2 = np.zeros(len(feature_channels), dtype=np.float64)
    total_pixels = 0

    for sample in sorted(train_samples, key=lambda s: (s.tile_id, s.y0, s.x0)):
        x, _ext, _bd, _dist = _load_patch_arrays(
            sample,
            patch_size=patch_size,
            tile_cache=tile_cache,
        )
        flat = x.reshape(x.shape[0], -1).astype(np.float64)
        sum_x += flat.sum(axis=1)
        sum_x2 += np.square(flat).sum(axis=1)
        total_pixels += int(flat.shape[1])

    if total_pixels <= 0:
        raise RuntimeError("Normalization stats require at least one pixel")
    mean = (sum_x / float(total_pixels)).astype(np.float32)
    var = np.clip((sum_x2 / float(total_pixels)) - np.square(mean.astype(np.float64)), 0.0, None)
    raw_std = np.sqrt(var).astype(np.float32)
    std = np.clip(raw_std, 1e-6, None)

    dead_channels = [
        (str(ch), float(mean[idx]), float(raw_std[idx]))
        for idx, ch in enumerate(feature_channels)
        if float(raw_std[idx]) < float(min_channel_std)
    ]
    if dead_channels:
        formatted = ", ".join(
            f"{name}(mean={m:.6f}, std={s:.6f})" for name, m, s in dead_channels
        )
        raise RuntimeError(
            "Dead channels detected in normalization stats: "
            f"{formatted}. Remove channels or provide real signal."
        )

    return {
        "channels": list(feature_channels),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _clip_feature_ranges(x: np.ndarray, feature_channels: tuple[str, ...]) -> np.ndarray:
    """Clamp channels to physically plausible ranges after augmentation."""
    ranges: dict[str, tuple[float, float]] = {
        "edge_composite": (0.0, 1.0),
        "max_ndvi": (0.0, 1.0),
        "mean_ndvi": (0.0, 1.0),
        "ndvi_std": (0.0, 1.0),
        "ndwi_mean": (-1.0, 1.0),
        "bsi_mean": (-1.0, 1.0),
        "scl_valid_fraction": (0.0, 1.0),
        "rgb_r": (0.0, 1.0),
        "rgb_g": (0.0, 1.0),
        "rgb_b": (0.0, 1.0),
        "s1_vv_mean": (-1.0, 1.0),
        "s1_vh_mean": (-1.0, 1.0),
        "ndvi_entropy": (0.0, 5.0),
        "mndwi_max": (-1.0, 1.0),
        "ndmi_mean": (-1.0, 1.0),
        "ndwi_median": (-1.0, 1.0),
        "green_median": (0.0, 1.0),
        "swir_median": (0.0, 1.0),
    }
    for idx, name in enumerate(feature_channels):
        lo, hi = ranges.get(str(name), (-1e3, 1e3))
        x[idx] = np.clip(x[idx], lo, hi)
    return x


def _apply_photometric_aug(x: np.ndarray, feature_channels: tuple[str, ...]) -> np.ndarray:
    """Apply mild radiometric jitter to spectral channels."""
    photo_names = {
        "max_ndvi",
        "mean_ndvi",
        "ndwi_mean",
        "bsi_mean",
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "mndwi_max",
        "ndmi_mean",
        "ndwi_median",
        "green_median",
        "swir_median",
    }
    photo_idx = tuple(idx for idx, name in enumerate(feature_channels) if name in photo_names)

    if photo_idx and random.random() < 0.7:
        gain = random.uniform(0.90, 1.10)
        bias = random.uniform(-0.05, 0.05)
        x[list(photo_idx)] = x[list(photo_idx)] * gain + bias

    if photo_idx and random.random() < 0.5:
        contrast = random.uniform(0.85, 1.15)
        mean = np.mean(x[list(photo_idx)], axis=(1, 2), keepdims=True)
        x[list(photo_idx)] = (x[list(photo_idx)] - mean) * contrast + mean

    if photo_idx and random.random() < 0.5:
        noise_std = random.uniform(0.003, 0.02)
        noise = np.random.normal(0.0, noise_std, size=x[list(photo_idx)].shape).astype(np.float32)
        x[list(photo_idx)] = x[list(photo_idx)] + noise

    return _clip_feature_ranges(x, feature_channels)


def _apply_elastic(
    x: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    distance: np.ndarray,
    *,
    alpha: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply smooth elastic deformation to features and targets."""
    h, w = extent.shape
    dx = gaussian_filter(
        (np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0),
        sigma=max(0.5, float(sigma)),
        mode="reflect",
    ) * float(alpha)
    dy = gaussian_filter(
        (np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0),
        sigma=max(0.5, float(sigma)),
        mode="reflect",
    ) * float(alpha)

    grid_y, grid_x = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    coords = np.asarray([grid_y + dy, grid_x + dx], dtype=np.float32)

    warped_x = np.empty_like(x)
    for c in range(x.shape[0]):
        warped_x[c] = map_coordinates(x[c], coords, order=1, mode="reflect")
    warped_extent = map_coordinates(extent.astype(np.float32), coords, order=0, mode="nearest")
    warped_boundary = map_coordinates(boundary.astype(np.float32), coords, order=0, mode="nearest")
    warped_distance = map_coordinates(distance.astype(np.float32), coords, order=1, mode="reflect")

    return (
        warped_x.astype(np.float32),
        (warped_extent > 0.5).astype(np.float32),
        (warped_boundary > 0.5).astype(np.float32),
        np.clip(warped_distance.astype(np.float32), 0.0, 1.0),
    )


def _apply_aug(
    x: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    distance: np.ndarray,
    feature_channels: tuple[str, ...],
    *,
    elastic_prob: float,
    elastic_alpha: float,
    elastic_sigma: float,
):
    x = _apply_photometric_aug(x, feature_channels)

    if random.random() < 0.5:
        x = x[:, :, ::-1]
        extent = extent[:, ::-1]
        boundary = boundary[:, ::-1]
        distance = distance[:, ::-1]
    if random.random() < 0.5:
        x = x[:, ::-1, :]
        extent = extent[::-1, :]
        boundary = boundary[::-1, :]
        distance = distance[::-1, :]
    if random.random() < 0.5:
        k = random.randint(0, 3)
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
        extent = np.rot90(extent, k=k).copy()
        boundary = np.rot90(boundary, k=k).copy()
        distance = np.rot90(distance, k=k).copy()
    if float(elastic_prob) > 0.0 and random.random() < float(elastic_prob):
        x, extent, boundary, distance = _apply_elastic(
            x,
            extent,
            boundary,
            distance,
            alpha=float(elastic_alpha),
            sigma=float(elastic_sigma),
        )
    return _clip_feature_ranges(x, feature_channels), extent, boundary, distance


class PatchDataset(Dataset):
    def __init__(
        self,
        samples: list[PatchSample],
        norm_stats: dict[str, list[float]],
        *,
        train: bool,
        feature_channels: tuple[str, ...],
        patch_size: int = 256,
        tile_cache_size: int = 2,
        elastic_prob: float = 0.0,
        elastic_alpha: float = 80.0,
        elastic_sigma: float = 10.0,
    ) -> None:
        self.samples = samples
        self.train = train
        self.feature_channels = feature_channels
        self.patch_size = patch_size
        self.tile_cache = _TileDataCache(
            feature_channels=feature_channels,
            max_items=tile_cache_size,
        )
        self.mean = np.asarray(norm_stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(norm_stats["std"], dtype=np.float32)[:, None, None]
        self.elastic_prob = float(max(0.0, min(1.0, elastic_prob)))
        self.elastic_alpha = float(max(1.0, elastic_alpha))
        self.elastic_sigma = float(max(0.5, elastic_sigma))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x, extent, boundary, distance = _load_patch_arrays(
            sample,
            patch_size=self.patch_size,
            tile_cache=self.tile_cache,
        )

        if self.train:
            x, extent, boundary, distance = _apply_aug(
                x,
                extent,
                boundary,
                distance,
                self.feature_channels,
                elastic_prob=self.elastic_prob,
                elastic_alpha=self.elastic_alpha,
                elastic_sigma=self.elastic_sigma,
            )

        x = (x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        target = {
            "extent": extent[None, ...].astype(np.float32),
            "boundary": boundary[None, ...].astype(np.float32),
            "distance": distance[None, ...].astype(np.float32),
        }
        weight = np.float32(sample.quality_weight)
        edge_valid_fraction = np.float32(sample.edge_valid_fraction)

        return (
            torch.from_numpy(x),
            {k: torch.from_numpy(v) for k, v in target.items()},
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(edge_valid_fraction, dtype=torch.float32),
        )


class TileGroupedBatchSampler(Sampler[list[int]]):
    """Group training patches by tile to avoid constant tile cache eviction."""

    def __init__(
        self,
        samples: list[PatchSample],
        *,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self.tile_to_indices: dict[str, list[int]] = {}
        for idx, sample in enumerate(samples):
            self.tile_to_indices.setdefault(sample.tile_id, []).append(idx)

    def __iter__(self):
        rnd = random.Random(self.seed + self._epoch)
        tile_ids = list(self.tile_to_indices.keys())
        rnd.shuffle(tile_ids)
        self._epoch += 1
        for tile_id in tile_ids:
            indices = list(self.tile_to_indices[tile_id])
            rnd.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.tile_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean(dim=1)


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * target).sum(dim=(2, 3))
        fp = (probs * (1.0 - target)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * target).sum(dim=(2, 3))
        score = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return 1.0 - score.mean(dim=1)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs)
        focal = ((1.0 - pt) ** self.gamma) * bce
        return focal.mean(dim=(1, 2, 3))


class MultiTaskLoss(nn.Module):
    def __init__(self, w_extent: float = 1.0, w_boundary: float = 2.5, w_distance: float = 0.75) -> None:
        super().__init__()
        self.w_extent = w_extent
        self.w_boundary = w_boundary
        self.w_distance = w_distance
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=2.0)
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        *,
        edge_valid_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce_extent = self.bce(preds["extent"], target["extent"]).mean(dim=(1, 2, 3))
        loss_extent = bce_extent + self.dice(preds["extent"], target["extent"])
        loss_boundary = (
            self.focal(preds["boundary"], target["boundary"])
            + self.tversky(preds["boundary"], target["boundary"])
            + 0.5 * self.dice(preds["boundary"], target["boundary"])
        )
        loss_distance = nn.functional.mse_loss(
            preds["distance"],
            target["distance"],
            reduction="none",
        ).mean(dim=(1, 2, 3))
        if edge_valid_fraction is None:
            edge_weight = torch.ones_like(loss_extent)
        else:
            edge_weight = torch.clamp(
                edge_valid_fraction.to(loss_extent.device).float().view(-1),
                0.0,
                1.0,
            )
        return (
            self.w_extent * loss_extent
            + edge_weight * (self.w_boundary * loss_boundary + self.w_distance * loss_distance)
        )


def _boundary_hd95_px(pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> float:
    pred_arr = pred_boundary.detach().cpu().numpy() > 0.5
    gt_arr = gt_boundary.detach().cpu().numpy() > 0.5
    if pred_arr.ndim == 4:
        pred_arr = pred_arr[:, 0, :, :]
        gt_arr = gt_arr[:, 0, :, :]
    elif pred_arr.ndim == 3:
        pred_arr = pred_arr[:, :, :]
        gt_arr = gt_arr[:, :, :]
    else:
        pred_arr = pred_arr[np.newaxis, ...]
        gt_arr = gt_arr[np.newaxis, ...]

    per_sample: list[float] = []
    for pred_mask, gt_mask in zip(pred_arr, gt_arr):
        if not np.any(pred_mask) or not np.any(gt_mask):
            continue
        dist_to_gt = distance_transform_edt(~gt_mask)
        dist_to_pred = distance_transform_edt(~pred_mask)
        samples = np.concatenate([dist_to_gt[pred_mask], dist_to_pred[gt_mask]])
        if samples.size == 0:
            continue
        per_sample.append(float(np.quantile(samples.astype(np.float32), 0.95)))
    if not per_sample:
        return 0.0
    return float(np.mean(per_sample))


def _metrics(preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, float | list[float] | None]:
    with torch.no_grad():
        extent = (torch.sigmoid(preds["extent"]) > 0.5).float()
        boundary = (torch.sigmoid(preds["boundary"]) > 0.5).float()
        gt_extent = target["extent"]
        gt_boundary = target["boundary"]

        tp_iou = (extent * gt_extent).sum().item()
        fp_iou = (extent * (1.0 - gt_extent)).sum().item()
        fn_iou = ((1.0 - extent) * gt_extent).sum().item()
        iou = tp_iou / (tp_iou + fp_iou + fn_iou + 1e-6)

        tp_f1 = (boundary * gt_boundary).sum().item()
        fp_f1 = (boundary * (1.0 - gt_boundary)).sum().item()
        fn_f1 = ((1.0 - boundary) * gt_boundary).sum().item()
        f1 = (2.0 * tp_f1) / (2.0 * tp_f1 + fp_f1 + fn_f1 + 1e-6)
        boundary_iou = tp_f1 / (tp_f1 + fp_f1 + fn_f1 + 1e-6)
        hd95_px = _boundary_hd95_px(boundary, gt_boundary)
        extent_arr = extent.detach().cpu().numpy()
        gt_extent_arr = gt_extent.detach().cpu().numpy()
        valid_area_ratios: list[float] = []
        empty_gt_count = 0.0
        for pred_mask, gt_mask in zip(extent_arr, gt_extent_arr):
            gt_area = float(np.count_nonzero(gt_mask > 0.5))
            if gt_area < 1.0:
                empty_gt_count += 1.0
                continue
            pred_area = float(np.count_nonzero(pred_mask > 0.5))
            valid_area_ratios.append(pred_area / max(gt_area, 1.0))
        area_ratio_pred_gt = float(np.mean(valid_area_ratios)) if valid_area_ratios else None
        area_ratio_valid_count = float(len(valid_area_ratios))
        area_ratio_px_median = float(np.median(valid_area_ratios)) if valid_area_ratios else None
        area_ratio_px_p90 = float(np.quantile(valid_area_ratios, 0.90)) if valid_area_ratios else None

    return {
        "iou_extent": float(iou),
        "iou_extent_px": float(iou),
        "f1_boundary": float(f1),
        "f1_boundary_px": float(f1),
        "boundary_iou": float(boundary_iou),
        "hd95_px": float(hd95_px),
        "area_ratio_px_median": area_ratio_px_median,
        "area_ratio_px_p90": area_ratio_px_p90,
        "area_ratio_pred_gt": area_ratio_pred_gt,
        "area_ratio_valid_count": area_ratio_valid_count,
        "empty_gt_count": float(empty_gt_count),
        "area_ratio_values": [float(v) for v in valid_area_ratios],
        "tp_extent": tp_iou,
        "fp_extent": fp_iou,
        "fn_extent": fn_iou,
        "tp_boundary": tp_f1,
        "fp_boundary": fp_f1,
        "fn_boundary": fn_f1,
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    *,
    progress_interval: int = 0,
    progress_prefix: str = "[eval]",
    progress_file: Path | None = None,
    progress_state: dict[str, object] | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tp_extent = 0.0
    total_fp_extent = 0.0
    total_fn_extent = 0.0
    total_tp_boundary = 0.0
    total_fp_boundary = 0.0
    total_fn_boundary = 0.0
    total_hd95 = 0.0
    total_area_ratio = 0.0
    total_area_ratio_valid = 0.0
    total_empty_gt = 0.0
    all_area_ratio_values: list[float] = []
    n = 0

    total_batches = max(1, len(loader))
    eval_started = time.monotonic()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader, start=1):
            x, target = batch_data[0], batch_data[1]
            edge_valid = batch_data[3] if len(batch_data) > 3 else None
            x = x.to(device)
            target_t = {k: v.to(device) for k, v in target.items()}
            preds = model(x)
            loss = criterion(
                preds,
                target_t,
                edge_valid_fraction=edge_valid.to(device) if edge_valid is not None else None,
            ).mean()
            mets = _metrics(preds, target_t)
            batch = x.shape[0]
            total_loss += float(loss.item()) * batch
            total_tp_extent += mets["tp_extent"]
            total_fp_extent += mets["fp_extent"]
            total_fn_extent += mets["fn_extent"]
            total_tp_boundary += mets["tp_boundary"]
            total_fp_boundary += mets["fp_boundary"]
            total_fn_boundary += mets["fn_boundary"]
            total_hd95 += mets["hd95_px"] * batch
            area_ratio_valid = float(mets.get("area_ratio_valid_count", 0.0))
            area_ratio_alias = mets.get("area_ratio_pred_gt")
            total_area_ratio += float(area_ratio_alias or 0.0) * area_ratio_valid
            total_area_ratio_valid += area_ratio_valid
            total_empty_gt += float(mets.get("empty_gt_count", 0.0))
            all_area_ratio_values.extend(float(v) for v in (mets.get("area_ratio_values") or []))
            n += batch

            if progress_interval > 0 and (batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches):
                elapsed = time.monotonic() - eval_started
                eta = (elapsed / batch_idx) * max(0, total_batches - batch_idx)
                print(
                    f"{progress_prefix} batch {batch_idx}/{total_batches} "
                    f"({(100.0 * batch_idx / total_batches):.1f}%) | "
                    f"elapsed={_format_eta(elapsed)} | eta={_format_eta(eta)}",
                    flush=True,
                )
                if progress_state is not None:
                    payload = dict(progress_state)
                    payload.update(
                        {
                            "stage": "validation",
                            "eval_batch": batch_idx,
                            "eval_batches_total": total_batches,
                            "eval_progress_pct": round(100.0 * batch_idx / total_batches, 2),
                            "eval_elapsed_s": round(elapsed, 2),
                            "eval_eta_s": round(eta, 2),
                        }
                    )
                    _write_progress_file(progress_file, payload)

    n = max(1, n)
    iou_extent = total_tp_extent / (total_tp_extent + total_fp_extent + total_fn_extent + 1e-6)
    f1_boundary = (2.0 * total_tp_boundary) / (2.0 * total_tp_boundary + total_fp_boundary + total_fn_boundary + 1e-6)
    boundary_iou = total_tp_boundary / (total_tp_boundary + total_fp_boundary + total_fn_boundary + 1e-6)
    area_ratio_pred_gt = total_area_ratio / total_area_ratio_valid if total_area_ratio_valid > 0.0 else None
    area_ratio_px_median = float(np.median(all_area_ratio_values)) if all_area_ratio_values else None
    area_ratio_px_p90 = float(np.quantile(all_area_ratio_values, 0.90)) if all_area_ratio_values else None
    return {
        "loss": total_loss / n,
        "iou_extent": iou_extent,
        "iou_extent_px": iou_extent,
        "f1_boundary": f1_boundary,
        "f1_boundary_px": f1_boundary,
        "boundary_iou": boundary_iou,
        "hd95_px": total_hd95 / n,
        "area_ratio_px_median": area_ratio_px_median,
        "area_ratio_px_p90": area_ratio_px_p90,
        "area_ratio_pred_gt": area_ratio_pred_gt,
        "area_ratio_valid_count": total_area_ratio_valid,
        "empty_gt_count": total_empty_gt,
    }


def _load_tile_quality_table(labels_dir: Path) -> dict[str, dict[str, float | bool]]:
    """Load per-tile quality flags/weights from weak labels summary CSV."""
    import csv

    summary_path = labels_dir / "weak_labels_summary.csv"
    table: dict[str, dict[str, float | bool]] = {}
    if not summary_path.exists():
        return table
    try:
        with open(summary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tile_id = row.get("tile_id", "")
                if not tile_id:
                    continue
                used_fallback = _to_bool(row.get("used_fallback"), default=False)
                quality_gate_failed = _to_bool(row.get("quality_gate_failed"), default=False)
                manual_gt_used = _to_bool(row.get("manual_gt_used"), default=False)
                temporal_amp = float(row.get("temporal_amp") or 0.0)
                temporal_entropy = float(row.get("temporal_entropy") or 0.0)
                table[tile_id] = {
                    "tile_weight": compute_tile_quality_weight(
                        used_fallback=used_fallback,
                        quality_gate_failed=quality_gate_failed,
                        manual_gt=manual_gt_used,
                    ),
                    "used_fallback": bool(used_fallback),
                    "quality_gate_failed": bool(quality_gate_failed),
                    "temporal_amp": float(temporal_amp),
                    "temporal_entropy": float(temporal_entropy),
                    "manual_gt_used": bool(manual_gt_used),
                    "gt_quality_tier": str(row.get("gt_quality_tier") or "weak_reference"),
                    "region_band": str(row.get("region_band") or "central"),
                    "region_boundary_profile_target": str(
                        row.get("region_boundary_profile_target") or "balanced"
                    ),
                    "error_mode_tag": str(row.get("error_mode_tag") or "none"),
                    "parcel_shape_class": str(row.get("parcel_shape_class") or "irregular_large"),
                    "adjacency_tag": str(row.get("adjacency_tag") or "none"),
                }
    except Exception:
        pass
    return table


def _read_manual_tile_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    raw = path.read_text(encoding="utf-8")
    ids: set[str] = set()
    for line in raw.splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        token = token.split(",")[0].strip()
        if token and token.lower() != "tile_id":
            ids.add(token)
    return ids


def _resolve_label_path_for_tile(
    tile_id: str,
    labels_dir: Path,
    *,
    manual_labels_dir: Path | None,
    manual_tile_ids: set[str],
) -> tuple[Path | None, bool]:
    """Resolve label path with manual GT taking precedence when available."""
    weak_path = labels_dir / f"{tile_id}_label.tif"
    manual_candidates: list[Path] = []
    if manual_labels_dir is not None:
        if not manual_tile_ids or tile_id in manual_tile_ids:
            manual_candidates = [
                manual_labels_dir / f"{tile_id}_label.tif",
                manual_labels_dir / f"{tile_id}.tif",
            ]
    for candidate in manual_candidates:
        if candidate.exists():
            return candidate, True
    if weak_path.exists():
        return weak_path, False
    return None, False


def _collect_samples(
    tiles_dir: Path,
    labels_dir: Path,
    patch_size: int,
    stride: int,
    *,
    feature_channels: tuple[str, ...],
    manual_labels_dir: Path | None = None,
    manual_tile_ids: set[str] | None = None,
    manual_gt_tier: str = "verified",
) -> tuple[list[PatchSample], dict[str, dict[str, float | bool]]]:
    samples: list[PatchSample] = []
    tile_quality = _load_tile_quality_table(labels_dir)
    manual_tile_ids = set(manual_tile_ids or set())
    manual_tier = str(manual_gt_tier or "verified")

    tile_paths = sorted(tiles_dir.glob("*.npz"))
    if not tile_paths:
        raise FileNotFoundError(f"No .npz tiles found in {tiles_dir}")

    for npz_path in tile_paths:
        tile_id = npz_path.stem
        label_path, manual_gt_used = _resolve_label_path_for_tile(
            tile_id,
            labels_dir,
            manual_labels_dir=manual_labels_dir,
            manual_tile_ids=manual_tile_ids,
        )
        if label_path is None:
            continue

        # Load tile once to build patch metadata; actual training reads patches lazily.
        x, edge_valid_mask = _build_feature_stack(npz_path, feature_channels=feature_channels)
        extent, boundary, distance = _build_targets(label_path)
        quality_entry = dict(tile_quality.get(tile_id, {}))
        if manual_gt_used:
            quality_entry["manual_gt_used"] = True
            quality_entry["gt_quality_tier"] = manual_tier
            quality_entry["tile_weight"] = compute_tile_quality_weight(
                used_fallback=bool(quality_entry.get("used_fallback", False)),
                quality_gate_failed=bool(quality_entry.get("quality_gate_failed", False)),
                manual_gt=True,
            )
            tile_quality[tile_id] = quality_entry
        tile_weight = float(quality_entry.get("tile_weight", 1.0))
        patches = _extract_patches(
            tile_id,
            x,
            extent,
            boundary,
            label_path,
            npz_path,
            edge_valid_mask,
            patch_size=patch_size,
            stride=stride,
        )
        for p in patches:
            p.quality_weight = tile_weight
        samples.extend(patches)

    if not samples:
        raise RuntimeError("No training samples found (check labels and tiles paths)")

    return samples, tile_quality


def _export_onnx(
    model: nn.Module,
    out_path: Path,
    patch_size: int,
    device: str,
    *,
    opset_version: int,
    n_channels: int,
) -> None:
    model.eval()
    model.to(device)

    class ExportWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            return out["extent"], out["boundary"], out["distance"]

    wrapper = ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, int(n_channels), patch_size, patch_size, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["extent", "boundary", "distance"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "extent": {0: "batch", 2: "height", 3: "width"},
            "boundary": {0: "batch", 2: "height", 3: "width"},
            "distance": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=int(opset_version),
        dynamo=False,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    feature_profile = resolve_feature_profile(getattr(args, "ml_feature_profile", "v2_16ch"))
    feature_channels = get_feature_channels(feature_profile)

    metadata = {
        "model_version": str(args.model_version),
        "train_data_version": str(args.train_data_version),
        "feature_stack_version": str(args.feature_stack_version),
        "feature_profile": feature_profile,
        "channels": list(feature_channels),
        "onnx_opset": int(args.onnx_opset),
        "weak_label_source": "weak_labels_summary.csv",
        "extent_threshold_used": float(getattr(args, "ml_extent_bin_threshold", 0.42)),
        "geometry_refine_profile": str(getattr(args, "geometry_refine_profile", "balanced")),
        "region_profile_mode": "auto_only",
        "distance_normalization": "inside_max_plus_outer_soft_band_v1",
        "dropout_bottleneck": float(args.dropout_bottleneck),
        "dropout_dec4": float(args.dropout_dec4),
        "dropout_dec3": float(args.dropout_dec3),
        "elastic_prob": float(args.elastic_prob),
        "elastic_alpha": float(args.elastic_alpha),
        "elastic_sigma": float(args.elastic_sigma),
        "min_channel_std": float(args.min_channel_std),
        "tile_cache_size": int(args.tile_cache_size),
        "num_workers": int(args.num_workers),
        "train_batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "manual_labels_dir": str(args.manual_labels_dir) if args.manual_labels_dir else None,
        "manual_tile_ids_path": str(args.manual_tile_ids) if args.manual_tile_ids else None,
        "manual_gt_tier": str(args.manual_gt_tier),
    }

    print(f"Loading weak-label tiles from: {args.tiles_dir}")
    manual_tile_ids = _read_manual_tile_ids(args.manual_tile_ids)
    samples, tile_quality = _collect_samples(
        args.tiles_dir,
        args.labels_dir,
        args.patch_size,
        args.stride,
        feature_channels=feature_channels,
        manual_labels_dir=args.manual_labels_dir,
        manual_tile_ids=manual_tile_ids,
        manual_gt_tier=args.manual_gt_tier,
    )
    tile_ids = sorted({s.tile_id for s in samples})
    split = _split_tile_ids(tile_ids, args.seed)

    train_samples_raw = [s for s in samples if s.tile_id in split["train"]]
    val_samples = [s for s in samples if s.tile_id in split["val"]]
    test_samples = [s for s in samples if s.tile_id in split["test"]]
    train_samples = _rebalance_train_samples(train_samples_raw, seed=args.seed)
    norm_stat_samples = _resolve_norm_stat_samples(train_samples_raw, train_samples)

    norm_stats = _compute_norm_stats(
        norm_stat_samples,
        feature_channels,
        patch_size=args.patch_size,
        tile_cache_size=max(1, min(int(args.tile_cache_size), 2)),
        min_channel_std=float(args.min_channel_std),
    )

    stats = {
        "n_tiles": len(tile_ids),
        "channels": list(feature_channels),
        "metadata": metadata,
        "norm_stats_source": "raw_train_split" if norm_stat_samples is train_samples_raw else "rebalanced_train_split",
        "tile_quality": [
            {
                "tile_id": tile_id,
                "tile_weight": float(tile_quality.get(tile_id, {}).get("tile_weight", 1.0)),
                "quality_gate_failed": bool(tile_quality.get(tile_id, {}).get("quality_gate_failed", False)),
                "used_fallback": bool(tile_quality.get(tile_id, {}).get("used_fallback", False)),
                "temporal_amp": float(tile_quality.get(tile_id, {}).get("temporal_amp", 0.0)),
                "temporal_entropy": float(tile_quality.get(tile_id, {}).get("temporal_entropy", 0.0)),
                "manual_gt_used": bool(tile_quality.get(tile_id, {}).get("manual_gt_used", False)),
                "gt_quality_tier": str(tile_quality.get(tile_id, {}).get("gt_quality_tier", "weak_reference")),
                "region_band": str(tile_quality.get(tile_id, {}).get("region_band", "central")),
                "region_boundary_profile_target": str(
                    tile_quality.get(tile_id, {}).get("region_boundary_profile_target", "balanced")
                ),
                "error_mode_tag": str(tile_quality.get(tile_id, {}).get("error_mode_tag", "none")),
                "parcel_shape_class": str(
                    tile_quality.get(tile_id, {}).get("parcel_shape_class", "irregular_large")
                ),
                "adjacency_tag": str(tile_quality.get(tile_id, {}).get("adjacency_tag", "none")),
            }
            for tile_id in tile_ids
        ],
        "split": {
            "train_tiles": sorted(split["train"]),
            "val_tiles": sorted(split["val"]),
            "test_tiles": sorted(split["test"]),
            "train_patches_raw": len(train_samples_raw),
            "train_patches": len(train_samples),
            "val_patches": len(val_samples),
            "test_patches": len(test_samples),
        },
        "norm": norm_stats,
    }
    manual_tiles_used = sorted(
        tile_id for tile_id in tile_ids if bool(tile_quality.get(tile_id, {}).get("manual_gt_used", False))
    )
    stats["manual_gt"] = {
        "manual_tiles_used": manual_tiles_used,
        "manual_tile_count": len(manual_tiles_used),
    }
    args.output_norm.parent.mkdir(parents=True, exist_ok=True)
    args.output_norm.write_text(
        json.dumps({**norm_stats, "metadata": metadata}, indent=2),
        encoding="utf-8",
    )
    if args.output_stats:
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"Dataset patches train/val/test: "
        f"{len(train_samples)}/{len(val_samples)}/{len(test_samples)} "
        f"(raw train={len(train_samples_raw)})"
    )
    if len(tile_ids) < int(args.min_tiles):
        raise RuntimeError(
            f"Insufficient tiles for retrain: got {len(tile_ids)}, "
            f"required at least {int(args.min_tiles)}."
        )
    if len(tile_ids) < 30:
        print(
            "⚠️  Low tile count for BoundaryUNet training: "
            f"{len(tile_ids)} tiles (recommended: 30+)."
        )
    if len(val_samples) < 10 or len(test_samples) < 10:
        print(
            "⚠️  Very small validation/test split: "
            f"val={len(val_samples)}, test={len(test_samples)} patches. "
            "Holdout metrics may be unstable."
        )

    train_ds = PatchDataset(
        train_samples,
        norm_stats,
        train=True,
        feature_channels=feature_channels,
        patch_size=args.patch_size,
        tile_cache_size=args.tile_cache_size,
        elastic_prob=float(args.elastic_prob),
        elastic_alpha=float(args.elastic_alpha),
        elastic_sigma=float(args.elastic_sigma),
    )
    val_ds = PatchDataset(
        val_samples,
        norm_stats,
        train=False,
        feature_channels=feature_channels,
        patch_size=args.patch_size,
        tile_cache_size=max(1, min(int(args.tile_cache_size), 2)),
        elastic_prob=0.0,
    )
    test_ds = PatchDataset(
        test_samples,
        norm_stats,
        train=False,
        feature_channels=feature_channels,
        patch_size=args.patch_size,
        tile_cache_size=max(1, min(int(args.tile_cache_size), 2)),
        elastic_prob=0.0,
    )

    train_batch_sampler = TileGroupedBatchSampler(
        train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = BoundaryUNet(
        in_channels=len(feature_channels),
        dropout_bottleneck=float(args.dropout_bottleneck),
        dropout_dec4=float(args.dropout_dec4),
        dropout_dec3=float(args.dropout_dec3),
    ).to(device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_state = None
    best_f1 = -1.0
    patience_left = args.patience
    accumulation_steps = max(1, int(args.accumulation_steps))
    progress_file = (
        args.progress_file
        if args.progress_file is not None
        else args.output_stats.with_name(args.output_stats.stem + "_progress.json")
    )
    training_started = time.monotonic()
    progress_interval = max(1, int(args.progress_interval))
    eval_progress_interval = max(1, int(args.eval_progress_interval))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_train = 0
        optimizer.zero_grad(set_to_none=True)
        num_train_batches = len(train_loader)
        epoch_started = time.monotonic()
        for batch_idx, batch_data in enumerate(train_loader, start=1):
            x, target, weights, edge_valid = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            x = x.to(device)
            target_t = {k: v.to(device) for k, v in target.items()}
            preds = model(x)
            loss_per_sample = criterion(
                preds,
                target_t,
                edge_valid_fraction=edge_valid.to(device),
            )
            sample_weights = weights.to(device).float()
            weight_sum = torch.clamp(sample_weights.sum(), min=1e-6)
            loss = torch.sum(loss_per_sample * sample_weights) / weight_sum
            (loss / float(accumulation_steps)).backward()
            should_step = (batch_idx % accumulation_steps == 0) or (batch_idx == num_train_batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * x.shape[0]
            n_train += x.shape[0]

            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == num_train_batches:
                epoch_elapsed = time.monotonic() - epoch_started
                epoch_eta = (epoch_elapsed / batch_idx) * max(0, num_train_batches - batch_idx)
                total_elapsed = time.monotonic() - training_started
                completed_epochs = (epoch - 1) + (batch_idx / max(1, num_train_batches))
                total_eta = None
                if completed_epochs > 0:
                    avg_epoch_time = total_elapsed / completed_epochs
                    total_eta = avg_epoch_time * max(0.0, args.epochs - completed_epochs)
                print(
                    f"[train] epoch {epoch:03d}/{args.epochs} | "
                    f"batch {batch_idx}/{num_train_batches} "
                    f"({(100.0 * batch_idx / num_train_batches):.1f}%) | "
                    f"loss={float(loss.item()):.4f} | "
                    f"elapsed={_format_eta(epoch_elapsed)} | "
                    f"eta_epoch={_format_eta(epoch_eta)} | "
                    f"eta_total={_format_eta(total_eta)}",
                    flush=True,
                )
                _write_progress_file(
                    progress_file,
                    {
                        "stage": "training",
                        "epoch": epoch,
                        "epochs_total": int(args.epochs),
                        "batch": batch_idx,
                        "batches_total": num_train_batches,
                        "epoch_progress_pct": round(100.0 * batch_idx / num_train_batches, 2),
                        "epoch_elapsed_s": round(epoch_elapsed, 2),
                        "epoch_eta_s": round(epoch_eta, 2),
                        "total_elapsed_s": round(total_elapsed, 2),
                        "total_eta_s": round(total_eta, 2) if total_eta is not None else None,
                        "train_loss_last": float(loss.item()),
                    },
                )

        scheduler.step()
        train_loss = total_loss / max(1, n_train)
        print(
            f"[eval] epoch {epoch:03d}/{args.epochs} | validating {len(val_loader)} batch(es)",
            flush=True,
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            progress_interval=eval_progress_interval,
            progress_prefix=f"[eval] epoch {epoch:03d}/{args.epochs}",
            progress_file=progress_file,
            progress_state={
                "epoch": epoch,
                "epochs_total": int(args.epochs),
                "train_loss": float(train_loss),
            },
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou_extent']:.4f} | "
            f"val_f1={val_metrics['f1_boundary']:.4f} | "
            f"val_boundary_iou={val_metrics['boundary_iou']:.4f} | "
            f"val_hd95_px={val_metrics['hd95_px']:.2f}"
        )

        if val_metrics["f1_boundary"] > best_f1:
            best_f1 = val_metrics["f1_boundary"]
            best_state = {
                "model_state": model.state_dict(),
                "norm_stats": norm_stats,
                "metrics": val_metrics,
                "epoch": epoch,
                "metadata": metadata,
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

        total_elapsed = time.monotonic() - training_started
        avg_epoch_time = total_elapsed / float(epoch)
        total_eta = avg_epoch_time * max(0.0, args.epochs - epoch)
        _write_progress_file(
            progress_file,
            {
                "stage": "epoch_complete",
                "epoch": epoch,
                "epochs_total": int(args.epochs),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_iou": float(val_metrics["iou_extent"]),
                "val_f1": float(val_metrics["f1_boundary"]),
                "total_elapsed_s": round(total_elapsed, 2),
                "total_eta_s": round(total_eta, 2),
                "best_f1": float(best_f1),
                "patience_left": int(patience_left),
            },
        )

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_model)
    print(f"Saved checkpoint: {args.output_model}")
    print(f"Saved norm stats: {args.output_norm}")
    model_meta_path = Path(str(args.output_model) + ".meta.json")
    model_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model metadata: {model_meta_path}")

    model.load_state_dict(best_state["model_state"])
    test_metrics = _evaluate(model, test_loader, criterion, device)
    print(
        f"Test metrics: loss={test_metrics['loss']:.4f}, "
        f"iou_extent={test_metrics['iou_extent']:.4f}, "
        f"f1_boundary={test_metrics['f1_boundary']:.4f}, "
        f"boundary_iou={test_metrics['boundary_iou']:.4f}, "
        f"hd95_px={test_metrics['hd95_px']:.2f}, "
        f"area_ratio_pred_gt={test_metrics['area_ratio_pred_gt']:.4f}"
    )

    if args.output_onnx:
        try:
            args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
            _export_onnx(
                model,
                args.output_onnx,
                args.patch_size,
                device,
                opset_version=int(args.onnx_opset),
                n_channels=len(feature_channels),
            )
            print(f"Saved ONNX: {args.output_onnx}")
            onnx_meta_path = Path(str(args.output_onnx) + ".meta.json")
            onnx_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved ONNX metadata: {onnx_meta_path}")
        except Exception as exc:
            print(f"ONNX export skipped: {exc}")

    if args.output_stats:
        final_stats = {
            **stats,
            "val_best": {
                **best_state["metrics"],
                "epoch": int(best_state["epoch"]),
            },
            "test_metrics": test_metrics,
        }
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
    _write_progress_file(
        progress_file,
        {
            "stage": "complete",
            "status": "done",
            "epoch": int(best_state["epoch"]),
            "epochs_total": int(args.epochs),
            "best_f1": float(best_f1),
            "test_f1": float(test_metrics["f1_boundary"]),
            "test_iou": float(test_metrics["iou_extent"]),
        },
    )



def parse_args() -> argparse.Namespace:
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Train BoundaryUNet from weak labels")
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "debug" / "runs" / "real_tiles",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "debug" / "runs" / "real_tiles_labels_weak",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "boundary_unet_v2.pth",
    )
    parser.add_argument(
        "--output-norm",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "boundary_unet_v2.norm.json",
    )
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "boundary_unet_v2.onnx",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=PROJECT_ROOT / "backend" / "debug" / "runs" / "boundary_dataset_stats.json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--tile-cache-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--dropout-bottleneck", type=float, default=0.30)
    parser.add_argument("--dropout-dec4", type=float, default=0.20)
    parser.add_argument("--dropout-dec3", type=float, default=0.10)
    parser.add_argument("--elastic-prob", type=float, default=0.30)
    parser.add_argument("--elastic-alpha", type=float, default=80.0)
    parser.add_argument("--elastic-sigma", type=float, default=10.0)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--eval-progress-interval", type=int, default=50)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--min-channel-std", type=float, default=1e-3)
    parser.add_argument("--manual-labels-dir", type=Path, default=None)
    parser.add_argument("--manual-tile-ids", type=Path, default=None)
    parser.add_argument("--manual-gt-tier", type=str, default="verified")
    parser.add_argument("--model-version", type=str, default=str(cfg.MODEL_VERSION))
    parser.add_argument("--train-data-version", type=str, default=str(cfg.TRAIN_DATA_VERSION))
    parser.add_argument("--feature-stack-version", type=str, default=str(cfg.FEATURE_STACK_VERSION))
    parser.add_argument("--onnx-opset", type=int, default=int(cfg.ONNX_OPSET_VERSION))
    parser.add_argument("--ml-extent-bin-threshold", type=float, default=float(getattr(cfg, "ML_EXTENT_BIN_THRESHOLD", 0.42)))
    parser.add_argument("--geometry-refine-profile", type=str, default=str(getattr(cfg, "GEOMETRY_REFINE_PROFILE", "balanced")))
    parser.add_argument("--ml-feature-profile", type=str, default=str(getattr(cfg, "ML_FEATURE_PROFILE", "v2_16ch")))
    parser.add_argument("--min-tiles", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
