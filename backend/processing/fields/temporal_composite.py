"""Multi-year temporal composite helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from processing.fields.edge_composite import build_multitemporal_edge_composite
from utils.nan_safe import nanmax_safe, nanmean_safe, nanstd_safe


def _safe_replace_year(value: datetime, target_year: int) -> datetime:
    """Replace year while handling Feb 29 for non-leap target years."""
    try:
        return value.replace(year=target_year)
    except ValueError:
        if value.month == 2 and value.day == 29:
            return value.replace(year=target_year, day=28)
        raise


def expand_dates_multi_year(
    date_from: datetime,
    date_to: datetime,
    years_back: int,
) -> list[tuple[datetime, datetime]]:
    """Expand a date range backwards by whole-year offsets."""
    ranges = [(date_from, date_to)]
    for year_offset in range(1, max(0, int(years_back)) + 1):
        target_year_from = date_from.year - year_offset
        target_year_to = date_to.year - year_offset
        ranges.append(
            (
                _safe_replace_year(date_from, target_year_from),
                _safe_replace_year(date_to, target_year_to),
            )
        )
    return ranges


def score_scene(cloud_cover: float, mean_ndvi: float) -> float:
    """Simple score: lower clouds and greener scenes rank higher."""
    return (1.0 - cloud_cover / 100.0) * 0.7 + min(float(mean_ndvi), 1.0) * 0.3


def _empty_composite(shape: tuple[int, int]) -> dict[str, Any]:
    """Return a zeroed composite when no valid scene survives filtering."""
    h, w = shape
    zeros = np.zeros((h, w), dtype=np.float32)
    return {
        "max_ndvi": zeros.copy(),
        "mean_ndvi": zeros.copy(),
        "ndvi_std": zeros.copy(),
        "edge_composite": zeros.copy(),
        "n_valid_scenes": 0,
        "date_ranges": [],
    }


def build_multiyear_composite(
    *,
    scene_loader=None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    cfg,
    ndvi_stack: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    edge_bands: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build a temporal composite from either scenes or already loaded arrays.

    The current runtime path uses the array branch to preserve the existing
    pipeline. The scene-loader branch keeps the interface needed for a later
    fully remote multi-year implementation.
    """
    if ndvi_stack is not None and valid_mask is not None:
        ndvi_stack = np.asarray(ndvi_stack, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if ndvi_stack.ndim == 2:
            ndvi_stack = ndvi_stack[np.newaxis, ...]
        if valid_mask.ndim == 2:
            valid_mask = valid_mask[np.newaxis, ...]
        if ndvi_stack.shape != valid_mask.shape:
            raise ValueError(
                "ndvi_stack and valid_mask must match, "
                f"got {ndvi_stack.shape} vs {valid_mask.shape}"
            )

        cube = np.where(valid_mask, ndvi_stack, np.nan)
        max_ndvi = nanmax_safe(cube, axis=0, fill_value=np.nan)
        mean_ndvi = nanmean_safe(cube, axis=0, fill_value=np.nan)
        ndvi_std = nanstd_safe(cube, axis=0, fill_value=np.nan)

        max_ndvi = np.where(np.isfinite(max_ndvi), max_ndvi, 0.0).astype(np.float32)
        mean_ndvi = np.where(np.isfinite(mean_ndvi), mean_ndvi, 0.0).astype(np.float32)
        ndvi_std = np.where(np.isfinite(ndvi_std), ndvi_std, 0.0).astype(np.float32)

        if edge_bands is None:
            edge_bands = {"ndvi": ndvi_stack}
        edge_composite = build_multitemporal_edge_composite(
            edge_bands,
            valid_mask,
            alpha=float(getattr(cfg, "EDGE_ALPHA", 0.7)),
            canny_sigma=float(getattr(cfg, "EDGE_CANNY_SIGMA", 1.2)),
            coverage_threshold=float(getattr(cfg, "EDGE_COVERAGE_THRESHOLD", 0.30)),
            binary_threshold=float(getattr(cfg, "EDGE_BINARY_THRESHOLD", 0.12)),
            closing_radius=int(getattr(cfg, "EDGE_CLOSING_RADIUS", 2)),
            soft_clip_percentile=(
                float(getattr(cfg, "EDGE_SOFT_CLIP_PERCENTILE", 95.0))
                if bool(getattr(cfg, "EDGE_SOFT_CLIP_ENABLED", True))
                else None
            ),
        )

        return {
            "max_ndvi": max_ndvi,
            "mean_ndvi": mean_ndvi,
            "ndvi_std": ndvi_std,
            "edge_composite": edge_composite.astype(np.float32),
            "n_valid_scenes": int(ndvi_stack.shape[0]),
            "date_ranges": [],
        }

    if scene_loader is None or date_from is None or date_to is None:
        raise ValueError("Provide either ndvi_stack+valid_mask or scene_loader+date range")

    invalid_scl = set(getattr(cfg, "TEMPORAL_SCL_INVALID", ()))
    date_ranges = expand_dates_multi_year(date_from, date_to, getattr(cfg, "TEMPORAL_YEARS_BACK", 0))

    ndvi_frames: list[np.ndarray] = []
    inferred_shape: tuple[int, int] | None = None
    for range_from, range_to in date_ranges:
        scenes = list(scene_loader(range_from, range_to) or [])
        scored = sorted(
            scenes,
            key=lambda scene: score_scene(
                float(scene.get("meta", {}).get("cloud_cover", 100.0)),
                float(np.nanmean(scene.get("ndvi", 0.0))),
            ),
            reverse=True,
        )
        best = scored[: int(getattr(cfg, "TEMPORAL_BEST_N_SCENES", 8))]

        for scene in best:
            ndvi = np.asarray(scene["ndvi"], dtype=np.float32).copy()
            if inferred_shape is None:
                inferred_shape = tuple(ndvi.shape)
            scl = np.asarray(scene.get("scl")) if "scl" in scene else None
            if scl is not None:
                ndvi[np.isin(scl, list(invalid_scl))] = np.nan
            ndvi[ndvi < float(getattr(cfg, "TEMPORAL_NDVI_VALID_MIN", 0.05))] = np.nan
            if not np.any(np.isfinite(ndvi)):
                continue
            ndvi_frames.append(ndvi)

    if not ndvi_frames:
        if inferred_shape is None:
            raise RuntimeError("No valid scenes found for the requested temporal composite")
        result = _empty_composite(inferred_shape)
        result["date_ranges"] = date_ranges
        return result

    cube = np.stack(ndvi_frames, axis=0)
    max_ndvi = nanmax_safe(cube, axis=0, fill_value=np.nan)
    mean_ndvi = nanmean_safe(cube, axis=0, fill_value=np.nan)
    ndvi_std = nanstd_safe(cube, axis=0, fill_value=np.nan)

    valid_mask = np.isfinite(cube)
    edge_composite = build_multitemporal_edge_composite(
        {"ndvi": np.nan_to_num(cube, nan=0.0)},
        valid_mask,
        alpha=float(getattr(cfg, "EDGE_ALPHA", 0.7)),
        canny_sigma=float(getattr(cfg, "EDGE_CANNY_SIGMA", 1.2)),
        coverage_threshold=float(getattr(cfg, "EDGE_COVERAGE_THRESHOLD", 0.30)),
        binary_threshold=float(getattr(cfg, "EDGE_BINARY_THRESHOLD", 0.12)),
        closing_radius=int(getattr(cfg, "EDGE_CLOSING_RADIUS", 2)),
        soft_clip_percentile=(
            float(getattr(cfg, "EDGE_SOFT_CLIP_PERCENTILE", 95.0))
            if bool(getattr(cfg, "EDGE_SOFT_CLIP_ENABLED", True))
            else None
        ),
    )

    return {
        "max_ndvi": np.where(np.isfinite(max_ndvi), max_ndvi, 0.0).astype(np.float32),
        "mean_ndvi": np.where(np.isfinite(mean_ndvi), mean_ndvi, 0.0).astype(np.float32),
        "ndvi_std": np.where(np.isfinite(ndvi_std), ndvi_std, 0.0).astype(np.float32),
        "edge_composite": edge_composite.astype(np.float32),
        "n_valid_scenes": len(ndvi_frames),
        "date_ranges": date_ranges,
    }
