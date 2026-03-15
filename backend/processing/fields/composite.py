"""Multi-temporal stack building and phenometric computation."""
import warnings

import numpy as np

from utils.nan_safe import nanmax_safe, nanmean_safe, nanmedian_safe, nanmin_safe, nanstd_safe

VALID_SCL_CLASSES = {4, 5, 6}


def _spatial_shape(stack: np.ndarray) -> tuple[int, int]:
    """Return (H, W) for a (T, H, W) stack, otherwise (0, 0)."""
    if getattr(stack, "ndim", 0) != 3:
        return 0, 0
    return int(stack.shape[1]), int(stack.shape[2])


def _clip_unit(values: np.ndarray | float) -> np.ndarray:
    """Clip score-like arrays into [0, 1]."""
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _normalize_score(values: np.ndarray, *, center: float | None = None) -> np.ndarray:
    """Normalize a 1D score array to [0, 1] while staying NaN-safe."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float64)

    work = arr[finite]
    if center is not None:
        scaled = 1.0 - np.abs(work - float(center))
        out = np.zeros_like(arr, dtype=np.float64)
        out[finite] = _clip_unit(scaled)
        return out

    lo = float(np.nanmin(work))
    hi = float(np.nanmax(work))
    out = np.zeros_like(arr, dtype=np.float64)
    if hi - lo <= 1e-9:
        out[finite] = 1.0
        return out
    out[finite] = (work - lo) / (hi - lo)
    return _clip_unit(out)


def _region_band_for_lat(aoi_lat: float, cfg=None) -> str:
    """Classify AOI latitude into a coarse phenology band."""
    from core.region import resolve_region_band

    south_max = float(getattr(cfg, "REGION_LAT_SOUTH_MAX", 48.0))
    north_min = float(getattr(cfg, "REGION_LAT_NORTH_MIN", 57.0))
    return resolve_region_band(aoi_lat, south_max=south_max, north_min=north_min)


def build_valid_mask_from_scl(scl_stack: np.ndarray) -> np.ndarray:
    """Build valid pixel mask from SCL stack.

    Args:
        scl_stack: (T, H, W) uint8 SCL values.

    Returns:
        (T, H, W) bool mask — True where pixel is valid.
    """
    mask = np.zeros_like(scl_stack, dtype=bool)
    for cls in VALID_SCL_CLASSES:
        mask |= scl_stack == cls
    return mask


def select_dates_by_coverage(
    valid_mask: np.ndarray,
    min_valid_pct: float = 0.5,
    n_dates: int = 7,
    min_good_dates: int | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | bool | list[float]]]:
    """Select dates with best coverage spread across season.

    Args:
        valid_mask: (T, H, W) bool.
        min_valid_pct: minimum fraction of valid pixels to keep date.
        n_dates: target number of dates.
        min_good_dates: minimum number of dates that should pass min_valid_pct.
        return_metadata: when True, also return selection diagnostics.

    Returns:
        1D array of selected date indices, optionally with metadata.
    """
    t_count = valid_mask.shape[0]
    coverages = valid_mask.reshape(t_count, -1).mean(axis=1)

    good_indices = np.where(coverages >= min_valid_pct)[0]
    low_quality_input = False
    if len(good_indices) == 0:
        good_indices = np.argsort(coverages)[-min(n_dates, t_count):]
        good_indices = np.sort(good_indices)
        low_quality_input = True

    if min_good_dates is not None and len(good_indices) < min_good_dates:
        low_quality_input = True

    if len(good_indices) <= n_dates:
        selected = good_indices
    else:
        quantiles = np.linspace(0, 1, n_dates)
        positions = quantiles * (len(good_indices) - 1)
        selected = good_indices[np.unique(np.round(positions).astype(int))]

    if not return_metadata:
        return selected

    metadata: dict[str, float | int | bool | list[float]] = {
        "coverages": coverages.astype(float).tolist(),
        "good_date_count": int(len(good_indices)),
        "selected_date_count": int(len(selected)),
        "low_quality_input": low_quality_input,
        "min_valid_pct": float(min_valid_pct),
    }
    return selected, metadata


def select_dates_adaptive(
    valid_mask: np.ndarray,
    indices: dict[str, np.ndarray] | None,
    time_windows: list[tuple[str, str]],
    aoi_lat: float,
    n_dates: int,
    min_good_dates: int,
    cfg,
    *,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | bool | str | list[float] | list[dict[str, float]]]]:
    """Select dates using coverage plus region-aware phenology heuristics.

    The selector stays deterministic and falls back to the existing coverage-only
    policy when the temporal stack is too weak.
    """
    t_count = int(valid_mask.shape[0])
    if t_count == 0:
        empty = np.array([], dtype=np.int32)
        if not return_metadata:
            return empty
        return empty, {
            "coverages": [],
            "score_total": [],
            "score_components": [],
            "good_date_count": 0,
            "selected_date_count": 0,
            "low_quality_input": True,
            "selected_date_confidence_mean": 0.0,
            "region_band": _region_band_for_lat(aoi_lat, cfg),
        }

    coverages = valid_mask.reshape(t_count, -1).mean(axis=1).astype(np.float32)
    min_valid_pct = float(getattr(cfg, "DATE_SELECTION_MIN_VALID_PCT", 0.50))
    region_band = _region_band_for_lat(aoi_lat, cfg)

    # Coverage score and borderline cloud penalty.
    coverage_score = _clip_unit((coverages - min_valid_pct) / max(1e-6, 1.0 - min_valid_pct))
    cloud_penalty = _clip_unit((min_valid_pct - coverages) / max(1e-6, min_valid_pct))

    # Region-aware phenology prefers later dates in the north and earlier in the south.
    default_center = (t_count - 1) / 2.0
    if region_band == "north":
        shift_days = float(getattr(cfg, "DATE_SELECTION_NORTH_SHIFT_DAYS", 14))
    elif region_band == "south":
        shift_days = float(getattr(cfg, "DATE_SELECTION_SOUTH_SHIFT_DAYS", -14))
    else:
        shift_days = 0.0
    shift_frac = np.clip(shift_days / 60.0, -0.5, 0.5)
    preferred_center = float(np.clip(default_center + shift_frac * max(t_count - 1, 1), 0.0, max(t_count - 1, 0)))
    season_position = np.arange(t_count, dtype=np.float32)
    season_score = _normalize_score(
        np.abs(season_position - preferred_center) / max(t_count - 1, 1),
        center=0.0,
    )

    phenology_score = season_score.copy()
    water_risk_penalty = np.zeros(t_count, dtype=np.float32)
    if isinstance(indices, dict) and indices:
        ndvi = np.asarray(indices.get("NDVI")) if "NDVI" in indices else None
        if ndvi is not None and ndvi.ndim == 3 and ndvi.shape == valid_mask.shape:
            masked_ndvi = np.where(valid_mask, ndvi, np.nan).reshape(t_count, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ndvi_p90 = np.nanpercentile(masked_ndvi, 90, axis=1)
                ndvi_std = np.nanstd(masked_ndvi, axis=1)
            np.nan_to_num(ndvi_p90, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(ndvi_std, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            ndvi_signal = 0.7 * _normalize_score(ndvi_p90) + 0.3 * _normalize_score(ndvi_std)
            phenology_score = _clip_unit(0.55 * season_score + 0.45 * ndvi_signal)

        ndwi = np.asarray(indices.get("NDWI")) if "NDWI" in indices else None
        mndwi = np.asarray(indices.get("MNDWI")) if "MNDWI" in indices else None
        if ndwi is not None and mndwi is not None and ndwi.ndim == 3 and mndwi.ndim == 3:
            masked_ndwi = np.where(valid_mask, ndwi, np.nan).reshape(t_count, -1)
            masked_mndwi = np.where(valid_mask, mndwi, np.nan).reshape(t_count, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ndwi_p90 = np.nanpercentile(masked_ndwi, 90, axis=1)
                mndwi_p90 = np.nanpercentile(masked_mndwi, 90, axis=1)
            np.nan_to_num(ndwi_p90, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(mndwi_p90, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            water_risk_penalty = _clip_unit(
                0.5 * _normalize_score(np.maximum(ndwi_p90, 0.0))
                + 0.5 * _normalize_score(np.maximum(mndwi_p90, 0.0))
            )

    # Greedy spacing bonus: prefer dates spread through the season.
    uniqueness_seed = np.zeros(t_count, dtype=np.float32)
    if t_count > 1:
        uniqueness_seed[0] = 1.0
        uniqueness_seed[-1] = 1.0
        for idx in range(1, t_count - 1):
            nearest_edge = min(idx, (t_count - 1) - idx)
            uniqueness_seed[idx] = min(1.0, nearest_edge / max((t_count - 1) / 2.0, 1.0))
    else:
        uniqueness_seed[0] = 1.0

    weight_coverage = float(getattr(cfg, "DATE_SELECTION_WEIGHT_COVERAGE", 0.40))
    weight_pheno = float(getattr(cfg, "DATE_SELECTION_WEIGHT_PHENO", 0.30))
    weight_uniqueness = float(getattr(cfg, "DATE_SELECTION_WEIGHT_UNIQUENESS", 0.20))
    weight_water = float(getattr(cfg, "DATE_SELECTION_WEIGHT_WATER_PENALTY", 0.10))
    total_no_uniqueness = (
        weight_coverage * coverage_score
        + weight_pheno * phenology_score
        - weight_water * water_risk_penalty
        - 0.10 * cloud_penalty
    )
    total = total_no_uniqueness + weight_uniqueness * uniqueness_seed

    good_indices = np.where(coverages >= min_valid_pct)[0]
    low_quality_input = bool(len(good_indices) < int(min_good_dates))
    candidate_pool = good_indices if len(good_indices) else np.arange(t_count)
    spacing = max(1, int(round(t_count / max(1, int(n_dates)))))

    selected: list[int] = []
    selected_reason = ["not_selected"] * t_count
    candidate_list = sorted(candidate_pool.tolist())

    # Greedy iterative selection: uniqueness is recomputed against already chosen windows.
    while len(selected) < min(int(n_dates), t_count):
        best_idx = None
        best_score = -1e9
        for idx in candidate_list:
            if idx in selected:
                continue
            if selected:
                nearest = min(abs(idx - chosen) for chosen in selected)
                uniqueness_dynamic = min(1.0, float(nearest) / float(max(spacing, 1)))
            else:
                uniqueness_dynamic = float(uniqueness_seed[idx])
            dynamic_total = float(total_no_uniqueness[idx] + weight_uniqueness * uniqueness_dynamic)
            if dynamic_total > best_score:
                best_score = dynamic_total
                best_idx = idx
            elif dynamic_total == best_score and best_idx is not None:
                # Deterministic tie-break: higher coverage first, then earlier date.
                if float(coverages[idx]) > float(coverages[best_idx]):
                    best_idx = idx
                elif float(coverages[idx]) == float(coverages[best_idx]) and int(idx) < int(best_idx):
                    best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
        selected_reason[int(best_idx)] = "high_score"

    if len(selected) < min(int(n_dates), t_count):
        ranking = sorted(
            candidate_pool.tolist(),
            key=lambda idx: (float(total[idx]), float(coverages[idx]), -float(idx)),
            reverse=True,
        )
        for idx in ranking:
            if idx in selected:
                continue
            selected.append(int(idx))
            if selected_reason[idx] == "not_selected":
                selected_reason[idx] = "season_spread"
            if len(selected) >= min(int(n_dates), t_count):
                break

    if len(selected) < int(min_good_dates) and bool(
        getattr(cfg, "DATE_SELECTION_ALLOW_LOW_CONFIDENCE_FALLBACK", True)
    ):
        low_quality_input = True
        fallback_ranking = sorted(range(t_count), key=lambda idx: float(coverages[idx]), reverse=True)
        for idx in fallback_ranking:
            if idx in selected:
                continue
            selected.append(int(idx))
            selected_reason[idx] = "fallback_low_quality"
            if len(selected) >= min(int(n_dates), t_count):
                break

    selected_arr = np.asarray(sorted(set(selected)), dtype=np.int32)
    if not return_metadata:
        return selected_arr

    score_components = [
        {
            "coverage": round(float(coverage_score[idx]), 6),
            "phenology": round(float(phenology_score[idx]), 6),
            "uniqueness": round(float(uniqueness_seed[idx]), 6),
            "water_risk_penalty": round(float(water_risk_penalty[idx]), 6),
            "cloud_penalty": round(float(cloud_penalty[idx]), 6),
        }
        for idx in range(t_count)
    ]
    metadata: dict[str, float | int | bool | str | list[float] | list[dict[str, float]] | list[str]] = {
        "coverages": coverages.astype(float).tolist(),
        "score_total": np.asarray(total, dtype=np.float64).astype(float).tolist(),
        "score_components": score_components,
        "selected_reason": selected_reason,
        "good_date_count": int(len(good_indices)),
        "selected_date_count": int(len(selected_arr)),
        "low_quality_input": low_quality_input,
        "selected_date_confidence_mean": float(np.mean(total[selected_arr])) if len(selected_arr) else 0.0,
        "region_band": region_band,
        "min_valid_pct": min_valid_pct,
    }
    return selected_arr, metadata


def compute_phenometrics(
    ndvi_stack: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-pixel phenological metrics from NDVI time series.

    Args:
        ndvi_stack: (T, H, W) float.
        valid_mask: (T, H, W) bool.

    Returns:
        dict with keys: ndvi_min, ndvi_max, ndvi_mean, ndvi_std, ndvi_delta.
        Each value is (H, W).
    """
    if valid_mask.size == 0 or not valid_mask.any():
        h, w = _spatial_shape(ndvi_stack)
        zeros = np.zeros((h, w), dtype=np.float32)
        return {
            "ndvi_min": zeros,
            "ndvi_max": zeros.copy(),
            "ndvi_mean": zeros.copy(),
            "ndvi_std": zeros.copy(),
            "ndvi_delta": zeros.copy(),
        }

    masked = np.where(valid_mask, ndvi_stack, np.nan)

    ndvi_min = nanmin_safe(masked, axis=0, fill_value=np.nan)
    ndvi_max = nanmax_safe(masked, axis=0, fill_value=np.nan)
    ndvi_mean = nanmean_safe(masked, axis=0, fill_value=np.nan)
    ndvi_std = nanstd_safe(masked, axis=0, fill_value=np.nan)
    del masked

    ndvi_delta = ndvi_max - ndvi_min

    # Use -0.1 as sentinel for pixels with no valid observations, so they
    # are distinguishable from actual bare soil (NDVI ~ 0).  Previously 0.0
    # was used, causing the ML model to treat cloudy/invalid pixels as bare
    # soil and producing false boundary detections in cloud-shadow areas.
    _NODATA_SENTINEL = -0.1
    np.nan_to_num(ndvi_min, copy=False, nan=_NODATA_SENTINEL, posinf=0.0, neginf=_NODATA_SENTINEL)
    np.nan_to_num(ndvi_max, copy=False, nan=_NODATA_SENTINEL, posinf=0.0, neginf=_NODATA_SENTINEL)
    np.nan_to_num(ndvi_mean, copy=False, nan=_NODATA_SENTINEL, posinf=0.0, neginf=_NODATA_SENTINEL)
    np.nan_to_num(ndvi_std, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(ndvi_delta, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "ndvi_mean": ndvi_mean,
        "ndvi_std": ndvi_std,
        "ndvi_delta": ndvi_delta,
    }


def build_median_composite(
    stack: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Build median composite from multi-temporal stack.

    Args:
        stack: (T, H, W) float.
        valid_mask: (T, H, W) bool.

    Returns:
        (H, W) median composite.
    """
    if valid_mask.size == 0 or not valid_mask.any():
        h, w = _spatial_shape(stack)
        return np.zeros((h, w), dtype=np.float32)

    masked = np.where(valid_mask, stack, np.nan)
    result = nanmedian_safe(masked, axis=0, fill_value=np.nan)
    del masked
    # Use -0.1 sentinel so invalid pixels are not confused with actual data.
    np.nan_to_num(result, copy=False, nan=-0.1, posinf=0.0, neginf=-0.1)
    return result
