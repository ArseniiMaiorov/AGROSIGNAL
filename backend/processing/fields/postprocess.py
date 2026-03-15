"""Postprocessing for crop candidate masks before watershed segmentation."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label as nd_label

from core.logging import get_logger
from processing.fields.boundary_fill import (
    boundary_to_regions,
    build_boundary_probability_map,
    filter_regions_by_phenology,
)
from processing.fields.boundary_smooth import clean_raster_mask, close_boundary_gaps
from processing.fields.field_grow import seeded_grow_into_field, seeded_grow_into_grass
from processing.fields.field_infill import infill_field_holes
from processing.fields.field_watershed import watershed_field_segmentation
from processing.fields.ndvi_phenology import compute_field_candidate, compute_phenology_masks
from processing.fields.owt import oriented_watershed
from processing.fields.phenoclassify import BUILTUP, CROP, FOREST, GRASS, WATER
from processing.fields.region_merge import hierarchical_merge, merge_crop_regions
from processing.priors.worldcover import load_worldcover_prior
from processing.fields.road_filter import build_road_mask
from processing.fields.road_osm import fetch_road_mask
from processing.fields.road_spectral import build_spectral_road_mask
from utils.raster import (
    count_components,
    count_small_components,
    empty_bool_like,
    remove_small_components,
    select_small_components,
)

logger = get_logger(__name__)


def _emit_progress(progress_callback: Callable[[str], None] | None, checkpoint: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(checkpoint)
    except Exception as exc:
        logger.debug(
            "postprocess_progress_callback_failed",
            checkpoint=checkpoint,
            error=str(exc),
        )


def _record_step(
    debug_masks: dict[str, np.ndarray] | None,
    debug_stats: dict[str, dict[str, int | float]] | None,
    step_name: str,
    mask: np.ndarray,
) -> None:
    """Capture a postprocess step for later inspection."""
    if debug_masks is not None:
        debug_masks[step_name] = mask.astype(np.uint8)
    if debug_stats is not None:
        debug_stats[step_name] = {
            "pixels": int(np.count_nonzero(mask)),
            "components": count_components(mask),
            "coverage_ratio": float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0,
        }


def _record_float_step(
    debug_masks: dict[str, np.ndarray] | None,
    debug_stats: dict[str, dict[str, int | float]] | None,
    step_name: str,
    array: np.ndarray,
) -> None:
    """Capture a floating-point raster debug layer."""
    if debug_masks is not None:
        debug_masks[step_name] = array.astype(np.float32)
    if debug_stats is not None:
        debug_stats[step_name] = {
            "min": float(np.nanmin(array)) if array.size else 0.0,
            "max": float(np.nanmax(array)) if array.size else 0.0,
            "mean": float(np.nanmean(array)) if array.size else 0.0,
        }


def _sanitize_added_pixels(
    mask: np.ndarray,
    hard_exclusion_mask: np.ndarray,
    debug_stats: dict[str, dict[str, int | float]] | None,
    step_name: str,
) -> tuple[np.ndarray, int]:
    """Clip newly added pixels against hard exclusions and record blocked count."""
    blocked_mask = mask & hard_exclusion_mask
    blocked_pixels = int(np.count_nonzero(blocked_mask))
    sanitized = mask & ~hard_exclusion_mask
    if debug_stats is not None:
        debug_stats[f"{step_name}_sanitize"] = {
            "blocked_pixels": blocked_pixels,
            "blocked_ratio": float(blocked_pixels / mask.size) if mask.size else 0.0,
        }
    return sanitized, blocked_pixels


def _edge_overlap_ratio(mask: np.ndarray, reference_mask: np.ndarray, *, buffer_px: int = 1) -> float:
    """Estimate how much the exterior edge leans into a reference mask."""
    mask = mask.astype(bool, copy=False)
    reference_mask = reference_mask.astype(bool, copy=False)
    if not np.any(mask) or not np.any(reference_mask):
        return 0.0
    ring = binary_dilation(mask, iterations=max(1, int(buffer_px))) & ~mask
    ring_pixels = int(np.count_nonzero(ring))
    if ring_pixels <= 0:
        return 0.0
    return float(np.count_nonzero(ring & reference_mask) / ring_pixels)


def _maybe_rollback_edge_drift(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    reference_mask: np.ndarray,
    cfg,
) -> tuple[np.ndarray, bool, float]:
    """Rollback the last mask expansion if the new boundary hugs the reference mask."""
    buffer_px = int(getattr(cfg, "ROAD_SNAP_REJECT_BUFFER_PX", 2))
    current_ratio = _edge_overlap_ratio(current_mask, reference_mask, buffer_px=buffer_px)
    if not bool(getattr(cfg, "ROAD_SNAP_REJECT_ENABLED", False)):
        return current_mask, False, current_ratio
    previous_ratio = _edge_overlap_ratio(previous_mask, reference_mask, buffer_px=buffer_px)
    max_ratio = float(getattr(cfg, "ROAD_SNAP_REJECT_MAX_OVERLAP_RATIO", 0.08))
    if current_ratio > max_ratio and current_ratio > (previous_ratio + 0.01):
        return previous_mask.astype(bool, copy=True), True, current_ratio
    return current_mask, False, current_ratio


def _bridge_near_components(
    mask: np.ndarray,
    ndvi: np.ndarray,
    barrier_mask: np.ndarray,
    boundary_prob: np.ndarray | None,
    cfg,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Bridge nearby likely-same-field fragments for south recall mode."""
    labeled, n_labels = nd_label(mask.astype(bool))
    if n_labels <= 1:
        return mask.astype(bool, copy=True), 0, 0

    max_gap_px = max(1, int(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_MAX_GAP_PX", 3)))
    max_ndvi_diff = float(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_MAX_NDVI_DIFF", 0.08))
    max_boundary_prob = float(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_MAX_BOUNDARY_PROB", 0.45))
    max_components = max(
        1,
        int(
            getattr(
                cfg,
                "SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS",
                min(400, int(getattr(cfg, "POST_MERGE_MAX_COMPONENTS", 1500))),
            )
        ),
    )
    if n_labels > max_components:
        logger.warning(
            "south_component_bridge_skipped",
            component_count=int(n_labels),
            max_components=max_components,
        )
        if progress_callback is not None:
            try:
                progress_callback(int(max_components), int(max_components))
            except Exception as exc:
                logger.debug("south_component_bridge_progress_failed", error=str(exc))
        return mask.astype(bool, copy=True), 0, 0

    result = mask.astype(bool, copy=True)
    added = 0
    bridges = 0
    components = [(labeled == idx) for idx in range(1, n_labels + 1)]
    component_means: list[float] = []
    for comp in components:
        valid = comp & np.isfinite(ndvi)
        component_means.append(float(np.nanmean(ndvi[valid])) if np.any(valid) else 0.0)

    last_emit_at = 0.0
    for left_idx, comp_i in enumerate(components):
        try:
            if not np.any(comp_i):
                continue
            dilated_i = binary_dilation(comp_i, iterations=max_gap_px)
            for right_idx in range(left_idx + 1, len(components)):
                comp_j = components[right_idx]
                if not np.any(comp_j):
                    continue
                if abs(component_means[left_idx] - component_means[right_idx]) > max_ndvi_diff:
                    continue
                dilated_j = binary_dilation(comp_j, iterations=max_gap_px)
                bridge = dilated_i & dilated_j & ~result
                if not np.any(bridge):
                    continue
                if np.any(bridge & barrier_mask):
                    continue
                if boundary_prob is not None:
                    bridge_bp = float(np.nanmean(boundary_prob[bridge])) if np.any(bridge) else 1.0
                    if bridge_bp > max_boundary_prob:
                        continue
                result |= bridge
                added += int(np.count_nonzero(bridge))
                bridges += 1
        finally:
            if progress_callback is not None:
                now = time.monotonic()
                done = left_idx + 1
                if done == 1 or done == len(components) or (now - last_emit_at) >= 2.0:
                    try:
                        progress_callback(int(done), int(len(components)))
                    except Exception as exc:
                        logger.debug("south_component_bridge_progress_failed", error=str(exc))
                    last_emit_at = now

    return result, added, bridges


def _build_osm_road_mask(
    candidate_mask: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    tile_transform,
    out_shape: tuple[int, int] | None,
    crs_epsg: int | str | None,
    cfg,
) -> np.ndarray:
    if (
        bbox is None
        or tile_transform is None
        or out_shape is None
        or crs_epsg is None
    ):
        return empty_bool_like(candidate_mask)

    return fetch_road_mask(
        bbox=bbox,
        transform=tile_transform,
        out_shape=out_shape,
        crs_epsg=crs_epsg,
        cfg=cfg,
    )


def _build_road_barrier(
    candidate_mask: np.ndarray,
    ndvi: np.ndarray,
    water_mask: np.ndarray,
    cfg,
    *,
    nir: np.ndarray | None = None,
    swir: np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    tile_transform=None,
    out_shape: tuple[int, int] | None = None,
    crs_epsg: int | str | None = None,
) -> np.ndarray:
    """Combine OSM and spectral road masks into one barrier mask."""
    road_input_ndvi = np.where(water_mask, np.inf, ndvi)
    osm_roads = _build_osm_road_mask(
        candidate_mask,
        bbox,
        tile_transform,
        out_shape,
        crs_epsg,
        cfg,
    )

    if nir is not None and swir is not None:
        spectral_roads = build_spectral_road_mask(road_input_ndvi, nir, swir, cfg)
    else:
        # Backward-compatible fallback for local runs that do not pass extra bands.
        spectral_roads = build_road_mask(road_input_ndvi, cfg)

    return (osm_roads | spectral_roads) & ~water_mask


def run_postprocess(
    candidate_mask: np.ndarray,
    water_mask: np.ndarray,
    classes: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    cfg,
    open_water_mask: np.ndarray | None = None,
    seasonal_wet_mask: np.ndarray | None = None,
    riparian_soft_mask: np.ndarray | None = None,
    riparian_hard_mask: np.ndarray | None = None,
    nir: np.ndarray | None = None,
    swir: np.ndarray | None = None,
    edge_composite: np.ndarray | None = None,
    boundary_prob: np.ndarray | None = None,
    ndvi_std: np.ndarray | None = None,
    worldcover_mask: np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    tile_transform=None,
    out_shape: tuple[int, int] | None = None,
    crs_epsg: int | str | None = None,
    return_debug_steps: bool = False,
    return_debug_stats: bool = False,
    return_candidate_masks: bool = False,
    region_profile: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, dict]]:
    """Run barrier construction and raster cleanup before segmentation."""
    for name, array in {
        "water_mask": water_mask,
        "classes": classes,
        "ndvi": ndvi,
        "ndwi": ndwi,
    }.items():
        if array.shape != candidate_mask.shape:
            raise ValueError(
                f"{name} must match candidate_mask shape, "
                f"got {array.shape} vs {candidate_mask.shape}"
            )

    if nir is not None and nir.shape != candidate_mask.shape:
        raise ValueError("nir must match candidate_mask shape")
    if swir is not None and swir.shape != candidate_mask.shape:
        raise ValueError("swir must match candidate_mask shape")
    if edge_composite is not None and edge_composite.shape != candidate_mask.shape:
        raise ValueError("edge_composite must match candidate_mask shape")
    if boundary_prob is not None and boundary_prob.shape != candidate_mask.shape:
        raise ValueError("boundary_prob must match candidate_mask shape")
    if ndvi_std is not None and ndvi_std.shape != candidate_mask.shape:
        raise ValueError("ndvi_std must match candidate_mask shape")
    if worldcover_mask is not None and worldcover_mask.shape != candidate_mask.shape:
        raise ValueError("worldcover_mask must match candidate_mask shape")
    for name, array in {
        "open_water_mask": open_water_mask,
        "seasonal_wet_mask": seasonal_wet_mask,
        "riparian_soft_mask": riparian_soft_mask,
        "riparian_hard_mask": riparian_hard_mask,
    }.items():
        if array is not None and array.shape != candidate_mask.shape:
            raise ValueError(f"{name} must match candidate_mask shape")

    candidate_mask = candidate_mask.astype(bool, copy=True)
    region_token = str(region_profile or "").strip().lower()
    region_actions: list[str] = []
    water_mask = water_mask.astype(bool, copy=False)
    open_water_mask = (
        np.asarray(open_water_mask, dtype=bool)
        if open_water_mask is not None
        else water_mask.astype(bool, copy=False)
    )
    seasonal_wet_mask = (
        np.asarray(seasonal_wet_mask, dtype=bool)
        if seasonal_wet_mask is not None
        else np.zeros_like(candidate_mask, dtype=bool)
    )
    riparian_soft_mask = (
        np.asarray(riparian_soft_mask, dtype=bool)
        if riparian_soft_mask is not None
        else np.zeros_like(candidate_mask, dtype=bool)
    )
    riparian_hard_mask = (
        np.asarray(riparian_hard_mask, dtype=bool)
        if riparian_hard_mask is not None
        else np.zeros_like(candidate_mask, dtype=bool)
    )
    capture_debug_stats = bool(return_debug_steps or return_debug_stats)
    debug_masks: dict[str, np.ndarray] | None = {} if return_debug_steps else None
    debug_stats: dict[str, dict[str, int | float]] | None = {} if capture_debug_stats else None
    phenology_masks = None
    if ndvi_std is not None:
        phenology_masks = compute_phenology_masks(ndvi, ndvi_std, cfg)

    _record_step(debug_masks, debug_stats, "step_00_candidate_initial", candidate_mask)
    _record_step(debug_masks, debug_stats, "step_00a_open_water_mask", open_water_mask)
    _record_step(debug_masks, debug_stats, "step_00b_seasonal_wet_mask", seasonal_wet_mask)
    _record_step(debug_masks, debug_stats, "step_00c_riparian_soft_mask", riparian_soft_mask)
    _record_step(debug_masks, debug_stats, "step_00d_riparian_hard_mask", riparian_hard_mask)

    _emit_progress(progress_callback, "road_barrier_start")
    road_mask = _build_road_barrier(
        candidate_mask,
        ndvi,
        open_water_mask,
        cfg,
        nir=nir,
        swir=swir,
        bbox=bbox,
        tile_transform=tile_transform,
        out_shape=out_shape or candidate_mask.shape,
        crs_epsg=crs_epsg,
    )
    _emit_progress(progress_callback, "road_barrier_done")
    initial_candidate_pixels = int(np.count_nonzero(candidate_mask))
    road_removed = int(np.count_nonzero(candidate_mask & road_mask))
    road_removed_ratio = (
        float(road_removed / max(initial_candidate_pixels, 1))
        if initial_candidate_pixels > 0
        else 0.0
    )
    road_barrier_retry_used = False
    large_field_px = max(
        1,
        int(
            float(getattr(cfg, "POST_LARGE_FIELD_RESCUE_MIN_AREA_HA", 2.0))
            * 10_000.0
            / max(float(getattr(cfg, "POST_PX_AREA_M2", 100)), 1.0)
        ),
    )
    if (
        bool(getattr(cfg, "ROAD_BARRIER_SOFT_RETRY_ENABLED", True))
        and initial_candidate_pixels >= large_field_px
        and road_removed_ratio > float(getattr(cfg, "ROAD_BARRIER_SOFT_RETRY_MIN_REMOVED_RATIO", 0.15))
        and np.any(road_mask)
    ):
        softened = binary_erosion(road_mask, iterations=1)
        if not np.any(softened):
            # Erosion collapsed the mask entirely — keep the original road_mask
            # instead of discarding it.  Using zeros here would disable road
            # removal completely, which is wrong.
            softened = road_mask.copy()
        softened_removed = int(np.count_nonzero(candidate_mask & softened))
        if softened_removed < road_removed:
            road_mask = softened.astype(bool, copy=False)
            road_removed = softened_removed
            road_removed_ratio = (
                float(softened_removed / max(initial_candidate_pixels, 1))
                if initial_candidate_pixels > 0
                else 0.0
            )
            road_barrier_retry_used = True
    logger.debug("postprocess_road_mask", removed_pixels=road_removed)
    _record_step(debug_masks, debug_stats, "step_01_road_mask", road_mask)
    if debug_stats is not None:
        debug_stats["step_01_road_mask_meta"] = {
            "removed_pixels": int(road_removed),
            "removed_ratio": float(road_removed_ratio),
            "soft_retry_used": int(road_barrier_retry_used),
        }

    # Road hard exclusion: remove road pixels from candidate mask
    if bool(getattr(cfg, "POST_ROAD_HARD_EXCLUSION", True)):
        candidate_mask &= ~road_mask
        _record_step(debug_masks, debug_stats, "step_01b_road_hard_exclusion", candidate_mask)

    forest_mask = classes == FOREST
    if phenology_masks is not None:
        forest_mask |= phenology_masks["is_forest"]
    forest_removed = int(np.count_nonzero(candidate_mask & forest_mask))
    logger.debug("postprocess_forest_mask", removed_pixels=forest_removed)
    _record_step(debug_masks, debug_stats, "step_02_forest_mask", forest_mask)

    builtup_mask = classes == BUILTUP
    is_water_mask = open_water_mask | (classes == WATER)
    _record_step(debug_masks, debug_stats, "step_02b_builtup_mask", builtup_mask)

    wc_prior_mask = np.zeros_like(candidate_mask, dtype=bool)
    if (
        worldcover_mask is not None
        and phenology_masks is not None
        and bool(getattr(cfg, "FRAMEWORK_USE_WEAK_WORLDCOVER", True))
        and bool(getattr(cfg, "USE_WEAK_WORLDCOVER_BARRIER", True))
    ):
        wc_prior_mask = load_worldcover_prior(worldcover_mask, phenology_masks, cfg)
    _record_step(debug_masks, debug_stats, "step_02c_worldcover_weak_prior", wc_prior_mask)

    # WC tree hard exclusion: WorldCover class 10 (tree cover) as hard barrier
    wc_tree_hard_enabled = bool(
        getattr(
            cfg,
            "WC_TREE_HARD_EXCLUSION",
            getattr(cfg, "WCTREEHARD_EXCLUSION", True),
        )
    )
    wc_tree_hard = np.zeros_like(candidate_mask, dtype=bool)
    if worldcover_mask is not None and wc_tree_hard_enabled:
        wc_tree_hard = worldcover_mask == 10
    wc_tree_removed = int(np.count_nonzero(candidate_mask & wc_tree_hard))
    _record_step(debug_masks, debug_stats, "step_02h_wc_tree_hard", wc_tree_hard)
    if np.any(wc_tree_hard):
        candidate_mask &= ~wc_tree_hard
        _record_step(debug_masks, debug_stats, "step_02i_candidate_minus_wc_tree", candidate_mask)

    if phenology_masks is not None:
        _record_step(debug_masks, debug_stats, "step_02d_pheno_field", phenology_masks["is_field"])
        _record_step(debug_masks, debug_stats, "step_02e_pheno_grass", phenology_masks["is_grass"])
        _record_step(debug_masks, debug_stats, "step_02f_pheno_forest", phenology_masks["is_forest"])

    grass_barrier_mask = (
        phenology_masks["is_grass"]
        if phenology_masks is not None
        else np.zeros_like(candidate_mask, dtype=bool)
    )
    hard_exclusion_mask = (
        is_water_mask
        | riparian_hard_mask
        | forest_mask
        | builtup_mask
        | wc_prior_mask
        | wc_tree_hard
    )
    grow_block_mask = road_mask | hard_exclusion_mask | seasonal_wet_mask
    barrier_mask = grow_block_mask | grass_barrier_mask
    _record_step(debug_masks, debug_stats, "step_02g_forbidden_mask", hard_exclusion_mask)
    _record_step(debug_masks, debug_stats, "step_03_barrier_mask", barrier_mask)
    field_candidate = None
    if phenology_masks is not None and ndvi_std is not None:
        field_candidate = compute_field_candidate(ndvi, ndvi_std, barrier_mask, cfg)
    else:
        field_candidate = np.zeros_like(candidate_mask, dtype=bool)

    crop_soft_mask = (
        candidate_mask
        | (classes == CROP)
        | (phenology_masks["is_field"] if phenology_masks is not None else False)
    )
    crop_soft_mask = crop_soft_mask.astype(bool, copy=False) & ~grow_block_mask
    _record_step(debug_masks, debug_stats, "step_03b_field_candidate", field_candidate)
    _record_step(debug_masks, debug_stats, "step_03c_crop_soft_mask", crop_soft_mask)
    legacy_seed_mask = field_candidate if np.any(field_candidate) else crop_soft_mask
    boundary_field_mask = np.zeros_like(candidate_mask, dtype=bool)
    owt_edge = None
    boundary_prob = (
        np.asarray(boundary_prob, dtype=np.float32)
        if boundary_prob is not None
        else None
    )
    boundary_regions = 0
    if boundary_prob is None and edge_composite is not None and ndvi_std is not None:
        owt_edge = oriented_watershed(edge_composite, ndvi, cfg=cfg)
        boundary_prob = build_boundary_probability_map(
            owt_edge,
            road_mask,
            is_water_mask | riparian_hard_mask,
            forest_mask,
            cfg,
        )
    if boundary_prob is not None and ndvi_std is not None:
        min_region_px = max(1, int(cfg.POST_MIN_FIELD_AREA_HA * 10_000 / cfg.POST_PX_AREA_M2))
        labeled_boundary, boundary_regions = boundary_to_regions(
            boundary_prob,
            min_region_px=min_region_px,
            boundary_thresh=getattr(cfg, "BOUNDARY_FILL_THRESH", None),
        )
        boundary_region_limit = max(
            1,
            int(
                getattr(
                    cfg,
                    "BOUNDARY_FILL_MAX_REGIONS",
                    max(2000, int(getattr(cfg, "POST_MERGE_MAX_COMPONENTS", 1500)) * 2),
                )
            ),
        )
        if boundary_regions > boundary_region_limit:
            logger.warning(
                "boundary_fill_region_cap_exceeded",
                region_count=int(boundary_regions),
                max_regions=boundary_region_limit,
            )
            if debug_stats is not None:
                debug_stats["boundary_fill_meta"] = {
                    "region_count": int(boundary_regions),
                    "max_regions": int(boundary_region_limit),
                    "skipped": 1,
                }
        elif boundary_regions > 0:
            try:
                def _boundary_merge_progress(done: int, total: int) -> None:
                    _emit_progress(progress_callback, f"boundary_merge_scan:{int(done)}:{int(total)}")

                labeled_boundary = hierarchical_merge(
                    labeled_boundary,
                    boundary_prob,
                    ndvi,
                    ndvi_std,
                    barrier_mask,
                    cfg,
                    region_profile=region_token,
                    progress_callback=_boundary_merge_progress,
                )
            except Exception as exc:
                logger.warning(
                    "hierarchical_merge_failed_fallback",
                    error=str(exc),
                    region_count=int(boundary_regions),
                )
            boundary_field_mask = filter_regions_by_phenology(
                labeled_boundary,
                ndvi,
                ndvi_std,
                barrier_mask,
                cfg,
            )
        _record_float_step(debug_masks, debug_stats, "owt_edge", owt_edge)
        _record_float_step(debug_masks, debug_stats, "boundary_prob", boundary_prob)
        _record_step(debug_masks, debug_stats, "field_mask_boundary", boundary_field_mask)
    _emit_progress(progress_callback, "boundary_fill_done")

    recovery_boundary_anchor = np.zeros_like(candidate_mask, dtype=bool)
    if bool(getattr(cfg, "RECOVERY_BOUNDARY_ANCHOR_ENABLED", False)) and np.any(boundary_field_mask):
        anchor_dilation_px = max(
            0,
            int(getattr(cfg, "RECOVERY_BOUNDARY_ANCHOR_DILATION_PX", 2)),
        )
        recovery_boundary_anchor = boundary_field_mask.copy()
        if anchor_dilation_px > 0:
            recovery_boundary_anchor = binary_dilation(
                recovery_boundary_anchor,
                iterations=anchor_dilation_px,
            )
        recovery_boundary_anchor &= ~grow_block_mask
        anchored_soft = (field_candidate | crop_soft_mask) & recovery_boundary_anchor
        candidate_mask = boundary_field_mask | anchored_soft
        _record_step(
            debug_masks,
            debug_stats,
            "step_03d_recovery_boundary_anchor",
            recovery_boundary_anchor,
        )
    else:
        candidate_mask = boundary_field_mask | (field_candidate & ~barrier_mask)
    if not np.any(candidate_mask):
        candidate_mask = legacy_seed_mask

    if return_debug_steps and bool(getattr(cfg, "DEBUG_COMPARE_VERSIONS", False)):
        _record_step(debug_masks, debug_stats, "v3_final_mask", legacy_seed_mask)
        _record_step(debug_masks, debug_stats, "v4_final_mask", candidate_mask)

    candidate_mask, blocked_after_barrier = _sanitize_added_pixels(
        candidate_mask,
        hard_exclusion_mask,
        debug_stats,
        "step_04_after_barrier",
    )
    _record_step(debug_masks, debug_stats, "step_04_after_barrier", candidate_mask)

    if not np.any(candidate_mask):
        logger.debug("postprocess_final_crop_pixels", crop_pixels=0)
        if debug_stats is not None:
            debug_stats["summary"] = {
                "barrier_pixels": int(np.count_nonzero(barrier_mask)),
                "barrier_ratio": float(np.count_nonzero(barrier_mask) / barrier_mask.size)
                if barrier_mask.size
                else 0.0,
                "hard_exclusion_pixels": int(np.count_nonzero(hard_exclusion_mask)),
                "grow_block_pixels": int(np.count_nonzero(grow_block_mask)),
                "road_pixels": int(np.count_nonzero(road_mask)),
                "road_removed_ratio": float(road_removed_ratio),
                "road_barrier_retry_used": int(road_barrier_retry_used),
                "road_snap_reject_used": 0,
                "road_edge_overlap_ratio": 0.0,
                "boundary_shift_to_road_ratio": 0.0,
                "forest_pixels": int(np.count_nonzero(forest_mask)),
                "water_pixels": int(np.count_nonzero(is_water_mask)),
                "open_water_pixels": int(np.count_nonzero(open_water_mask)),
                "seasonal_wet_pixels": int(np.count_nonzero(seasonal_wet_mask)),
                "riparian_soft_pixels": int(np.count_nonzero(riparian_soft_mask)),
                "riparian_hard_pixels": int(np.count_nonzero(riparian_hard_mask)),
                "water_edge_overlap_ratio": 0.0,
                "hydro_rescue_used": 0,
                "builtup_pixels": int(np.count_nonzero(builtup_mask)),
                "worldcover_weak_prior_pixels": int(np.count_nonzero(wc_prior_mask)),
                "wctree_enabled": int(wc_tree_hard_enabled),
                "wctree_pixels": int(np.count_nonzero(wc_tree_hard)),
                "wctree_removed_pixels": int(wc_tree_removed),
                "blocked_after_barrier_pixels": int(blocked_after_barrier),
                "grass_barrier_pixels": int(np.count_nonzero(grass_barrier_mask)),
                "field_candidate_pixels": int(np.count_nonzero(field_candidate)),
                "crop_soft_pixels": int(np.count_nonzero(crop_soft_mask)),
                "boundary_field_pixels": int(np.count_nonzero(boundary_field_mask)),
                "boundary_regions": int(boundary_regions),
                "small_refine_pixels": 0,
                "min_px": 0,
                "components_before_merge": 0,
                "components_after_merge": 0,
                "components_after_clean": 0,
                "components_after_grow": 0,
                "components_after_gap_close": 0,
                "components_after_watershed": 0,
                "split_risk_score": 0.0,
                "shrink_risk_score": 0.0,
                "region_profile_applied": region_token or "balanced",
                "region_profile_actions": list(region_actions),
                "bridge_added_pixels": 0,
                "bridge_pairs": 0,
            }
        if capture_debug_stats or return_candidate_masks:
            payload = {
                "masks": debug_masks or {},
                "stats": debug_stats or {},
            }
            if return_candidate_masks:
                payload["candidate_masks"] = {
                    "field_candidate": field_candidate.astype(np.uint8, copy=False),
                    "crop_soft_mask": crop_soft_mask.astype(np.uint8, copy=False),
                    "boundary_field_mask": boundary_field_mask.astype(np.uint8, copy=False),
                    "recovery_boundary_anchor": recovery_boundary_anchor.astype(np.uint8, copy=False),
                    "legacy_seed_mask": legacy_seed_mask.astype(np.uint8, copy=False),
                    "final_candidate_mask": candidate_mask.astype(np.uint8, copy=False),
                    "barrier_mask": barrier_mask.astype(np.uint8, copy=False),
                }
            return candidate_mask, payload
        return candidate_mask

    before_clean = candidate_mask.copy()
    clean_debug = None
    if return_debug_steps:
        candidate_mask, clean_debug = clean_raster_mask(
            candidate_mask,
            cfg,
            hard_exclusion_mask=hard_exclusion_mask,
            return_debug=True,
        )
    else:
        candidate_mask = clean_raster_mask(candidate_mask, cfg, hard_exclusion_mask=hard_exclusion_mask)
    candidate_mask, _blocked_after_clean = _sanitize_added_pixels(
        candidate_mask,
        hard_exclusion_mask,
        debug_stats,
        "step_05_after_clean",
    )
    morph_delta = int(np.count_nonzero(candidate_mask)) - int(np.count_nonzero(before_clean))
    # Rollback morphological clean if it caused excessive area loss
    before_clean_area = int(np.count_nonzero(before_clean))
    if before_clean_area > 0:
        clean_area_ratio = int(np.count_nonzero(candidate_mask)) / before_clean_area
        clean_rollback_min_area_ratio = 0.90
        if region_token == "south_recall":
            clean_rollback_min_area_ratio = float(
                getattr(cfg, "SOUTH_CLEAN_ROLLBACK_MIN_AREA_RATIO", 0.75)
            )
        elif region_token == "north_boundary":
            clean_rollback_min_area_ratio = float(
                getattr(cfg, "NORTH_CLEAN_ROLLBACK_MIN_AREA_RATIO", 0.92)
            )
        if clean_area_ratio < clean_rollback_min_area_ratio:
            logger.warning(
                "postprocess_clean_rollback",
                clean_area_ratio=round(clean_area_ratio, 3),
            )
            candidate_mask = before_clean
            candidate_mask &= ~hard_exclusion_mask
            if region_token == "north_boundary":
                region_actions.append("north_clean_rollback")
    logger.debug("postprocess_morph_close", pixels_added=max(0, morph_delta))
    if clean_debug is not None:
        _record_step(
            debug_masks,
            debug_stats,
            "step_05a_clean_holes_skipped_due_to_forbidden",
            clean_debug.get("holes_skipped_due_to_forbidden", np.zeros_like(candidate_mask, dtype=bool)),
        )
    _record_step(debug_masks, debug_stats, "step_05_after_clean", candidate_mask)
    _emit_progress(progress_callback, "clean_done")

    growable_mask = None
    if phenology_masks is not None:
        soft_field = (
            (classes == GRASS)
            & (ndvi > float(cfg.PHENO_FIELD_MAX_NDVI_MIN) - 0.15)
            & (ndvi_std > float(cfg.PHENO_FIELD_NDVI_STD_MIN) - 0.07)
            & ~grow_block_mask
        )
        growable_mask = field_candidate | crop_soft_mask | soft_field

    before_grow = candidate_mask.copy()
    if ndvi_std is not None:
        candidate_mask, grown_added = seeded_grow_into_field(
            candidate_mask,
            ndvi,
            ndvi_std,
            grow_block_mask,
            cfg,
            growable_mask=growable_mask,
            region_profile=region_token,
            boundary_prob=boundary_prob,
        )
    else:
        candidate_mask, grown_added = seeded_grow_into_grass(
            candidate_mask,
            classes,
            ndvi,
            grow_block_mask,
            cfg,
            growable_mask=growable_mask,
            region_profile=region_token,
            boundary_prob=boundary_prob,
        )
    candidate_mask, _blocked_after_grow = _sanitize_added_pixels(
        candidate_mask,
        hard_exclusion_mask,
        debug_stats,
        "step_06_after_grow",
    )
    candidate_mask, road_snap_reject_used, road_edge_overlap_ratio = _maybe_rollback_edge_drift(
        candidate_mask,
        before_grow,
        road_mask,
        cfg,
    )
    logger.debug("postprocess_seeded_grow", pixels_added=grown_added)
    _record_step(debug_masks, debug_stats, "step_06_after_grow", candidate_mask)

    hydro_rescue_used = False
    if (
        bool(getattr(cfg, "BOUNDARY_OUTER_EXPAND_WATER_AWARE", True))
        and bool(getattr(cfg, "HYDRO_FIELD_NEAR_WATER_RESCUE_ENABLED", True))
        and np.any(riparian_soft_mask)
        and np.any(candidate_mask)
    ):
        rescue_iters = min(
            int(getattr(cfg, "BOUNDARY_OUTER_EXPAND_MAX_PX", 4)),
            max(0, int(getattr(cfg, "BOUNDARY_OUTER_EXPAND_NEAR_WATER_MAX_PX", 2))),
        )
        if rescue_iters > 0:
            rescue_ring = binary_dilation(candidate_mask, iterations=rescue_iters) & ~candidate_mask
            hydro_allow = riparian_soft_mask & ~open_water_mask & ~riparian_hard_mask
            hydro_allow &= np.isfinite(ndvi)
            hydro_allow &= ndvi >= float(
                getattr(cfg, "HYDRO_FIELD_NEAR_WATER_MIN_NDVI_MAX", 0.40)
            )
            if ndvi_std is not None:
                hydro_allow &= np.isfinite(ndvi_std)
                hydro_allow &= ndvi_std >= float(
                    getattr(cfg, "HYDRO_FIELD_NEAR_WATER_MIN_NDVI_STD", 0.12)
                )
            if boundary_prob is not None:
                boundary_gate = float(
                    np.clip(
                        np.nanpercentile(np.asarray(boundary_prob, dtype=np.float32), 70),
                        0.35,
                        0.55,
                    )
                )
                hydro_allow &= np.asarray(boundary_prob, dtype=np.float32) <= boundary_gate
            hydro_added = rescue_ring & hydro_allow
            if np.any(hydro_added):
                candidate_mask |= hydro_added
                candidate_mask, _blocked_after_hydro = _sanitize_added_pixels(
                    candidate_mask,
                    hard_exclusion_mask,
                    debug_stats,
                    "step_06b_after_hydro_rescue",
                )
                hydro_rescue_used = True
                _record_step(debug_masks, debug_stats, "step_06b_after_hydro_rescue", candidate_mask)
    _emit_progress(progress_callback, "grow_done")

    before_gap_close = candidate_mask.copy()
    gap_added = 0
    if edge_composite is not None:
        def _gap_close_progress(done: int, total: int) -> None:
            _emit_progress(progress_callback, f"gap_close_scan:{int(done)}:{int(total)}")

        gap_debug = None
        if return_debug_steps:
            _cbg = close_boundary_gaps(
                candidate_mask,
                edge_composite,
                cfg,
                hard_exclusion_mask=hard_exclusion_mask,
                return_debug=True,
                region_profile=region_token,
                progress_callback=_gap_close_progress,
            )
            if isinstance(_cbg, tuple):
                if len(_cbg) == 3:
                    candidate_mask, gap_added, gap_debug = _cbg
                else:
                    candidate_mask, gap_added = _cbg
            else:
                candidate_mask, gap_added = _cbg, 0
        else:
            _cbg = close_boundary_gaps(
                candidate_mask,
                edge_composite,
                cfg,
                hard_exclusion_mask=hard_exclusion_mask,
                region_profile=region_token,
                progress_callback=_gap_close_progress,
            )
            if isinstance(_cbg, tuple):
                if len(_cbg) == 3:
                    candidate_mask, gap_added, _gap_debug_unused = _cbg
                else:
                    candidate_mask, gap_added = _cbg
            else:
                candidate_mask, gap_added = _cbg, 0
        candidate_mask, _blocked_after_gap = _sanitize_added_pixels(
            candidate_mask,
            hard_exclusion_mask,
            debug_stats,
            "step_07_after_gap_close",
        )
        # Road drift check after gap-close
        if road_mask is not None and np.any(road_mask):
            candidate_mask, gap_road_reject, gap_road_ratio = _maybe_rollback_edge_drift(
                candidate_mask,
                before_gap_close,
                road_mask,
                cfg,
            )
            if gap_road_reject:
                road_snap_reject_used = True
                logger.debug("postprocess_gap_close_road_drift_rollback", ratio=round(gap_road_ratio, 4))
        if gap_debug is not None:
            _record_step(
                debug_masks,
                debug_stats,
                "step_07a_gap_holes_skipped_due_to_forbidden",
                gap_debug.get("holes_skipped_due_to_forbidden", np.zeros_like(candidate_mask, dtype=bool)),
            )
    logger.debug("postprocess_gap_close", pixels_added=gap_added)
    _record_step(debug_masks, debug_stats, "step_07_after_gap_close", candidate_mask)
    _emit_progress(progress_callback, "gap_close_done")

    skip_watershed_for_largest = bool(
        getattr(cfg, "FRAMEWORK_SKIP_WATERSHED_FOR_LARGEST", True)
    )
    small_refine_max_px = max(1, int(5_000 / cfg.POST_PX_AREA_M2))
    if skip_watershed_for_largest:
        small_components = select_small_components(candidate_mask, small_refine_max_px)
        stable_mask = candidate_mask & ~small_components
    else:
        small_components = candidate_mask.copy()
        stable_mask = np.zeros_like(candidate_mask, dtype=bool)
    _record_step(debug_masks, debug_stats, "step_07b_small_components", small_components)

    # Build allow_infill_mask for smart infill: only fill holes where pixels are crop-like
    allow_infill_mask = (classes == CROP).copy()
    if phenology_masks is not None:
        allow_infill_mask |= phenology_masks["is_field"]
    # Exclude forest/water/bare from infill
    if phenology_masks is not None:
        allow_infill_mask &= ~phenology_masks["is_forest"]
        allow_infill_mask &= ~phenology_masks.get("is_bare", np.zeros_like(candidate_mask, dtype=bool))

    small_components, infill_added = infill_field_holes(
        small_components,
        hard_exclusion_mask,
        cfg,
        allow_infill_mask=allow_infill_mask if ndvi_std is not None else None,
        progress_callback=(
            lambda done, total: _emit_progress(
                progress_callback,
                f"infill_scan:{int(done)}:{int(total)}",
            )
        ),
    )
    candidate_mask = stable_mask | small_components
    candidate_mask, _blocked_after_infill = _sanitize_added_pixels(
        candidate_mask,
        hard_exclusion_mask,
        debug_stats,
        "step_08_after_infill",
    )
    logger.debug("postprocess_infill", pixels_added=infill_added)
    _record_step(debug_masks, debug_stats, "step_08_after_infill", candidate_mask)
    _emit_progress(progress_callback, "infill_done")

    bridge_added = 0
    bridge_pairs = 0
    if (
        region_token == "south_recall"
        and bool(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_ENABLED", True))
        and np.any(small_components)
    ):
        small_components, bridge_added, bridge_pairs = _bridge_near_components(
            small_components,
            ndvi,
            barrier_mask,
            boundary_prob,
            cfg,
            progress_callback=(
                lambda done, total: _emit_progress(
                    progress_callback,
                    f"bridge_scan:{int(done)}:{int(total)}",
                )
            ),
        )
        if bridge_added > 0:
            small_components &= ~hard_exclusion_mask
            candidate_mask = stable_mask | small_components
            candidate_mask, _blocked_after_bridge = _sanitize_added_pixels(
                candidate_mask,
                hard_exclusion_mask,
                debug_stats,
                "step_08b_after_component_bridge",
            )
            region_actions.append("south_bridge")
            _record_step(debug_masks, debug_stats, "step_08b_after_component_bridge", candidate_mask)
    _emit_progress(progress_callback, "bridge_done")

    components_before = count_components(small_components)
    if boundary_prob is not None and ndvi_std is not None and np.any(small_components):
        labeled_small, _ = nd_label(small_components.astype(bool))
        try:
            def _merge_progress(done: int, total: int) -> None:
                _emit_progress(progress_callback, f"merge_scan:{int(done)}:{int(total)}")

            labeled_small = hierarchical_merge(
                labeled_small,
                boundary_prob,
                ndvi,
                ndvi_std,
                barrier_mask,
                cfg,
                region_profile=region_token,
                progress_callback=_merge_progress,
            )
        except Exception as exc:
            logger.warning("hierarchical_merge_small_failed", error=str(exc))
        small_components = labeled_small > 0
    else:
        def _merge_progress(done: int, total: int) -> None:
            _emit_progress(progress_callback, f"merge_scan:{int(done)}:{int(total)}")

        small_components = merge_crop_regions(
            small_components,
            ndvi,
            barrier_mask,
            cfg,
            region_profile=region_token,
            progress_callback=_merge_progress,
            boundary_prob=boundary_prob,
        )
    small_components &= ~hard_exclusion_mask
    candidate_mask = stable_mask | small_components
    candidate_mask, _blocked_after_merge = _sanitize_added_pixels(
        candidate_mask,
        hard_exclusion_mask,
        debug_stats,
        "step_09_after_merge",
    )
    components_after = count_components(small_components)
    logger.debug("postprocess_components_before_merge", component_count=components_before)
    logger.debug("postprocess_components_after_merge", component_count=components_after)
    _record_step(debug_masks, debug_stats, "step_09_after_merge", candidate_mask)
    _emit_progress(progress_callback, "merge_done")

    before_watershed = candidate_mask.copy()
    skip_watershed = False
    if edge_composite is not None and np.any(small_components):
        small_edge_mean = (
            float(np.nanmean(edge_composite[small_components]))
            if np.any(small_components)
            else 1.0
        )
        if (
            region_token == "south_recall"
            and bool(getattr(cfg, "SOUTH_SKIP_WATERSHED_LARGE_COMPONENTS", True))
        ):
            weak_edge_threshold = max(0.02, float(getattr(cfg, "EDGE_WEAK_THRESHOLD", 0.1)) * 0.6)
            if small_edge_mean <= weak_edge_threshold:
                skip_watershed = True
                region_actions.append("south_skip_watershed_large_component")
        elif (
            region_token == "north_boundary"
            and bool(getattr(cfg, "NORTH_WATERSHED_WEAKEN_ENABLED", True))
        ):
            weak_edge_threshold = max(0.02, float(getattr(cfg, "EDGE_WEAK_THRESHOLD", 0.1)) * 0.5)
            skip_watershed = small_edge_mean <= weak_edge_threshold
    if edge_composite is not None and np.any(small_components) and not skip_watershed:
        small_components, _labels = watershed_field_segmentation(
            small_components,
            ndvi,
            edge_composite,
            cfg,
        )
        small_components &= ~hard_exclusion_mask
        candidate_mask = stable_mask | small_components
        candidate_mask, _blocked_after_watershed = _sanitize_added_pixels(
            candidate_mask,
            hard_exclusion_mask,
            debug_stats,
            "step_10_after_watershed",
        )
    else:
        candidate_mask, _blocked_after_watershed = _sanitize_added_pixels(
            candidate_mask,
            hard_exclusion_mask,
            debug_stats,
            "step_10_after_watershed",
        )
    # Road drift check after watershed
    if road_mask is not None and np.any(road_mask):
        candidate_mask, ws_road_reject, ws_road_ratio = _maybe_rollback_edge_drift(
            candidate_mask,
            before_watershed,
            road_mask,
            cfg,
        )
        if ws_road_reject:
            road_snap_reject_used = True
            logger.debug("postprocess_watershed_road_drift_rollback", ratio=round(ws_road_ratio, 4))
    _record_step(debug_masks, debug_stats, "step_10_after_watershed", candidate_mask)

    water_edge_overlap_ratio = _edge_overlap_ratio(
        candidate_mask,
        open_water_mask | riparian_hard_mask,
        buffer_px=1,
    )
    boundary_shift_to_road_ratio = road_edge_overlap_ratio

    min_field_ha = float(cfg.POST_MIN_FIELD_AREA_HA)
    if region_token == "south_recall":
        min_field_ha = float(getattr(cfg, "SOUTH_POST_MIN_FIELD_AREA_HA", min_field_ha))
    min_px = max(1, int(min_field_ha * 10_000 / cfg.POST_PX_AREA_M2))
    small_removed = count_small_components(candidate_mask, min_px)
    candidate_mask = remove_small_components(candidate_mask, min_px)
    logger.debug("postprocess_small_objects_removed", object_count=small_removed)
    logger.debug(
        "postprocess_final_crop_pixels",
        crop_pixels=int(np.count_nonzero(candidate_mask)),
    )
    _record_step(debug_masks, debug_stats, "step_11_after_small_remove", candidate_mask)
    _emit_progress(progress_callback, "watershed_done")
    if debug_stats is not None:
        components_after_clean = int(debug_stats.get("step_05_after_clean", {}).get("components", 0))
        components_after_grow = int(debug_stats.get("step_06_after_grow", {}).get("components", 0))
        components_after_gap_close = int(debug_stats.get("step_07_after_gap_close", {}).get("components", 0))
        components_after_infill = int(debug_stats.get("step_08_after_infill", {}).get("components", 0))
        components_after_watershed = int(debug_stats.get("step_10_after_watershed", {}).get("components", 0))
        split_risk_score = float(components_after_watershed / max(components_after, 1))
        shrink_risk_score = float(
            max(0.0, 1.0 - (np.count_nonzero(candidate_mask) / max(initial_candidate_pixels, 1)))
        )
        debug_stats["summary"] = {
            "barrier_pixels": int(np.count_nonzero(barrier_mask)),
            "barrier_ratio": float(np.count_nonzero(barrier_mask) / barrier_mask.size)
            if barrier_mask.size
            else 0.0,
            "hard_exclusion_pixels": int(np.count_nonzero(hard_exclusion_mask)),
            "grow_block_pixels": int(np.count_nonzero(grow_block_mask)),
            "road_pixels": int(np.count_nonzero(road_mask)),
            "road_removed_ratio": float(road_removed_ratio),
            "road_barrier_retry_used": int(road_barrier_retry_used),
            "road_snap_reject_used": int(road_snap_reject_used),
            "road_edge_overlap_ratio": float(road_edge_overlap_ratio),
            "boundary_shift_to_road_ratio": float(boundary_shift_to_road_ratio),
            "forest_pixels": int(np.count_nonzero(forest_mask)),
            "water_pixels": int(np.count_nonzero(is_water_mask)),
            "open_water_pixels": int(np.count_nonzero(open_water_mask)),
            "seasonal_wet_pixels": int(np.count_nonzero(seasonal_wet_mask)),
            "riparian_soft_pixels": int(np.count_nonzero(riparian_soft_mask)),
            "riparian_hard_pixels": int(np.count_nonzero(riparian_hard_mask)),
            "water_edge_overlap_ratio": float(water_edge_overlap_ratio),
            "hydro_rescue_used": int(hydro_rescue_used),
            "builtup_pixels": int(np.count_nonzero(builtup_mask)),
            "worldcover_weak_prior_pixels": int(np.count_nonzero(wc_prior_mask)),
            "wctree_enabled": int(wc_tree_hard_enabled),
            "wctree_pixels": int(np.count_nonzero(wc_tree_hard)),
            "wctree_removed_pixels": int(wc_tree_removed),
            "blocked_after_barrier_pixels": int(blocked_after_barrier),
            "grass_barrier_pixels": int(np.count_nonzero(grass_barrier_mask)),
            "field_candidate_pixels": int(np.count_nonzero(field_candidate)),
            "crop_soft_pixels": int(np.count_nonzero(crop_soft_mask)),
                "boundary_field_pixels": int(np.count_nonzero(boundary_field_mask)),
                "recovery_boundary_anchor_pixels": int(np.count_nonzero(recovery_boundary_anchor)),
                "boundary_regions": int(boundary_regions),
            "small_refine_pixels": int(np.count_nonzero(small_components)),
            "min_px": int(min_px),
            "components_before_merge": int(components_before),
            "components_after_merge": int(components_after),
            "components_after_clean": components_after_clean,
            "components_after_grow": components_after_grow,
            "components_after_gap_close": components_after_gap_close,
            "components_after_infill": components_after_infill,
            "components_after_watershed": components_after_watershed,
            "split_risk_score": split_risk_score,
            "shrink_risk_score": shrink_risk_score,
            "region_profile_applied": region_token or "balanced",
            "region_profile_actions": list(region_actions),
            "bridge_added_pixels": int(bridge_added),
            "bridge_pairs": int(bridge_pairs),
        }

    candidate_mask = candidate_mask.astype(bool)
    _emit_progress(progress_callback, "finalize_done")
    if capture_debug_stats or return_candidate_masks:
        payload = {
            "masks": debug_masks or {},
            "stats": debug_stats or {},
        }
        if return_candidate_masks:
            payload["candidate_masks"] = {
                "field_candidate": field_candidate.astype(np.uint8, copy=False),
                "crop_soft_mask": crop_soft_mask.astype(np.uint8, copy=False),
                    "boundary_field_mask": boundary_field_mask.astype(np.uint8, copy=False),
                    "recovery_boundary_anchor": recovery_boundary_anchor.astype(np.uint8, copy=False),
                    "legacy_seed_mask": legacy_seed_mask.astype(np.uint8, copy=False),
                    "final_candidate_mask": candidate_mask.astype(np.uint8, copy=False),
                    "barrier_mask": barrier_mask.astype(np.uint8, copy=False),
                }
        return candidate_mask, payload
    return candidate_mask
