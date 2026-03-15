"""Region growing and graph merge for fragmented crop masks."""
from __future__ import annotations

import time
from typing import Callable

import networkx as nx
import numpy as np
from scipy.ndimage import find_objects, label as nd_label
from skimage.morphology import closing, dilation, disk

from core.logging import get_logger

logger = get_logger(__name__)


def _emit_progress(
    progress_callback: Callable[[int, int], None] | None,
    *,
    done: int,
    total: int,
    last_emit_at: float,
    force: bool = False,
    min_interval_s: float = 2.0,
) -> float:
    if progress_callback is None or total <= 0:
        return last_emit_at
    now = time.monotonic()
    if not force and done > 1 and done < total and (now - last_emit_at) < float(min_interval_s):
        return last_emit_at
    try:
        progress_callback(int(done), int(total))
    except Exception as exc:
        logger.debug(
            "region_merge_progress_callback_failed",
            done=int(done),
            total=int(total),
            error=str(exc),
        )
    return now


def _expand_slice(
    row_slice: slice,
    col_slice: slice,
    shape: tuple[int, int],
    pad: int,
) -> tuple[slice, slice]:
    """Expand a 2D slice by a pixel padding while staying in bounds."""
    height, width = shape
    return (
        slice(max(0, row_slice.start - pad), min(height, row_slice.stop + pad)),
        slice(max(0, col_slice.start - pad), min(width, col_slice.stop + pad)),
    )


def merge_crop_regions(
    candidate_mask: np.ndarray,
    ndvi: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
    region_profile: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    boundary_prob: np.ndarray | None = None,
) -> np.ndarray:
    """Merge nearby crop fragments with similar NDVI into larger regions.

    This implementation avoids storing a full-sized boolean mask per component,
    which explodes memory on 2048x2048 tiles.
    """
    if candidate_mask.ndim != 2:
        raise ValueError(f"candidate_mask must be 2D, got shape={candidate_mask.shape}")
    if ndvi.shape != candidate_mask.shape:
        raise ValueError(
            "ndvi must match candidate_mask shape, "
            f"got ndvi={ndvi.shape}, candidate_mask={candidate_mask.shape}"
        )
    if barrier_mask.shape != candidate_mask.shape:
        raise ValueError(
            "barrier_mask must match candidate_mask shape, "
            f"got barrier_mask={barrier_mask.shape}, candidate_mask={candidate_mask.shape}"
        )

    candidate_mask = candidate_mask.astype(bool, copy=False)
    barrier_mask = barrier_mask.astype(bool, copy=False)
    if not np.any(candidate_mask):
        return np.zeros_like(candidate_mask, dtype=bool)

    labeled, n_labels = nd_label(candidate_mask)
    if n_labels <= 1:
        return candidate_mask.copy()

    token = str(region_profile or "").strip().lower()
    radius = max(0, int(cfg.POST_MERGE_BUFFER_PX))
    selem = disk(radius) if radius > 0 else None
    max_components = max(1, int(getattr(cfg, "POST_MERGE_MAX_COMPONENTS", 1500)))
    overlap_min = float(getattr(cfg, "POST_MERGE_OVERLAP_MIN", 0.30))
    ndvi_diff_max = float(getattr(cfg, "POST_MERGE_NDVI_DIFF_MAX", 0.12))
    if token == "south_recall":
        overlap_min = min(
            overlap_min,
            float(getattr(cfg, "SOUTH_MERGE_MIN_OVERLAP_RATIO", 0.03)),
        )
        ndvi_diff_max = max(
            ndvi_diff_max,
            float(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_MAX_NDVI_DIFF", 0.08)),
        )

    if n_labels > max_components:
        logger.warning(
            "postprocess_region_merge_skipped",
            component_count=n_labels,
            max_components=max_components,
        )
        return candidate_mask.copy()

    flat_labels = labeled.ravel()
    finite_ndvi = np.isfinite(ndvi)
    label_counts = np.bincount(flat_labels, minlength=n_labels + 1)
    ndvi_counts = np.bincount(flat_labels[finite_ndvi.ravel()], minlength=n_labels + 1)
    ndvi_sums = np.bincount(
        flat_labels[finite_ndvi.ravel()],
        weights=ndvi.ravel()[finite_ndvi.ravel()],
        minlength=n_labels + 1,
    )
    ndvi_means = np.zeros(n_labels + 1, dtype=np.float32)
    valid_mean = ndvi_counts > 0
    ndvi_means[valid_mean] = ndvi_sums[valid_mean] / ndvi_counts[valid_mean]

    objects = find_objects(labeled, max_label=n_labels)
    expanded_slices: dict[int, tuple[slice, slice]] = {}
    label_slices: dict[int, tuple[slice, slice]] = {}

    for component_id, obj_slice in enumerate(objects, start=1):
        if obj_slice is None:
            continue
        row_slice, col_slice = obj_slice
        label_slices[component_id] = (row_slice, col_slice)
        expanded_slices[component_id] = _expand_slice(
            row_slice,
            col_slice,
            candidate_mask.shape,
            radius,
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(1, n_labels + 1))

    component_ids = [component_id for component_id in range(1, n_labels + 1) if component_id in label_slices]
    total_steps = max(1, len(component_ids) * 2)
    last_emit_at = 0.0

    for index, component_id in enumerate(component_ids, start=1):
        try:
            expanded_row, expanded_col = expanded_slices[component_id]
            neighborhood = labeled[expanded_row, expanded_col]
            neighbor_ids = np.unique(neighborhood)
            neighbor_ids = neighbor_ids[(neighbor_ids > component_id)]
            if neighbor_ids.size == 0:
                continue

            area_i = int(label_counts[component_id])
            if area_i <= 0:
                continue

            for other_id in neighbor_ids.tolist():
                if other_id not in expanded_slices:
                    continue

                pair_row = slice(
                    min(expanded_slices[component_id][0].start, expanded_slices[other_id][0].start),
                    max(expanded_slices[component_id][0].stop, expanded_slices[other_id][0].stop),
                )
                pair_col = slice(
                    min(expanded_slices[component_id][1].start, expanded_slices[other_id][1].start),
                    max(expanded_slices[component_id][1].stop, expanded_slices[other_id][1].stop),
                )

                local_labels = labeled[pair_row, pair_col]
                mask_i_local = local_labels == component_id
                mask_j_local = local_labels == other_id

                if radius > 0:
                    dilated_i_local = dilation(mask_i_local, selem)
                    dilated_j_local = dilation(mask_j_local, selem)
                else:
                    dilated_i_local = mask_i_local
                    dilated_j_local = mask_j_local

                overlap = int(np.count_nonzero(dilated_i_local & mask_j_local))
                if overlap <= 0:
                    continue

                bridge = dilated_i_local & dilated_j_local
                bridge_sum = int(np.count_nonzero(bridge))
                if bridge_sum <= 0:
                    continue

                barrier_ratio = float(
                    np.count_nonzero(bridge & barrier_mask[pair_row, pair_col]) / bridge_sum
                )
                if barrier_ratio > cfg.POST_MERGE_BARRIER_RATIO:
                    continue

                min_area = min(area_i, int(label_counts[other_id]))
                if min_area <= 0:
                    continue

                overlap_ratio = overlap / min_area
                ndvi_diff = abs(float(ndvi_means[component_id]) - float(ndvi_means[other_id]))

                # Check boundary strength between regions to prevent merging across field edges
                boundary_ok = True
                if boundary_prob is not None:
                    bridge_boundary = boundary_prob[pair_row, pair_col][bridge]
                    if bridge_boundary.size > 0:
                        mean_bp = float(np.nanmean(bridge_boundary))
                        merge_bp_thresh = float(getattr(cfg, "MERGE_BOUNDARY_THRESH", 0.25))
                        if mean_bp > merge_bp_thresh:
                            boundary_ok = False

                if overlap_ratio >= overlap_min and ndvi_diff <= ndvi_diff_max and boundary_ok:
                    graph.add_edge(component_id, other_id)
        finally:
            last_emit_at = _emit_progress(
                progress_callback,
                done=index,
                total=total_steps,
                last_emit_at=last_emit_at,
                force=index == 1 or index == len(component_ids),
            )

    merged = np.zeros_like(candidate_mask, dtype=bool)
    connected_components = [list(component) for component in nx.connected_components(graph)]
    for index, component in enumerate(connected_components, start=1):
        component = list(component)
        if not component:
            continue

        row_start = min(label_slices[idx][0].start for idx in component if idx in label_slices)
        row_stop = max(label_slices[idx][0].stop for idx in component if idx in label_slices)
        col_start = min(label_slices[idx][1].start for idx in component if idx in label_slices)
        col_stop = max(label_slices[idx][1].stop for idx in component if idx in label_slices)

        if radius > 0:
            row_start = max(0, row_start - radius)
            row_stop = min(candidate_mask.shape[0], row_stop + radius)
            col_start = max(0, col_start - radius)
            col_stop = min(candidate_mask.shape[1], col_stop + radius)

        local_labels = labeled[row_start:row_stop, col_start:col_stop]
        local_mask = np.isin(local_labels, component)
        local_barrier = barrier_mask[row_start:row_stop, col_start:col_stop]

        if len(component) > 1 and radius > 0:
            local_mask = closing(local_mask, selem)
        local_mask &= ~local_barrier

        merged[row_start:row_stop, col_start:col_stop] |= local_mask

        last_emit_at = _emit_progress(
            progress_callback,
            done=len(component_ids) + index,
            total=total_steps,
            last_emit_at=last_emit_at,
            force=index == len(connected_components),
        )

    _emit_progress(
        progress_callback,
        done=total_steps,
        total=total_steps,
        last_emit_at=last_emit_at,
        force=True,
    )

    return merged


def _premerge_tiny_regions(
    labeled: np.ndarray,
    ndvi: np.ndarray,
    barrier_mask: np.ndarray,
    target_count: int,
    cfg,
) -> np.ndarray:
    """Merge regions smaller than a threshold into their nearest neighbour.

    This reduces the total region count so that hierarchical_merge can proceed
    instead of being skipped entirely.  The merge only happens across non-barrier
    boundaries and with similar NDVI.
    """
    result = labeled.astype(np.int32, copy=True)
    max_label = int(result.max())
    if max_label < 2:
        return result

    flat = result.ravel()
    counts = np.bincount(flat, minlength=max_label + 1)

    finite_ndvi = np.isfinite(ndvi)
    ndvi_counts = np.bincount(flat[finite_ndvi.ravel()], minlength=max_label + 1).astype(np.float64)
    ndvi_sums = np.bincount(
        flat[finite_ndvi.ravel()],
        weights=ndvi.ravel()[finite_ndvi.ravel()],
        minlength=max_label + 1,
    )
    region_ndvi = np.zeros(max_label + 1, dtype=np.float32)
    valid = ndvi_counts > 0
    region_ndvi[valid] = (ndvi_sums[valid] / ndvi_counts[valid]).astype(np.float32)

    # Determine size threshold: merge everything below this until we have <= target_count
    n_regions = int(np.count_nonzero(counts[1:] > 0))
    min_px = int(getattr(cfg, "POST_PREMERGE_MIN_PX", 10))
    ndvi_diff_max = float(getattr(cfg, "POST_MERGE_NDVI_DIFF_MAX", 0.12))
    selem = disk(1)

    # Sort tiny regions from smallest to largest
    tiny_ids = [
        int(lid) for lid in range(1, max_label + 1)
        if 0 < counts[lid] < min_px
    ]
    tiny_ids.sort(key=lambda lid: int(counts[lid]))

    objects = find_objects(result, max_label=max_label)
    merged_count = 0

    for lid in tiny_ids:
        if n_regions <= target_count:
            break
        obj_slice = objects[lid - 1]
        if obj_slice is None:
            continue
        row_s, col_s = obj_slice
        # Expand by 1 pixel
        r0 = max(0, row_s.start - 1)
        r1 = min(result.shape[0], row_s.stop + 1)
        c0 = max(0, col_s.start - 1)
        c1 = min(result.shape[1], col_s.stop + 1)

        local = result[r0:r1, c0:c1]
        mask_i = local == lid
        if not np.any(mask_i):
            continue

        dilated = dilation(mask_i, selem)
        neighbor_labels = np.unique(local[dilated & ~mask_i])
        neighbor_labels = neighbor_labels[neighbor_labels > 0]
        neighbor_labels = neighbor_labels[neighbor_labels != lid]

        if neighbor_labels.size == 0:
            continue

        # Pick the neighbour with closest NDVI and no barrier between
        local_barrier = barrier_mask[r0:r1, c0:c1]
        best_neighbour = None
        best_diff = 999.0
        for nid in neighbor_labels.tolist():
            nid = int(nid)
            bridge = dilated & (local == nid)
            if np.any(bridge & local_barrier):
                continue
            diff = abs(float(region_ndvi[lid]) - float(region_ndvi[nid]))
            if diff < best_diff:
                best_diff = diff
                best_neighbour = nid

        if best_neighbour is not None and best_diff <= ndvi_diff_max * 1.5:
            result[result == lid] = best_neighbour
            # Update stats
            counts[best_neighbour] += counts[lid]
            counts[lid] = 0
            n_regions -= 1
            merged_count += 1

    if merged_count > 0:
        logger.info(
            "premerge_tiny_regions_done",
            merged=merged_count,
            remaining_regions=n_regions,
        )
    return result


def hierarchical_merge(
    labeled: np.ndarray,
    boundary_prob: np.ndarray,
    max_ndvi: np.ndarray,
    ndvi_std: np.ndarray,
    barrier_mask: np.ndarray,
    cfg,
    region_profile: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Merge over-segmented labels when their shared boundary is weak."""
    if not (
        labeled.shape
        == boundary_prob.shape
        == max_ndvi.shape
        == ndvi_std.shape
        == barrier_mask.shape
    ):
        raise ValueError("all hierarchical_merge inputs must share the same shape")

    result = labeled.astype(np.int32, copy=True)
    if result.max() < 2:
        return result

    token = str(region_profile or "").strip().lower()
    merge_boundary_thresh = float(getattr(cfg, "MERGE_BOUNDARY_THRESH", 0.30))
    merge_ndvi_diff_max = float(getattr(cfg, "POST_MERGE_NDVI_DIFF_MAX", 0.12))
    merge_barrier_ratio = float(getattr(cfg, "POST_MERGE_BARRIER_RATIO", 0.20))
    if token == "south_recall":
        merge_boundary_thresh = min(
            merge_boundary_thresh + 0.05,
            float(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_MAX_BOUNDARY_PROB", 0.45)),
        )
        merge_ndvi_diff_max = max(
            merge_ndvi_diff_max,
            float(getattr(cfg, "SOUTH_COMPONENT_BRIDGE_MAX_NDVI_DIFF", 0.08)),
        )
    elif token == "north_boundary":
        merge_boundary_thresh = float(
            getattr(cfg, "NORTH_MERGE_BOUNDARY_THRESH", merge_boundary_thresh)
        )

    region_ids = [int(v) for v in np.unique(result) if v > 0]
    if len(region_ids) < 2:
        return result
    max_regions = max(
        1,
        int(getattr(cfg, "POST_MERGE_MAX_COMPONENTS", 600)),
    )
    if len(region_ids) > max_regions:
        # Progressive merge: pre-merge tiny regions (<10px) into nearest
        # neighbour to reduce count below the limit, instead of skipping entirely.
        logger.info(
            "hierarchical_merge_progressive_premerge",
            region_count=len(region_ids),
            max_regions=max_regions,
        )
        result = _premerge_tiny_regions(result, max_ndvi, barrier_mask, max_regions, cfg)
        region_ids = [int(v) for v in np.unique(result) if v > 0]
        if len(region_ids) < 2:
            return result
        if len(region_ids) > max_regions:
            logger.warning(
                "hierarchical_merge_still_over_limit",
                region_count=len(region_ids),
                max_regions=max_regions,
            )
            # Even after pre-merge, still over limit — process the largest
            # regions only instead of skipping entirely.
            label_counts = np.bincount(result.ravel())
            # Keep only top max_regions labels by area
            top_labels = set(
                np.argsort(label_counts)[-max_regions:]
            )
            top_labels.discard(0)
            mask_keep = np.isin(result, list(top_labels))
            result[~mask_keep] = 0
            region_ids = sorted(top_labels)

    max_label = int(result.max())
    flat_labels = result.ravel()
    finite_ndvi = np.isfinite(max_ndvi)
    ndvi_counts = np.bincount(flat_labels[finite_ndvi.ravel()], minlength=max_label + 1).astype(np.float32)
    ndvi_sums = np.bincount(
        flat_labels[finite_ndvi.ravel()],
        weights=max_ndvi.ravel()[finite_ndvi.ravel()],
        minlength=max_label + 1,
    )
    region_ndvi = np.zeros(max_label + 1, dtype=np.float32)
    np.divide(ndvi_sums, np.maximum(ndvi_counts, 1.0), out=region_ndvi, where=ndvi_counts > 0)

    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(region_ids)
    selem = disk(1)
    objects = find_objects(result, max_label=max_label)
    label_slices: dict[int, tuple[slice, slice]] = {}
    expanded_slices: dict[int, tuple[slice, slice]] = {}
    for region_id, obj_slice in enumerate(objects, start=1):
        if obj_slice is None:
            continue
        row_slice, col_slice = obj_slice
        label_slices[region_id] = (row_slice, col_slice)
        expanded_slices[region_id] = _expand_slice(row_slice, col_slice, result.shape, 1)

    total_steps = max(1, len(region_ids) * 2)
    last_emit_at = 0.0
    for index, region_id in enumerate(region_ids, start=1):
        try:
            if region_id not in label_slices:
                continue

            expanded_row, expanded_col = expanded_slices[region_id]
            local_labels = result[expanded_row, expanded_col]
            mask_i_local = local_labels == region_id
            if not np.any(mask_i_local):
                continue

            dilated_i_local = dilation(mask_i_local, selem)
            neighbor_ids = np.unique(local_labels[dilated_i_local & ~mask_i_local])
            neighbor_ids = neighbor_ids[neighbor_ids > region_id]

            for other_id in neighbor_ids.tolist():
                other_id = int(other_id)
                if other_id not in expanded_slices:
                    continue

                pair_row = slice(
                    min(expanded_slices[region_id][0].start, expanded_slices[other_id][0].start),
                    max(expanded_slices[region_id][0].stop, expanded_slices[other_id][0].stop),
                )
                pair_col = slice(
                    min(expanded_slices[region_id][1].start, expanded_slices[other_id][1].start),
                    max(expanded_slices[region_id][1].stop, expanded_slices[other_id][1].stop),
                )

                pair_labels = result[pair_row, pair_col]
                mask_i_pair = pair_labels == region_id
                mask_j_pair = pair_labels == other_id
                shared = dilation(mask_i_pair, selem) & mask_j_pair
                if np.count_nonzero(shared) < 2:
                    continue

                mean_bp = float(np.nanmean(boundary_prob[pair_row, pair_col][shared]))
                barrier_r = float(np.mean(barrier_mask[pair_row, pair_col][shared]))
                if barrier_r > merge_barrier_ratio:
                    continue

                ndvi_diff = abs(float(region_ndvi[region_id]) - float(region_ndvi[other_id]))
                if mean_bp < merge_boundary_thresh and ndvi_diff < merge_ndvi_diff_max:
                    merge_graph.add_edge(region_id, other_id)
        finally:
            last_emit_at = _emit_progress(
                progress_callback,
                done=index,
                total=total_steps,
                last_emit_at=last_emit_at,
                force=index == 1 or index == len(region_ids),
            )

    connected_components = [sorted(int(v) for v in component) for component in nx.connected_components(merge_graph)]
    for index, component in enumerate(connected_components, start=1):
        component = sorted(int(v) for v in component)
        if len(component) < 2:
            last_emit_at = _emit_progress(
                progress_callback,
                done=len(region_ids) + index,
                total=total_steps,
                last_emit_at=last_emit_at,
                force=index == len(connected_components),
            )
            continue
        target_id = component[0]
        for region_id in component[1:]:
            result[result == region_id] = target_id

        last_emit_at = _emit_progress(
            progress_callback,
            done=len(region_ids) + index,
            total=total_steps,
            last_emit_at=last_emit_at,
            force=index == len(connected_components),
        )

    _emit_progress(
        progress_callback,
        done=total_steps,
        total=total_steps,
        last_emit_at=last_emit_at,
        force=True,
    )

    return result
