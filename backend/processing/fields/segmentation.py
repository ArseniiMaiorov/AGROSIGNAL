"""Watershed segmentation of agricultural fields."""
from typing import Callable

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from processing.fields.owt import oriented_watershed

ProgressCallback = Callable[[str, int, int], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int,
) -> None:
    if progress_callback is None:
        return
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    progress_callback(str(stage), safe_completed, safe_total)


def _connected_components(mask: np.ndarray) -> np.ndarray:
    labels, _ = ndimage.label(mask.astype(bool))
    return labels.astype(np.int32, copy=False)


def _region_means(labels: np.ndarray, values: np.ndarray | None) -> dict[int, float]:
    if values is None:
        return {}
    out: dict[int, float] = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label <= 0:
            continue
        region_values = values[labels == label]
        if region_values.size == 0:
            out[int(label)] = 0.0
            continue
        finite = region_values[np.isfinite(region_values)]
        out[int(label)] = float(np.mean(finite)) if finite.size else 0.0
    return out


def _collect_zero_boundary_pairs(
    labels: np.ndarray,
    boundary_prob: np.ndarray | None,
    edge_score: np.ndarray | None,
    ndvi_means: dict[int, float],
    ndvi_std_means: dict[int, float],
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, float | int]]:
    stats: dict[tuple[int, int], dict[str, float | int]] = {}
    boundary_pixels = np.argwhere(labels == 0)
    if boundary_pixels.size == 0:
        return []

    total_pixels = int(boundary_pixels.shape[0])
    _emit_progress(progress_callback, "pair_scan", 0, total_pixels)
    for pixel_index, (row, col) in enumerate(boundary_pixels, start=1):
        y0 = max(0, int(row) - 1)
        y1 = min(labels.shape[0], int(row) + 2)
        x0 = max(0, int(col) - 1)
        x1 = min(labels.shape[1], int(col) + 2)
        neighborhood = labels[y0:y1, x0:x1]
        unique = np.unique(neighborhood)
        unique = unique[unique > 0]
        if unique.size < 2:
            continue

        boundary_value = (
            float(boundary_prob[row, col])
            if boundary_prob is not None and np.isfinite(boundary_prob[row, col])
            else 0.0
        )
        edge_value = (
            float(edge_score[row, col])
            if edge_score is not None and np.isfinite(edge_score[row, col])
            else 0.0
        )

        for left_index in range(len(unique) - 1):
            left = int(unique[left_index])
            for right_index in range(left_index + 1, len(unique)):
                right = int(unique[right_index])
                key = (left, right) if left < right else (right, left)
                item = stats.setdefault(
                    key,
                    {
                        "left": key[0],
                        "right": key[1],
                        "shared_boundary_px": 0,
                        "boundary_prob_sum": 0.0,
                        "edge_score_sum": 0.0,
                    },
                )
                item["shared_boundary_px"] = int(item["shared_boundary_px"]) + 1
                item["boundary_prob_sum"] = float(item["boundary_prob_sum"]) + boundary_value
                item["edge_score_sum"] = float(item["edge_score_sum"]) + edge_value
        if pixel_index % 2048 == 0 or pixel_index == total_pixels:
            _emit_progress(progress_callback, "pair_scan", pixel_index, total_pixels)

    pairs: list[dict[str, float | int]] = []
    for item in stats.values():
        shared = max(int(item["shared_boundary_px"]), 1)
        left = int(item["left"])
        right = int(item["right"])
        boundary_mean = float(item["boundary_prob_sum"]) / shared
        edge_mean = float(item["edge_score_sum"]) / shared
        ndvi_delta = abs(float(ndvi_means.get(left, 0.0)) - float(ndvi_means.get(right, 0.0)))
        ndvi_std_delta = abs(
            float(ndvi_std_means.get(left, 0.0)) - float(ndvi_std_means.get(right, 0.0))
        )
        split_score = (
            0.35 * boundary_mean
            + 0.25 * edge_mean
            + 0.20 * ndvi_delta
            + 0.20 * ndvi_std_delta
        )
        pairs.append(
            {
                "left": left,
                "right": right,
                "shared_boundary_px": shared,
                "boundary_prob_mean": boundary_mean,
                "edge_score_mean": edge_mean,
                "ndvi_delta": ndvi_delta,
                "ndvi_std_delta": ndvi_std_delta,
                "feature_delta": max(ndvi_delta, ndvi_std_delta),
                "split_score": split_score,
            }
        )
    return pairs


def _qualifies_as_strong_split(pair: dict[str, float | int], cfg) -> bool:
    return (
        int(pair["shared_boundary_px"]) >= int(getattr(cfg, "SELECTIVE_SPLIT_MIN_SHARED_BOUNDARY_PX", 24))
        and float(pair["boundary_prob_mean"]) >= float(getattr(cfg, "SELECTIVE_SPLIT_MIN_BOUNDARY_PROB", 0.58))
        and float(pair["edge_score_mean"]) >= float(getattr(cfg, "SELECTIVE_SPLIT_MIN_EDGE_SCORE", 0.22))
        and float(pair["feature_delta"]) >= float(getattr(cfg, "SELECTIVE_SPLIT_MIN_FEATURE_DELTA", 0.12))
        and float(pair["split_score"]) >= float(getattr(cfg, "SELECTIVE_SPLIT_SCORE_MIN", 0.62))
    )


def _merge_weak_watershed_pairs(
    labels: np.ndarray,
    pair_stats: list[dict[str, float | int]],
    *,
    strong_pairs: set[tuple[int, int]],
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    positive = np.unique(labels)
    positive = positive[positive > 0]
    if positive.size == 0:
        return labels.astype(np.int32, copy=False)

    parent = {int(label): int(label) for label in positive.tolist()}

    def find(label: int) -> int:
        root = parent[label]
        while root != parent[root]:
            parent[root] = parent[parent[root]]
            root = parent[root]
        while label != root:
            next_label = parent[label]
            parent[label] = root
            label = next_label
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    total_pairs = max(len(pair_stats), 1)
    _emit_progress(progress_callback, "pair_merge", 0, total_pairs)
    for pair_index, pair in enumerate(pair_stats, start=1):
        key = tuple(sorted((int(pair["left"]), int(pair["right"]))))
        if key in strong_pairs:
            continue
        union(key[0], key[1])
        if pair_index % 256 == 0 or pair_index == total_pairs:
            _emit_progress(progress_callback, "pair_merge", pair_index, total_pairs)

    grouped = np.zeros_like(labels, dtype=np.int32)
    positive_mask = labels > 0
    if np.any(positive_mask):
        remapped = np.vectorize(lambda value: find(int(value)), otypes=[np.int32])(labels[positive_mask])
        grouped[positive_mask] = remapped

    zero_pixels = np.argwhere(grouped == 0)
    total_zero = max(int(zero_pixels.shape[0]), 1)
    _emit_progress(progress_callback, "zero_fill", 0, total_zero)
    for zero_index, (row, col) in enumerate(zero_pixels, start=1):
        y0 = max(0, int(row) - 1)
        y1 = min(grouped.shape[0], int(row) + 2)
        x0 = max(0, int(col) - 1)
        x1 = min(grouped.shape[1], int(col) + 2)
        unique = np.unique(grouped[y0:y1, x0:x1])
        unique = unique[unique > 0]
        if unique.size == 1:
            grouped[row, col] = int(unique[0])
        if zero_index % 4096 == 0 or zero_index == total_zero:
            _emit_progress(progress_callback, "zero_fill", zero_index, total_zero)

    relabeled = np.zeros_like(grouped, dtype=np.int32)
    next_label = 1
    for parent_label in sorted(int(item) for item in np.unique(grouped) if item > 0):
        components, count = ndimage.label(grouped == parent_label)
        for component_id in range(1, count + 1):
            relabeled[components == component_id] = next_label
            next_label += 1
    return relabeled


def _selective_split_refine(
    labels: np.ndarray,
    seg_mask: np.ndarray,
    *,
    boundary_prob: np.ndarray | None,
    edge_score: np.ndarray | None,
    ndvi: np.ndarray | None,
    ndvi_std: np.ndarray | None,
    cfg,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool | None]]:
    diagnostics: dict[str, float | int | str | bool | None] = {
        "watershed_applied": False,
        "watershed_skipped_reason": None,
        "watershed_rollback_reason": None,
        "split_score_p50": 0.0,
        "split_score_p90": 0.0,
        "components_before_watershed": int(np.max(_connected_components(seg_mask))),
        "components_after_watershed": int(labels.max()),
    }
    if cfg is None or not bool(getattr(cfg, "SELECTIVE_SPLIT_ENABLED", False)):
        diagnostics["watershed_applied"] = bool(labels.max() > 0)
        return labels.astype(np.int32, copy=False), diagnostics

    base_labels = _connected_components(seg_mask)
    base_count = int(base_labels.max())
    if labels.max() <= 0:
        diagnostics["watershed_skipped_reason"] = "empty_watershed_labels"
        diagnostics["components_after_watershed"] = base_count
        return base_labels, diagnostics

    ndvi_means = _region_means(labels, ndvi)
    ndvi_std_means = _region_means(labels, ndvi_std)
    pair_stats = _collect_zero_boundary_pairs(
        labels,
        boundary_prob,
        edge_score,
        ndvi_means,
        ndvi_std_means,
        progress_callback=progress_callback,
    )
    split_scores = np.asarray([float(item["split_score"]) for item in pair_stats], dtype=np.float32)
    if split_scores.size:
        diagnostics["split_score_p50"] = round(float(np.nanpercentile(split_scores, 50)), 4)
        diagnostics["split_score_p90"] = round(float(np.nanpercentile(split_scores, 90)), 4)
    if not pair_stats:
        diagnostics["watershed_skipped_reason"] = "no_internal_boundaries"
        diagnostics["components_after_watershed"] = base_count
        return base_labels, diagnostics

    strong_pairs = {
        tuple(sorted((int(pair["left"]), int(pair["right"]))))
        for pair in pair_stats
        if _qualifies_as_strong_split(pair, cfg)
    }
    if bool(getattr(cfg, "WATERSHED_ONLY_IF_SPLIT_SCORE", True)) and not strong_pairs:
        diagnostics["watershed_skipped_reason"] = "no_strong_internal_boundary"
        diagnostics["components_after_watershed"] = base_count
        return base_labels, diagnostics

    merged_labels = _merge_weak_watershed_pairs(
        labels,
        pair_stats,
        strong_pairs=strong_pairs,
        progress_callback=progress_callback,
    )
    merged_count = int(merged_labels.max())
    diagnostics["watershed_applied"] = True
    diagnostics["components_after_watershed"] = merged_count

    boundary_strengths = np.asarray(
        [float(pair["boundary_prob_mean"]) for pair in pair_stats],
        dtype=np.float32,
    )
    mean_boundary_strength = float(np.nanmean(boundary_strengths)) if boundary_strengths.size else 0.0
    rollback_ratio = float(getattr(cfg, "WATERSHED_ROLLBACK_COMPONENT_RATIO_MAX", 1.8))
    rollback_boundary_max = float(
        getattr(cfg, "WATERSHED_ROLLBACK_MAX_INTERNAL_BOUNDARY_CONF", 0.55)
    )
    if (
        base_count > 0
        and merged_count > int(np.ceil(base_count * rollback_ratio))
        and mean_boundary_strength < rollback_boundary_max
    ):
        diagnostics["watershed_applied"] = False
        diagnostics["watershed_rollback_reason"] = "oversegmentation_low_boundary_conf"
        diagnostics["components_after_watershed"] = base_count
        return base_labels, diagnostics

    return merged_labels, diagnostics


def build_watershed_surface(
    candidate_mask: np.ndarray,
    edge_composite: np.ndarray,
    lambda_edge: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build watershed flooding surface.

    S(x,y) = -D(x,y) + lambda * E(x,y)

    Args:
        candidate_mask: (H, W) bool — True where candidate field pixel.
        edge_composite: (H, W) float edge composite [0, 1].
        lambda_edge: weight for edge barrier.

    Returns:
        (surface, distance_map): both (H, W) float.
    """
    distance = ndimage.distance_transform_edt(candidate_mask)
    surface = -distance + lambda_edge * edge_composite
    return surface, distance


def find_markers(
    distance_map: np.ndarray,
    candidate_mask: np.ndarray,
    min_distance: int = 15,
    *,
    seed_mode: str = "auto",
    custom_points: list[tuple[int, int]] | None = None,
    grid_step: int | None = None,
) -> np.ndarray:
    """Find markers for watershed via peak_local_max on distance transform.

    Args:
        distance_map: (H, W) distance transform.
        candidate_mask: (H, W) bool mask.
        min_distance: minimum pixel distance between markers.

    Returns:
        (H, W) labeled markers (0 = background).
    """
    normalized_mode = str(seed_mode or "auto").lower()
    if normalized_mode in {"distance"}:
        normalized_mode = "auto"
    elif normalized_mode in {"edges"}:
        normalized_mode = "grid"

    marker_mask = np.zeros_like(candidate_mask, dtype=bool)

    if normalized_mode == "custom":
        for row, col in custom_points or []:
            if 0 <= row < candidate_mask.shape[0] and 0 <= col < candidate_mask.shape[1]:
                if candidate_mask[row, col]:
                    marker_mask[row, col] = True
        if not marker_mask.any():
            normalized_mode = "auto"

    if normalized_mode == "grid":
        step = max(4, int(grid_step or max(min_distance * 2, 8)))
        y0 = step // 2
        x0 = step // 2
        ys = np.arange(y0, candidate_mask.shape[0], step, dtype=np.int32)
        xs = np.arange(x0, candidate_mask.shape[1], step, dtype=np.int32)
        if ys.size > 0 and xs.size > 0:
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            valid = candidate_mask[grid_y, grid_x]
            marker_mask[grid_y[valid], grid_x[valid]] = True
        if not marker_mask.any():
            normalized_mode = "auto"

    if normalized_mode == "auto":
        coords = peak_local_max(
            distance_map,
            min_distance=min_distance,
            labels=candidate_mask.astype(np.int32),
        )
        if len(coords) > 0:
            marker_mask[coords[:, 0], coords[:, 1]] = True

    markers, _ = ndimage.label(marker_mask)
    return markers


def watershed_segment(
    edge_composite: np.ndarray,
    candidate_mask: np.ndarray,
    osm_mask: np.ndarray | None = None,
    lambda_edge: float = 0.5,
    min_distance: int = 15,
    seed_mode: str = "auto",
    custom_seed_points: list[tuple[int, int]] | None = None,
    grid_step: int | None = None,
    precomputed_distance: np.ndarray | None = None,
    ndvi: np.ndarray | None = None,
    ndvi_std: np.ndarray | None = None,
    boundary_prob: np.ndarray | None = None,
    owt_edge: np.ndarray | None = None,
    cfg=None,
    return_diagnostics: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str | bool | None]]:
    """Run watershed segmentation on edge composite.

    Args:
        edge_composite: (H, W) float edge composite.
        candidate_mask: (H, W) bool — True for candidate field pixels.
        osm_mask: (H, W) bool — True for pixels to exclude (roads, buildings, etc.).
        lambda_edge: weight for edge barrier in surface.
        min_distance: min distance between watershed markers (pixels).

    Returns:
        (H, W) label array — each connected region has a unique integer label.
    """
    seg_mask = candidate_mask.copy()
    if osm_mask is not None:
        seg_mask = seg_mask & ~osm_mask

    if not seg_mask.any():
        empty = np.zeros_like(candidate_mask, dtype=np.int32)
        diagnostics = {
            "watershed_applied": False,
            "watershed_skipped_reason": "empty_segmentation_mask",
            "watershed_rollback_reason": None,
            "split_score_p50": 0.0,
            "split_score_p90": 0.0,
            "components_before_watershed": 0,
            "components_after_watershed": 0,
        }
        return (empty, diagnostics) if return_diagnostics else empty
    _emit_progress(progress_callback, "surface", 1, 4)

    edge_barrier = edge_composite
    if owt_edge is not None:
        if owt_edge.shape != candidate_mask.shape:
            raise ValueError("owt_edge must match candidate_mask shape")
        edge_barrier = owt_edge.astype(np.float32, copy=False)
    elif ndvi is not None:
        if ndvi.shape != candidate_mask.shape:
            raise ValueError("ndvi must match candidate_mask shape")
        edge_barrier = oriented_watershed(edge_composite, ndvi, cfg=cfg)

    if precomputed_distance is not None:
        if precomputed_distance.shape != seg_mask.shape:
            raise ValueError("precomputed_distance must match candidate_mask shape")
        distance = np.maximum(precomputed_distance.astype(np.float32, copy=False), 0.0)
        # Blend ML distance with EDT for more robust markers
        edt_distance = ndimage.distance_transform_edt(seg_mask).astype(np.float32)
        if edt_distance.max() > 0:
            edt_distance /= edt_distance.max()
        if distance.max() > 0:
            distance_norm = distance / distance.max()
        else:
            distance_norm = distance
        blended_distance = 0.6 * distance_norm + 0.4 * edt_distance
        surface = -blended_distance + (lambda_edge * edge_barrier)
    else:
        surface, distance = build_watershed_surface(seg_mask, edge_barrier, lambda_edge)
        blended_distance = distance
    _emit_progress(progress_callback, "surface", 2, 4)

    markers = find_markers(
        blended_distance if precomputed_distance is not None else distance,
        seg_mask,
        min_distance,
        seed_mode=seed_mode,
        custom_points=custom_seed_points,
        grid_step=grid_step,
    )
    _emit_progress(progress_callback, "markers", 3, 4)

    if markers.max() == 0:
        labels = _connected_components(seg_mask)
        diagnostics = {
            "watershed_applied": False,
            "watershed_skipped_reason": "markers_unavailable",
            "watershed_rollback_reason": None,
            "split_score_p50": 0.0,
            "split_score_p90": 0.0,
            "components_before_watershed": int(labels.max()),
            "components_after_watershed": int(labels.max()),
        }
        return (labels, diagnostics) if return_diagnostics else labels

    labels = watershed(surface, markers, mask=seg_mask, watershed_line=True)
    _emit_progress(progress_callback, "watershed", 4, 4)
    refined_labels, diagnostics = _selective_split_refine(
        labels.astype(np.int32, copy=False),
        seg_mask,
        boundary_prob=boundary_prob,
        edge_score=edge_barrier,
        ndvi=ndvi,
        ndvi_std=ndvi_std,
        cfg=cfg,
        progress_callback=progress_callback,
    )
    return (refined_labels, diagnostics) if return_diagnostics else refined_labels
