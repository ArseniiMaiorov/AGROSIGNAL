"""Supervised candidate ranker for field detection v4.

Replaces binary accept/reject with a continuous score per candidate polygon.
Each candidate gets 19+ features computed from edge composite, spectral indices,
WorldCover priors, shape metrics, and inter-branch agreement. A LightGBM/XGBoost
model (or rule-based fallback) ranks candidates, then non-maximum suppression
keeps the best non-overlapping set.

Features computed per candidate:
- edge_mean, edge_p90: edge composite statistics within polygon
- boundary_closure_ratio: fraction of perimeter with strong edge signal
- owt_strength_mean: oriented watershed strength along boundary
- ndvi_delta_mean, ndre_delta_mean: interior-exterior spectral contrast
- ndvi_std_mean: internal homogeneity
- worldcover_cropland_frac, worldcover_noncrop_frac: land cover priors
- road_overlap_ratio: road barrier overlap
- compactness, rectangularity, hole_ratio, perimeter_area_ratio: shape
- agreement_branch_count: how many branches proposed this region
- agreement_iou_max: best IoU with any candidate from another branch
- tile_quality_score: ambient tile quality
- selected_dates_count: number of valid scenes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.ndimage import label as nd_label

from core.logging import get_logger

logger = get_logger(__name__)


def _emit_progress(
    progress_callback: Callable[[int, int, str], None] | None,
    completed: int,
    total: int,
    stage: str,
) -> None:
    if progress_callback is None:
        return
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    try:
        progress_callback(safe_completed, safe_total, str(stage))
    except Exception as exc:
        logger.warning(
            "candidate_ranker_progress_callback_failed",
            stage=str(stage),
            completed=safe_completed,
            total=safe_total,
            error=str(exc),
        )


@dataclass(slots=True)
class CandidatePolygon:
    """A single field candidate from one detection branch."""
    mask: np.ndarray  # (H, W) bool
    branch: str  # "boundary", "crop_region", "refine"
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    reject_reason: str | None = None
    label_id: int = 0  # connected component ID


@dataclass(slots=True)
class RankedCandidate:
    candidate: CandidatePolygon
    rank: int
    keep: bool
    suppressed_by: int | None = None  # label_id of suppressor


def compute_candidate_features(
    candidate: CandidatePolygon,
    *,
    edge_composite: np.ndarray,
    ndvi: np.ndarray,
    ndvi_std: np.ndarray | None = None,
    ndre: np.ndarray | None = None,
    boundary_prob: np.ndarray | None = None,
    worldcover: np.ndarray | None = None,
    road_mask: np.ndarray | None = None,
    tile_quality_score: float = 1.0,
    selected_dates_count: int = 0,
) -> dict[str, float]:
    """Compute ranking features for a single candidate polygon."""
    mask = candidate.mask.astype(bool)
    area_px = int(np.count_nonzero(mask))
    if area_px == 0:
        return {"area_px": 0.0, "reject": 1.0}

    feats: dict[str, float] = {}
    feats["area_px"] = float(area_px)

    # --- Edge features ---
    ec_inside = edge_composite[mask]
    ec_finite = ec_inside[np.isfinite(ec_inside)]
    feats["edge_mean"] = float(np.mean(ec_finite)) if ec_finite.size > 0 else 0.0
    feats["edge_p90"] = float(np.percentile(ec_finite, 90)) if ec_finite.size > 0 else 0.0

    # Boundary closure: fraction of perimeter pixels with strong edge
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(mask, iterations=1)
    perimeter = mask & ~interior
    perimeter_px = int(np.count_nonzero(perimeter))
    if perimeter_px > 0:
        edge_on_perimeter = edge_composite[perimeter]
        edge_thresh = float(np.percentile(ec_finite, 50)) if ec_finite.size > 10 else 0.1
        feats["boundary_closure_ratio"] = float(np.mean(edge_on_perimeter > edge_thresh))
    else:
        feats["boundary_closure_ratio"] = 0.0

    # --- Spectral contrast (interior vs exterior ring) ---
    from scipy.ndimage import binary_dilation
    exterior_ring = binary_dilation(mask, iterations=2) & ~mask
    exterior_px = int(np.count_nonzero(exterior_ring))

    ndvi_inside = float(np.nanmean(ndvi[mask]))
    ndvi_outside = float(np.nanmean(ndvi[exterior_ring])) if exterior_px > 10 else ndvi_inside
    feats["ndvi_delta_mean"] = ndvi_inside - ndvi_outside

    if ndre is not None:
        ndre_inside = float(np.nanmean(ndre[mask]))
        ndre_outside = float(np.nanmean(ndre[exterior_ring])) if exterior_px > 10 else ndre_inside
        feats["ndre_delta_mean"] = ndre_inside - ndre_outside
    else:
        feats["ndre_delta_mean"] = 0.0

    # --- Internal homogeneity ---
    if ndvi_std is not None:
        feats["ndvi_std_mean"] = float(np.nanmean(ndvi_std[mask]))
    else:
        feats["ndvi_std_mean"] = float(np.nanstd(ndvi[mask]))

    # --- WorldCover priors ---
    if worldcover is not None:
        wc_inside = worldcover[mask]
        feats["worldcover_cropland_frac"] = float(np.mean(wc_inside == 40))
        feats["worldcover_noncrop_frac"] = float(np.mean(np.isin(wc_inside, [50, 60, 70, 80, 90, 95, 100])))
    else:
        feats["worldcover_cropland_frac"] = 0.5
        feats["worldcover_noncrop_frac"] = 0.0

    # --- Road overlap ---
    if road_mask is not None:
        feats["road_overlap_ratio"] = float(np.mean(road_mask[mask]))
    else:
        feats["road_overlap_ratio"] = 0.0

    # --- Shape metrics ---
    feats["perimeter_area_ratio"] = float(perimeter_px) / max(float(area_px), 1.0)

    # Compactness: 4π·area / perimeter²
    pi = float(np.pi)
    feats["compactness"] = (4.0 * pi * float(area_px)) / max(float(perimeter_px) ** 2, 1.0)

    # Rectangularity: area / bounding_box_area
    rows, cols = np.where(mask)
    if rows.size > 0:
        bbox_area = float((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
        feats["rectangularity"] = float(area_px) / max(bbox_area, 1.0)
    else:
        feats["rectangularity"] = 0.0

    # Hole ratio
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(mask)
    filled_area = int(np.count_nonzero(filled))
    feats["hole_ratio"] = 1.0 - float(area_px) / max(float(filled_area), 1.0)

    # --- Boundary probability (from ML model) ---
    if boundary_prob is not None:
        bp_perimeter = boundary_prob[perimeter] if perimeter_px > 0 else np.array([0.0])
        feats["owt_strength_mean"] = float(np.mean(bp_perimeter))
    else:
        feats["owt_strength_mean"] = 0.0

    # --- Tile-level context ---
    feats["tile_quality_score"] = tile_quality_score
    feats["selected_dates_count"] = float(selected_dates_count)

    candidate.features = feats
    return feats


def compute_branch_agreement(
    candidates: list[CandidatePolygon],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """Compute inter-branch agreement for each candidate.

    For each candidate, counts how many other branches have a candidate
    with IoU > 0.3, and records the max IoU.
    """
    total = len(candidates)
    _emit_progress(progress_callback, 0, total, "branch_agreement")
    for idx, c in enumerate(candidates, start=1):
        other_branches = [o for o in candidates if o.branch != c.branch and o.mask.shape == c.mask.shape]
        c_area = int(np.count_nonzero(c.mask))
        if c_area == 0:
            c.features["agreement_branch_count"] = 0.0
            c.features["agreement_iou_max"] = 0.0
            _emit_progress(progress_callback, idx, total, "branch_agreement")
            continue

        agreeing_branches: set[str] = set()
        best_iou = 0.0
        for o in other_branches:
            o_area = int(np.count_nonzero(o.mask))
            if o_area == 0:
                continue
            intersection = int(np.count_nonzero(c.mask & o.mask))
            union = c_area + o_area - intersection
            iou = float(intersection) / max(float(union), 1.0)
            if iou > 0.3:
                agreeing_branches.add(o.branch)
            best_iou = max(best_iou, iou)

        c.features["agreement_branch_count"] = float(len(agreeing_branches))
        c.features["agreement_iou_max"] = best_iou
        _emit_progress(progress_callback, idx, total, "branch_agreement")


def score_candidates_rule_based(candidates: list[CandidatePolygon]) -> list[CandidatePolygon]:
    """Rule-based candidate scoring (fallback when no trained model available).

    Score = weighted combination of features, designed to reward:
    - High edge strength at boundary (closure)
    - Strong spectral contrast (NDVI/NDRE delta)
    - High cropland fraction from WorldCover
    - Low road overlap
    - Good shape (compact, not too many holes)
    - Multi-branch agreement
    """
    for c in candidates:
        f = c.features
        if f.get("area_px", 0) == 0:
            c.score = 0.0
            c.reject_reason = "zero_area"
            continue

        score = 0.0
        # Edge closure: 0-1 → 0-25 points
        score += float(f.get("boundary_closure_ratio", 0.0)) * 25.0
        # Edge strength: 0-20 points
        score += min(float(f.get("edge_p90", 0.0)) * 100.0, 20.0)
        # NDVI contrast: typically -0.3 to +0.3 → 0-20 points
        score += max(0.0, float(f.get("ndvi_delta_mean", 0.0)) * 50.0 + 5.0)
        # WorldCover cropland: 0-15 points
        score += float(f.get("worldcover_cropland_frac", 0.5)) * 15.0
        # Compactness: 0-10 points
        score += min(float(f.get("compactness", 0.0)) * 10.0, 10.0)
        # Agreement: 0-10 points (max 2 other branches)
        score += min(float(f.get("agreement_branch_count", 0.0)) * 5.0, 10.0)

        # Penalties
        score -= float(f.get("road_overlap_ratio", 0.0)) * 15.0
        score -= float(f.get("worldcover_noncrop_frac", 0.0)) * 12.0
        score -= float(f.get("hole_ratio", 0.0)) * 8.0

        # Normalize to 0-1
        c.score = float(np.clip(score / 100.0, 0.0, 1.0))

    return candidates


def rank_and_suppress(
    candidates: list[CandidatePolygon],
    *,
    min_score: float = 0.25,
    iou_threshold: float = 0.40,
    cfg=None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[RankedCandidate]:
    """Rank candidates by score and suppress overlapping lower-scored ones.

    Non-maximum suppression at the raster level: for each pair of overlapping
    candidates, keep the one with higher score if IoU exceeds threshold.
    """
    if cfg is not None:
        min_score = float(getattr(cfg, "CANDIDATE_MIN_SCORE", min_score))
        iou_threshold = float(getattr(cfg, "CANDIDATE_NMS_IOU_THRESHOLD", iou_threshold))

    # Filter low scores
    viable = [c for c in candidates if c.score >= min_score]
    rejected = [c for c in candidates if c.score < min_score]
    _emit_progress(progress_callback, 0, len(viable), "rank_prepare")
    for r in rejected:
        r.reject_reason = r.reject_reason or "below_min_score"

    # Sort by score descending
    viable.sort(key=lambda c: c.score, reverse=True)

    results: list[RankedCandidate] = []
    kept_masks: list[tuple[int, np.ndarray, int]] = []  # (idx, mask, area)

    for rank, c in enumerate(viable):
        c_area = int(np.count_nonzero(c.mask))
        if c_area == 0:
            results.append(RankedCandidate(candidate=c, rank=rank, keep=False))
            _emit_progress(progress_callback, rank + 1, len(viable), "suppress")
            continue

        suppressed_by = None
        for kept_idx, kept_mask, kept_area in kept_masks:
            intersection = int(np.count_nonzero(c.mask & kept_mask))
            union = c_area + kept_area - intersection
            iou = float(intersection) / max(float(union), 1.0)
            if iou > iou_threshold:
                suppressed_by = kept_idx
                break

        if suppressed_by is not None:
            c.reject_reason = f"suppressed_by_{suppressed_by}"
            results.append(RankedCandidate(candidate=c, rank=rank, keep=False, suppressed_by=suppressed_by))
        else:
            kept_masks.append((rank, c.mask, c_area))
            results.append(RankedCandidate(candidate=c, rank=rank, keep=True))
        _emit_progress(progress_callback, rank + 1, len(viable), "suppress")

    # Add rejected candidates
    for r in rejected:
        results.append(RankedCandidate(candidate=r, rank=len(viable), keep=False))

    logger.info(
        "candidate_ranking_complete",
        total=len(candidates),
        viable=len(viable),
        kept=sum(1 for r in results if r.keep),
    )
    _emit_progress(progress_callback, len(viable), len(viable), "done")
    return results
