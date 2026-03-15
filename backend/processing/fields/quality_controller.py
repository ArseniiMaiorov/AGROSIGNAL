"""Tile quality controller — decides detection mode per tile.

Assesses observation quality and routes to the appropriate detection path:
- normal: full 3-branch fusion pipeline
- boundary_recovery: boundary-first only (weak phenology but stable edges)
- degraded_output: minimal processing with high uncertainty flag
- skip_tile: too few valid observations to produce any reliable output

Quality signals integrated:
- Valid observation count & coverage fraction
- Edge composite strength (multi-temporal gradient persistence)
- Cloud/shadow interference from SCL
- Phenology signal clarity (NDVI temporal variability)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


class TileQualityMode(str, Enum):
    NORMAL = "normal"
    BOUNDARY_RECOVERY = "boundary_recovery"
    DEGRADED_OUTPUT = "degraded_output"
    SKIP_TILE = "skip_tile"


@dataclass(slots=True)
class TileQualityReport:
    mode: TileQualityMode
    coverage_fraction: float
    valid_scene_count: int
    edge_strength_mean: float
    edge_strength_p90: float
    ndvi_temporal_std_mean: float
    cloud_interference_fraction: float
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "coverage_fraction": round(self.coverage_fraction, 4),
            "valid_scene_count": self.valid_scene_count,
            "edge_strength_mean": round(self.edge_strength_mean, 4),
            "edge_strength_p90": round(self.edge_strength_p90, 4),
            "ndvi_temporal_std_mean": round(self.ndvi_temporal_std_mean, 4),
            "cloud_interference_fraction": round(self.cloud_interference_fraction, 4),
            "reasons": self.reasons,
            "diagnostics": self.diagnostics,
        }


def assess_tile_quality(
    *,
    valid_mask: np.ndarray,
    edge_composite: np.ndarray,
    ndvi_stack: np.ndarray | None = None,
    scl_stack: np.ndarray | None = None,
    cfg,
) -> TileQualityReport:
    """Assess tile data quality and determine the detection mode.

    Parameters
    ----------
    valid_mask : (T, H, W) or (H, W) bool — per-pixel/scene validity from SCL
    edge_composite : (H, W) float — multi-temporal edge strength
    ndvi_stack : (T, H, W) float — NDVI per scene (optional)
    scl_stack : (T, H, W) uint8 — SCL values per scene (optional)
    cfg : settings object
    """
    min_coverage = float(getattr(cfg, "QC_MIN_COVERAGE_FRACTION", 0.20))
    min_scenes = int(getattr(cfg, "QC_MIN_VALID_SCENES", 3))
    min_edge_p90 = float(getattr(cfg, "QC_MIN_EDGE_P90", 0.08))
    min_ndvi_std = float(getattr(cfg, "QC_MIN_NDVI_TEMPORAL_STD", 0.04))
    boundary_recovery_edge_p90 = float(getattr(cfg, "QC_BOUNDARY_RECOVERY_EDGE_P90", 0.12))

    reasons: list[str] = []

    # --- Coverage fraction ---
    vm = np.asarray(valid_mask, dtype=bool)
    if vm.ndim == 3:
        coverage_per_scene = np.array([float(np.mean(vm[t])) for t in range(vm.shape[0])])
        valid_scene_count = int(np.sum(coverage_per_scene >= 0.30))
        coverage_fraction = float(np.mean(coverage_per_scene))
    else:
        coverage_fraction = float(np.mean(vm))
        valid_scene_count = 1 if coverage_fraction >= 0.30 else 0

    # --- Edge composite strength ---
    ec = np.asarray(edge_composite, dtype=np.float32)
    finite_ec = ec[np.isfinite(ec)]
    if finite_ec.size > 0:
        edge_strength_mean = float(np.mean(finite_ec))
        edge_strength_p90 = float(np.percentile(finite_ec, 90))
    else:
        edge_strength_mean = 0.0
        edge_strength_p90 = 0.0

    # --- NDVI temporal variability ---
    if ndvi_stack is not None and ndvi_stack.ndim == 3 and ndvi_stack.shape[0] >= 2:
        with np.errstate(all="ignore"):
            ndvi_std = np.nanstd(ndvi_stack, axis=0)
        ndvi_temporal_std_mean = float(np.nanmean(ndvi_std))
    else:
        ndvi_temporal_std_mean = 0.0

    # --- Cloud interference ---
    if scl_stack is not None and scl_stack.ndim == 3:
        cloud_classes = np.isin(scl_stack, [8, 9, 10, 3])  # cloud high/med/cirrus + shadow
        cloud_interference_fraction = float(np.mean(cloud_classes))
    else:
        cloud_interference_fraction = 0.0

    # --- Decision logic ---
    if valid_scene_count < 1 or coverage_fraction < 0.05:
        mode = TileQualityMode.SKIP_TILE
        reasons.append("insufficient_valid_observations")
    elif valid_scene_count < min_scenes and coverage_fraction < min_coverage:
        mode = TileQualityMode.DEGRADED_OUTPUT
        reasons.append("low_coverage_and_few_scenes")
    elif edge_strength_p90 >= boundary_recovery_edge_p90 and ndvi_temporal_std_mean < min_ndvi_std:
        # Strong edges but weak phenology → boundary-first recovery
        mode = TileQualityMode.BOUNDARY_RECOVERY
        reasons.append("strong_edges_weak_phenology")
    elif coverage_fraction < min_coverage:
        mode = TileQualityMode.DEGRADED_OUTPUT
        reasons.append("low_coverage")
    elif edge_strength_p90 < min_edge_p90 and ndvi_temporal_std_mean < min_ndvi_std:
        mode = TileQualityMode.DEGRADED_OUTPUT
        reasons.append("weak_edges_and_weak_phenology")
    else:
        mode = TileQualityMode.NORMAL

    report = TileQualityReport(
        mode=mode,
        coverage_fraction=coverage_fraction,
        valid_scene_count=valid_scene_count,
        edge_strength_mean=edge_strength_mean,
        edge_strength_p90=edge_strength_p90,
        ndvi_temporal_std_mean=ndvi_temporal_std_mean,
        cloud_interference_fraction=cloud_interference_fraction,
        reasons=reasons,
        diagnostics={
            "min_coverage_threshold": min_coverage,
            "min_scenes_threshold": min_scenes,
            "min_edge_p90_threshold": min_edge_p90,
            "min_ndvi_std_threshold": min_ndvi_std,
        },
    )

    logger.info(
        "tile_quality_assessed",
        mode=mode.value,
        coverage=round(coverage_fraction, 3),
        scenes=valid_scene_count,
        edge_p90=round(edge_strength_p90, 4),
        ndvi_std=round(ndvi_temporal_std_mean, 4),
    )
    return report
