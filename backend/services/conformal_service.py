"""Split-conformal prediction intervals for yield forecasts.

Implements proper conformal inference (Vovk et al., 2005; Lei et al., 2018):
1. Calibration residuals are computed on a held-out calibration set
2. At inference, the (1-α) quantile of calibration residuals gives the interval width
3. Intervals are stratified by crop × region bucket for better conditional coverage

This replaces ad-hoc confidence heuristics with statistically grounded intervals.

References:
- Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random World
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-Free Predictive Inference
- Romano, Patterson & Candès (2019). Conformalized Quantile Regression
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ConformalCalibrationSet:
    """Stored calibration residuals for a crop × region bucket."""
    crop_code: str
    region_key: str  # e.g., "lat_45_55"
    residuals: list[float]  # |actual - predicted| values from calibration split
    model_version: str
    n_calibration: int

    @property
    def is_sufficient(self) -> bool:
        return len(self.residuals) >= 5


@dataclass(slots=True)
class PredictionInterval:
    lower: float
    upper: float
    width: float
    coverage_target: float
    method: str  # "conformal", "conformal_global_fallback", "heuristic"
    calibration_size: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": round(self.lower, 2),
            "upper": round(self.upper, 2),
            "width": round(self.width, 2),
            "coverage_target": self.coverage_target,
            "method": self.method,
            "calibration_size": self.calibration_size,
            "confidence": round(self.confidence, 3),
        }


class ConformalService:
    """Manages conformal calibration sets and produces prediction intervals."""

    def __init__(self) -> None:
        # In-memory calibration store (populated from DB or training pipeline)
        self._calibration_sets: dict[str, ConformalCalibrationSet] = {}

    def register_calibration_set(self, cal_set: ConformalCalibrationSet) -> None:
        key = f"{cal_set.crop_code}:{cal_set.region_key}"
        self._calibration_sets[key] = cal_set
        logger.info(
            "conformal_calibration_registered",
            crop=cal_set.crop_code,
            region=cal_set.region_key,
            n=len(cal_set.residuals),
        )

    def compute_interval(
        self,
        *,
        crop_code: str,
        region_key: str | None,
        point_estimate: float,
        coverage: float = 0.90,
        model_version: str | None = None,
    ) -> PredictionInterval:
        """Compute prediction interval using conformal inference.

        Falls back to global calibration set or heuristic if no bucket match.
        """
        # Try crop × region bucket
        cal_set = self._find_calibration_set(crop_code, region_key)

        if cal_set is not None and cal_set.is_sufficient:
            return self._conformal_interval(cal_set, point_estimate, coverage)

        # Fallback: global calibration set for this crop
        cal_set_global = self._find_calibration_set(crop_code, "global")
        if cal_set_global is not None and cal_set_global.is_sufficient:
            interval = self._conformal_interval(cal_set_global, point_estimate, coverage)
            interval.method = "conformal_global_fallback"
            # Widen by 15% for non-stratified set
            interval.width *= 1.15
            interval.lower = max(0.0, point_estimate - interval.width / 2)
            interval.upper = point_estimate + interval.width / 2
            return interval

        # Fallback: heuristic
        return self._heuristic_interval(point_estimate, coverage)

    def compute_interval_from_loo_residuals(
        self,
        *,
        loo_residuals: list[float],
        point_estimate: float,
        coverage: float = 0.90,
    ) -> PredictionInterval:
        """Compute interval directly from LOO residuals (no stored calibration set needed)."""
        if len(loo_residuals) < 3:
            return self._heuristic_interval(point_estimate, coverage)

        arr = np.asarray([abs(r) for r in loo_residuals], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return self._heuristic_interval(point_estimate, coverage)

        # Conformal quantile with finite-sample correction
        n = arr.size
        q_level = min(coverage * (1.0 + 1.0 / n), 1.0)
        width = float(np.quantile(arr, q_level))

        # Empirical coverage on calibration set
        emp_coverage = float(np.mean(arr <= width))

        confidence = self._coverage_to_confidence(emp_coverage, n, coverage)

        return PredictionInterval(
            lower=round(max(0.0, point_estimate - width), 2),
            upper=round(point_estimate + width, 2),
            width=round(width, 2),
            coverage_target=coverage,
            method="conformal_loo",
            calibration_size=n,
            confidence=confidence,
        )

    def _find_calibration_set(
        self, crop_code: str, region_key: str | None,
    ) -> ConformalCalibrationSet | None:
        if region_key:
            key = f"{crop_code}:{region_key}"
            if key in self._calibration_sets:
                return self._calibration_sets[key]
        return None

    def _conformal_interval(
        self,
        cal_set: ConformalCalibrationSet,
        point_estimate: float,
        coverage: float,
    ) -> PredictionInterval:
        arr = np.asarray(cal_set.residuals, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = arr.size

        # Finite-sample corrected quantile (Lei et al. 2018)
        q_level = min(coverage * (1.0 + 1.0 / n), 1.0)
        width = float(np.quantile(arr, q_level))

        emp_coverage = float(np.mean(arr <= width))
        confidence = self._coverage_to_confidence(emp_coverage, n, coverage)

        return PredictionInterval(
            lower=round(max(0.0, point_estimate - width), 2),
            upper=round(point_estimate + width, 2),
            width=round(width, 2),
            coverage_target=coverage,
            method="conformal",
            calibration_size=n,
            confidence=confidence,
        )

    def _heuristic_interval(self, point_estimate: float, coverage: float) -> PredictionInterval:
        """Fallback when no calibration data is available."""
        # Use coefficient of variation typical for crop yields (~15-25%)
        cv = 0.20
        z = 1.645 if coverage >= 0.90 else 1.28  # normal quantile approximation
        width = point_estimate * cv * z
        width = max(width, 250.0)  # minimum 250 kg/ha

        return PredictionInterval(
            lower=round(max(0.0, point_estimate - width), 2),
            upper=round(point_estimate + width, 2),
            width=round(width, 2),
            coverage_target=coverage,
            method="heuristic",
            calibration_size=0,
            confidence=round(0.40 * coverage, 3),
        )

    @staticmethod
    def _coverage_to_confidence(empirical_coverage: float, n: int, target: float) -> float:
        """Convert empirical coverage and sample size to confidence score."""
        # Coverage closeness to target
        coverage_quality = 1.0 - abs(empirical_coverage - target) / target
        # Sample size factor: need 10+ for reasonable calibration
        sample_factor = min(n / 15.0, 1.0)
        return float(np.clip(coverage_quality * sample_factor, 0.20, 0.95))

    @staticmethod
    def region_key_from_latitude(latitude: float | None) -> str:
        """Assign latitude to a regional bucket."""
        if latitude is None:
            return "global"
        lat = abs(latitude)
        if lat < 40:
            return "lat_lt_40"
        elif lat < 48:
            return "lat_40_48"
        elif lat < 55:
            return "lat_48_55"
        elif lat < 62:
            return "lat_55_62"
        else:
            return "lat_gt_62"
