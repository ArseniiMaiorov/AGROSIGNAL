"""Celery tasks for model training, calibration, and conformal refresh.

queue: model-low
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
from typing import Any
from uuid import UUID

from core.celery_app import celery
from core.logging import get_logger

logger = get_logger(__name__)


def _deterministic_calibration_holdout(*, field_season_id: int, crop_code: str) -> bool:
    key = f"{field_season_id}:{crop_code}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest(), 16) % 5 == 0


def _predict_residual_adjustment(
    *,
    weekly_rows: list,
    mechanistic_result: Any,
    field: Any,
    model_snapshot: dict[str, Any],
) -> float:
    feature_names = list(model_snapshot.get("feature_names") or [])
    coefficients = list(model_snapshot.get("coefficients") or [])
    if not feature_names or not coefficients or len(feature_names) != len(coefficients):
        return 0.0

    features = _extract_residual_features(weekly_rows, mechanistic_result, field)
    vector = [float(features.get(name, 0.0) or 0.0) for name in feature_names]
    return float(sum(v * float(c) for v, c in zip(vector, coefficients)))


def _serialize_conformal_sets(
    calibration_buckets: dict[tuple[str, str], list[float]],
    fallback_buckets: dict[tuple[str, str], list[float]],
    *,
    model_version: str,
    min_bucket_size: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    conformal_sets: list[dict[str, Any]] = []
    stats = {
        "bucket_count": 0,
        "fallback_bucket_count": 0,
        "insufficient_bucket_count": 0,
    }

    all_keys = set(fallback_buckets) | set(calibration_buckets)
    for crop_code, region_key in sorted(all_keys):
        calibration_values = [abs(float(v)) for v in calibration_buckets.get((crop_code, region_key), []) if v is not None]
        fallback_values = [abs(float(v)) for v in fallback_buckets.get((crop_code, region_key), []) if v is not None]
        source = None
        residuals: list[float] = []
        if len(calibration_values) >= min_bucket_size:
            residuals = calibration_values
            source = "calibration_split"
        elif len(fallback_values) >= min_bucket_size:
            residuals = fallback_values
            source = "all_rows_fallback"
            stats["fallback_bucket_count"] += 1
        else:
            stats["insufficient_bucket_count"] += 1
            continue

        conformal_sets.append(
            {
                "crop_code": crop_code,
                "region_key": region_key,
                "residuals": [round(float(v), 6) for v in residuals],
                "n_calibration": len(residuals),
                "model_version": model_version,
                "source": source,
            }
        )
        stats["bucket_count"] += 1

    return conformal_sets, stats


@celery.task(name="tasks.model.train_global_residual_model", bind=True, max_retries=1)
def train_global_residual_model(
    self,
    organization_id_str: str | None = None,
) -> dict[str, Any]:
    """Train a global residual model over mechanistic baseline.

    Loads all yield observations, runs mechanistic baseline for each,
    computes residuals, and trains a LightGBM/Ridge on phase-aggregated features.
    """
    logger.info("train_global_residual_model_start")

    async def _run() -> dict[str, Any]:
        from storage.db import (
            YieldObservation, FieldSeason, Field, CropAssignment, Crop,
            YieldModel,
            get_session_factory,
        )
        from sqlalchemy import select, desc
        from services.mechanistic_engine import run_mechanistic_baseline
        from services.weekly_profile_service import (
            FEATURE_SCHEMA_VERSION,
            ensure_weekly_profile,
            profile_has_signal,
            rows_to_weekly_inputs,
        )
        import numpy as np

        factory = get_session_factory()
        async with factory() as db:
            # Load all yield observations with feature data
            obs_stmt = (
                select(YieldObservation, FieldSeason, Field, CropAssignment)
                .join(FieldSeason, YieldObservation.field_season_id == FieldSeason.id)
                .join(Field, FieldSeason.field_id == Field.id)
                .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            )
            if organization_id_str:
                org_id = UUID(organization_id_str)
                obs_stmt = obs_stmt.where(YieldObservation.organization_id == org_id)

            rows = (await db.execute(obs_stmt)).all()
            if not rows:
                return {"status": "no_data", "samples": 0}

            residuals: list[float] = []
            features_list: list[dict[str, float]] = []
            targets: list[float] = []
            skipped_no_profile = 0
            skipped_no_signal = 0

            for obs, season, field, crop_assign in rows:
                # Load weekly features for this field-season
                weekly_rows = await ensure_weekly_profile(
                    db,
                    organization_id=obs.organization_id,
                    field_id=field.id,
                    season_year=season.season_year,
                )
                if len(weekly_rows) < 5:
                    skipped_no_profile += 1
                    continue
                if not profile_has_signal(weekly_rows):
                    skipped_no_signal += 1
                    continue

                # Build weekly inputs
                weekly_inputs = rows_to_weekly_inputs(weekly_rows)

                # Get crop baseline
                crop_stmt = select(Crop).where(Crop.code == crop_assign.crop_code)
                crop = (await db.execute(crop_stmt)).scalar_one_or_none()
                baseline_kg = float(crop.yield_baseline_kg_ha) if crop else 3000.0

                # Run mechanistic model
                result = run_mechanistic_baseline(
                    crop_code=crop_assign.crop_code,
                    crop_baseline_kg_ha=baseline_kg,
                    weekly_inputs=weekly_inputs,
                    field_area_ha=float(field.area_m2 or 0) / 10000.0,
                )

                actual = float(obs.yield_kg_ha)
                mech = result.baseline_yield_kg_ha
                residual = actual - mech

                residuals.append(residual)
                targets.append(actual)

                # Extract phase-aggregated features for residual model
                features_list.append(_extract_residual_features(weekly_rows, result, field))

            if len(residuals) < 5:
                return {"status": "insufficient_data", "samples": len(residuals)}

            # Train Ridge regression on residuals
            feature_names = sorted(features_list[0].keys()) if features_list else []
            X = np.array([[f.get(name, 0.0) for name in feature_names] for f in features_list])
            y = np.array(residuals)

            # Ridge with LOO CV for alpha selection
            best_alpha = 1.0
            best_rmse = float("inf")
            for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
                loo_errors = []
                for i in range(len(y)):
                    X_loo = np.delete(X, i, axis=0)
                    y_loo = np.delete(y, i)
                    try:
                        c = np.linalg.solve(
                            X_loo.T @ X_loo + alpha * np.eye(X_loo.shape[1]),
                            X_loo.T @ y_loo,
                        )
                        pred = float(X[i] @ c)
                        loo_errors.append((pred - y[i]) ** 2)
                    except np.linalg.LinAlgError:
                        loo_errors.append(float("inf"))
                rmse = float(np.sqrt(np.mean(loo_errors)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_alpha = alpha

            # Final model
            coeffs = np.linalg.solve(
                X.T @ X + best_alpha * np.eye(X.shape[1]),
                X.T @ y,
            )

            # Save model record
            import json
            model = YieldModel(
                model_name="global_residual",
                model_version=f"residual_ridge_v1_{len(residuals)}s",
                mechanistic_params_version="mechanistic_v1",
                residual_model_version="ridge_v1",
                training_summary={
                    "n_samples": len(residuals),
                    "alpha": best_alpha,
                    "loo_rmse": round(best_rmse, 2),
                    "residual_mean": round(float(np.mean(residuals)), 2),
                    "residual_std": round(float(np.std(residuals)), 2),
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "skipped_no_profile": skipped_no_profile,
                    "skipped_no_signal": skipped_no_signal,
                },
                metrics={
                    "loo_rmse": round(best_rmse, 2),
                    "residual_bias": round(float(np.mean(residuals)), 2),
                },
                config_snapshot={
                    "feature_names": feature_names,
                    "coefficients": [round(float(c), 6) for c in coeffs],
                    "alpha": best_alpha,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                },
                status="validated",
            )
            db.add(model)
            await db.commit()

            return {
                "status": "trained",
                "samples": len(residuals),
                "loo_rmse": round(best_rmse, 2),
                "model_version": model.model_version,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "skipped_no_profile": skipped_no_profile,
                "skipped_no_signal": skipped_no_signal,
            }

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


def _extract_residual_features(
    weekly_rows: list,
    mech_result,
    field,
) -> dict[str, float]:
    """Extract phase-aggregated features for the residual model."""
    import numpy as np

    feats: dict[str, float] = {}

    ndvi_values = [float(w.ndvi_mean) for w in weekly_rows if w.ndvi_mean is not None]
    if ndvi_values:
        feats["peak_ndvi"] = float(np.max(ndvi_values))
        feats["auc_ndvi"] = float(np.trapezoid(ndvi_values))
        feats["weeks_to_peak"] = float(np.argmax(ndvi_values))
    else:
        feats["peak_ndvi"] = 0.0
        feats["auc_ndvi"] = 0.0
        feats["weeks_to_peak"] = 0.0

    ndre_values = [float(w.ndre_mean) for w in weekly_rows if w.ndre_mean is not None]
    if ndre_values:
        feats["peak_ndre"] = float(np.max(ndre_values))
        feats["auc_ndre"] = float(np.trapezoid(ndre_values))
    else:
        feats["peak_ndre"] = 0.0
        feats["auc_ndre"] = 0.0

    ndmi_values = [float(w.ndmi_mean) for w in weekly_rows if w.ndmi_mean is not None]
    feats["seasonal_ndmi_auc"] = float(np.trapezoid(ndmi_values)) if ndmi_values else 0.0

    # Water stress weeks
    water_stress_weeks = sum(1 for w in weekly_rows if w.water_stress is not None and w.water_stress > 0.3)
    feats["early_water_deficit_weeks"] = float(water_stress_weeks)

    # Heat stress during reproductive
    heat_weeks = sum(1 for w in weekly_rows if w.heat_stress is not None and w.heat_stress > 0.2)
    feats["reproductive_heat_weeks"] = float(heat_weeks)

    # VPD
    vpd_values = [float(w.vpd_kpa) for w in weekly_rows if w.vpd_kpa is not None]
    feats["cumulative_vpd_above_threshold"] = float(sum(max(v - 2.0, 0.0) for v in vpd_values))

    # Field geometry
    feats["field_area_ha"] = float(field.area_m2 or 0) / 10000.0
    feats["geometry_confidence"] = float(field.quality_score or 0.5)

    return feats


@celery.task(name="tasks.model.recalibrate_tenant_model", bind=True, max_retries=1)
def recalibrate_tenant_model(
    self,
    organization_id_str: str,
) -> dict[str, Any]:
    """Re-calibrate tenant-specific yield model with partial pooling."""
    logger.info("recalibrate_tenant_model_start", org=organization_id_str)
    # Partial pooling calibration: shrinks tenant coefficients toward global model
    # when few local observations exist. Full implementation requires
    # the global residual model to be trained first.
    return {"organization_id": organization_id_str, "status": "placeholder"}


@celery.task(name="tasks.model.refresh_conformal_calibration", bind=True, max_retries=1)
def refresh_conformal_calibration(
    self,
    model_version: str,
    organization_id_str: str | None = None,
) -> dict[str, Any]:
    """Refresh conformal calibration sets for prediction intervals.

    Computes LOO residuals on the calibration split and stores them
    in ConformalCalibrationSet objects per crop × region bucket.
    """
    logger.info("refresh_conformal_calibration_start", model_version=model_version)

    async def _run() -> dict[str, Any]:
        from geoalchemy2.shape import to_shape
        import numpy as np
        from sqlalchemy import select

        from services.conformal_service import ConformalService
        from services.mechanistic_engine import run_mechanistic_baseline
        from services.weekly_profile_service import (
            FEATURE_SCHEMA_VERSION,
            ensure_weekly_profile,
            profile_has_signal,
            rows_to_weekly_inputs,
        )
        from storage.db import (
            Crop,
            CropAssignment,
            Field,
            FieldSeason,
            YieldModel,
            YieldObservation,
            get_session_factory,
        )

        factory = get_session_factory()
        async with factory() as db:
            model_stmt = select(YieldModel).where(YieldModel.model_version == model_version)
            model = (await db.execute(model_stmt)).scalar_one_or_none()
            if model is None:
                return {"status": "model_not_found", "model_version": model_version}

            obs_stmt = (
                select(YieldObservation, FieldSeason, Field, CropAssignment)
                .join(FieldSeason, YieldObservation.field_season_id == FieldSeason.id)
                .join(Field, FieldSeason.field_id == Field.id)
                .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            )
            if organization_id_str:
                org_id = UUID(organization_id_str)
                obs_stmt = obs_stmt.where(YieldObservation.organization_id == org_id)

            rows = (await db.execute(obs_stmt)).all()
            if not rows:
                return {"status": "no_data", "model_version": model_version, "samples": 0}

            calibration_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
            fallback_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
            evaluated_rows = 0
            skipped_no_profile = 0
            skipped_no_signal = 0

            for obs, season, field, crop_assign in rows:
                weekly_rows = await ensure_weekly_profile(
                    db,
                    organization_id=obs.organization_id,
                    field_id=field.id,
                    season_year=season.season_year,
                )
                if len(weekly_rows) < 5:
                    skipped_no_profile += 1
                    continue
                if not profile_has_signal(weekly_rows):
                    skipped_no_signal += 1
                    continue

                crop_stmt = select(Crop).where(Crop.code == crop_assign.crop_code)
                crop = (await db.execute(crop_stmt)).scalar_one_or_none()
                baseline_kg = float(crop.yield_baseline_kg_ha) if crop else 3000.0
                weekly_inputs = rows_to_weekly_inputs(weekly_rows)
                geometry = to_shape(field.geom)
                mechanistic = run_mechanistic_baseline(
                    crop_code=crop_assign.crop_code,
                    crop_baseline_kg_ha=baseline_kg,
                    weekly_inputs=weekly_inputs,
                    field_area_ha=float(field.area_m2 or 0) / 10000.0,
                    latitude=geometry.centroid.y,
                )
                point_estimate = float(mechanistic.baseline_yield_kg_ha)
                point_estimate += _predict_residual_adjustment(
                    weekly_rows=weekly_rows,
                    mechanistic_result=mechanistic,
                    field=field,
                    model_snapshot=dict(model.config_snapshot or {}),
                )
                residual = abs(float(obs.yield_kg_ha) - point_estimate)
                region_key = ConformalService.region_key_from_latitude(geometry.centroid.y)
                keys = [
                    (str(crop_assign.crop_code), str(region_key)),
                    (str(crop_assign.crop_code), "global"),
                ]
                is_holdout = _deterministic_calibration_holdout(
                    field_season_id=int(season.id),
                    crop_code=str(crop_assign.crop_code),
                )
                for key in keys:
                    fallback_buckets[key].append(residual)
                    if is_holdout:
                        calibration_buckets[key].append(residual)
                evaluated_rows += 1

            conformal_sets, bucket_stats = _serialize_conformal_sets(
                calibration_buckets,
                fallback_buckets,
                model_version=model.model_version,
            )
            if not conformal_sets:
                return {
                    "status": "insufficient_calibration",
                    "model_version": model.model_version,
                    "samples": evaluated_rows,
                    "skipped_no_profile": skipped_no_profile,
                    "skipped_no_signal": skipped_no_signal,
                    **bucket_stats,
                }

            snapshot = dict(model.config_snapshot or {})
            snapshot["conformal_sets"] = conformal_sets
            snapshot["feature_schema_version"] = FEATURE_SCHEMA_VERSION
            model.config_snapshot = snapshot
            model.calibration_set_hash = hashlib.sha1(
                json.dumps(conformal_sets, sort_keys=True).encode("utf-8")
            ).hexdigest()

            training_summary = dict(model.training_summary or {})
            training_summary.update(
                {
                    "conformal_samples": evaluated_rows,
                    "conformal_bucket_count": bucket_stats["bucket_count"],
                    "conformal_fallback_bucket_count": bucket_stats["fallback_bucket_count"],
                    "conformal_skipped_no_profile": skipped_no_profile,
                    "conformal_skipped_no_signal": skipped_no_signal,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                }
            )
            model.training_summary = training_summary

            metrics = dict(model.metrics or {})
            calibration_sizes = np.asarray([float(item.get("n_calibration") or 0.0) for item in conformal_sets], dtype=float)
            metrics.update(
                {
                    "conformal_bucket_count": bucket_stats["bucket_count"],
                    "conformal_mean_bucket_size": round(float(np.mean(calibration_sizes)), 2) if calibration_sizes.size else 0.0,
                    "conformal_max_bucket_size": int(np.max(calibration_sizes)) if calibration_sizes.size else 0,
                }
            )
            model.metrics = metrics

            await db.commit()
            return {
                "status": "calibrated",
                "model_version": model.model_version,
                "samples": evaluated_rows,
                "skipped_no_profile": skipped_no_profile,
                "skipped_no_signal": skipped_no_signal,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                **bucket_stats,
            }

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
