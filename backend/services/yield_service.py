"""Hybrid yield prediction service: global baseline + tenant calibration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from geoalchemy2.shape import to_shape
import numpy as np
from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from services.conformal_service import ConformalCalibrationSet, ConformalService
from services.crop_service import CropService
from services.field_analytics_service import FieldAnalyticsService
from services.forecast_curve import build_forecast_curve_points
from services.payload_meta import build_freshness
from services.message_codes import classify_support_reason
from services.temporal_analytics_service import TemporalAnalyticsService, normalize_driver_breakdown
from services.trust_service import describe_prediction_operational_tier
from services.weather_service import WeatherService
from services.weekly_profile_service import (
    FEATURE_SCHEMA_VERSION,
    current_season_year,
    ensure_weekly_profile,
    load_crop_hint,
    profile_has_signal,
    rows_to_weekly_inputs,
    summarize_geometry_quality,
)
from storage.db import (
    Crop,
    CropAssignment,
    Field,
    FieldFeatureWeekly,
    FieldSeason,
    ManagementEvent,
    SoilProfile,
    WeatherDaily,
    YieldModel,
    YieldObservation,
    YieldPrediction,
)

logger = get_logger(__name__)

CALIBRATION_FEATURE_NAMES: tuple[str, ...] = (
    "crop_baseline",
    "field_area_ha",
    "compactness",
    "soil_organic_matter_pct",
    "soil_ph",
    "soil_n_ppm",
    "soil_p_ppm",
    "soil_k_ppm",
    "management_total_amount",
    "historical_field_mean_yield",
)

OBSERVED_FEATURE_NAMES: tuple[str, ...] = (
    "current_ndvi_mean",
    "current_ndmi_mean",
    "current_ndre_mean",
    "ndvi_auc",
    "ndvi_peak",
    "ndvi_mean_season",
    "current_soil_moisture",
    "current_vpd_mean",
    "current_precipitation_mm",
    "current_wind_speed_m_s",
    "seasonal_gdd_sum",
    "seasonal_precipitation_mm",
    "seasonal_temperature_mean_c",
    "seasonal_vpd_mean",
    "latitude",
)

ALL_FEATURE_NAMES: tuple[str, ...] = CALIBRATION_FEATURE_NAMES + OBSERVED_FEATURE_NAMES


@dataclass(slots=True)
class TrainingRow:
    features: dict[str, float | None]
    target: float


class YieldService:
    """Prediction service with objective baseline and optional tenant residual calibration."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self.crop_service = CropService(db)
        self.weather_service = WeatherService(db)
        self.field_analytics_service = FieldAnalyticsService(db)
        self.temporal_analytics_service = TemporalAnalyticsService(db)

    async def get_or_create_prediction(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        field = await self._get_field(field_id, organization_id=organization_id)
        crop = await self._resolve_crop(crop_code)
        if not refresh:
            existing = await self._get_latest_prediction(field_id, crop.id, organization_id=organization_id)
            if existing is not None:
                return await self._prediction_to_dict(existing, crop)

        prediction = await self._build_prediction(field, crop, organization_id=organization_id)
        self.db.add(prediction)
        await self.db.flush()
        logger.info(
            "yield_prediction_created",
            field_id=str(field_id),
            crop_code=crop.code,
            yield_kg_ha=prediction.estimated_yield_kg_ha,
            supported=bool((prediction.details or {}).get("supported")),
            confidence_tier=(prediction.details or {}).get("confidence_tier"),
        )
        return await self._prediction_to_dict(prediction, crop)

    async def estimate_prediction(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None = None,
        scenario_adjustments: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        field = await self._get_field(field_id, organization_id=organization_id)
        crop = await self._resolve_crop(crop_code)
        transient_prediction = await self._build_prediction(
            field,
            crop,
            organization_id=organization_id,
            scenario_adjustments=scenario_adjustments,
            persist=False,
        )
        return await self._prediction_to_dict(transient_prediction, crop)

    async def _get_field(self, field_id: UUID, *, organization_id: UUID) -> Field:
        field = (
            await self.db.execute(
                select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
            )
        ).scalar_one_or_none()
        if field is None:
            raise ValueError("Поле не найдено")
        if str(getattr(field, "source", "") or "").strip().lower() == "autodetect_preview":
            raise ValueError("Preview-контур из быстрого режима нельзя использовать для прогноза. Запустите Standard или Quality.")
        return field

    async def _resolve_crop(self, crop_code: str | None) -> Crop:
        effective_code = crop_code or self.settings.DEFAULT_CROP_CODE
        crop = await self.crop_service.get_crop_by_code(effective_code)
        if crop is None:
            raise ValueError("Культура не найдена")
        return crop

    async def build_forecast_curve(
        self,
        field: Field,
        crop: Crop,
        *,
        organization_id: UUID,
        temperature_delta_c: float = 0.0,
        extra_precip_total_mm: float = 0.0,
        precipitation_factor: float | None = None,
    ) -> dict[str, Any]:
        geom = to_shape(field.geom)
        centroid = geom.centroid
        forecast_payload = await self.weather_service.get_forecast(
            centroid.y,
            centroid.x,
            days=10,
            organization_id=organization_id,
        )
        base_temp_c = float(getattr(crop, "base_temp_c", None) or 5.0)
        return {
            "provider": forecast_payload.get("provider"),
            "days": int(forecast_payload.get("days") or 10),
            "base_temp_c": base_temp_c,
            "freshness": forecast_payload.get("freshness") or {},
            "error": forecast_payload.get("error"),
            "points": build_forecast_curve_points(
                list(forecast_payload.get("forecast") or []),
                base_temp_c=base_temp_c,
                temperature_delta_c=temperature_delta_c,
                extra_precip_total_mm=extra_precip_total_mm,
                precipitation_factor=precipitation_factor,
            ),
        }

    async def _get_latest_prediction(
        self,
        field_id: UUID,
        crop_id: int,
        *,
        organization_id: UUID,
    ) -> YieldPrediction | None:
        stmt = (
            select(YieldPrediction)
            .where(YieldPrediction.organization_id == organization_id)
            .where(YieldPrediction.field_id == field_id)
            .where(YieldPrediction.crop_id == crop_id)
            .order_by(desc(YieldPrediction.prediction_date))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        if record.prediction_date.date() != datetime.now(timezone.utc).date():
            return None
        return record

    async def _load_conformal_service(
        self,
        *,
        organization_id: UUID,
        preferred_model_version: str | None = None,
    ) -> tuple[ConformalService | None, dict[str, Any]]:
        stmt = (
            select(YieldModel)
            .where(YieldModel.status.in_(("deployed", "validated")))
            .where(
                or_(
                    YieldModel.organization_id == organization_id,
                    YieldModel.organization_id.is_(None),
                )
            )
            .order_by(desc(YieldModel.deployed_at), desc(YieldModel.created_at))
        )
        rows = list((await self.db.execute(stmt)).scalars().all())
        if not rows:
            return None, {"model_version": None, "n_sets": 0}

        def _sort_key(model: YieldModel) -> tuple[int, int, int, float]:
            deployed_rank = 0 if str(model.status or "") == "deployed" else 1
            org_rank = 0 if model.organization_id == organization_id else 1
            preferred_rank = 0 if preferred_model_version and model.model_version == preferred_model_version else 1
            ts = model.deployed_at or model.created_at
            timestamp = ts.timestamp() if ts is not None else 0.0
            return (preferred_rank, org_rank, deployed_rank, -timestamp)

        rows.sort(key=_sort_key)

        service = ConformalService()
        seen_keys: set[str] = set()
        loaded_sets = 0
        source_model_version: str | None = None

        for model in rows:
            raw_sets = (model.config_snapshot or {}).get("conformal_sets") or []
            if isinstance(raw_sets, dict):
                raw_sets = list(raw_sets.values())
            if not isinstance(raw_sets, list):
                continue
            model_loaded = 0
            for payload in raw_sets:
                if not isinstance(payload, dict):
                    continue
                crop_code = str(payload.get("crop_code") or "").strip()
                region_key = str(payload.get("region_key") or "").strip()
                residuals_raw = payload.get("residuals") or []
                if not crop_code or not region_key or not isinstance(residuals_raw, list):
                    continue
                key = f"{crop_code}:{region_key}"
                if key in seen_keys:
                    continue
                residuals = []
                for value in residuals_raw:
                    try:
                        residuals.append(abs(float(value)))
                    except Exception:
                        continue
                if len(residuals) < 1:
                    continue
                service.register_calibration_set(
                    ConformalCalibrationSet(
                        crop_code=crop_code,
                        region_key=region_key,
                        residuals=residuals,
                        model_version=str(payload.get("model_version") or model.model_version),
                        n_calibration=int(payload.get("n_calibration") or len(residuals)),
                    )
                )
                seen_keys.add(key)
                loaded_sets += 1
                model_loaded += 1
            if model_loaded > 0 and source_model_version is None:
                source_model_version = model.model_version

        if loaded_sets == 0:
            return None, {"model_version": None, "n_sets": 0}
        return service, {"model_version": source_model_version, "n_sets": loaded_sets}

    async def _build_prediction(
        self,
        field: Field,
        crop: Crop,
        *,
        organization_id: UUID,
        scenario_adjustments: dict[str, float] | None = None,
        persist: bool = True,
    ) -> YieldPrediction:
        prediction_date = datetime.now(timezone.utc)
        geom = to_shape(field.geom)
        centroid = geom.centroid
        weather = await self.weather_service.get_current_weather(centroid.y, centroid.x, organization_id=organization_id)
        current_metrics, _raw_values = await self.field_analytics_service._collect_field_metrics(field)
        seasonal_weather = await self._seasonal_weather_summary(
            field.id,
            crop.code,
            organization_id=organization_id,
        )

        # Fetch cumulative NDVI (area under curve) for richer vegetation signal
        try:
            ndvi_auc_data = await self.temporal_analytics_service.compute_ndvi_auc(
                field.id, organization_id=organization_id,
            )
        except Exception:
            ndvi_auc_data = {}

        current_features, feature_gaps = await self._build_current_features(
            field,
            crop,
            organization_id=organization_id,
            current_metrics=current_metrics,
            weather=weather,
            scenario_adjustments=scenario_adjustments,
            latitude=centroid.y,
            seasonal_weather=seasonal_weather,
            ndvi_auc_data=ndvi_auc_data,
        )
        applicability_features = current_features
        applicability_feature_gaps = feature_gaps
        if scenario_adjustments:
            applicability_features, applicability_feature_gaps = await self._build_current_features(
                field,
                crop,
                organization_id=organization_id,
                current_metrics=current_metrics,
                weather=weather,
                scenario_adjustments=None,
                latitude=centroid.y,
                seasonal_weather=seasonal_weather,
                ndvi_auc_data=ndvi_auc_data,
            )
        training_rows = await self._load_training_rows(organization_id=organization_id, crop_code=crop.code)
        crop_suitability = _evaluate_crop_suitability(crop, applicability_features)

        coverage_score = round(
            float(sum(1 for value in applicability_features.values() if value is not None))
            / float(max(len(ALL_FEATURE_NAMES), 1)),
            4,
        )
        objective_feature_count = sum(
            1
            for name in OBSERVED_FEATURE_NAMES + ("soil_organic_matter_pct", "soil_ph", "historical_field_mean_yield")
            if applicability_features.get(name) is not None
        )
        global_estimate, global_factors = _estimate_global_baseline(
            crop,
            current_features,
            weather=weather,
            current_metrics=current_metrics,
            scenario_adjustments=scenario_adjustments,
        )

        # Weekly mechanistic baseline is the primary path when a sufficient
        # profile is available. The heuristic baseline remains only as fallback.
        mechanistic_used = False
        weekly_rows = []
        weekly_geometry_summary: dict[str, float | None] = {}
        try:
            from services.mechanistic_engine import run_mechanistic_baseline

            current_year = current_season_year()
            weekly_rows = await ensure_weekly_profile(
                self.db,
                organization_id=organization_id,
                field_id=field.id,
                season_year=current_year,
            )
            weekly_geometry_summary = summarize_geometry_quality(weekly_rows)
            for key in ("geometry_confidence", "tta_consensus", "boundary_uncertainty"):
                summary_value = weekly_geometry_summary.get(key)
                if summary_value is not None:
                    current_features[key] = float(summary_value)
                    applicability_features[key] = float(summary_value)
            weekly_inputs = rows_to_weekly_inputs(weekly_rows)
            if len(weekly_inputs) >= 5 and profile_has_signal(weekly_rows):
                mech_result = run_mechanistic_baseline(
                    crop_code=crop.code,
                    crop_baseline_kg_ha=float(crop.yield_baseline_kg_ha),
                    weekly_inputs=weekly_inputs,
                    soil_ph=current_features.get("soil_ph"),
                    soil_om_pct=current_features.get("soil_organic_matter_pct"),
                    soil_n_ppm=current_features.get("soil_n_ppm"),
                    soil_texture_code=current_features.get("_soil_texture_code"),
                    field_area_ha=float(field.area_m2 or 0) / 10000.0,
                    compactness=current_features.get("compactness"),
                    latitude=centroid.y,
                )
                global_estimate = float(mech_result.baseline_yield_kg_ha)
                global_factors["mechanistic_engine"] = round(
                    mech_result.baseline_yield_kg_ha / max(float(crop.yield_baseline_kg_ha), 1.0), 4
                )
                global_factors["heuristic_fallback"] = 0.0
                mechanistic_used = True
        except Exception:
            pass  # Silently fall back to heuristic baseline

        training_domain = {
            "crop_code": crop.code,
            "samples": len(training_rows),
            "calibration_feature_names": list(CALIBRATION_FEATURE_NAMES),
            "observed_feature_names": list(OBSERVED_FEATURE_NAMES),
        }

        supported = False
        confidence_tier = "unsupported"
        support_reason: str | None = None
        estimated_yield = 0.0
        confidence = 0.0
        interval_width = 0.0
        explanation_summary = "Прогноз недоступен: не хватает объективных полевых данных."
        explanation_drivers = []
        model_version = "unsupported_v3"
        geometry_quality_impact = {
            "geometry_confidence": applicability_features.get("geometry_confidence"),
            "tta_consensus": applicability_features.get("tta_consensus"),
            "boundary_uncertainty": applicability_features.get("boundary_uncertainty"),
            "penalty_factor": 1.0,
            "status": "neutral",
        }

        if len(training_rows) >= 5:
            x_train, y_train, medians = _materialize_training_matrix(training_rows)
            global_train = np.asarray(
                [
                    _estimate_global_baseline_from_features(
                        crop_baseline=float(row.features.get("crop_baseline") or crop.yield_baseline_kg_ha),
                        features=row.features,
                    )
                    for row in training_rows
                ],
                dtype=float,
            )
            residual_target = y_train - global_train

            # Ridge regression instead of raw LLSQ to prevent coefficient
            # explosion with small sample sizes (N=5-15).  Alpha is chosen
            # via leave-one-out cross-validation when we have enough samples.
            n_samples = x_train.shape[0]
            if n_samples >= 8:
                # Try a few alpha values, pick the one with lowest LOO RMSE
                best_alpha = 1.0
                best_loo_rmse = float("inf")
                for alpha_candidate in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
                    loo_errors = []
                    for i in range(n_samples):
                        x_loo = np.delete(x_train, i, axis=0)
                        y_loo = np.delete(residual_target, i)
                        n_feat = x_loo.shape[1]
                        try:
                            c = np.linalg.solve(
                                x_loo.T @ x_loo + alpha_candidate * np.eye(n_feat),
                                x_loo.T @ y_loo,
                            )
                            pred_i = float(np.dot(x_train[i], c))
                            loo_errors.append((pred_i - residual_target[i]) ** 2)
                        except np.linalg.LinAlgError:
                            loo_errors.append(float("inf"))
                    loo_rmse = float(np.sqrt(np.mean(loo_errors)))
                    if loo_rmse < best_loo_rmse:
                        best_loo_rmse = loo_rmse
                        best_alpha = alpha_candidate
                alpha = best_alpha
            else:
                alpha = 10.0  # Strong regularization with few samples

            n_feat = x_train.shape[1]
            try:
                coeffs = np.linalg.solve(
                    x_train.T @ x_train + alpha * np.eye(n_feat),
                    x_train.T @ residual_target,
                )
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(x_train, residual_target, rcond=None)[0]

            x_current = _vectorize_current(current_features, medians)
            residual_adjustment = float(np.dot(x_current, coeffs))
            estimated_yield = round(max(0.0, global_estimate + residual_adjustment), 2)
            residuals = np.abs(np.dot(x_train, coeffs) - residual_target)

            # Prediction interval via conformal inference.
            # Use LOO residuals when available (more honest than in-sample).
            if n_samples >= 8:
                # Compute LOO residuals with the best alpha
                loo_residuals = []
                for i in range(n_samples):
                    x_loo = np.delete(x_train, i, axis=0)
                    y_loo = np.delete(residual_target, i)
                    n_f = x_loo.shape[1]
                    try:
                        c_loo = np.linalg.solve(
                            x_loo.T @ x_loo + alpha * np.eye(n_f),
                            x_loo.T @ y_loo,
                        )
                        loo_residuals.append(abs(float(np.dot(x_train[i], c_loo)) - residual_target[i]))
                    except np.linalg.LinAlgError:
                        loo_residuals.append(float("inf"))
                loo_arr = np.asarray(loo_residuals, dtype=float)
                interval_width = float(np.quantile(loo_arr[np.isfinite(loo_arr)], 0.9)) if np.any(np.isfinite(loo_arr)) else max(250.0, global_estimate * 0.15)

                # Conformal coverage: fraction of LOO predictions within interval
                coverage = float(np.mean(loo_arr[np.isfinite(loo_arr)] <= interval_width)) if np.any(np.isfinite(loo_arr)) else 0.5
            else:
                interval_width = float(np.quantile(residuals, 0.9)) if residuals.size else max(250.0, global_estimate * 0.15)
                # With few samples, use Bayesian credible interval interpretation
                coverage = float(2.0 * _norm_cdf(estimated_yield / max(2.0 * interval_width, 1.0)) - 1.0)

            # Sample-size penalty: small calibration sets are unreliable
            sample_factor = min(n_samples / 15.0, 1.0)  # full trust at 15+ samples
            confidence = round(
                float(np.clip(coverage * sample_factor, 0.25, 0.95)),
                3,
            )
            supported = True
            confidence_tier = "tenant_calibrated"
            model_version = self.settings.YIELD_MODEL_VERSION or "hybrid_global_tenant_v3"
            explanation_summary = (
                "Прогноз собран в два этапа: глобальный базовый уровень по признакам культуры, "
                "почвы, погоды и спутниковым данным, а также калибровка по остаткам "
                "на исторических сезонах арендатора."
            )
            explanation_drivers = [
                {"label": label, "effect": round(float(effect - 1.0), 4)}
                for label, effect in global_factors.items()
            ]
            explanation_drivers.append(
                {
                    "label": "tenant_residual_calibration",
                    "effect": round(float(residual_adjustment / max(estimated_yield, 1.0)), 4),
                }
            )
        elif 1 <= len(training_rows) < 5 and coverage_score >= 0.30 and objective_feature_count >= 1:
            # Micro-calibration: 1-4 historical seasons → simple mean-ratio scale on global baseline.
            # Much more reliable than pure global baseline for established fields with a few seasons.
            global_hist_estimates = [
                _estimate_global_baseline_from_features(
                    crop_baseline=float(row.features.get("crop_baseline") or crop.yield_baseline_kg_ha),
                    features=row.features,
                )
                for row in training_rows
            ]
            global_hist_mean = float(np.mean(global_hist_estimates)) if global_hist_estimates else global_estimate
            observed_hist_mean = float(np.mean([row.target for row in training_rows]))
            # Scale factor: clipped to [0.60, 1.60] to prevent extrapolation with few samples
            micro_scale = float(np.clip(observed_hist_mean / max(global_hist_mean, 1.0), 0.60, 1.60))
            estimated_yield = round(max(0.0, global_estimate * micro_scale), 2)
            n_micro = len(training_rows)
            scale_trust = float(np.clip(n_micro / 5.0, 0.20, 0.90))
            interval_width = max(estimated_yield * (0.20 - scale_trust * 0.05), 280.0)
            confidence = round(float(np.clip(0.32 + coverage_score * 0.18 + scale_trust * 0.14, 0.30, 0.76)), 3)
            supported = True
            confidence_tier = "satellite_anchored"
            model_version = "satellite_anchored_v1"
            explanation_summary = (
                f"Прогноз: глобальный базовый уровень, откалиброванный по {n_micro} историческому(-им) сезону(-ам). "
                "Для полноценной калибровки необходимо накопить данные минимум за 5 сезонов."
            )
            explanation_drivers = [
                {"label": label, "effect": round(float(effect - 1.0), 4)}
                for label, effect in global_factors.items()
            ]
            explanation_drivers.append({
                "label": "micro_season_scale",
                "effect": round(float(micro_scale - 1.0), 4),
            })
        elif coverage_score >= 0.42 and objective_feature_count >= 2:
            estimated_yield = round(global_estimate, 2)
            interval_width = max(estimated_yield * (0.22 - min(coverage_score, 0.9) * 0.08), 320.0)
            # Base confidence: higher weight on feature coverage + raised ceiling (0.82 vs 0.72)
            # because objective global baseline with rich satellite coverage is reliable
            confidence = round(float(max(0.28, min(0.82, 0.40 + coverage_score * 0.32))), 3)
            supported = True
            confidence_tier = "global_baseline"
            model_version = "global_baseline_v3"
            explanation_summary = (
                "Прогноз построен на объективном базовом уровне: "
                "исходные данные культуры, геометрия поля, почвенные характеристики, "
                "наблюдённые погодные условия и актуальные спутниковые метрики поля. "
                "Калибровка по данным арендатора не применялась."
            )
            explanation_drivers = [
                {"label": label, "effect": round(float(effect - 1.0), 4)}
                for label, effect in global_factors.items()
            ]
        else:
            support_reason = (
                "Недостаточно наблюдаемых данных для построения базового прогноза и нет достаточной истории "
                "для калибровки. Загрузите историю урожайности, план посева, анализы почвы или события управления."
            )

        # Geometry confidence penalty: low detection quality degrades yield confidence
        # because area, compactness, and metric extraction are all unreliable.
        geom_conf = applicability_features.get("geometry_confidence")
        if geom_conf is not None and supported and float(geom_conf) < 0.6:
            geom_penalty = max(0.65, float(geom_conf) / 0.6)  # 0.65-1.0 multiplier
            confidence = round(confidence * geom_penalty, 3)
            geometry_quality_impact = {
                "geometry_confidence": float(geom_conf),
                "tta_consensus": applicability_features.get("tta_consensus"),
                "boundary_uncertainty": applicability_features.get("boundary_uncertainty"),
                "penalty_factor": round(float(geom_penalty), 4),
                "status": "penalized",
            }
            explanation_drivers.append(
                {"label": "geometry_confidence_penalty", "effect": round(geom_penalty - 1.0, 4)}
            )
        elif geom_conf is not None:
            geom_conf_f = float(geom_conf)
            # Bonus for high geometric quality: reliable boundary → reliable area/metric extraction
            geom_bonus = min(0.08, max(0.0, (geom_conf_f - 0.80) * 0.40)) if geom_conf_f > 0.80 else 0.0
            tta_raw = applicability_features.get("tta_consensus")
            tta_agr = float(tta_raw) if tta_raw is not None else 0.0
            tta_bonus = min(0.04, max(0.0, (tta_agr - 0.85) * 0.20)) if tta_agr > 0.85 else 0.0
            if geom_bonus + tta_bonus > 0.0 and supported:
                confidence = round(min(0.82, confidence + geom_bonus + tta_bonus), 3)
            geometry_quality_impact = {
                "geometry_confidence": geom_conf_f,
                "tta_consensus": tta_agr if tta_raw is not None else None,
                "boundary_uncertainty": applicability_features.get("boundary_uncertainty"),
                "penalty_factor": 1.0,
                "geom_bonus": round(geom_bonus, 4),
                "tta_bonus": round(tta_bonus, 4),
                "status": "bonus_applied" if (geom_bonus + tta_bonus) > 0 else "applied_no_penalty",
            }

        suitability_status = str(crop_suitability.get("status") or "unknown")
        if supported and suitability_status == "low":
            confidence = round(min(confidence, 0.58 if confidence_tier == "global_baseline" else 0.76), 3)
            interval_width = interval_width * 1.25 if interval_width else 450.0
            explanation_drivers.append({"label": "crop_suitability_penalty", "effect": round(float(crop_suitability.get("yield_factor", 1.0) - 1.0), 4)})
        severe_unsuitable = suitability_status == "unsuitable" and len(training_rows) < 15
        if severe_unsuitable:
            supported = False
            confidence_tier = "unsupported"
            estimated_yield = 0.0
            confidence = 0.0
            prediction_interval = {"lower": None, "upper": None}
            support_reason = (
                crop_suitability.get("support_reason")
                or "Культура находится вне агроклиматической пригодности для этого поля и не будет оцениваться автоматически."
            )
            explanation_summary = "Прогноз отключён: пригодность культуры выходит за пределы агроклиматического домена модели."
            model_version = "unsupported_v4"

        prediction_interval = {
            "lower": round(max(0.0, estimated_yield - interval_width), 2) if supported else None,
            "upper": round(max(estimated_yield, estimated_yield + interval_width), 2) if supported else None,
        }

        # --- V4: Conformal prediction interval enhancement ---
        base_interval_method = (
            "tenant_calibrated_loo"
            if confidence_tier == "tenant_calibrated"
            else "global_baseline_empirical"
            if confidence_tier == "global_baseline"
            else "unsupported"
        )
        conformal_method = None
        conformal_model_version = None
        conformal_calibration_size = 0
        conformal_region_key = None
        if supported and estimated_yield > 0:
            try:
                conformal_svc, conformal_meta = await self._load_conformal_service(
                    organization_id=organization_id,
                    preferred_model_version=model_version,
                )
                if conformal_svc is None:
                    raise RuntimeError("conformal calibration unavailable")
                conformal_region_key = conformal_svc.region_key_from_latitude(centroid.y)
                conformal_interval = conformal_svc.compute_interval(
                    crop_code=crop.code,
                    region_key=conformal_region_key,
                    point_estimate=estimated_yield,
                    model_version=str(conformal_meta.get("model_version") or model_version),
                )
                if (
                    conformal_interval is not None
                    and str(conformal_interval.method).startswith("conformal")
                    and conformal_interval.confidence > 0.5
                ):
                    prediction_interval = {
                        "lower": round(conformal_interval.lower, 2),
                        "upper": round(conformal_interval.upper, 2),
                    }
                    conformal_method = conformal_interval.method
                    interval_width = conformal_interval.width
                    conformal_model_version = str(conformal_meta.get("model_version") or model_version)
                    conformal_calibration_size = int(conformal_interval.calibration_size or 0)
            except Exception:
                pass  # Fall back to existing interval

        data_quality = {
            "status": confidence_tier,
            "valid_feature_count": sum(1 for value in applicability_features.values() if value is not None),
            "has_grid_metrics": bool(current_metrics),
            "coverage_metrics": sorted(current_metrics.keys()),
            "prediction_interval": prediction_interval,
            "confidence_reason": (
                "Применена калибровка по данным арендатора поверх глобального базового уровня."
                if confidence_tier == "tenant_calibrated"
                else "Использован только глобальный базовый уровень без калибровки."
                if confidence_tier == "global_baseline"
                else support_reason or "Прогноз выключен: входные данные вне применимости модели."
            ),
            "confidence_tier": confidence_tier,
            "crop_suitability_status": suitability_status,
            "geometry_confidence": float(geom_conf) if geom_conf is not None else None,
            "interval_method": conformal_method or base_interval_method,
        }
        details = {
            "supported": supported,
            "weather_snapshot": weather,
            "seasonal_weather": seasonal_weather,
            "current_metrics": current_metrics,
            "training_samples": len(training_rows),
            "prediction_interval": prediction_interval,
            "dataset_version": self.settings.TRAIN_DATA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "confidence_tier": confidence_tier,
            "scenario_adjustments": scenario_adjustments or {},
            "support_reason": support_reason,
            "crop_suitability": crop_suitability,
            "geometry_quality_impact": geometry_quality_impact,
            "weekly_geometry_summary": weekly_geometry_summary,
            "weekly_profile_rows": len(weekly_rows),
            "mechanistic_primary_used": mechanistic_used,
            "conformal_interval": {
                "method": conformal_method or base_interval_method,
                "model_version": conformal_model_version,
                "calibration_size": conformal_calibration_size,
                "region_key": conformal_region_key,
            },
            "applicability_valid_feature_count": sum(
                1 for value in applicability_features.values() if value is not None
            ),
            "applicability_feature_gaps": sorted(applicability_feature_gaps),
            "applicability_feature_available": sorted(set(ALL_FEATURE_NAMES) - set(applicability_feature_gaps)),
        }
        explanation = {
            "summary": explanation_summary,
            "drivers": explanation_drivers,
            "baseline_reference_kg_ha": float(crop.yield_baseline_kg_ha),
            "crop_suitability_status": suitability_status,
        }
        input_features = {
            key: (round(float(value), 4) if value is not None else None)
            for key, value in current_features.items()
        }

        return YieldPrediction(
            organization_id=organization_id if persist else organization_id,
            field_id=field.id,
            crop_id=crop.id,
            prediction_date=prediction_date,
            estimated_yield_kg_ha=estimated_yield,
            confidence=confidence,
            model_version=model_version,
            details=details,
            input_features=input_features,
            explanation=explanation,
            data_quality=data_quality,
        )

    async def _build_current_features(
        self,
        field: Field,
        crop: Crop,
        *,
        organization_id: UUID,
        current_metrics: dict[str, Any] | None = None,
        weather: dict[str, Any] | None = None,
        scenario_adjustments: dict[str, float] | None = None,
        latitude: float | None = None,
        seasonal_weather: dict[str, float | int | None] | None = None,
        ndvi_auc_data: dict[str, float | None] | None = None,
    ) -> tuple[dict[str, float | None], set[str]]:
        current_metrics = current_metrics or {}
        weather = weather or {}
        seasonal_weather = seasonal_weather or {}
        field_area_ha = float(field.area_m2 or 0.0) / 10000.0
        compactness = _compactness(field.area_m2, field.perimeter_m)
        soil = await self._latest_soil(field.id, organization_id=organization_id)
        management_total = await self._field_management_total(field.id, organization_id=organization_id)
        management_by_type = await self._field_management_by_type(field.id, organization_id=organization_id)
        historical_field_mean = await self._historical_field_mean_yield(field.id, crop.code, organization_id=organization_id)
        hist_mean_full, hist_n_seasons, hist_latest_year = await self._historical_field_yield_meta(
            field.id, crop.code, organization_id=organization_id,
        )
        soil_organic_matter = None if soil is None else soil.organic_matter_pct
        soil_ph = None if soil is None else soil.ph
        soil_n_ppm = None if soil is None else soil.n_ppm
        soil_p_ppm = None if soil is None else soil.p_ppm
        soil_k_ppm = None if soil is None else soil.k_ppm
        soil_texture = None if soil is None else soil.texture_class

        if scenario_adjustments:
            irrigation_pct = float(scenario_adjustments.get("irrigation_pct") or 0.0)
            fertilizer_pct = float(scenario_adjustments.get("fertilizer_pct") or 0.0)
            # Management total: interpret percentage as absolute activity units.
            # Saturating curve prevents unrealistic scaling.
            if management_total is None and (irrigation_pct or fertilizer_pct):
                management_total = max(0.0, abs(irrigation_pct) * 0.10 + abs(fertilizer_pct) * 0.15)
            elif management_total is not None:
                base = float(management_total)
                delta = abs(irrigation_pct) * 0.10 + abs(fertilizer_pct) * 0.15
                management_total = max(0.0, base + delta)

            # Nutrient levels: percentage applied as actual fraction of current value.
            # No artificial dampening — let the Mitscherlich curves in _soil_factor
            # handle diminishing returns and toxicity naturally.
            # Clip ranges are wide enough to reach toxic/lethal thresholds.
            if soil_n_ppm is not None:
                soil_n_ppm = float(np.clip(float(soil_n_ppm) * (1.0 + fertilizer_pct / 100.0), 2.0, 80.0))
            if soil_p_ppm is not None:
                soil_p_ppm = float(np.clip(float(soil_p_ppm) * (1.0 + fertilizer_pct / 100.0), 2.0, 65.0))
            if soil_k_ppm is not None:
                soil_k_ppm = float(np.clip(float(soil_k_ppm) * (1.0 + fertilizer_pct / 100.0), 4.0, 90.0))

        current_precipitation = _metric_mean(current_metrics, "precipitation")
        if current_precipitation is None:
            current_precipitation = weather.get("precipitation_mm")
        current_soil_moisture = _metric_mean(current_metrics, "soil_moisture")
        if current_soil_moisture is None:
            current_soil_moisture = weather.get("soil_moisture")
        current_vpd = _metric_mean(current_metrics, "vpd")
        current_wind = _metric_mean(current_metrics, "wind")
        if current_wind is None:
            current_wind = weather.get("wind_speed_m_s")

        if scenario_adjustments:
            expected_rain_mm = float(scenario_adjustments.get("expected_rain_mm") or 0.0)
            irrigation_pct = float(scenario_adjustments.get("irrigation_pct") or 0.0)
            if current_precipitation is None and expected_rain_mm > 0:
                current_precipitation = expected_rain_mm
            elif current_precipitation is not None:
                current_precipitation = max(0.0, float(current_precipitation) + expected_rain_mm)
            # Soil moisture: saturating model based on FAO-56.
            # Rain adds ~0.08-0.10 per 50mm for loam soils (depends on texture).
            # Irrigation fills deficit toward field capacity (0.35) with diminishing returns.
            # Total moisture is physically bounded by saturation (~0.55).
            field_capacity = 0.35
            sm_base = float(current_soil_moisture) if current_soil_moisture is not None else 0.18
            rain_addition = expected_rain_mm * 0.002 * (1.0 - sm_base / 0.55)  # ~0.1 per 50mm, less if already wet
            irrig_deficit = max(field_capacity - sm_base, 0.0)
            irrig_addition = irrig_deficit * (1.0 - float(np.exp(-0.03 * max(irrigation_pct, 0.0))))
            sm_new = sm_base + rain_addition + irrig_addition
            # Drainage: excess above field capacity drains partly (5-20mm/day equivalent)
            if sm_new > field_capacity:
                excess = sm_new - field_capacity
                sm_new = field_capacity + excess * 0.6  # 40% drains away
            if current_soil_moisture is not None or irrigation_pct or expected_rain_mm:
                current_soil_moisture = float(np.clip(sm_new, 0.02, 0.60))

            # Solar radiation: scale by cloud_cover_factor (1.0 = no change).
            # cloud_cover_factor > 1.0 = clearer skies (more radiation),
            # cloud_cover_factor < 1.0 = overcast scenario (less radiation).
            cloud_cover_factor = float(scenario_adjustments.get("cloud_cover_factor") or 1.0)
            if cloud_cover_factor != 1.0:
                _sr = _as_optional_float(seasonal_weather.get("solar_radiation_mean"))
                if _sr is not None and _sr > 0.0:
                    seasonal_weather = dict(seasonal_weather)
                    # _sr is weekly MJ/m²; keep in weekly units, clip at 245 MJ/week (35 MJ/day * 7)
                    seasonal_weather["solar_radiation_mean"] = float(np.clip(_sr * cloud_cover_factor, 0.0, 245.0))

        geometry_confidence = float(field.quality_score) if field.quality_score is not None else None
        ndvi_auc_data = ndvi_auc_data or {}

        features: dict[str, float | None] = {
            "crop_baseline": float(crop.yield_baseline_kg_ha),
            "field_area_ha": field_area_ha if field_area_ha > 0 else None,
            "compactness": compactness,
            "geometry_confidence": geometry_confidence,
            "soil_organic_matter_pct": soil_organic_matter,
            "soil_ph": soil_ph,
            "soil_n_ppm": soil_n_ppm,
            "soil_p_ppm": soil_p_ppm,
            "soil_k_ppm": soil_k_ppm,
            "_soil_texture_code": _texture_to_code(soil_texture),
            "management_total_amount": management_total,
            "_mgmt_irrigation": management_by_type.get("irrigation") or management_by_type.get("полив"),
            "_mgmt_fertilizer": management_by_type.get("fertilizer") or management_by_type.get("удобрение") or management_by_type.get("fertilization"),
            "_mgmt_pesticide": management_by_type.get("pesticide") or management_by_type.get("пестицид") or management_by_type.get("protection"),
            "_mgmt_event_count": float(sum(1 for _ in management_by_type.values())) if management_by_type else None,
            "historical_field_mean_yield": historical_field_mean,
            "current_ndvi_mean": _metric_mean(current_metrics, "ndvi"),
            "current_ndmi_mean": _metric_mean(current_metrics, "ndmi"),
            "current_ndre_mean": _metric_mean(current_metrics, "ndre"),
            "ndvi_auc": _as_optional_float(ndvi_auc_data.get("ndvi_auc")),
            "ndvi_peak": _as_optional_float(ndvi_auc_data.get("ndvi_peak")),
            "ndvi_mean_season": _as_optional_float(ndvi_auc_data.get("ndvi_mean_season")),
            "current_soil_moisture": current_soil_moisture,
            "current_vpd_mean": current_vpd,
            "current_precipitation_mm": current_precipitation,
            "current_wind_speed_m_s": current_wind,
            "seasonal_gdd_sum": _as_optional_float(seasonal_weather.get("gdd_sum")),
            "seasonal_precipitation_mm": _as_optional_float(seasonal_weather.get("precipitation_sum")),
            "seasonal_temperature_mean_c": _as_optional_float(seasonal_weather.get("temperature_mean_c")),
            "seasonal_vpd_mean": _as_optional_float(seasonal_weather.get("vpd_mean")),
            "seasonal_observed_days": _as_optional_float(seasonal_weather.get("observed_days")),
            "seasonal_solar_radiation_mean": _as_optional_float(seasonal_weather.get("solar_radiation_mean")),
            "latitude": _as_optional_float(latitude),
            "_hist_n_seasons": float(hist_n_seasons) if hist_n_seasons else None,
            "_hist_latest_year": float(hist_latest_year) if hist_latest_year else None,
        }
        feature_gaps = {name for name, value in features.items() if value is None}
        return features, feature_gaps

    async def _load_training_rows(self, *, organization_id: UUID, crop_code: str) -> list[TrainingRow]:
        stmt = (
            select(YieldObservation, FieldSeason, Field)
            .join(FieldSeason, YieldObservation.field_season_id == FieldSeason.id)
            .join(Field, FieldSeason.field_id == Field.id)
            .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            .where(YieldObservation.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(Field.organization_id == organization_id)
            .where(CropAssignment.organization_id == organization_id)
            .where(CropAssignment.crop_code == crop_code)
        )
        rows = (await self.db.execute(stmt)).all()
        results: list[TrainingRow] = []
        for yield_row, season, field in rows:
            soil = await self._latest_soil(field.id, organization_id=organization_id)
            management_total = await self._season_management_total(season.id, organization_id=organization_id)
            historical_field_mean = await self._historical_field_mean_yield(field.id, crop_code, organization_id=organization_id)
            results.append(
                TrainingRow(
                    features={
                        "crop_baseline": float((yield_row.payload or {}).get("crop_baseline") or 0.0),
                        "field_area_ha": float(field.area_m2 or 0.0) / 10000.0,
                        "compactness": _compactness(field.area_m2, field.perimeter_m),
                        "soil_organic_matter_pct": None if soil is None else soil.organic_matter_pct,
                        "soil_ph": None if soil is None else soil.ph,
                        "soil_n_ppm": None if soil is None else soil.n_ppm,
                        "soil_p_ppm": None if soil is None else soil.p_ppm,
                        "soil_k_ppm": None if soil is None else soil.k_ppm,
                        "management_total_amount": management_total,
                        "historical_field_mean_yield": historical_field_mean,
                    },
                    target=float(yield_row.yield_kg_ha),
                )
            )
        if not results:
            return []
        if any(row.features["crop_baseline"] in (None, 0.0) for row in results):
            baseline_map = await self._crop_baselines()
            for row in results:
                if row.features["crop_baseline"] in (None, 0.0):
                    row.features["crop_baseline"] = float(baseline_map.get(crop_code) or 0.0)
        return results

    async def _crop_baselines(self) -> dict[str, float]:
        rows = (await self.db.execute(select(Crop.code, Crop.yield_baseline_kg_ha))).all()
        return {str(code): float(value) for code, value in rows}

    async def _latest_soil(self, field_id: UUID, *, organization_id: UUID) -> SoilProfile | None:
        stmt = (
            select(SoilProfile)
            .where(SoilProfile.organization_id == organization_id)
            .where(SoilProfile.field_id == field_id)
            .order_by(desc(SoilProfile.sampled_at))
            .limit(1)
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def _field_management_total(self, field_id: UUID, *, organization_id: UUID) -> float | None:
        stmt = (
            select(func.sum(ManagementEvent.amount))
            .join(FieldSeason, ManagementEvent.field_season_id == FieldSeason.id)
            .where(ManagementEvent.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
        )
        value = (await self.db.execute(stmt)).scalar_one_or_none()
        return None if value is None else float(value)

    async def _field_management_by_type(self, field_id: UUID, *, organization_id: UUID) -> dict[str, float]:
        """Return management event totals grouped by event_type."""
        stmt = (
            select(ManagementEvent.event_type, func.sum(ManagementEvent.amount))
            .join(FieldSeason, ManagementEvent.field_season_id == FieldSeason.id)
            .where(ManagementEvent.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .group_by(ManagementEvent.event_type)
        )
        rows = (await self.db.execute(stmt)).all()
        return {str(t).lower(): float(a) for t, a in rows if a is not None}

    async def _season_management_total(self, field_season_id: int, *, organization_id: UUID) -> float | None:
        stmt = (
            select(func.sum(ManagementEvent.amount))
            .where(ManagementEvent.organization_id == organization_id)
            .where(ManagementEvent.field_season_id == field_season_id)
        )
        value = (await self.db.execute(stmt)).scalar_one_or_none()
        return None if value is None else float(value)

    async def _historical_field_mean_yield(self, field_id: UUID, crop_code: str, *, organization_id: UUID) -> float | None:
        stmt = (
            select(func.avg(YieldObservation.yield_kg_ha))
            .join(FieldSeason, YieldObservation.field_season_id == FieldSeason.id)
            .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            .where(YieldObservation.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(CropAssignment.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .where(CropAssignment.crop_code == crop_code)
        )
        value = (await self.db.execute(stmt)).scalar_one_or_none()
        return None if value is None else float(value)

    async def _historical_field_yield_meta(
        self, field_id: UUID, crop_code: str, *, organization_id: UUID,
    ) -> tuple[float | None, int, int | None]:
        """Return (mean_yield, n_seasons, most_recent_year)."""
        stmt = (
            select(
                func.avg(YieldObservation.yield_kg_ha),
                func.count(func.distinct(FieldSeason.season_year)),
                func.max(FieldSeason.season_year),
            )
            .join(FieldSeason, YieldObservation.field_season_id == FieldSeason.id)
            .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            .where(YieldObservation.organization_id == organization_id)
            .where(FieldSeason.organization_id == organization_id)
            .where(CropAssignment.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .where(CropAssignment.crop_code == crop_code)
        )
        row = (await self.db.execute(stmt)).first()
        if row is None or row[0] is None:
            return None, 0, None
        return float(row[0]), int(row[1] or 0), int(row[2]) if row[2] is not None else None

    async def _seasonal_weather_summary(
        self,
        field_id: UUID,
        crop_code: str,
        *,
        organization_id: UUID,
    ) -> dict[str, float | int | None]:
        season_stmt = (
            select(FieldSeason.id, FieldSeason.season_year)
            .join(CropAssignment, CropAssignment.field_season_id == FieldSeason.id)
            .where(FieldSeason.organization_id == organization_id)
            .where(CropAssignment.organization_id == organization_id)
            .where(FieldSeason.field_id == field_id)
            .where(CropAssignment.crop_code == crop_code)
            .order_by(desc(FieldSeason.season_year))
            .limit(1)
        )
        season_row = (await self.db.execute(season_stmt)).first()
        if season_row is None:
            # No FieldSeason — fall back directly to FieldFeatureWeekly
            fallback_year = current_season_year()
            await ensure_weekly_profile(
                self.db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=fallback_year,
            )
            return await self._seasonal_weather_from_weekly(field_id, fallback_year, organization_id=organization_id)
        field_season_id, season_year = season_row
        summary_stmt = (
            select(
                func.sum(WeatherDaily.gdd),
                func.sum(WeatherDaily.precipitation_mm),
                func.avg(WeatherDaily.temperature_mean_c),
                func.avg(WeatherDaily.vpd),
                func.avg(WeatherDaily.soil_moisture),
                func.count(WeatherDaily.id),
            )
            .where(WeatherDaily.organization_id == organization_id)
            .where(WeatherDaily.field_season_id == field_season_id)
        )
        summary_row = (await self.db.execute(summary_stmt)).first()
        if summary_row is None:
            return {}
        gdd_sum, precipitation_sum, temperature_mean_c, vpd_mean, soil_moisture_mean, observed_days = summary_row
        result = {
            "season_year": int(season_year),
            "gdd_sum": _as_optional_float(gdd_sum),
            "precipitation_sum": _as_optional_float(precipitation_sum),
            "temperature_mean_c": _as_optional_float(temperature_mean_c),
            "vpd_mean": _as_optional_float(vpd_mean),
            "soil_moisture_mean": _as_optional_float(soil_moisture_mean),
            "observed_days": int(observed_days or 0),
        }
        # If WeatherDaily has no data, fall back to FieldFeatureWeekly aggregates
        if not any(result.get(k) is not None for k in ("gdd_sum", "precipitation_sum", "temperature_mean_c")):
            await ensure_weekly_profile(
                self.db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=int(season_year),
            )
            fallback = await self._seasonal_weather_from_weekly(field_id, int(season_year), organization_id=organization_id)
            for key, value in fallback.items():
                if result.get(key) is None and value is not None:
                    result[key] = value
        return result

    async def _seasonal_weather_from_weekly(
        self,
        field_id: UUID,
        season_year: int,
        *,
        organization_id: UUID,
    ) -> dict[str, Any]:
        """Fallback: aggregate seasonal weather metrics from FieldFeatureWeekly."""
        from datetime import date as _date
        season_from = _date(season_year, 1, 1)
        season_to = _date(season_year, 12, 31)
        stmt = (
            select(
                func.sum(FieldFeatureWeekly.gdd),
                func.sum(FieldFeatureWeekly.precipitation_mm),
                func.avg(FieldFeatureWeekly.tmean_c),
                func.avg(FieldFeatureWeekly.vpd_kpa),
                func.avg(FieldFeatureWeekly.soil_moisture),
                func.avg(FieldFeatureWeekly.solar_radiation_mj),
                func.count(FieldFeatureWeekly.id),
            )
            .where(FieldFeatureWeekly.organization_id == organization_id)
            .where(FieldFeatureWeekly.field_id == field_id)
            .where(FieldFeatureWeekly.season_year == season_year)
            .where(FieldFeatureWeekly.week_start >= season_from)
            .where(FieldFeatureWeekly.week_start <= season_to)
        )
        row = (await self.db.execute(stmt)).first()
        if not row:
            return {}
        gdd_sum, precipitation_sum, temperature_mean_c, vpd_mean, soil_moisture_mean, solar_radiation_mean, row_count = row
        return {
            "season_year": season_year,
            "gdd_sum": _as_optional_float(gdd_sum),
            "precipitation_sum": _as_optional_float(precipitation_sum),
            "temperature_mean_c": _as_optional_float(temperature_mean_c),
            "vpd_mean": _as_optional_float(vpd_mean),
            "soil_moisture_mean": _as_optional_float(soil_moisture_mean),
            "solar_radiation_mean": _as_optional_float(solar_radiation_mean),
            "observed_days": int(row_count or 0) * 7,
        }

    async def _prediction_to_dict(self, prediction: YieldPrediction, crop: Crop) -> dict[str, Any]:
        prediction_date = prediction.prediction_date or datetime.now(timezone.utc)
        details = dict(prediction.details or {})
        supported = bool(details.get("supported"))
        data_quality = dict(prediction.data_quality or {})
        input_features = dict(prediction.input_features or {})
        missing = [name for name, value in input_features.items() if value is None]
        applicability_missing = list(details.get("applicability_feature_gaps") or missing)
        applicability_available = list(
            details.get("applicability_feature_available") or sorted(set(input_features) - set(applicability_missing))
        )
        applicability_valid_feature_count = int(
            details.get("applicability_valid_feature_count")
            or data_quality.get("valid_feature_count")
            or len(applicability_available)
        )
        confidence_tier = str(
            details.get("confidence_tier")
            or data_quality.get("confidence_tier")
            or ("tenant_calibrated" if supported else "unsupported")
        )
        support_reason = None if supported else details.get("support_reason") or data_quality.get("confidence_reason")
        support_reason_code, support_reason_params = classify_support_reason(support_reason)
        crop_suitability = dict(details.get("crop_suitability") or {})
        trust_meta = describe_prediction_operational_tier(
            supported=supported,
            confidence_tier=confidence_tier,
            crop_suitability=crop_suitability,
            support_reason=support_reason,
        )
        field_id = prediction.field_id
        crop_code = crop.code
        field = await self._get_field(field_id, organization_id=prediction.organization_id)
        crop_hint = await load_crop_hint(
            self.db,
            organization_id=prediction.organization_id,
            field_id=field_id,
        )
        weekly_rows = await ensure_weekly_profile(
            self.db,
            organization_id=prediction.organization_id,
            field_id=field_id,
        )
        geometry_summary = summarize_geometry_quality(weekly_rows)
        try:
            analytics = await self.temporal_analytics_service.get_temporal_analytics(
                field_id,
                organization_id=prediction.organization_id,
                crop_code=crop_code,
            )
        except Exception as exc:
            logger.warning("temporal_analytics_unavailable", field_id=str(field_id), error=str(exc))
            analytics = {}
        try:
            zones = await self.temporal_analytics_service.get_management_zones(
                field_id,
                organization_id=prediction.organization_id,
                prediction_payload={
                    "estimated_yield_kg_ha": prediction.estimated_yield_kg_ha,
                    "confidence": prediction.confidence,
                },
            )
        except Exception as exc:
            logger.warning("management_zones_unavailable", field_id=str(field_id), error=str(exc))
            zones = {}
        try:
            forecast_curve = await self.build_forecast_curve(
                field,
                crop,
                organization_id=prediction.organization_id,
            )
        except Exception as exc:
            logger.warning("prediction_forecast_curve_unavailable", field_id=str(field_id), error=str(exc))
            forecast_curve = {}
        normalized_drivers = normalize_driver_breakdown(
            list((prediction.explanation or {}).get("drivers") or []),
            baseline_yield_kg_ha=float(prediction.estimated_yield_kg_ha or 0.0),
            baseline_inputs=input_features,
            source="yield_model",
        )
        explanation = dict(prediction.explanation or {})
        explanation["drivers"] = normalized_drivers
        return {
            "id": prediction.id,
            "field_id": str(field_id),
            "crop": {
                "id": crop.id,
                "code": crop.code,
                "name": crop.name,
            },
            "prediction_date": prediction_date.isoformat(),
            "estimated_yield_kg_ha": prediction.estimated_yield_kg_ha,
            "confidence": prediction.confidence,
            "confidence_tier": confidence_tier,
            "model_version": prediction.model_version,
            "details": details,
            "input_features": input_features,
            "explanation": explanation,
            "data_quality": data_quality,
            "prediction_interval": dict(data_quality.get("prediction_interval") or details.get("prediction_interval") or {}),
            "model_applicability": {
                "supported": supported,
                "coverage_score": applicability_valid_feature_count / max(len(ALL_FEATURE_NAMES), 1),
                "feature_gaps": applicability_missing,
                "confidence_tier": confidence_tier,
            },
            "training_domain": {
                "samples": int(details.get("training_samples") or 0),
                "crop_code": crop.code,
                "confidence_tier": confidence_tier,
            },
            "feature_coverage": {
                "available": applicability_available,
                "missing": applicability_missing,
            },
            "crop_suitability": crop_suitability,
            "crop_hint": crop_hint,
            "seasonal_series": dict(analytics.get("seasonal_series") or {}),
            "phenology": dict(analytics.get("phenology") or {}),
            "anomalies": list(analytics.get("anomalies") or []),
            "water_balance": dict(analytics.get("water_balance") or {}),
            "risk": dict(analytics.get("risk") or {}),
            "history_trend": dict(analytics.get("history_trend") or {}),
            "forecast_curve": forecast_curve,
            "management_zone_summary": dict(zones.get("summary") or {}),
            "driver_breakdown": normalized_drivers,
            "geometry_quality_impact": {
                **dict(details.get("geometry_quality_impact") or {}),
                **{k: v for k, v in geometry_summary.items() if v is not None},
            },
            "support_reason": support_reason,
            "support_reason_code": support_reason_code,
            "support_reason_params": support_reason_params,
            "operational_tier": trust_meta.get("operational_tier"),
            "review_required": bool(trust_meta.get("review_required")),
            "review_reason": trust_meta.get("review_reason"),
            "review_reason_code": trust_meta.get("review_reason_code"),
            "review_reason_params": trust_meta.get("review_reason_params") or {},
            "freshness": build_freshness(
                provider="yield_model",
                fetched_at=prediction_date,
                cache_written_at=prediction_date,
                model_version=prediction.model_version,
                dataset_version=details.get("dataset_version"),
            ),
        }


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    payload = metrics.get(key)
    if not isinstance(payload, dict):
        return None
    value = payload.get("mean")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _compactness(area_m2: float | None, perimeter_m: float | None) -> float | None:
    if not area_m2 or not perimeter_m:
        return None
    return max(0.1, min(1.0, (4.0 * np.pi * float(area_m2)) / max(float(perimeter_m) ** 2, 1.0)))


def _estimate_global_baseline(
    crop: Crop,
    features: dict[str, float | None],
    *,
    weather: dict[str, Any] | None = None,
    current_metrics: dict[str, Any] | None = None,
    scenario_adjustments: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Agronomic yield estimation based on Liebig's Law of the Minimum.

    References:
        - Mitscherlich (1909) diminishing returns for nutrient response
        - Cerrato & Blackmer (1990) nitrogen response plateau models
        - FAO-56 (Allen et al., 1998) water stress coefficients
        - RUSLE (Renard et al., 1997) soil erosion factors
        - Lobell et al. (2011) VPD heat stress on crop yields
    """
    del weather, current_metrics
    crop_baseline = float(features.get("crop_baseline") or crop.yield_baseline_kg_ha or 0.0)

    # --- Core agronomic factors (each represents a limiting resource) ---
    crop_code = str(crop.code or "").strip().lower()
    factors = {
        "area_shape": _area_shape_factor(features.get("field_area_ha"), features.get("compactness")),
        "soil_profile": _soil_factor(features, crop_code),
        "management": _management_factor(features.get("management_total_amount"), features),
        "vegetation_signal": _vegetation_factor(features, crop),
        "hydro_stress": _hydro_factor(features),
        "wind_stress": _wind_factor(features.get("current_wind_speed_m_s")),
        "climate_suitability": _climate_suitability_factor(features, crop),
    }

    # Solar radiation: Monteith (1977) RUE theory — radiation limits photosynthesis
    # and is a first-order constraint when light is scarce (heavy cloud cover seasons).
    # Only applied when ERA5-derived seasonal mean is available (> 0).
    _solar_rad = features.get("seasonal_solar_radiation_mean")
    if _solar_rad is not None and float(_solar_rad) > 0.0:
        # seasonal_solar_radiation_mean is weekly MJ/m²; _radiation_factor expects daily MJ/m²/day
        factors["solar_radiation"] = _radiation_factor(float(_solar_rad) / 7.0)

    # --- Scenario-driven factors ---
    scenario_factors = _scenario_response_factors(features, scenario_adjustments or {})
    factors.update(scenario_factors)

    # --- Interaction penalties (cross-factor stress amplification) ---
    interaction = _interaction_penalty(features, scenario_adjustments or {})
    if interaction != 1.0:
        factors["interaction_penalty"] = interaction

    # Liebig: yield limited by the worst factor, not just the product.
    # We use a weighted blend: 60 % geometric mean × 40 % minimum factor,
    # so a single bad factor drags yield down hard (as in real fields).
    factor_values = list(factors.values())
    geometric = float(np.prod(np.asarray(factor_values, dtype=float)) ** (1.0 / max(len(factor_values), 1)))
    liebig_min = float(min(factor_values))
    combined = 0.60 * geometric + 0.40 * liebig_min

    estimate = crop_baseline * float(np.clip(combined, 0.15, 1.45))

    historical_mean = features.get("historical_field_mean_yield")
    if historical_mean is not None:
        # Exponential time decay: recent observations anchor more strongly.
        # Half-life ~2.3 years (decay=0.3), so 5-year-old data has ~22% weight.
        current_year = datetime.now(timezone.utc).year
        n_seasons = int(features.get("_hist_n_seasons") or 1)
        latest_year = features.get("_hist_latest_year")
        years_ago = max(0, current_year - int(latest_year)) if latest_year is not None else 3
        decay = float(np.exp(-0.3 * years_ago))
        # Scale by number of seasons: 3+ recent seasons → full 25%; 1 season → ~8%
        season_factor = min(n_seasons / 3.0, 1.0)
        anchor_weight = 0.25 * decay * season_factor
        estimate = estimate * (1.0 - anchor_weight) + float(historical_mean) * anchor_weight
        factors["historical_anchor"] = max(0.5, min(1.5, float(historical_mean) / max(crop_baseline, 1.0)))

    return round(max(0.0, estimate), 2), factors


def _estimate_global_baseline_from_features(
    *,
    crop_baseline: float,
    features: dict[str, float | None],
) -> float:
    stub_crop = Crop(
        id=0,
        code="train",
        name="train",
        category="train",
        yield_baseline_kg_ha=crop_baseline,
        ndvi_target=0.68,
        base_temp_c=5.0,
        description=None,
    )
    estimate, _ = _estimate_global_baseline(stub_crop, features)
    return float(estimate)


# ---------------------------------------------------------------------------
# Mitscherlich response helper: Y = peak * (1 - e^(-c*(x - x0)))
# Produces classic diminishing-returns curve with toxicity penalty past optimum.
# ---------------------------------------------------------------------------

def _mitscherlich_response(
    value: float,
    *,
    x_zero: float,
    x_optimal: float,
    x_toxic: float,
    x_lethal: float,
    min_factor: float,
    peak_factor: float,
    toxic_factor: float,
    lethal_factor: float,
) -> float:
    """Asymmetric response: Mitscherlich rise → plateau → toxicity decline.

    Below x_zero:  min_factor (severe deficiency)
    x_zero → x_optimal: Mitscherlich curve (diminishing returns)
    x_optimal → x_toxic: plateau (peak_factor)
    x_toxic → x_lethal: linear decline to toxic_factor, then to lethal_factor
    """
    x = float(value)
    if x <= x_zero:
        return float(min_factor)
    if x <= x_optimal:
        # Mitscherlich: 1 - e^(-c*dx), scaled to reach peak_factor at x_optimal
        dx = (x - x_zero) / max(x_optimal - x_zero, 1e-6)
        # c chosen so that at dx=1 we reach ~95% of peak
        response = 1.0 - np.exp(-3.0 * dx)
        return float(min_factor + response * (peak_factor - min_factor))
    if x <= x_toxic:
        return float(peak_factor)
    if x <= x_lethal:
        ratio = (x - x_toxic) / max(x_lethal - x_toxic, 1e-6)
        return float(peak_factor - ratio * (peak_factor - toxic_factor))
    # Beyond lethal
    over = min((x - x_lethal) / max(x_lethal, 1e-6), 1.0)
    return float(max(lethal_factor, toxic_factor - over * (toxic_factor - lethal_factor)))


_CROP_NUTRIENT_PROFILES: dict[str, dict[str, dict]] = {
    "wheat": {
        "soil_n_ppm": dict(x_zero=3.0, x_optimal=22.0, x_toxic=42.0, x_lethal=58.0,
                           min_factor=0.40, peak_factor=1.10, toxic_factor=0.82, lethal_factor=0.55),
        "soil_p_ppm": dict(x_zero=2.0, x_optimal=16.0, x_toxic=38.0, x_lethal=52.0,
                           min_factor=0.50, peak_factor=1.07, toxic_factor=0.86, lethal_factor=0.62),
        "soil_k_ppm": dict(x_zero=5.0, x_optimal=25.0, x_toxic=50.0, x_lethal=72.0,
                           min_factor=0.50, peak_factor=1.07, toxic_factor=0.84, lethal_factor=0.60),
    },
    "corn": {
        "soil_n_ppm": dict(x_zero=4.0, x_optimal=28.0, x_toxic=50.0, x_lethal=65.0,
                           min_factor=0.35, peak_factor=1.12, toxic_factor=0.80, lethal_factor=0.50),
        "soil_p_ppm": dict(x_zero=2.0, x_optimal=18.0, x_toxic=40.0, x_lethal=55.0,
                           min_factor=0.45, peak_factor=1.08, toxic_factor=0.85, lethal_factor=0.60),
        "soil_k_ppm": dict(x_zero=5.0, x_optimal=28.0, x_toxic=55.0, x_lethal=75.0,
                           min_factor=0.45, peak_factor=1.08, toxic_factor=0.82, lethal_factor=0.58),
    },
    "soy": {
        "soil_n_ppm": dict(x_zero=2.0, x_optimal=10.0, x_toxic=28.0, x_lethal=42.0,
                           min_factor=0.55, peak_factor=1.05, toxic_factor=0.85, lethal_factor=0.65),
        "soil_p_ppm": dict(x_zero=2.0, x_optimal=15.0, x_toxic=36.0, x_lethal=50.0,
                           min_factor=0.50, peak_factor=1.06, toxic_factor=0.88, lethal_factor=0.65),
        "soil_k_ppm": dict(x_zero=5.0, x_optimal=24.0, x_toxic=50.0, x_lethal=70.0,
                           min_factor=0.50, peak_factor=1.06, toxic_factor=0.85, lethal_factor=0.62),
    },
    "barley": {
        "soil_n_ppm": dict(x_zero=3.0, x_optimal=18.0, x_toxic=38.0, x_lethal=52.0,
                           min_factor=0.42, peak_factor=1.08, toxic_factor=0.84, lethal_factor=0.58),
        "soil_p_ppm": dict(x_zero=2.0, x_optimal=14.0, x_toxic=35.0, x_lethal=48.0,
                           min_factor=0.50, peak_factor=1.06, toxic_factor=0.88, lethal_factor=0.65),
        "soil_k_ppm": dict(x_zero=5.0, x_optimal=22.0, x_toxic=48.0, x_lethal=68.0,
                           min_factor=0.50, peak_factor=1.06, toxic_factor=0.85, lethal_factor=0.62),
    },
    "sunflower": {
        "soil_n_ppm": dict(x_zero=3.0, x_optimal=16.0, x_toxic=35.0, x_lethal=50.0,
                           min_factor=0.45, peak_factor=1.07, toxic_factor=0.84, lethal_factor=0.58),
        "soil_p_ppm": dict(x_zero=2.0, x_optimal=12.0, x_toxic=32.0, x_lethal=48.0,
                           min_factor=0.50, peak_factor=1.05, toxic_factor=0.88, lethal_factor=0.65),
        "soil_k_ppm": dict(x_zero=5.0, x_optimal=20.0, x_toxic=45.0, x_lethal=65.0,
                           min_factor=0.50, peak_factor=1.06, toxic_factor=0.85, lethal_factor=0.62),
    },
    "rapeseed": {
        "soil_n_ppm": dict(x_zero=3.0, x_optimal=24.0, x_toxic=45.0, x_lethal=60.0,
                           min_factor=0.38, peak_factor=1.10, toxic_factor=0.82, lethal_factor=0.55),
        "soil_p_ppm": dict(x_zero=2.0, x_optimal=16.0, x_toxic=38.0, x_lethal=52.0,
                           min_factor=0.50, peak_factor=1.07, toxic_factor=0.86, lethal_factor=0.62),
        "soil_k_ppm": dict(x_zero=5.0, x_optimal=26.0, x_toxic=52.0, x_lethal=72.0,
                           min_factor=0.48, peak_factor=1.07, toxic_factor=0.84, lethal_factor=0.60),
    },
}

_DEFAULT_NUTRIENT_CONFIG: dict[str, dict] = {
    "soil_n_ppm": dict(x_zero=3.0, x_optimal=18.0, x_toxic=38.0, x_lethal=55.0,
                       min_factor=0.40, peak_factor=1.08, toxic_factor=0.85, lethal_factor=0.60),
    "soil_p_ppm": dict(x_zero=2.0, x_optimal=14.0, x_toxic=35.0, x_lethal=50.0,
                       min_factor=0.50, peak_factor=1.06, toxic_factor=0.88, lethal_factor=0.65),
    "soil_k_ppm": dict(x_zero=5.0, x_optimal=22.0, x_toxic=48.0, x_lethal=70.0,
                       min_factor=0.50, peak_factor=1.06, toxic_factor=0.85, lethal_factor=0.62),
}

# Aliases for Cyrillic and alternative crop names
_CROP_CODE_ALIASES: dict[str, str] = {
    "maize": "corn", "кукуруза": "corn",
    "soybean": "soy", "соя": "soy",
    "ячмень": "barley", "oats": "barley", "овес": "barley",
    "canola": "rapeseed", "рапс": "rapeseed",
    "пшеница": "wheat",
    "подсолнечник": "sunflower",
}


def _crop_nutrient_configs(crop_code: str) -> dict[str, dict]:
    """Return crop-specific Mitscherlich nutrient parameters."""
    code = str(crop_code or "").strip().lower()
    code = _CROP_CODE_ALIASES.get(code, code)
    return _CROP_NUTRIENT_PROFILES.get(code, _DEFAULT_NUTRIENT_CONFIG)


def _area_shape_factor(field_area_ha: float | None, compactness: float | None) -> float:
    """Edge effects reduce yield on small/irregular fields (border spray drift,
    machinery turning losses). Based on Sparkes et al. (1998) field edge yield loss."""
    area = 1.0 if field_area_ha is None else float(field_area_ha)
    compact = 0.65 if compactness is None else float(compactness)
    # Small fields (<2 ha) lose up to 15% from edge effects
    area_term = float(np.clip(0.85 + min(area, 50.0) / 50.0 * 0.15, 0.85, 1.0))
    # Irregular shapes (low compactness) lose up to 12% from machinery inefficiency
    shape_term = float(np.clip(0.88 + compact * 0.12, 0.88, 1.0))
    return float(np.clip(area_term * shape_term, 0.76, 1.0))


def _texture_to_code(texture: str | None) -> float | None:
    """Encode soil texture class to numeric code for feature vector.

    Codes ordered by water-holding capacity (low→high):
    1=sand, 2=loamy_sand, 3=sandy_loam, 4=loam, 5=silt_loam,
    6=clay_loam, 7=silty_clay_loam, 8=clay
    """
    if texture is None:
        return None
    t = str(texture).strip().lower().replace(" ", "_").replace("-", "_")
    _MAP = {
        "sand": 1.0, "песок": 1.0, "песчаная": 1.0,
        "loamy_sand": 2.0, "супесь": 2.0, "супесчаная": 2.0,
        "sandy_loam": 3.0, "легкий_суглинок": 3.0, "легкая_суглинистая": 3.0,
        "loam": 4.0, "суглинок": 4.0, "суглинистая": 4.0,
        "silt_loam": 5.0, "пылеватый_суглинок": 5.0,
        "silt": 5.0, "пылеватая": 5.0,
        "clay_loam": 6.0, "тяжелый_суглинок": 6.0, "тяжелая_суглинистая": 6.0,
        "silty_clay_loam": 7.0, "глинистый_суглинок": 7.0,
        "silty_clay": 7.5, "пылеватая_глина": 7.5,
        "sandy_clay": 6.5, "песчаная_глина": 6.5,
        "clay": 8.0, "глина": 8.0, "глинистая": 8.0,
        "чернозем": 4.5, "chernozem": 4.5,
    }
    return _MAP.get(t)


def _texture_nutrient_availability(texture_code: float | None) -> float:
    """Soil texture affects nutrient availability and water-holding capacity.

    Sandy soils (code 1-2): nutrients leach quickly, low CEC → lower availability
    Loam soils (code 3-5): optimal CEC and drainage → best availability
    Clay soils (code 6-8): high CEC but nutrients can be fixed → moderate availability

    Based on Brady & Weil (2017) "The Nature and Properties of Soils":
    - CEC increases sand→clay, but P/K fixation also increases in clay
    - Plant-available water peaks in loam/silt-loam soils
    """
    if texture_code is None:
        return 1.0
    tc = float(texture_code)
    if tc <= 1.5:
        return 0.82  # Sandy: low CEC, rapid leaching
    if tc <= 3.0:
        return float(0.82 + (tc - 1.5) / 1.5 * 0.13)  # Sandy loam: improving
    if tc <= 5.5:
        return 1.05  # Loam/silt-loam: optimal
    if tc <= 7.0:
        return float(1.05 - (tc - 5.5) / 1.5 * 0.10)  # Clay-loam: P fixation
    # Heavy clay: nutrient fixation, poor drainage
    return float(np.clip(0.95 - (tc - 7.0) * 0.10, 0.78, 0.95))


def _soil_factor(features: dict[str, float | None], crop_code: str = "") -> float:
    """Soil nutrient response using Liebig's Law of the Minimum.

    Each nutrient has a Mitscherlich response curve with toxicity at excess levels.
    Thresholds are crop-specific (wheat needs more N than soy, etc.).
    Soil texture modifies nutrient availability via CEC and drainage properties.

    References:
    - Cerrato & Blackmer (1990): N response plateau models
    - Barrow (1980): P toxicity (Zn/Fe lockout)
    - Mengel & Kirkby (2001): K interactions (Mg/Ca deficiency)
    - Brady & Weil (2017): texture-dependent CEC and nutrient availability
    """
    terms = []

    organic = features.get("soil_organic_matter_pct")
    if organic is not None:
        terms.append(_mitscherlich_response(
            float(organic),
            x_zero=0.3, x_optimal=2.5, x_toxic=5.5, x_lethal=8.0,
            min_factor=0.55, peak_factor=1.08, toxic_factor=1.02, lethal_factor=0.95,
        ))

    ph = features.get("soil_ph")
    if ph is not None:
        ph_val = float(ph)
        if ph_val < 5.0:
            terms.append(float(np.clip(0.35 + (ph_val - 4.0) * 0.45, 0.35, 0.80)))
        elif ph_val < 6.0:
            terms.append(float(0.80 + (ph_val - 5.0) * 0.25))
        elif ph_val <= 7.2:
            terms.append(1.05)
        elif ph_val <= 8.0:
            terms.append(float(1.05 - (ph_val - 7.2) * 0.19))
        else:
            terms.append(float(np.clip(0.90 - (ph_val - 8.0) * 0.35, 0.40, 0.90)))

    # NPK: Mitscherlich with toxicity — crop-specific thresholds
    nutrient_configs = _crop_nutrient_configs(crop_code)
    for key, cfg in nutrient_configs.items():
        value = features.get(key)
        if value is None:
            continue
        terms.append(_mitscherlich_response(float(value), **cfg))

    # Soil texture modifies effective nutrient availability
    texture_code = features.get("_soil_texture_code")
    texture_avail = _texture_nutrient_availability(texture_code)
    if texture_avail != 1.0:
        terms.append(texture_avail)

    if not terms:
        return 1.0

    # Liebig's Law: the most deficient nutrient limits yield
    return float(np.clip(min(terms), 0.35, 1.10))


def _management_factor(value: float | None, features: dict[str, float | None] | None = None) -> float:
    """Management intensity response with event-type differentiation.

    When typed management data is available (irrigation, fertilizer, pesticide),
    each type has its own Mitscherlich response curve. This is more accurate than
    treating all management events as equivalent.

    Without typed data, falls back to total management intensity.
    """
    features = features or {}
    irrig = features.get("_mgmt_irrigation")
    fert = features.get("_mgmt_fertilizer")
    pest = features.get("_mgmt_pesticide")

    # If we have typed management data, compute per-type factors
    if any(v is not None for v in [irrig, fert, pest]):
        typed_terms = []
        if irrig is not None:
            # Irrigation: optimal ~5-15 units, over-irrigation causes waterlogging
            typed_terms.append(_mitscherlich_response(
                float(irrig),
                x_zero=0.0, x_optimal=8.0, x_toxic=20.0, x_lethal=35.0,
                min_factor=0.80, peak_factor=1.08, toxic_factor=0.85, lethal_factor=0.60,
            ))
        if fert is not None:
            # Fertilizer: optimal ~5-12 units, over-fertilization causes salt stress
            typed_terms.append(_mitscherlich_response(
                float(fert),
                x_zero=0.0, x_optimal=10.0, x_toxic=25.0, x_lethal=40.0,
                min_factor=0.75, peak_factor=1.12, toxic_factor=0.78, lethal_factor=0.50,
            ))
        if pest is not None:
            # Pesticide: moderate application is protective, too much causes toxicity
            typed_terms.append(_mitscherlich_response(
                float(pest),
                x_zero=0.0, x_optimal=5.0, x_toxic=15.0, x_lethal=25.0,
                min_factor=0.85, peak_factor=1.05, toxic_factor=0.82, lethal_factor=0.60,
            ))
        if typed_terms:
            # Geometric mean: all management types contribute independently
            return float(np.clip(
                float(np.prod(np.asarray(typed_terms, dtype=float)) ** (1.0 / len(typed_terms))),
                0.45, 1.12,
            ))

    # Fallback: total management intensity
    if value is None:
        return 1.0
    # management_total_amount is stored in raw event units (kg/ha seeds,
    # mm irrigation, etc.). Divide by 20 to map to Mitscherlich [0–50] scale:
    #   0 raw    →  0 (no inputs → min_factor 0.70)
    #   200 raw  → 10 (typical sowing 200 kg/ha → x_optimal, peak 1.10)
    #   600 raw  → 30 (intensive season → approaching x_toxic)
    #  1000 raw  → 50 (x_lethal, floor 0.55)
    return _mitscherlich_response(
        float(value) / 20.0,
        x_zero=0.0, x_optimal=10.0, x_toxic=30.0, x_lethal=50.0,
        min_factor=0.70, peak_factor=1.10, toxic_factor=0.80, lethal_factor=0.55,
    )


def _radiation_factor(solar_rad_mj_day: float) -> float:
    """Solar radiation availability factor for yield (1.0 = optimal ~18 MJ/m²/day).

    Based on Monteith (1977) radiation use efficiency (RUE) theory and
    light-response curves for temperate field crops (Duncan et al. 1967).
    Mean growing-season daily radiation for mid-latitude crops:
      Wheat/barley: optimal 14–18 MJ/m²/day
      Maize/sunflower: optimal 18–22 MJ/m²/day

    Calibration:
      0        : no data → neutral (0.80, avoid over-penalising missing data)
      0–5      : heavy overcast / polar winter → severe shade stress
      5–10     : persistent cloud, below-optimal photosynthesis
      10–16    : moderate light, most crops near saturation
      16–22    : optimal range (cereal crops)
      > 22     : slight photo-inhibition / heat load above crop optimum

    Source: Allen et al. 1998 (FAO-56); Sinclair & Muchow 1999 (RUE review)
    """
    v = float(solar_rad_mj_day)
    if v <= 0.0:
        return 0.80          # Missing data: neutral — avoid false penalty
    if v < 5.0:
        return float(0.50 + v / 5.0 * 0.30)              # 0.50 → 0.80
    if v < 10.0:
        return float(0.80 + (v - 5.0) / 5.0 * 0.15)      # 0.80 → 0.95
    if v < 16.0:
        return float(0.95 + (v - 10.0) / 6.0 * 0.07)     # 0.95 → 1.02
    if v <= 22.0:
        return 1.02
    return float(max(0.90, 1.02 - (v - 22.0) * 0.020))   # slight decline


def _nitrogen_status_from_ndre(ndre: float) -> float:
    """Leaf nitrogen status proxy from Red Edge NDVI (Gitelson et al. 2003).

    NDRE = (NIR - RedEdge) / (NIR + RedEdge) correlates linearly with
    chlorophyll content and thus canopy N availability (Schlemmer et al. 2013,
    Remote Sensing of Environment 133:128-135).

    Calibration thresholds based on cereal crop studies:
      NDRE < 0.10 : severe N deficiency → factor 0.55
      0.10-0.20   : moderate deficiency → factor 0.55–0.78
      0.20-0.35   : sub-optimal         → factor 0.78–0.97
      0.35-0.50   : optimal range       → factor 0.97–1.05
      > 0.50      : luxury uptake / saturation → slight decline to 1.00
    """
    v = float(ndre)
    if v < 0.10:
        return 0.55
    if v < 0.20:
        return float(0.55 + (v - 0.10) / 0.10 * 0.23)
    if v < 0.35:
        return float(0.78 + (v - 0.20) / 0.15 * 0.19)
    if v <= 0.50:
        return float(0.97 + (v - 0.35) / 0.15 * 0.08)
    # Luxury uptake / canopy saturation — slight decline
    return float(np.clip(1.05 - (v - 0.50) * 0.25, 0.90, 1.05))


def _vegetation_factor(features: dict[str, float | None], crop: Crop) -> float:
    """Vegetation health from satellite data (NDVI, NDMI, NDRE, cumulative NDVI AUC).

    Uses both current NDVI snapshot and cumulative NDVI (area under curve)
    over the growing season for more robust yield estimation.

    Current NDVI: real-time canopy state (Monteith 1977 radiation use efficiency)
    Cumulative NDVI (AUC): total photosynthetic activity over the season
      (Lobell 2013, Burke & Lobell 2017 — AUC is a stronger predictor than
       any single NDVI observation because it captures the full growth trajectory)

    NDMI: water stress indicator from near-infrared
    """
    ndvi = features.get("current_ndvi_mean")
    ndmi = features.get("current_ndmi_mean")
    ndre = features.get("current_ndre_mean")
    ndvi_auc = features.get("ndvi_auc")
    ndvi_peak = features.get("ndvi_peak")
    ndvi_mean_season = features.get("ndvi_mean_season")

    if ndvi is None and ndmi is None and ndvi_auc is None and ndre is None:
        return 1.0

    terms = []

    if ndvi is not None:
        ndvi_val = float(ndvi)
        target = max(float(crop.ndvi_target or 0.68), 0.3)

        if ndvi_val < 0.15:
            terms.append(0.30)
        elif ndvi_val < 0.35:
            terms.append(float(0.30 + (ndvi_val - 0.15) / 0.20 * 0.40))
        elif ndvi_val <= target:
            ratio = (ndvi_val - 0.35) / max(target - 0.35, 0.01)
            terms.append(float(0.70 + ratio * 0.38))
        elif ndvi_val <= target + 0.15:
            terms.append(1.08)
        else:
            over = (ndvi_val - target - 0.15) / 0.15
            terms.append(float(np.clip(1.08 - over * 0.15, 0.85, 1.08)))

    # Cumulative NDVI AUC: captures total season photosynthetic capacity.
    # Typical AUC for a healthy cereal field (20-week season): ~12-16 NDVI·weeks
    # A stressed field: ~6-10 NDVI·weeks
    auc_term = None
    if ndvi_auc is not None and float(ndvi_auc) > 0:
        auc_val = float(ndvi_auc)
        # Normalize: optimal AUC ~14 NDVI·weeks for typical crop
        if auc_val < 4.0:
            auc_term = 0.40  # Very poor season
        elif auc_val < 8.0:
            auc_term = float(0.40 + (auc_val - 4.0) / 4.0 * 0.35)  # Stressed
        elif auc_val < 14.0:
            auc_term = float(0.75 + (auc_val - 8.0) / 6.0 * 0.33)  # Good
        elif auc_val <= 20.0:
            auc_term = 1.08  # Excellent
        else:
            # Very high AUC: possible weed growth or delayed senescence
            auc_term = float(np.clip(1.08 - (auc_val - 20.0) / 10.0 * 0.10, 0.90, 1.08))

    # Peak NDVI: proxy for maximum canopy development
    peak_term = None
    if ndvi_peak is not None:
        peak_val = float(ndvi_peak)
        target = max(float(crop.ndvi_target or 0.68), 0.3)
        if peak_val < 0.40:
            peak_term = 0.50
        elif peak_val < target:
            peak_term = float(0.50 + (peak_val - 0.40) / max(target - 0.40, 0.01) * 0.55)
        else:
            peak_term = 1.05

    if ndmi is not None:
        ndmi_val = float(ndmi)
        if ndmi_val < -0.3:
            terms.append(0.50)
        elif ndmi_val < 0.0:
            terms.append(float(0.50 + (ndmi_val + 0.3) / 0.3 * 0.40))
        elif ndmi_val <= 0.4:
            terms.append(float(0.90 + ndmi_val / 0.4 * 0.15))
        elif ndmi_val <= 0.55:
            terms.append(1.05)
        else:
            terms.append(float(np.clip(1.05 - (ndmi_val - 0.55) * 1.5, 0.55, 1.05)))

    # Weighted combination: AUC > current NDVI > NDMI > peak
    # AUC is the strongest predictor (Burke & Lobell 2017),
    # current NDVI captures real-time state,
    # NDMI captures water stress
    if auc_term is not None and len(terms) >= 2:
        # Have AUC + current NDVI + NDMI
        result = auc_term * 0.40 + terms[0] * 0.35 + terms[1] * 0.20
        if peak_term is not None:
            result += peak_term * 0.05
        else:
            result += auc_term * 0.05
    elif auc_term is not None and len(terms) == 1:
        # Have AUC + one of (NDVI, NDMI)
        result = auc_term * 0.45 + terms[0] * 0.45
        if peak_term is not None:
            result += peak_term * 0.10
        else:
            result += auc_term * 0.10
    elif auc_term is not None:
        # Have only AUC
        result = auc_term * 0.85
        if peak_term is not None:
            result += peak_term * 0.15
        else:
            result += auc_term * 0.15
    elif len(terms) == 2:
        # Original: NDVI 70% + NDMI 30%
        result = terms[0] * 0.7 + terms[1] * 0.3
    elif len(terms) == 1:
        result = terms[0]
    else:
        result = 1.0

    # NDRE nitrogen status: multiplicative modifier on the vegetation factor.
    # When soil_n_ppm is available, NDRE weight is halved to avoid double-counting N.
    # Source: Gitelson et al. 2003; Schlemmer et al. 2013 — NDRE ↔ leaf [N]
    if ndre is not None:
        ndre_val = float(ndre)
        ndre_factor = _nitrogen_status_from_ndre(ndre_val)
        # Weight: 0.17 normally, halved to 0.09 when soil N test is available
        soil_n = features.get("soil_n_ppm")
        ndre_weight = 0.09 if (soil_n is not None and float(soil_n) > 0) else 0.17
        result = result * (1.0 - ndre_weight) + ndre_factor * ndre_weight

    return float(np.clip(result, 0.25, 1.15))


def _hydro_factor(features: dict[str, float | None]) -> float:
    """Water stress factor based on FAO-56 framework (Allen et al. 1998).

    Soil moisture:
    - < 0.08: permanent wilting point → severe stress (Ks ≈ 0)
    - 0.08-0.18: water stress zone (Ks < 1)
    - 0.18-0.35: readily available water (Ks = 1)
    - 0.35-0.50: above field capacity → O₂ deficit
    - > 0.50: waterlogged → anaerobic root damage

    VPD (Lobell et al. 2011):
    - < 0.4 kPa: humid, fungal disease risk
    - 0.4-2.0 kPa: optimal transpiration
    - > 2.5 kPa: stomatal closure → photosynthesis drops
    - > 4.0 kPa: extreme heat stress

    Precipitation:
    - < 3 mm: drought conditions
    - 5-25 mm: optimal
    - > 50 mm: flood/erosion risk
    - > 100 mm: severe flooding
    """
    factors = []

    soil_moist = features.get("current_soil_moisture")
    if soil_moist is not None:
        sm = float(soil_moist)
        if sm < 0.05:
            factors.append(0.20)  # Below wilting point
        elif sm < 0.10:
            factors.append(float(0.20 + (sm - 0.05) / 0.05 * 0.35))  # Severe stress
        elif sm < 0.18:
            factors.append(float(0.55 + (sm - 0.10) / 0.08 * 0.40))  # Moderate stress
        elif sm <= 0.35:
            factors.append(float(np.clip(0.95 + (sm - 0.18) / 0.17 * 0.10, 0.95, 1.05)))  # Optimal
        elif sm <= 0.48:
            # Above field capacity — O₂ deficit begins
            factors.append(float(1.05 - (sm - 0.35) / 0.13 * 0.45))
        else:
            # Waterlogged — anaerobic conditions (Setter & Waters 2003)
            factors.append(float(np.clip(0.60 - (sm - 0.48) * 2.0, 0.15, 0.60)))

    vpd = features.get("current_vpd_mean")
    if vpd is not None:
        vpd_val = float(vpd)
        if vpd_val < 0.3:
            factors.append(0.85)  # Too humid — disease pressure
        elif vpd_val < 0.5:
            factors.append(float(0.85 + (vpd_val - 0.3) / 0.2 * 0.15))
        elif vpd_val <= 2.0:
            factors.append(1.0)  # Optimal transpiration
        elif vpd_val <= 3.5:
            # Stomatal closure begins (Lobell et al. 2011: ~5% yield loss per kPa above 2.0)
            factors.append(float(1.0 - (vpd_val - 2.0) * 0.12))
        else:
            # Extreme heat stress
            factors.append(float(np.clip(0.82 - (vpd_val - 3.5) * 0.20, 0.30, 0.82)))

    precipitation = features.get("current_precipitation_mm")
    if precipitation is not None:
        p = float(precipitation)
        if p < 2.0:
            factors.append(0.65)  # Drought
        elif p < 5.0:
            factors.append(float(0.65 + (p - 2.0) / 3.0 * 0.25))
        elif p <= 25.0:
            factors.append(float(np.clip(0.90 + (p - 5.0) / 20.0 * 0.12, 0.90, 1.02)))
        elif p <= 50.0:
            factors.append(float(1.02 - (p - 25.0) / 25.0 * 0.22))
        elif p <= 100.0:
            # Heavy rain: erosion, nutrient leaching
            factors.append(float(0.80 - (p - 50.0) / 50.0 * 0.30))
        else:
            # Flooding
            factors.append(float(np.clip(0.50 - (p - 100.0) / 100.0 * 0.25, 0.15, 0.50)))

    if not factors:
        return 1.0

    # Water stress: worst factor dominates (Liebig's Law for water)
    return float(np.clip(min(factors), 0.15, 1.08))


def _wind_factor(value: float | None) -> float:
    """Wind damage (lodging risk, mechanical stress).
    Based on Berry et al. (2004) lodging risk model for cereals.
    - < 5 m/s: negligible
    - 5-10 m/s: minor stress, beneficial air movement
    - 10-15 m/s: moderate lodging risk
    - > 15 m/s: severe lodging, crop damage
    """
    if value is None:
        return 1.0
    w = float(value)
    if w <= 5.0:
        return 1.0
    if w <= 10.0:
        return float(1.0 - (w - 5.0) / 5.0 * 0.08)
    if w <= 15.0:
        return float(0.92 - (w - 10.0) / 5.0 * 0.18)
    # Severe wind
    return float(np.clip(0.74 - (w - 15.0) / 10.0 * 0.24, 0.35, 0.74))


def _crop_suitability_profile(crop: Crop) -> dict[str, float | str]:
    code = str(crop.code or "").strip().lower()
    if code in {"corn", "maize", "кукуруза"}:
        return {
            "family": "warm_season",
            "lat_soft_max": 55.0,
            "lat_hard_max": 60.0,
            "gdd_min": 1500.0,
            "gdd_optimal": 2200.0,
            "precip_min": 220.0,
            "precip_optimal": 420.0,
            "precip_high": 780.0,
        }
    if code in {"soy", "soybean", "соя"}:
        return {
            "family": "warm_season",
            "lat_soft_max": 54.0,
            "lat_hard_max": 58.0,
            "gdd_min": 1650.0,
            "gdd_optimal": 2300.0,
            "precip_min": 240.0,
            "precip_optimal": 430.0,
            "precip_high": 760.0,
        }
    if code in {"barley", "ячмень", "oats", "овес"}:
        return {
            "family": "cool_season",
            "lat_soft_max": 65.0,
            "lat_hard_max": 69.0,
            "gdd_min": 850.0,
            "gdd_optimal": 1350.0,
            "precip_min": 160.0,
            "precip_optimal": 320.0,
            "precip_high": 700.0,
        }
    if code in {"rapeseed", "canola", "рапс"}:
        return {
            "family": "temperate_oilseed",
            "lat_soft_max": 61.0,
            "lat_hard_max": 66.0,
            "gdd_min": 1050.0,
            "gdd_optimal": 1650.0,
            "precip_min": 200.0,
            "precip_optimal": 360.0,
            "precip_high": 720.0,
        }
    return {
        "family": "temperate_grain",
        "lat_soft_max": 61.0,
        "lat_hard_max": 67.0,
        "gdd_min": 950.0,
        "gdd_optimal": 1600.0,
        "precip_min": 170.0,
        "precip_optimal": 340.0,
        "precip_high": 720.0,
    }


def _evaluate_crop_suitability(crop: Crop, features: dict[str, float | None]) -> dict[str, Any]:
    profile = _crop_suitability_profile(crop)
    latitude = _as_optional_float(features.get("latitude"))
    gdd_sum = _as_optional_float(features.get("seasonal_gdd_sum"))
    precipitation_sum = _as_optional_float(features.get("seasonal_precipitation_mm"))
    temp_mean = _as_optional_float(features.get("seasonal_temperature_mean_c"))
    observed_days = int(_as_optional_float(features.get("seasonal_observed_days")) or 0)

    score = 1.0
    warnings: list[str] = []
    reasons: list[str] = []

    lat_soft_max = float(profile["lat_soft_max"])
    lat_hard_max = float(profile["lat_hard_max"])
    if latitude is not None:
        abs_lat = abs(latitude)
        if abs_lat > lat_hard_max:
            score *= 0.18
            warnings.append("Широтная зона находится за пределами устойчивой агроклиматической пригодности культуры.")
            reasons.append("latitude_out_of_range")
        elif abs_lat > lat_soft_max:
            ratio = (abs_lat - lat_soft_max) / max(lat_hard_max - lat_soft_max, 1e-6)
            score *= float(np.clip(0.85 - ratio * 0.45, 0.40, 0.85))
            warnings.append("Широта выше оптимального диапазона культуры: ожидается сокращение вегетационного окна.")
            reasons.append("latitude_penalty")

    gdd_min = float(profile["gdd_min"])
    gdd_optimal = float(profile["gdd_optimal"])
    if gdd_sum is not None and gdd_sum > 0:
        if gdd_sum < gdd_min:
            score *= float(np.clip(0.25 + (gdd_sum / max(gdd_min, 1.0)) * 0.55, 0.20, 0.80))
            warnings.append("Сезонная сумма GDD ниже минимально комфортной для культуры.")
            reasons.append("gdd_deficit")
        elif gdd_sum < gdd_optimal:
            ratio = (gdd_sum - gdd_min) / max(gdd_optimal - gdd_min, 1e-6)
            score *= float(np.clip(0.88 + ratio * 0.12, 0.88, 1.0))
        elif gdd_sum > gdd_optimal * 1.35:
            score *= 0.92
            warnings.append("Сумма GDD заметно выше оптимума: возможен ускоренный стрессовый цикл культуры.")
            reasons.append("gdd_excess")
    elif str(profile["family"]) == "warm_season":
        score *= 0.80
        warnings.append("Для теплолюбивой культуры нет сезонного GDD-профиля: доверие к прогнозу снижено.")
        reasons.append("gdd_missing")

    precip_min = float(profile["precip_min"])
    precip_optimal = float(profile["precip_optimal"])
    precip_high = float(profile["precip_high"])
    if precipitation_sum is not None and precipitation_sum > 0:
        if precipitation_sum < precip_min:
            deficit = (precip_min - precipitation_sum) / max(precip_min, 1.0)
            score *= float(np.clip(0.92 - deficit * 0.40, 0.45, 0.92))
            warnings.append("Сезонный водный баланс ниже комфортного диапазона культуры.")
            reasons.append("precipitation_deficit")
        elif precipitation_sum > precip_high:
            excess = (precipitation_sum - precip_high) / max(precip_high, 1.0)
            score *= float(np.clip(0.95 - excess * 0.30, 0.50, 0.95))
            warnings.append("Осадков больше оптимума: растёт риск переувлажнения и потерь от водного стресса.")
            reasons.append("precipitation_excess")
        elif precipitation_sum < precip_optimal:
            ratio = (precipitation_sum - precip_min) / max(precip_optimal - precip_min, 1e-6)
            score *= float(np.clip(0.92 + ratio * 0.08, 0.92, 1.0))

    if temp_mean is not None and str(profile["family"]) == "warm_season" and temp_mean < 11.0:
        score *= 0.75
        warnings.append("Средняя температура сезона слишком низка для теплолюбивой культуры.")
        reasons.append("temperature_deficit")

    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.82:
        status = "high"
    elif score >= 0.62:
        status = "moderate"
    elif score >= 0.35:
        status = "low"
    else:
        status = "unsuitable"

    _recommendations = {
        "high": "Климатические условия соответствуют оптимуму для данной культуры. Ограничивающих факторов не выявлено.",
        "moderate": "Культура адаптирована к условиям региона, однако отдельные агроклиматические факторы снижают потенциал урожайности.",
        "low": "Возможно выращивание при интенсивной агротехнике: орошение, защита от стресса, подбор сортов.",
        "unsuitable": "Агроклиматические условия за пределами допустимого диапазона для культуры. Рекомендуется выбор альтернативной культуры.",
    }
    return {
        "status": status,
        "score": round(score, 4),
        "yield_factor": round(float(np.clip(0.35 + score * 0.70, 0.20, 1.05)), 4),
        "warnings": warnings,
        "reasons": reasons,
        "recommendation": _recommendations.get(status, ""),
        "latitude": latitude,
        "seasonal_gdd_sum": gdd_sum,
        "seasonal_precipitation_mm": precipitation_sum,
        "seasonal_temperature_mean_c": temp_mean,
        "observed_days": observed_days,
        "support_reason": warnings[0] if status in {"low", "unsuitable"} and warnings else None,
    }


def _climate_suitability_factor(features: dict[str, float | None], crop: Crop) -> float:
    suitability = _evaluate_crop_suitability(crop, features)
    return float(np.clip(float(suitability.get("yield_factor") or 1.0), 0.20, 1.05))


def _interaction_penalty(
    features: dict[str, float | None],
    scenario_adjustments: dict[str, float],
) -> float:
    """Cross-factor interaction penalties grounded in agronomic research.

    Real fields suffer compounding stress when multiple factors are adverse:
    1. Drought × high fertilizer = salt stress (osmotic shock, Munns 2002)
    2. Excess water × high N = nitrate leaching + anaerobic loss (Di & Cameron 2002)
    3. High VPD × low soil moisture = amplified transpiration deficit
    4. Soil compaction × excess water = severe O₂ deficit
    """
    penalty = 1.0
    soil_moisture = features.get("current_soil_moisture")
    vpd = features.get("current_vpd_mean")
    soil_n = features.get("soil_n_ppm")
    fertilizer_pct = float(scenario_adjustments.get("fertilizer_pct") or 0.0)
    irrigation_pct = float(scenario_adjustments.get("irrigation_pct") or 0.0)

    # 1. Drought × Fertilizer = salt stress
    if soil_moisture is not None and float(soil_moisture) < 0.12:
        if fertilizer_pct > 15 or (soil_n is not None and float(soil_n) > 35):
            # High salt concentration in dry soil damages roots
            severity = min((35.0 - min(float(soil_moisture), 0.12) * 100.0) / 35.0, 1.0)
            fert_load = max(fertilizer_pct, 0.0) / 50.0
            penalty *= float(np.clip(1.0 - severity * fert_load * 0.30, 0.55, 1.0))

    # 2. Excess water × high N = leaching / denitrification
    if soil_moisture is not None and float(soil_moisture) > 0.40:
        n_excess = max(fertilizer_pct, 0.0) / 100.0
        if soil_n is not None:
            n_excess += max(float(soil_n) - 30.0, 0.0) / 30.0
        if n_excess > 0:
            wetness = (float(soil_moisture) - 0.40) / 0.20
            penalty *= float(np.clip(1.0 - wetness * n_excess * 0.25, 0.60, 1.0))

    # 3. High VPD × low soil moisture = amplified transpiration deficit
    if vpd is not None and float(vpd) > 2.0 and soil_moisture is not None and float(soil_moisture) < 0.18:
        vpd_stress = (float(vpd) - 2.0) / 2.0
        drought_stress = (0.18 - float(soil_moisture)) / 0.15
        penalty *= float(np.clip(1.0 - vpd_stress * drought_stress * 0.20, 0.55, 1.0))

    # 4. Over-irrigation on already wet soil (compaction + runoff)
    if soil_moisture is not None and float(soil_moisture) > 0.35 and irrigation_pct > 20:
        penalty *= float(np.clip(1.0 - (irrigation_pct - 20.0) / 80.0 * 0.18, 0.75, 1.0))

    return float(np.clip(penalty, 0.40, 1.0))


def _scenario_response_factors(
    features: dict[str, float | None],
    scenario_adjustments: dict[str, float],
) -> dict[str, float]:
    """Scenario modeling with agronomic response curves.

    Each scenario parameter follows a non-monotonic response:
    - Small increases help (diminishing returns)
    - Large increases cause penalties (toxicity, waterlogging, compaction)

    Based on:
    - Mitscherlich (1909) nutrient response curves
    - FAO-56 water balance framework
    - Cerrato & Blackmer (1990) N response plateau model
    - RUSLE (Renard et al. 1997) erosion from rainfall
    """
    if not scenario_adjustments:
        return {}

    factors: dict[str, float] = {}
    soil_moisture = features.get("current_soil_moisture")
    precipitation = features.get("current_precipitation_mm")
    soil_n = features.get("soil_n_ppm")

    # --- Irrigation (% change) ---
    irrigation_pct = float(scenario_adjustments.get("irrigation_pct") or 0.0)
    if irrigation_pct:
        sm = float(soil_moisture) if soil_moisture is not None else 0.22
        if irrigation_pct > 0:
            # Benefit depends on current soil moisture deficit
            deficit = max(0.30 - sm, 0.0)  # Distance from optimal
            # Mitscherlich: benefit = A * (1 - e^(-c * irrigation))
            # but only up to field capacity; beyond that, waterlogging
            benefit = deficit * 2.0 * (1.0 - np.exp(-0.04 * irrigation_pct))
            # Waterlogging penalty: saturating soil moisture model
            fc = 0.35  # field capacity
            irrig_add = max(fc - sm, 0.0) * (1.0 - float(np.exp(-0.03 * irrigation_pct)))
            effective_sm = sm + irrig_add
            if effective_sm > 0.40:
                waterlog = (effective_sm - 0.40) / 0.20
                penalty = waterlog * 0.35
            else:
                penalty = 0.0
            factors["scenario_irrigation"] = float(np.clip(1.0 + benefit - penalty, 0.50, 1.12))
        else:
            # Reducing irrigation when soil is already dry = worse
            if sm < 0.20:
                factors["scenario_irrigation"] = float(np.clip(1.0 + irrigation_pct / 100.0 * 0.30, 0.60, 1.0))
            else:
                # Soil is wet enough; reducing irrigation is mostly fine
                factors["scenario_irrigation"] = float(np.clip(1.0 + irrigation_pct / 200.0 * 0.05, 0.90, 1.0))

    # --- Fertilizer (% change) ---
    fertilizer_pct = float(scenario_adjustments.get("fertilizer_pct") or 0.0)
    if fertilizer_pct:
        current_n = float(soil_n) if soil_n is not None else 15.0
        # Ensure a minimum baseline so that percentage changes produce
        # non-zero effects even when current_n is very low or missing.
        current_n = max(current_n, 8.0)
        if fertilizer_pct > 0:
            # Cerrato & Blackmer (1990): quadratic-plateau N response
            # Effective N after addition
            effective_n = current_n * (1.0 + fertilizer_pct / 100.0)
            if effective_n <= 25.0:
                # Below optimum: Mitscherlich benefit
                gain = (1.0 - np.exp(-0.08 * fertilizer_pct)) * 0.15
                factors["scenario_fertilizer"] = float(np.clip(1.0 + gain, 1.0, 1.15))
            elif effective_n <= 40.0:
                # Plateau: minimal additional benefit
                gain = 0.15 * np.exp(-0.05 * (effective_n - 25.0))
                factors["scenario_fertilizer"] = float(np.clip(1.0 + gain, 0.95, 1.15))
            elif effective_n <= 55.0:
                # Toxicity zone: lodging risk, root burn
                toxicity = (effective_n - 40.0) / 15.0
                factors["scenario_fertilizer"] = float(np.clip(1.0 - toxicity * 0.25, 0.75, 1.0))
            else:
                # Severe over-fertilization: salt stress, crop death
                factors["scenario_fertilizer"] = float(np.clip(0.75 - (effective_n - 55.0) / 30.0 * 0.30, 0.35, 0.75))
        else:
            # Reducing fertilizer when soil N is already low = worse
            if current_n < 15.0:
                factors["scenario_fertilizer"] = float(np.clip(1.0 + fertilizer_pct / 100.0 * 0.25, 0.60, 1.0))
            else:
                factors["scenario_fertilizer"] = float(np.clip(1.0 + fertilizer_pct / 200.0 * 0.10, 0.85, 1.0))

    # --- Expected rain (mm) ---
    expected_rain_mm = float(scenario_adjustments.get("expected_rain_mm") or 0.0)
    if expected_rain_mm:
        sm = float(soil_moisture) if soil_moisture is not None else 0.22
        current_precip = float(precipitation) if precipitation is not None else 10.0
        total_water = current_precip + expected_rain_mm

        if total_water <= 5.0:
            factors["scenario_rainfall"] = float(np.clip(0.65 + total_water / 5.0 * 0.25, 0.65, 0.90))
        elif total_water <= 30.0:
            factors["scenario_rainfall"] = float(np.clip(0.90 + (total_water - 5.0) / 25.0 * 0.12, 0.90, 1.02))
        elif total_water <= 60.0:
            factors["scenario_rainfall"] = float(1.02 - (total_water - 30.0) / 30.0 * 0.20)
        elif total_water <= 120.0:
            # Heavy rain: RUSLE-style erosion + nutrient leaching
            factors["scenario_rainfall"] = float(0.82 - (total_water - 60.0) / 60.0 * 0.32)
        else:
            factors["scenario_rainfall"] = float(np.clip(0.50 - (total_water - 120.0) / 100.0 * 0.20, 0.20, 0.50))

        # Soil erosion component from heavy rainfall (simplified RUSLE R-factor)
        if expected_rain_mm > 30.0:
            erosion = min((expected_rain_mm - 30.0) / 70.0, 1.0) ** 1.5 * 0.15
            factors["scenario_erosion"] = float(np.clip(1.0 - erosion, 0.70, 1.0))

    # --- Temperature delta (°C deviation from normal) ---
    temp_delta = float(scenario_adjustments.get("temperature_delta_c") or 0.0)
    if temp_delta:
        if temp_delta > 0:
            # Heat stress: each +1°C above optimal reduces yield ~5% (Lobell et al. 2011)
            if temp_delta <= 2.0:
                factors["scenario_temperature"] = float(np.clip(1.0 - temp_delta * 0.03, 0.94, 1.0))
            elif temp_delta <= 5.0:
                factors["scenario_temperature"] = float(0.94 - (temp_delta - 2.0) * 0.06)
            else:
                factors["scenario_temperature"] = float(np.clip(0.76 - (temp_delta - 5.0) * 0.08, 0.30, 0.76))
        else:
            # Cold stress: frost damage, slow growth
            cold = abs(temp_delta)
            if cold <= 2.0:
                factors["scenario_temperature"] = float(np.clip(1.0 - cold * 0.02, 0.96, 1.0))
            elif cold <= 5.0:
                factors["scenario_temperature"] = float(0.96 - (cold - 2.0) * 0.07)
            else:
                # Frost damage
                factors["scenario_temperature"] = float(np.clip(0.75 - (cold - 5.0) * 0.10, 0.25, 0.75))

    # --- Planting density (% change from standard) ---
    density_pct = float(scenario_adjustments.get("planting_density_pct") or 0.0)
    if density_pct:
        if density_pct > 0:
            # More plants: competition for light/water/nutrients
            if density_pct <= 15:
                factors["scenario_density"] = float(1.0 + density_pct / 15.0 * 0.05)
            elif density_pct <= 40:
                factors["scenario_density"] = float(1.05 - (density_pct - 15.0) / 25.0 * 0.15)
            else:
                # Severe overcrowding
                factors["scenario_density"] = float(np.clip(0.90 - (density_pct - 40.0) / 60.0 * 0.25, 0.55, 0.90))
        else:
            # Fewer plants: reduced canopy, wasted resources
            under = abs(density_pct)
            factors["scenario_density"] = float(np.clip(1.0 - under / 100.0 * 0.30, 0.60, 1.0))

    # --- Tillage type (0=no-till, 1=minimum, 2=conventional, 3=deep) ---
    tillage = scenario_adjustments.get("tillage_type")
    if tillage is not None:
        tillage_val = int(float(tillage))
        # No-till preserves soil structure, moisture (Pittelkow et al. 2015)
        # Conventional tillage: short-term aeration benefit but long-term degradation
        tillage_map = {0: 1.02, 1: 1.0, 2: 0.96, 3: 0.90}
        factors["scenario_tillage"] = tillage_map.get(tillage_val, 1.0)

    # --- Pest pressure (0=low, 1=moderate, 2=high, 3=severe) ---
    pest = scenario_adjustments.get("pest_pressure")
    if pest is not None:
        pest_val = int(float(pest))
        # Oerke (2006): crop losses 20-40% without protection
        pest_map = {0: 1.0, 1: 0.92, 2: 0.78, 3: 0.55}
        factors["scenario_pest"] = pest_map.get(pest_val, 1.0)

    # --- Soil compaction index (0-1 scale) ---
    compaction = scenario_adjustments.get("soil_compaction")
    if compaction is not None:
        comp_val = float(np.clip(float(compaction), 0.0, 1.0))
        # Hamza & Anderson (2005): compaction reduces root growth, water infiltration
        factors["scenario_compaction"] = float(np.clip(1.0 - comp_val * 0.35, 0.65, 1.0))

    return factors


def _optimum_band_factor(
    value: float,
    *,
    min_value: float,
    low: float,
    high: float,
    max_value: float,
    min_factor: float,
    peak_factor: float,
    high_end_factor: float,
) -> float:
    x = float(np.clip(value, min_value, max_value))
    if x <= low:
        if low <= min_value:
            return float(peak_factor)
        ratio = (x - min_value) / max(low - min_value, 1e-6)
        return float(min_factor + ratio * (peak_factor - min_factor))
    if x <= high:
        return float(peak_factor)
    ratio = (x - high) / max(max_value - high, 1e-6)
    return float(peak_factor - ratio * (peak_factor - high_end_factor))


def _materialize_training_matrix(rows: list[TrainingRow]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    medians: dict[str, float] = {}
    columns: list[list[float]] = []
    for name in CALIBRATION_FEATURE_NAMES:
        values = [float(row.features.get(name)) for row in rows if row.features.get(name) is not None]
        medians[name] = float(np.median(np.asarray(values, dtype=float))) if values else 0.0
    for row in rows:
        columns.append(
            [
                1.0,
                *[
                    float(row.features.get(name) if row.features.get(name) is not None else medians[name])
                    for name in CALIBRATION_FEATURE_NAMES
                ],
            ]
        )
    x_train = np.asarray(columns, dtype=float)
    y_train = np.asarray([row.target for row in rows], dtype=float)
    return x_train, y_train, medians


def _vectorize_current(features: dict[str, float | None], medians: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            *[
                float(features.get(name) if features.get(name) is not None else medians[name])
                for name in CALIBRATION_FEATURE_NAMES
            ],
        ],
        dtype=float,
    )
