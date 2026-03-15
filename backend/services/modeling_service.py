"""Сервис сценарного моделирования с guardrails по области применимости."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from services.payload_meta import build_freshness
from services.message_codes import classify_support_reason, normalize_risk_level
from services.temporal_analytics_service import normalize_driver_breakdown
from services.trust_service import describe_prediction_operational_tier
from services.weekly_profile_service import (
    FEATURE_SCHEMA_VERSION,
    current_season_year,
    ensure_weekly_profile,
    load_crop_hint,
    profile_has_signal,
    rows_to_weekly_inputs,
    summarize_geometry_quality,
)
from storage.db import Crop, ScenarioRun
from services.yield_service import YieldService

logger = get_logger(__name__)


# Vegetation index sensitivities to agronomic factors.
# Each entry: how much a 100% factor change shifts the index value relative to
# baseline. Signs matter: negative = factor harms the index at high values.
# yield_sens: correlation with yield ratio (already embeds non-linear effects).
# irr/fert/rain: direct moisture/nutrition effects (normalized 0-1 scale).
# temp: per-degree-Celsius impact on the index.
_VI_SENS: dict[str, dict[str, float]] = {
    # solar: sensitivity of VI to cloud_cover_factor deviation from 1.0
    # +0.1 per unit solar factor for NDVI (radiation drives photosynthesis, Monteith 1977)
    # NDMI/NDWI: less sensitive to radiation, more to water
    "ndvi":         {"yield": 0.62, "irr": 0.12, "fert": 0.18, "rain": 0.10, "temp": -0.025, "solar": 0.10},
    "ndmi":         {"yield": 0.28, "irr": 0.38, "fert": 0.04, "rain": 0.32, "temp": -0.018, "solar": 0.03},
    "ndwi":         {"yield": 0.18, "irr": 0.48, "fert": 0.00, "rain": 0.38, "temp": -0.010, "solar": 0.02},
    "mndwi":        {"yield": 0.18, "irr": 0.48, "fert": 0.00, "rain": 0.38, "temp": -0.010, "solar": 0.02},
    "soil_moisture":{"yield": 0.20, "irr": 0.42, "fert": 0.04, "rain": 0.36, "temp": -0.018, "solar": 0.00},
    "bsi":          {"yield": -0.38, "irr": -0.18, "fert": -0.08, "rain": -0.14, "temp": 0.010, "solar": -0.04},
    "evi":          {"yield": 0.55, "irr": 0.14, "fert": 0.20, "rain": 0.12, "temp": -0.022, "solar": 0.09},
    "savi":         {"yield": 0.55, "irr": 0.14, "fert": 0.20, "rain": 0.12, "temp": -0.022, "solar": 0.09},
}
_VI_SENS_DEFAULT: dict[str, float] = {"yield": 0.40, "irr": 0.18, "fert": 0.12, "rain": 0.16, "temp": -0.018, "solar": 0.05}
_VI_CLAMP: dict[str, tuple[float, float]] = {
    "ndvi": (-0.2, 1.0), "ndmi": (-0.5, 0.8), "ndwi": (-0.5, 0.8), "mndwi": (-0.5, 0.8),
    "bsi": (-0.5, 1.0), "soil_moisture": (0.02, 0.65),
}
_VI_DEFAULT_CLAMP = (-1.0, 1.0)


def _driver_identity(raw: dict[str, Any] | None) -> str:
    payload = raw or {}
    return str(
        payload.get("input_key")
        or payload.get("driver_id")
        or payload.get("input")
        or payload.get("factor")
        or payload.get("label")
        or ""
    ).strip()


def _build_delta_driver_breakdown(
    *,
    baseline_drivers: list[dict[str, Any]] | None,
    scenario_drivers: list[dict[str, Any]] | None,
    baseline_inputs: dict[str, Any] | None,
    scenario_inputs: dict[str, Any] | None,
    baseline_yield_kg_ha: float,
    scenario_yield_kg_ha: float,
) -> list[dict[str, Any]]:
    baseline_map = {
        key: item
        for item in (baseline_drivers or [])
        if (key := _driver_identity(item))
    }
    scenario_map = {
        key: item
        for item in (scenario_drivers or [])
        if (key := _driver_identity(item))
    }
    raw_delta: list[dict[str, Any]] = []
    active_yield = max(float(scenario_yield_kg_ha or baseline_yield_kg_ha or 0.0), 1.0)

    for key in sorted(set(baseline_map) | set(scenario_map)):
        baseline_driver = baseline_map.get(key) or {}
        scenario_driver = scenario_map.get(key) or {}
        baseline_effect = float(baseline_driver.get("effect_kg_ha") or 0.0)
        scenario_effect = float(scenario_driver.get("effect_kg_ha") or 0.0)
        delta_effect = scenario_effect - baseline_effect
        if abs(delta_effect) < 0.5:
            continue
        label = (
            scenario_driver.get("label")
            or baseline_driver.get("label")
            or scenario_driver.get("factor")
            or baseline_driver.get("factor")
            or key
        )
        raw_delta.append(
            {
                "driver_id": scenario_driver.get("driver_id") or baseline_driver.get("driver_id") or key,
                "label": label,
                "input_key": scenario_driver.get("input_key") or baseline_driver.get("input_key") or key,
                "effect_kg_ha": round(delta_effect, 2),
                "effect_pct": round((delta_effect / active_yield) * 100.0, 2),
                "source": "scenario_delta",
                "confidence": min(
                    float(scenario_driver.get("confidence") or 0.72),
                    float(baseline_driver.get("confidence") or 0.72),
                ),
            }
        )

    return normalize_driver_breakdown(
        raw_delta,
        baseline_yield_kg_ha=baseline_yield_kg_ha,
        scenario_yield_kg_ha=scenario_yield_kg_ha,
        baseline_inputs=baseline_inputs or {},
        scenario_inputs=scenario_inputs or {},
        source="scenario_delta",
    )


def _project_scenario_series(
    baseline: dict,
    yield_ratio: float,
    factors: dict[str, float],
) -> dict:
    """Build synthetic 'with-intervention' time series by adjusting baseline metrics.

    Uses yield_ratio (scenario_yield / baseline_yield) as the primary signal,
    which already embeds non-linear Mitscherlich effects (excess is harmful too).
    Direct moisture effects are layered on top for NDMI/NDWI/soil_moisture.

    Args:
        baseline: seasonal_series dict with ``metrics[].points[].{observed_at, value}``
        yield_ratio: scenario_yield / baseline_yield. >1 improvement, <1 decline.
        factors: scenario_adj dict (irrigation_pct, fertilizer_pct, expected_rain_mm, …)
    """
    import copy
    if not baseline or not isinstance(baseline.get("metrics"), list):
        return dict(baseline)

    proj = copy.deepcopy(baseline)
    dy = max(-0.80, min(1.50, yield_ratio - 1.0))  # clamp unrealistic ratios

    # Normalize factor inputs to [-1, 1] or [0, 1] ranges
    irr_n = float(factors.get("irrigation_pct", 0)) / 100.0
    fert_n = float(factors.get("fertilizer_pct", 0)) / 100.0
    # rain: centre at 50 mm baseline (typical Sentinel coverage), range ±300 mm
    rain_n = (float(factors.get("expected_rain_mm", 50)) - 50.0) / 300.0
    rain_n = max(-0.5, min(0.5, rain_n))
    temp_d = float(factors.get("temperature_delta_c", 0))
    # cloud_cover_factor: 1.0 = baseline, >1 = clearer skies, <1 = overcast
    # Normalised as deviation from 1.0 so 0 = no change
    solar_d = float(factors.get("cloud_cover_factor", 1.0)) - 1.0

    for metric_entry in proj["metrics"]:
        mid = str(metric_entry.get("metric", ""))
        s = _VI_SENS.get(mid, _VI_SENS_DEFAULT)
        # Composite adjustment on the index VALUE (additive, domain-aware)
        adj = (
            dy      * s["yield"]
            + irr_n * s["irr"]
            + fert_n * s["fert"]
            + rain_n * s["rain"]
            + temp_d * s["temp"]
            + solar_d * s.get("solar", 0.05)
        )
        lo, hi = _VI_CLAMP.get(mid, _VI_DEFAULT_CLAMP)
        for pt in metric_entry.get("points") or []:
            v = pt.get("value")
            if v is not None:
                pt["value"] = round(max(lo, min(hi, float(v) + adj * abs(float(v)))), 4)

    return proj


class ModelingService:
    """Оценка влияния сценариев на прогноз урожая с анализом чувствительности."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.yield_service = YieldService(db)

    @staticmethod
    def _ensure_operational_field(field) -> None:
        if field is None:
            raise ValueError("Поле не найдено")
        if str(getattr(field, "source", "") or "").strip().lower() == "autodetect_preview":
            raise ValueError("Preview-контур из быстрого режима нельзя использовать для сценариев. Запустите Standard или Quality.")

    def _build_mechanistic_scenario_events(
        self,
        *,
        irrigation_pct: float,
        fertilizer_pct: float,
        expected_rain_mm: float,
        temperature_delta_c: float,
        planting_density_pct: float,
        tillage_type: int | None,
        pest_pressure: int | None,
        soil_compaction: float | None,
        precipitation_factor: float | None,
        sowing_shift_days: int | None,
        irrigation_events: list[dict[str, Any]] | None,
        fertilizer_events: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "irrigation_pct": float(irrigation_pct),
            "fertilizer_pct": float(fertilizer_pct),
            "expected_rain_mm": float(expected_rain_mm),
            "temperature_delta_c": float(temperature_delta_c),
            "planting_density_pct": float(planting_density_pct),
        }
        if precipitation_factor is not None:
            payload["precipitation_factor"] = float(precipitation_factor)
        if sowing_shift_days is not None:
            payload["sowing_shift_days"] = int(sowing_shift_days)
        if tillage_type is not None:
            payload["tillage_type"] = int(tillage_type)
        if pest_pressure is not None:
            payload["pest_pressure"] = int(pest_pressure)
        if soil_compaction is not None:
            payload["soil_compaction"] = float(soil_compaction)
        if irrigation_events:
            payload["irrigation_events"] = [dict(event) for event in irrigation_events]
        if fertilizer_events:
            payload["fertilizer_events"] = [dict(event) for event in fertilizer_events]
        return payload

    async def _build_scenario_forecast_curve(
        self,
        *,
        field,
        crop,
        organization_id: UUID,
        temperature_delta_c: float = 0.0,
        expected_rain_mm: float = 0.0,
        precipitation_factor: float | None = None,
    ) -> dict[str, Any]:
        baseline_curve = await self.yield_service.build_forecast_curve(
            field,
            crop,
            organization_id=organization_id,
        )
        scenario_curve = await self.yield_service.build_forecast_curve(
            field,
            crop,
            organization_id=organization_id,
            temperature_delta_c=temperature_delta_c,
            extra_precip_total_mm=expected_rain_mm,
            precipitation_factor=precipitation_factor,
        )
        return {
            "provider": scenario_curve.get("provider") or baseline_curve.get("provider"),
            "days": scenario_curve.get("days") or baseline_curve.get("days"),
            "base_temp_c": scenario_curve.get("base_temp_c") or baseline_curve.get("base_temp_c"),
            "freshness": scenario_curve.get("freshness") or baseline_curve.get("freshness") or {},
            "error": scenario_curve.get("error") or baseline_curve.get("error"),
            "baseline_points": list(baseline_curve.get("points") or []),
            "scenario_points": list(scenario_curve.get("points") or []),
        }

    async def simulate(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
        irrigation_pct: float,
        fertilizer_pct: float,
        expected_rain_mm: float,
        temperature_delta_c: float = 0.0,
        precipitation_factor: float | None = None,
        planting_density_pct: float = 0.0,
        sowing_shift_days: int | None = None,
        tillage_type: int | None = None,
        pest_pressure: int | None = None,
        soil_compaction: float | None = None,
        cloud_cover_factor: float = 1.0,
        irrigation_events: list[dict[str, Any]] | None = None,
        fertilizer_events: list[dict[str, Any]] | None = None,
        scenario_name: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        if self.db is None:
            return await self._simulate_scalar(
                field_id,
                organization_id=organization_id,
                crop_code=crop_code,
                irrigation_pct=irrigation_pct,
                fertilizer_pct=fertilizer_pct,
                expected_rain_mm=expected_rain_mm,
                temperature_delta_c=temperature_delta_c,
                precipitation_factor=precipitation_factor,
                planting_density_pct=planting_density_pct,
                sowing_shift_days=sowing_shift_days,
                tillage_type=tillage_type,
                pest_pressure=pest_pressure,
                soil_compaction=soil_compaction,
                cloud_cover_factor=cloud_cover_factor,
                scenario_name=scenario_name,
                save=save,
            )

        current_year = current_season_year()
        weekly_rows = await ensure_weekly_profile(
            self.db,
            organization_id=organization_id,
            field_id=field_id,
            season_year=current_year,
        )
        mechanistic_capable = bool(
            irrigation_events
            or fertilizer_events
            or precipitation_factor is not None
            or temperature_delta_c
        )
        unsupported_mechanistic_overrides = bool(
            irrigation_pct
            or fertilizer_pct
            or expected_rain_mm
            or planting_density_pct
            or sowing_shift_days is not None
            or tillage_type is not None
            or pest_pressure is not None
            or soil_compaction is not None
        )
        if (
            len(weekly_rows) >= 3
            and profile_has_signal(weekly_rows)
            and mechanistic_capable
            and not unsupported_mechanistic_overrides
        ):
            return await self.simulate_mechanistic(
                field_id,
                organization_id=organization_id,
                crop_code=crop_code,
                scenario_events=self._build_mechanistic_scenario_events(
                    irrigation_pct=irrigation_pct,
                    fertilizer_pct=fertilizer_pct,
                    expected_rain_mm=expected_rain_mm,
                    temperature_delta_c=temperature_delta_c,
                    precipitation_factor=precipitation_factor,
                    planting_density_pct=planting_density_pct,
                    sowing_shift_days=sowing_shift_days,
                    tillage_type=tillage_type,
                    pest_pressure=pest_pressure,
                    soil_compaction=soil_compaction,
                    irrigation_events=irrigation_events,
                    fertilizer_events=fertilizer_events,
                ),
                scenario_name=scenario_name,
                save=save,
                degraded_fallback=True,
            )

        return await self._simulate_scalar(
            field_id,
            organization_id=organization_id,
            crop_code=crop_code,
            irrigation_pct=irrigation_pct,
            fertilizer_pct=fertilizer_pct,
            expected_rain_mm=expected_rain_mm,
            temperature_delta_c=temperature_delta_c,
            precipitation_factor=precipitation_factor,
            planting_density_pct=planting_density_pct,
            sowing_shift_days=sowing_shift_days,
            tillage_type=tillage_type,
            pest_pressure=pest_pressure,
            soil_compaction=soil_compaction,
            cloud_cover_factor=cloud_cover_factor,
            scenario_name=scenario_name,
            save=save,
        )

    async def _simulate_scalar(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
        irrigation_pct: float,
        fertilizer_pct: float,
        expected_rain_mm: float,
        temperature_delta_c: float = 0.0,
        precipitation_factor: float | None = None,
        planting_density_pct: float = 0.0,
        sowing_shift_days: int | None = None,
        tillage_type: int | None = None,
        pest_pressure: int | None = None,
        soil_compaction: float | None = None,
        cloud_cover_factor: float = 1.0,
        scenario_name: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        baseline = await self.yield_service.get_or_create_prediction(
            field_id,
            organization_id=organization_id,
            crop_code=crop_code,
            refresh=False,
        )

        # When baseline management is zero (rainfed / no fertilizer), percentage
        # adjustments produce zero effect.  Convert percentage params to minimum
        # absolute floors so the model actually responds.
        effective_irrigation_pct = irrigation_pct
        effective_fertilizer_pct = fertilizer_pct

        # If the baseline has no management data (common for new fields), treat
        # the percentage as an absolute delta in sensible units.
        baseline_mgmt = float(
            (baseline.get("input_features") or {}).get("management_total_amount") or 0.0
        )
        if baseline_mgmt < 1.0 and (irrigation_pct > 0 or fertilizer_pct > 0):
            # For fields with no management history, interpret percentage as
            # absolute reference against crop-typical norms:
            # irrigation 10% → ~30 mm, fertilizer 10% → ~15 kg N/ha
            effective_irrigation_pct = irrigation_pct  # pass through, yield_service handles
            effective_fertilizer_pct = fertilizer_pct

        scenario_adj: dict[str, float] = {
            "irrigation_pct": effective_irrigation_pct,
            "fertilizer_pct": effective_fertilizer_pct,
            "expected_rain_mm": expected_rain_mm,
        }
        if temperature_delta_c:
            scenario_adj["temperature_delta_c"] = temperature_delta_c
        if precipitation_factor is not None:
            scenario_adj["precipitation_factor"] = float(precipitation_factor)
        if planting_density_pct:
            scenario_adj["planting_density_pct"] = planting_density_pct
        if tillage_type is not None:
            scenario_adj["tillage_type"] = float(tillage_type)
        if pest_pressure is not None:
            scenario_adj["pest_pressure"] = float(pest_pressure)
        if soil_compaction is not None:
            scenario_adj["soil_compaction"] = soil_compaction
        if cloud_cover_factor != 1.0:
            scenario_adj["cloud_cover_factor"] = float(cloud_cover_factor)

        scenario_prediction = await self.yield_service.estimate_prediction(
            field_id,
            organization_id=organization_id,
            crop_code=crop_code,
            scenario_adjustments=scenario_adj,
        )
        _baseline_yield_raw = baseline.get("estimated_yield_kg_ha")
        _scenario_yield_raw = scenario_prediction.get("estimated_yield_kg_ha")
        if _baseline_yield_raw is None:
            raise ValueError("Базовый прогноз не содержит оценки урожайности; попробуйте обновить прогноз поля.")
        if _scenario_yield_raw is None:
            raise ValueError("Сценарный расчёт не вернул оценку урожайности.")
        baseline_yield = float(_baseline_yield_raw)
        scenario_yield = float(_scenario_yield_raw)
        baseline_supported = bool((baseline.get("model_applicability") or {}).get("supported"))
        scenario_supported = bool((scenario_prediction.get("model_applicability") or {}).get("supported"))
        supported = baseline_supported and scenario_supported
        guardrails = {
            "irrigation_pct_abs_max": 80.0,
            "fertilizer_pct_abs_max": 80.0,
            "expected_rain_mm_max": 200.0,
            "temperature_delta_c_abs_max": 8.0,
            "planting_density_pct_abs_max": 80.0,
            "tillage_type_max": 3,
            "pest_pressure_max": 3,
            "soil_compaction_max": 1.0,
        }
        within_guardrails = (
            abs(float(irrigation_pct)) <= guardrails["irrigation_pct_abs_max"]
            and abs(float(fertilizer_pct)) <= guardrails["fertilizer_pct_abs_max"]
            and float(expected_rain_mm) <= guardrails["expected_rain_mm_max"]
            and abs(float(temperature_delta_c)) <= guardrails["temperature_delta_c_abs_max"]
            and abs(float(planting_density_pct)) <= guardrails["planting_density_pct_abs_max"]
        )

        support_reason = None
        delta_pct = 0.0
        delta_kg = 0.0
        constraint_warnings: list[str] = []
        counterfactual_feature_diff = {
            "management_delta_pct": round(float(irrigation_pct + fertilizer_pct), 3),
            "expected_rain_mm": round(float(expected_rain_mm), 3),
            "temperature_delta_c": round(float(temperature_delta_c), 2),
            "planting_density_pct": round(float(planting_density_pct), 2),
            "tillage_type": tillage_type,
            "pest_pressure": pest_pressure,
            "soil_compaction": round(float(soil_compaction), 3) if soil_compaction is not None else None,
            "baseline_confidence_tier": baseline.get("confidence_tier"),
            "scenario_confidence_tier": scenario_prediction.get("confidence_tier"),
        }
        if not supported:
            scenario_yield = baseline_yield
            support_reason = (
                scenario_prediction.get("support_reason")
                or baseline.get("support_reason")
                or "Базовый прогноз не поддерживается моделью."
            )
            constraint_warnings.append(support_reason)
        elif not within_guardrails:
            supported = False
            support_reason = "Сценарий выходит за observed training envelope и не будет оценён автоматически."
            constraint_warnings.append(support_reason)
        else:
            scenario_yield = round(float(scenario_yield), 2)
            delta_pct = round((scenario_yield / max(baseline_yield, 1.0) - 1.0) * 100.0, 2)
            delta_kg = round(scenario_yield - baseline_yield, 2)
        support_reason_code, support_reason_params = classify_support_reason(support_reason)

        crop_suitability = dict(
            scenario_prediction.get("crop_suitability")
            or baseline.get("crop_suitability")
            or {}
        )
        for warning in crop_suitability.get("warnings") or []:
            if isinstance(warning, str) and warning not in constraint_warnings:
                constraint_warnings.append(warning)

        # Risk assessment based on both magnitude and direction
        abs_delta = abs(delta_pct)
        if abs_delta <= 3:
            risk_level = "минимальный"
        elif abs_delta <= 8:
            risk_level = "низкий"
        elif abs_delta <= 15:
            risk_level = "умеренный"
        elif abs_delta <= 25:
            risk_level = "повышенный"
        else:
            risk_level = "критический"

        # Build risk comment based on scenario drivers
        delta_drivers = _build_delta_driver_breakdown(
            baseline_drivers=list(baseline.get("driver_breakdown") or baseline.get("explanation", {}).get("drivers") or []),
            scenario_drivers=list(scenario_prediction.get("driver_breakdown") or scenario_prediction.get("explanation", {}).get("drivers") or []),
            baseline_inputs=baseline.get("input_features") or {},
            scenario_inputs=scenario_prediction.get("input_features") or {},
            baseline_yield_kg_ha=baseline_yield,
            scenario_yield_kg_ha=scenario_yield if supported else baseline_yield,
        )
        risk_comment = _build_risk_comment(
            supported=supported,
            delta_pct=delta_pct,
            irrigation_pct=irrigation_pct,
            fertilizer_pct=fertilizer_pct,
            expected_rain_mm=expected_rain_mm,
            temperature_delta_c=temperature_delta_c,
            scenario_factors=delta_drivers,
        )
        normalized_drivers = delta_drivers
        baseline_series = dict(baseline.get("seasonal_series") or {})
        # In scalar mode scenario_prediction never has its own seasonal_series,
        # so we synthesise a projected time series from baseline + factor deltas.
        # yield_ratio encodes the non-linear Mitscherlich response (excess is harmful).
        if supported and baseline_series.get("metrics"):
            _yield_ratio = scenario_yield / max(baseline_yield, 1.0)
            scenario_series = _project_scenario_series(baseline_series, _yield_ratio, scenario_adj)
        else:
            scenario_series = dict(baseline_series)
        forecast_curve: dict[str, Any] = {}
        if self.db is not None:
            try:
                field = await self.yield_service._get_field(field_id, organization_id=organization_id)
                effective_crop_code = crop_code or ((baseline.get("crop") or {}).get("code"))
                crop = await self.yield_service._resolve_crop(effective_crop_code)
                forecast_curve = await self._build_scenario_forecast_curve(
                    field=field,
                    crop=crop,
                    organization_id=organization_id,
                    temperature_delta_c=temperature_delta_c,
                    expected_rain_mm=expected_rain_mm,
                    precipitation_factor=precipitation_factor,
                )
            except Exception as exc:
                logger.warning("scenario_forecast_curve_unavailable", field_id=str(field_id), error=str(exc))
        normalized_drivers = sorted(
            [item for item in normalized_drivers if abs(float(item.get("effect_kg_ha") or 0.0)) >= 0.5],
            key=lambda item: abs(float(item.get("effect_kg_ha") or 0.0)),
            reverse=True,
        )

        payload = {
            "field_id": str(field_id),
            "baseline_yield_kg_ha": baseline_yield,
            "scenario_yield_kg_ha": scenario_yield,
            "predicted_yield_change_pct": delta_pct,
            "factors": {k: round(v, 3) if isinstance(v, float) else v for k, v in scenario_adj.items()},
            "scenario_name": scenario_name or "Сценарий",
            "model_version": scenario_prediction.get("model_version") or baseline.get("model_version", "unsupported_scenario_v3"),
            "engine_mode": "scalar_sensitivity",
            "confidence_tier": (
                scenario_prediction.get("confidence_tier")
                if supported
                else baseline.get("confidence_tier") if baseline_supported else "unsupported"
            ),
            "assumptions": {
                "baseline_source": "latest_prediction",
                "counterfactual_mode": "agronomic_response_v3",
                "requires_supported_baseline": True,
                "response_model": "Mitscherlich + Liebig minimum law",
                "interaction_effects": True,
            },
            "comparison": {
                "delta_kg_ha": delta_kg,
                "delta_pct": delta_pct,
                "factor_breakdown": normalized_drivers,
            },
            "risk_summary": {
                "level": risk_level,
                "level_code": normalize_risk_level(risk_level),
                "comment": risk_comment,
                "confidence_note": (
                    "Модель учитывает перекрёстные взаимодействия факторов (засуха×удобрения, переувлажнение×азот)."
                    if supported
                    else None
                ),
            },
            "supported": supported,
            "model_applicability": scenario_prediction.get("model_applicability") or {},
            "training_domain": scenario_prediction.get("training_domain") or {},
            "feature_coverage": scenario_prediction.get("feature_coverage") or {},
            "crop_suitability": crop_suitability,
            "crop_hint": scenario_prediction.get("crop_hint") or baseline.get("crop_hint") or {},
            "observed_range_guardrails": guardrails,
            "counterfactual_feature_diff": counterfactual_feature_diff,
            "scenario_time_series": {
                "baseline": baseline_series,
                "scenario": scenario_series,
            },
            "forecast_curve": forecast_curve,
            "scenario_water_balance": dict(scenario_prediction.get("water_balance") or {}),
            "scenario_risk_projection": dict(scenario_prediction.get("risk") or {}),
            "driver_breakdown": normalized_drivers,
            "geometry_quality_impact": (
                scenario_prediction.get("geometry_quality_impact")
                or baseline.get("geometry_quality_impact")
                or {}
            ),
            "constraint_warnings": constraint_warnings,
            "support_reason": support_reason,
            "support_reason_code": support_reason_code,
            "support_reason_params": support_reason_params,
            "freshness": build_freshness(
                provider="scenario_model",
                fetched_at=scenario_prediction.get("prediction_date") or baseline.get("prediction_date"),
                cache_written_at=scenario_prediction.get("prediction_date") or baseline.get("prediction_date"),
                model_version=scenario_prediction.get("model_version") or baseline.get("model_version"),
                dataset_version=((scenario_prediction.get("freshness") or {}).get("dataset_version")),
            ),
        }
        trust_meta = describe_prediction_operational_tier(
            supported=supported,
            confidence_tier=payload.get("confidence_tier"),
            crop_suitability=crop_suitability,
            support_reason=support_reason,
        )
        payload.update(
            {
                "operational_tier": trust_meta.get("operational_tier"),
                "review_required": bool(trust_meta.get("review_required")),
                "review_reason": trust_meta.get("review_reason"),
                "review_reason_code": trust_meta.get("review_reason_code"),
                "review_reason_params": trust_meta.get("review_reason_params") or {},
            }
        )
        if save and self.db is not None:
            saved_id = await self._save_scenario(
                organization_id=organization_id,
                field_id=field_id,
                crop_code=crop_code,
                scenario_name=payload["scenario_name"],
                baseline=baseline,
                payload=payload,
            )
            payload["saved_scenario_id"] = saved_id
        return payload

    async def sensitivity_sweep(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
        base_adjustments: dict[str, float],
        sweep_param: str,
        sweep_values: list[float],
    ) -> dict[str, Any]:
        """Run the scenario model for each sweep value to build a response curve."""
        baseline = await self.yield_service.get_or_create_prediction(
            field_id, organization_id=organization_id, crop_code=crop_code, refresh=False,
        )
        baseline_yield = float(baseline["estimated_yield_kg_ha"])

        points: list[dict[str, float]] = []
        for val in sweep_values:
            adj = {**base_adjustments, sweep_param: val}
            pred = await self.yield_service.estimate_prediction(
                field_id, organization_id=organization_id, crop_code=crop_code,
                scenario_adjustments=adj,
            )
            points.append({
                "param_value": round(float(val), 2),
                "yield_kg_ha": round(float(pred["estimated_yield_kg_ha"]), 2),
            })

        return {
            "field_id": str(field_id),
            "sweep_param": sweep_param,
            "baseline_yield_kg_ha": baseline_yield,
            "points": points,
        }

    async def simulate_mechanistic(
        self,
        field_id: UUID,
        *,
        organization_id: UUID,
        crop_code: str | None,
        scenario_events: dict[str, Any] | None = None,
        scenario_name: str | None = None,
        save: bool = True,
        degraded_fallback: bool = True,
    ) -> dict[str, Any]:
        """Run counterfactual simulation using the weekly mechanistic engine.

        Unlike `simulate()` which adjusts scalar factors, this method runs
        the full weekly mechanistic model with event-based interventions:
        - irrigation_events: [{week, amount_mm}]
        - fertilizer_events: [{week, n_kg_ha}]
        - temperature_delta_c: global weather shift
        - precipitation_factor: multiplicative weather shift

        Returns weekly trace comparison (baseline vs scenario) with per-week
        stress breakdown and driver identification.
        """
        from services.mechanistic_engine import run_mechanistic_baseline
        from storage.db import Field, Crop
        from geoalchemy2.shape import to_shape
        import copy

        field = (await self.db.execute(
            select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
        )).scalar_one_or_none()
        self._ensure_operational_field(field)

        crop = None
        effective_code = crop_code or "wheat"
        if crop_code:
            crop = (await self.db.execute(select(Crop).where(Crop.code == crop_code))).scalar_one_or_none()
        if crop is None:
            crop = (await self.db.execute(select(Crop).where(Crop.code == "wheat"))).scalar_one_or_none()

        baseline_kg = float(crop.yield_baseline_kg_ha) if crop else 3000.0
        field_area_ha = float(field.area_m2 or 0) / 10000.0
        centroid = to_shape(field.geom).centroid
        baseline_prediction = await self.yield_service.get_or_create_prediction(
            field_id,
            organization_id=organization_id,
            crop_code=crop_code,
            refresh=False,
        )

        # Load weekly features
        current_year = current_season_year()
        weekly_rows = await ensure_weekly_profile(
            self.db,
            organization_id=organization_id,
            field_id=field_id,
            season_year=current_year,
        )

        if len(weekly_rows) < 3 or not profile_has_signal(weekly_rows):
            if degraded_fallback:
                return await self._simulate_scalar(
                    field_id,
                    organization_id=organization_id,
                    crop_code=crop_code,
                    irrigation_pct=float((scenario_events or {}).get("irrigation_pct", 0)),
                    fertilizer_pct=float((scenario_events or {}).get("fertilizer_pct", 0)),
                    expected_rain_mm=float((scenario_events or {}).get("expected_rain_mm", 0)),
                    temperature_delta_c=float((scenario_events or {}).get("temperature_delta_c", 0)),
                    precipitation_factor=(scenario_events or {}).get("precipitation_factor"),
                    scenario_name=scenario_name,
                    save=save,
                )
            support_reason = "Недельный профиль поля недостаточен для mechanistic scenario simulation."
            support_reason_code, support_reason_params = classify_support_reason(support_reason)
            trust_meta = describe_prediction_operational_tier(
                supported=False,
                confidence_tier="unsupported",
                crop_suitability=baseline_prediction.get("crop_suitability") or {},
                support_reason=support_reason,
            )
            try:
                forecast_curve = await self._build_scenario_forecast_curve(
                    field=field,
                    crop=crop,
                    organization_id=organization_id,
                    temperature_delta_c=float((scenario_events or {}).get("temperature_delta_c", 0.0)),
                    expected_rain_mm=float((scenario_events or {}).get("expected_rain_mm", 0.0)),
                    precipitation_factor=(scenario_events or {}).get("precipitation_factor"),
                )
            except Exception as exc:
                logger.warning("mechanistic_forecast_curve_unavailable", field_id=str(field_id), error=str(exc))
                forecast_curve = {}
            payload = {
                "field_id": str(field_id),
                "baseline_yield_kg_ha": float(baseline_prediction.get("estimated_yield_kg_ha") or 0.0),
                "scenario_yield_kg_ha": float(baseline_prediction.get("estimated_yield_kg_ha") or 0.0),
                "predicted_yield_change_pct": 0.0,
                "factors": {},
                "scenario_name": scenario_name or "Механистический сценарий",
                "model_version": "mechanistic_scenario_v1",
                "engine_mode": "mechanistic_event",
                "confidence_tier": "unsupported",
                "assumptions": {
                    "baseline_source": "weekly_profile",
                    "counterfactual_mode": "mechanistic_weekly",
                    "requires_supported_baseline": True,
                },
                "comparison": {"delta_kg_ha": 0.0, "delta_pct": 0.0, "factor_breakdown": []},
                "risk_summary": {"level": "неопределённый", "level_code": "unknown", "comment": support_reason},
                "supported": False,
                "model_applicability": {"supported": False, "support_reason": support_reason},
                "training_domain": dict(baseline_prediction.get("training_domain") or {}),
                "feature_coverage": dict(baseline_prediction.get("feature_coverage") or {}),
                "crop_suitability": dict(baseline_prediction.get("crop_suitability") or {}),
                "crop_hint": dict(baseline_prediction.get("crop_hint") or {}),
                "observed_range_guardrails": {"weekly_profile_required": True},
                "counterfactual_feature_diff": {"scenario_events": dict(scenario_events or {})},
                "scenario_time_series": {},
                "forecast_curve": forecast_curve,
                "scenario_water_balance": {},
                "scenario_risk_projection": {},
                "baseline_trace": [],
                "scenario_trace": [],
                "trace_supported": False,
                "engine_version": "mechanistic_v1",
                "weeks_simulated": 0,
                "driver_breakdown": [],
                "geometry_quality_impact": dict(baseline_prediction.get("geometry_quality_impact") or {}),
                "constraint_warnings": [support_reason],
                "support_reason": support_reason,
                "support_reason_code": support_reason_code,
                "support_reason_params": support_reason_params,
                "freshness": build_freshness(
                    provider="scenario_model",
                    fetched_at=baseline_prediction.get("prediction_date"),
                    cache_written_at=baseline_prediction.get("prediction_date"),
                    model_version="mechanistic_scenario_v1",
                    dataset_version=((baseline_prediction.get("freshness") or {}).get("dataset_version")),
                ),
                **trust_meta,
            }
            return payload

        # Build weekly inputs from features
        baseline_inputs = rows_to_weekly_inputs(weekly_rows)
        geometry_summary = (
            baseline_prediction.get("geometry_quality_impact")
            or summarize_geometry_quality(weekly_rows)
        )
        crop_hint = (
            baseline_prediction.get("crop_hint")
            or await load_crop_hint(
                self.db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=current_year,
            )
        )

        # Run baseline
        baseline_result = run_mechanistic_baseline(
            crop_code=effective_code,
            crop_baseline_kg_ha=baseline_kg,
            weekly_inputs=baseline_inputs,
            field_area_ha=field_area_ha,
            latitude=centroid.y,
        )

        # Apply scenario events
        scenario_inputs = [copy.copy(inp) for inp in baseline_inputs]
        events = scenario_events or {}

        # Temperature shift
        temp_delta = float(events.get("temperature_delta_c", 0.0))
        if temp_delta:
            for inp in scenario_inputs:
                inp.tmean_c += temp_delta
                inp.tmax_c += temp_delta
                inp.tmin_c += temp_delta

        # Precipitation factor
        precip_factor = float(events.get("precipitation_factor", 1.0))
        if precip_factor != 1.0:
            for inp in scenario_inputs:
                inp.precipitation_mm *= precip_factor

        # Event-based irrigation
        for evt in events.get("irrigation_events", []):
            week = int(evt.get("week", 0))
            amount = float(evt.get("amount_mm", 0.0))
            for inp in scenario_inputs:
                if inp.week == week:
                    inp.irrigation_mm += amount

        # Event-based fertilizer
        for evt in events.get("fertilizer_events", []):
            week = int(evt.get("week", 0))
            amount = float(evt.get("n_kg_ha", 0.0))
            for inp in scenario_inputs:
                if inp.week == week:
                    inp.n_applied_kg_ha += amount

        # Run scenario
        scenario_result = run_mechanistic_baseline(
            crop_code=effective_code,
            crop_baseline_kg_ha=baseline_kg,
            weekly_inputs=scenario_inputs,
            field_area_ha=field_area_ha,
            latitude=centroid.y,
        )

        baseline_yield = baseline_result.baseline_yield_kg_ha
        scenario_yield = scenario_result.baseline_yield_kg_ha
        delta_kg = round(scenario_yield - baseline_yield, 2)
        delta_pct = round((delta_kg / max(baseline_yield, 1.0)) * 100.0, 2)

        # Build driver breakdown from trace comparison
        raw_drivers = []
        b_final = baseline_result.final_state
        s_final = scenario_result.final_state
        stress_attrs = {"water_stress", "heat_stress", "vpd_stress", "nutrient_stress"}
        raw_deltas: list[tuple[str, str, float, float, float, float]] = []
        for attr, label in [
            ("water_stress", "water_stress"),
            ("heat_stress", "heat_stress"),
            ("vpd_stress", "vpd_stress"),
            ("nutrient_stress", "nutrient_stress"),
            ("canopy_cover", "canopy_cover"),
            ("biomass_proxy", "biomass"),
        ]:
            b_val = getattr(b_final, attr, 0.0)
            s_val = getattr(s_final, attr, 0.0)
            delta = s_val - b_val
            if abs(delta) > 0.005:
                contribution = -delta if attr in stress_attrs else delta
                raw_deltas.append((attr, label, float(b_val), float(s_val), float(delta), float(contribution)))

        total_contribution = sum(abs(item[5]) for item in raw_deltas) or 1.0
        for attr, label, b_val, s_val, delta, contribution in raw_deltas:
            signed_effect = delta_kg * (abs(contribution) / total_contribution)
            if contribution < 0:
                signed_effect *= -1.0
            raw_drivers.append(
                {
                    "driver_id": attr,
                    "label": label,
                    "input_key": attr,
                    "baseline_value": round(b_val, 4),
                    "scenario_value": round(s_val, 4),
                    "delta_input": round(delta, 4),
                    "effect_kg_ha": round(signed_effect, 2),
                    "effect_pct": round((signed_effect / max(baseline_yield, 1.0)) * 100.0, 2),
                    "source": "mechanistic_trace",
                    "confidence": 0.78,
                }
            )
        drivers = normalize_driver_breakdown(
            raw_drivers,
            baseline_yield_kg_ha=baseline_yield,
            scenario_yield_kg_ha=scenario_yield,
            source="mechanistic_trace",
        )

        baseline_supported = bool((baseline_prediction.get("model_applicability") or {}).get("supported"))
        supported = baseline_supported and len(baseline_inputs) >= 3
        support_reason = None if supported else (
            baseline_prediction.get("support_reason")
            or "Базовый прогноз по выбранной культуре не поддержан."
        )
        support_reason_code, support_reason_params = classify_support_reason(support_reason)
        crop_suitability = dict(baseline_prediction.get("crop_suitability") or {})

        abs_delta_pct = abs(delta_pct)
        if abs_delta_pct <= 3:
            risk_level = "минимальный"
        elif abs_delta_pct <= 8:
            risk_level = "низкий"
        elif abs_delta_pct <= 15:
            risk_level = "умеренный"
        elif abs_delta_pct <= 25:
            risk_level = "повышенный"
        else:
            risk_level = "критический"
        risk_comment = _build_risk_comment(
            supported=supported,
            delta_pct=delta_pct,
            irrigation_pct=float(events.get("irrigation_pct", 0.0)),
            fertilizer_pct=float(events.get("fertilizer_pct", 0.0)),
            expected_rain_mm=float(events.get("expected_rain_mm", 0.0)),
            temperature_delta_c=float(events.get("temperature_delta_c", 0.0)),
            scenario_factors=drivers,
        )
        scalar_factors: dict[str, float] = {}
        for key in ("temperature_delta_c", "precipitation_factor", "irrigation_pct", "fertilizer_pct", "expected_rain_mm"):
            if key in events and isinstance(events.get(key), (int, float)):
                scalar_factors[key] = round(float(events[key]), 4)
        if events.get("irrigation_events"):
            scalar_factors["irrigation_events_total_mm"] = round(
                sum(float((item or {}).get("amount_mm") or 0.0) for item in events.get("irrigation_events") or []),
                4,
            )
            scalar_factors["irrigation_events_count"] = float(len(events.get("irrigation_events") or []))
        if events.get("fertilizer_events"):
            scalar_factors["fertilizer_events_total_n_kg_ha"] = round(
                sum(float((item or {}).get("n_kg_ha") or 0.0) for item in events.get("fertilizer_events") or []),
                4,
            )
            scalar_factors["fertilizer_events_count"] = float(len(events.get("fertilizer_events") or []))

        baseline_trace = list(baseline_result.trace)
        scenario_trace = list(scenario_result.trace)
        water_balance_projection = {
            "baseline": [
                {"week": item.get("week"), "root_zone_water_mm": item.get("root_zone_water_mm")}
                for item in baseline_trace
            ],
            "scenario": [
                {"week": item.get("week"), "root_zone_water_mm": item.get("root_zone_water_mm")}
                for item in scenario_trace
            ],
            "delta_final_root_zone_water_mm": round(
                float(s_final.root_zone_water_mm) - float(b_final.root_zone_water_mm),
                2,
            ),
        }
        risk_projection = {
            "baseline": {
                "water_stress": round(float(b_final.water_stress), 4),
                "heat_stress": round(float(b_final.heat_stress), 4),
                "vpd_stress": round(float(b_final.vpd_stress), 4),
                "nutrient_stress": round(float(b_final.nutrient_stress), 4),
                "yield_potential_remaining": round(float(b_final.yield_potential_remaining), 4),
            },
            "scenario": {
                "water_stress": round(float(s_final.water_stress), 4),
                "heat_stress": round(float(s_final.heat_stress), 4),
                "vpd_stress": round(float(s_final.vpd_stress), 4),
                "nutrient_stress": round(float(s_final.nutrient_stress), 4),
                "yield_potential_remaining": round(float(s_final.yield_potential_remaining), 4),
            },
        }
        try:
            forecast_curve = await self._build_scenario_forecast_curve(
                field=field,
                crop=crop,
                organization_id=organization_id,
                temperature_delta_c=float(events.get("temperature_delta_c", 0.0)),
                expected_rain_mm=float(events.get("expected_rain_mm", 0.0)),
                precipitation_factor=events.get("precipitation_factor"),
            )
        except Exception as exc:
            logger.warning("mechanistic_forecast_curve_unavailable", field_id=str(field_id), error=str(exc))
            forecast_curve = {}
        counterfactual_feature_diff = {
            "selected_crop_code": effective_code,
            "weeks_in_profile": len(weekly_rows),
            "geometry_confidence": geometry_summary.get("geometry_confidence"),
            "tta_consensus": geometry_summary.get("tta_consensus"),
            "boundary_uncertainty": geometry_summary.get("boundary_uncertainty"),
            "precipitation_factor": float(events.get("precipitation_factor", 1.0)),
            "temperature_delta_c": float(events.get("temperature_delta_c", 0.0)),
            "irrigation_events_count": len(events.get("irrigation_events") or []),
            "fertilizer_events_count": len(events.get("fertilizer_events") or []),
        }
        constraint_warnings: list[str] = []
        if events.get("sowing_shift_days") is not None:
            constraint_warnings.append("Сдвиг срока сева сохранён как advisory-параметр, но ещё не моделируется mechanistic engine.")
        if not supported and support_reason:
            constraint_warnings.append(support_reason)

        payload = {
            "field_id": str(field_id),
            "engine_version": "mechanistic_v1",
            "engine_mode": "mechanistic_event",
            "baseline_yield_kg_ha": baseline_yield,
            "scenario_yield_kg_ha": scenario_yield,
            "predicted_yield_change_pct": delta_pct,
            "factors": scalar_factors,
            "scenario_name": scenario_name or "Механистический сценарий",
            "model_version": "mechanistic_scenario_v1",
            "confidence_tier": (
                baseline_prediction.get("confidence_tier")
                if supported
                else "unsupported"
            ),
            "assumptions": {
                "baseline_source": "weekly_profile",
                "counterfactual_mode": "mechanistic_weekly",
                "requires_supported_baseline": True,
                "response_model": "weekly mechanistic baseline",
            },
            "comparison": {
                "delta_kg_ha": delta_kg,
                "delta_pct": delta_pct,
                "factor_breakdown": drivers,
            },
            "risk_summary": {
                "level": risk_level,
                "level_code": normalize_risk_level(risk_level),
                "comment": risk_comment,
            },
            "supported": supported,
            "trace_supported": True,
            "model_applicability": (
                dict(baseline_prediction.get("model_applicability") or {})
                if supported
                else {
                    **dict(baseline_prediction.get("model_applicability") or {}),
                    "supported": False,
                }
            ),
            "training_domain": dict(baseline_prediction.get("training_domain") or {}),
            "feature_coverage": {
                **dict(baseline_prediction.get("feature_coverage") or {}),
                "weekly_profile_rows": len(weekly_rows),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            },
            "crop_suitability": crop_suitability,
            "crop_hint": crop_hint,
            "observed_range_guardrails": {
                "weekly_profile_required": True,
                "explicit_event_support": True,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            },
            "counterfactual_feature_diff": counterfactual_feature_diff,
            "scenario_time_series": {
                "baseline": baseline_trace,
                "scenario": scenario_trace,
            },
            "forecast_curve": forecast_curve,
            "scenario_water_balance": water_balance_projection,
            "scenario_risk_projection": risk_projection,
            "baseline_trace": baseline_result.trace,
            "scenario_trace": scenario_result.trace,
            "driver_breakdown": drivers,
            "geometry_quality_impact": geometry_summary,
            "constraint_warnings": constraint_warnings,
            "support_reason": support_reason,
            "support_reason_code": support_reason_code,
            "support_reason_params": support_reason_params,
            "freshness": build_freshness(
                provider="scenario_model",
                fetched_at=baseline_prediction.get("prediction_date"),
                cache_written_at=baseline_prediction.get("prediction_date"),
                model_version="mechanistic_scenario_v1",
                dataset_version=((baseline_prediction.get("freshness") or {}).get("dataset_version")),
            ),
            "weeks_simulated": len(baseline_result.trace),
        }
        trust_meta = describe_prediction_operational_tier(
            supported=supported,
            confidence_tier=payload.get("confidence_tier"),
            crop_suitability=crop_suitability,
            support_reason=support_reason,
        )
        payload.update(trust_meta)
        payload["review_reason_code"] = trust_meta.get("review_reason_code")
        payload["review_reason_params"] = trust_meta.get("review_reason_params") or {}

        if save:
            saved_id = await self._save_scenario(
                organization_id=organization_id,
                field_id=field_id,
                crop_code=crop_code,
                scenario_name=payload["scenario_name"],
                baseline={
                    **dict(baseline_prediction),
                    "estimated_yield_kg_ha": baseline_yield,
                },
                payload=payload,
            )
            payload["saved_scenario_id"] = saved_id

        return payload

    async def list_scenarios(self, field_id: UUID, *, organization_id: UUID) -> list[dict[str, Any]]:
        result = await self.db.execute(
            select(ScenarioRun)
            .where(ScenarioRun.organization_id == organization_id)
            .where(ScenarioRun.field_id == field_id)
            .order_by(desc(ScenarioRun.created_at))
        )
        items = []
        for row in result.scalars().all():
            items.append(
                {
                    "id": row.id,
                    "field_id": str(row.field_id),
                    "scenario_name": row.scenario_name,
                    "model_version": row.model_version,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "parameters": dict(row.parameters or {}),
                    "baseline_snapshot": dict(row.baseline_snapshot or {}),
                    "result_snapshot": dict(row.result_snapshot or {}),
                    "delta_pct": row.delta_pct,
                    "freshness": build_freshness(
                        provider="scenario_model",
                        fetched_at=row.created_at,
                        cache_written_at=row.created_at,
                        model_version=row.model_version,
                        dataset_version=((row.result_snapshot or {}).get("freshness") or {}).get("dataset_version"),
                    ),
                }
            )
        return items

    async def _save_scenario(
        self,
        *,
        organization_id: UUID,
        field_id: UUID,
        crop_code: str | None,
        scenario_name: str,
        baseline: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        crop_id = None
        if crop_code:
            crop_result = await self.db.execute(select(Crop).where(Crop.code == crop_code))
            crop = crop_result.scalar_one_or_none()
            crop_id = crop.id if crop is not None else None

        baseline_id = baseline.get("id")
        scenario = ScenarioRun(
            organization_id=organization_id,
            field_id=field_id,
            crop_id=crop_id,
            baseline_prediction_id=baseline_id,
            scenario_name=scenario_name,
            model_version=payload.get("model_version") or "unsupported_scenario_v2",
            parameters=dict(payload.get("factors") or {}),
            baseline_snapshot=dict(baseline),
            result_snapshot=dict(payload),
            delta_pct=payload.get("predicted_yield_change_pct"),
        )
        self.db.add(scenario)
        await self.db.flush()
        return int(scenario.id)


def _build_risk_comment(
    *,
    supported: bool,
    delta_pct: float,
    irrigation_pct: float,
    fertilizer_pct: float,
    expected_rain_mm: float,
    temperature_delta_c: float,
    scenario_factors: list[dict],
) -> str:
    """Generate a human-readable risk comment based on scenario analysis."""
    if not supported:
        return "Сценарий не поддержан: решение требует ручной агрономической оценки."

    warnings: list[str] = []

    # Detect specific agronomic risks
    if fertilizer_pct > 30:
        warnings.append("риск переудобрения (токсичность азота, полегание)")
    if fertilizer_pct > 50:
        warnings.append("критическое превышение норм удобрений — ожидается засоление почвы")

    if irrigation_pct > 35:
        warnings.append("риск переувлажнения (анаэробные условия для корней)")
    if irrigation_pct > 60:
        warnings.append("высокий риск заболачивания и вымывания питательных веществ")

    if expected_rain_mm > 50:
        warnings.append("повышенный риск водной эрозии почвы (RUSLE)")
    if expected_rain_mm > 100:
        warnings.append("критический риск затопления и потери урожая")

    if temperature_delta_c > 3:
        warnings.append("тепловой стресс: снижение фотосинтеза, стерильность пыльцы")
    elif temperature_delta_c < -3:
        warnings.append("риск заморозков: повреждение вегетативных тканей")

    # Check for dangerous interactions
    if irrigation_pct > 20 and fertilizer_pct > 20:
        warnings.append("взаимодействие: избыток воды + удобрения → вымывание нитратов (Di & Cameron 2002)")

    # Find the worst-performing factor
    neg_drivers = []
    for driver in scenario_factors:
        raw_effect = driver.get("effect_kg_ha")
        if raw_effect is None:
            raw_effect = driver.get("effect_pct")
        if raw_effect is None:
            raw_effect = driver.get("effect")
        try:
            effect = float(raw_effect or 0.0)
        except Exception:
            effect = 0.0
        if effect < 0:
            neg_drivers.append({**driver, "_normalized_effect": effect})
    if neg_drivers:
        worst = min(neg_drivers, key=lambda d: float(d.get("_normalized_effect", 0)))
        worst_label = worst.get("label", "")
        if "interaction" in worst_label:
            warnings.append("обнаружено перекрёстное взаимодействие стресс-факторов")

    if not warnings:
        if delta_pct > 0:
            return "Сценарий показывает умеренный рост урожайности в пределах агрономических норм."
        elif delta_pct < -3:
            return "Сценарий показывает снижение урожайности. Рекомендуется пересмотреть параметры."
        else:
            return "Сценарий близок к базовому прогнозу. Изменения минимальны."

    return "Агрономические предупреждения: " + "; ".join(warnings) + "."
