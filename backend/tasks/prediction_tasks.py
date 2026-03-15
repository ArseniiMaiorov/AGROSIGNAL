"""Celery tasks for yield prediction and scenario simulation.

queue: prediction-medium / modeling-medium
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from core.celery_app import celery
from core.logging import get_logger

logger = get_logger(__name__)


@celery.task(name="tasks.predictions.refresh_field_prediction", bind=True, max_retries=2)
def refresh_field_prediction(
    self,
    field_id_str: str,
    organization_id_str: str,
    crop_code: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Refresh yield prediction for a field using the mechanistic pipeline.

    Steps:
    1. Load weekly features from FieldFeatureWeekly
    2. Run mechanistic baseline
    3. Apply global residual model (if available)
    4. Apply tenant calibration (if available)
    5. Compute conformal prediction interval
    6. Save to YieldPrediction + PredictionRun
    """
    field_id = UUID(field_id_str)
    organization_id = UUID(organization_id_str)

    async def _run() -> dict[str, Any]:
        from storage.db import get_session_factory
        from services.yield_service import YieldService
        from services.weekly_profile_service import (
            FEATURE_SCHEMA_VERSION,
            current_season_year,
            ensure_weekly_profile,
            load_crop_hint,
        )

        factory = get_session_factory()
        async with factory() as db:
            season_year = current_season_year()
            weekly_rows = await ensure_weekly_profile(
                db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=season_year,
            )
            crop_hint = await load_crop_hint(
                db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=season_year,
            )
            service = YieldService(db)
            result = await service.get_or_create_prediction(
                field_id,
                organization_id=organization_id,
                crop_code=crop_code,
                refresh=force,
            )
            return {
                "field_id": field_id_str,
                "estimated_yield_kg_ha": result.get("estimated_yield_kg_ha"),
                "confidence": result.get("confidence"),
                "model_version": result.get("model_version"),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "weekly_profile_rows": len(weekly_rows),
                "crop_hint": crop_hint,
            }

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@celery.task(name="tasks.modeling.simulate_scenario_forward", bind=True, max_retries=1)
def simulate_scenario_forward(
    self,
    scenario_run_id_str: str,
    organization_id_str: str,
) -> dict[str, Any]:
    """Run a counterfactual scenario simulation using the mechanistic engine.

    Loads the scenario spec from ScenarioRun, applies interventions to weekly
    inputs, re-runs the mechanistic model, and stores the result trace.
    """
    organization_id = UUID(organization_id_str)
    scenario_run_id = int(scenario_run_id_str)

    async def _run() -> dict[str, Any]:
        from storage.db import (
            ScenarioRun, Field, Crop,
            get_session_factory,
        )
        from sqlalchemy import select
        from services.modeling_service import ModelingService
        from services.weekly_profile_service import (
            FEATURE_SCHEMA_VERSION,
            current_season_year,
            ensure_weekly_profile,
            profile_has_signal,
        )

        factory = get_session_factory()
        async with factory() as db:
            # Load scenario run
            scenario = (await db.execute(
                select(ScenarioRun)
                .where(ScenarioRun.id == scenario_run_id)
                .where(ScenarioRun.organization_id == organization_id)
            )).scalar_one_or_none()
            if scenario is None:
                return {"error": "scenario_not_found"}

            field = (await db.execute(
                select(Field).where(Field.id == scenario.field_id)
            )).scalar_one_or_none()
            if field is None:
                return {"error": "field_not_found"}

            crop = (await db.execute(
                select(Crop).where(Crop.id == scenario.crop_id)
            )).scalar_one_or_none()

            params = dict(scenario.parameters or {})
            crop_code = str(crop.code if crop else "wheat")

            current_year = current_season_year()
            weekly_rows = await ensure_weekly_profile(
                db,
                organization_id=organization_id,
                field_id=field.id,
                season_year=current_year,
            )

            if len(weekly_rows) < 3 or not profile_has_signal(weekly_rows):
                scenario.result_snapshot = {
                    "error": "insufficient_weekly_features",
                    "supported": False,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "weekly_profile_rows": len(weekly_rows),
                }
                await db.commit()
                return {"error": "insufficient_weekly_features"}

            service = ModelingService(db)
            payload = await service.simulate_mechanistic(
                field.id,
                organization_id=organization_id,
                crop_code=crop_code,
                scenario_events=params,
                scenario_name=scenario.scenario_name,
                save=False,
                degraded_fallback=False,
            )

            payload["feature_schema_version"] = FEATURE_SCHEMA_VERSION
            payload["weekly_profile_rows"] = len(weekly_rows)
            scenario.result_snapshot = dict(payload)
            scenario.baseline_snapshot = {
                "yield_kg_ha": payload.get("baseline_yield_kg_ha"),
                "confidence_tier": payload.get("confidence_tier"),
                "trace_supported": payload.get("trace_supported"),
                "weeks_simulated": payload.get("weeks_simulated"),
            }
            scenario.delta_pct = round(float(payload.get("predicted_yield_change_pct") or 0.0), 2)
            scenario.model_version = str(payload.get("model_version") or "mechanistic_scenario_v1")
            await db.commit()

            return {
                "scenario_run_id": scenario_run_id,
                "baseline_yield": payload.get("baseline_yield_kg_ha"),
                "scenario_yield": payload.get("scenario_yield_kg_ha"),
                "delta_pct": round(float(payload.get("predicted_yield_change_pct") or 0.0), 2),
                "supported": bool(payload.get("supported")),
                "engine_version": payload.get("engine_version"),
            }

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
