"""Celery tasks for feature materialization and detection scoring.

queue: features-medium
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from core.celery_app import celery
from core.logging import get_logger

logger = get_logger(__name__)


@celery.task(name="tasks.features.backfill_weekly_features", bind=True, max_retries=2)
def backfill_weekly_features(
    self,
    field_id_str: str,
    season_year: int,
    organization_id_str: str,
) -> dict[str, Any]:
    """Materialize the canonical weekly feature profile for a field-season."""
    field_id = UUID(field_id_str)
    organization_id = UUID(organization_id_str)

    async def _run() -> dict[str, Any]:
        from services.weekly_profile_service import materialize_weekly_profile
        from storage.db import get_session_factory

        factory = get_session_factory()
        async with factory() as db:
            return await materialize_weekly_profile(
                db,
                organization_id=organization_id,
                field_id=field_id,
                season_year=season_year,
            )

    logger.info("backfill_weekly_features_start", field_id=field_id_str, season_year=season_year)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@celery.task(name="tasks.features.recompute_detection_scores", bind=True, max_retries=1)
def recompute_detection_scores(
    self,
    run_id_str: str,
    organization_id_str: str,
) -> dict[str, Any]:
    """Re-score detection candidates for a completed AOI run.

    Loads candidate features from DB, applies the latest ranker model,
    and updates scores/rankings.
    """
    logger.info("recompute_detection_scores_start", run_id=run_id_str)
    # Placeholder: when a trained LightGBM ranker model is available,
    # this task loads it and re-scores all candidates for the run.
    return {"run_id": run_id_str, "status": "not_yet_implemented"}
