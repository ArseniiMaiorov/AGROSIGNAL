"""Async prediction and scenario tasks."""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any
from uuid import UUID

from core.celery_app import celery
from core.logging import get_logger
from services.async_job_service import append_job_log, build_job_meta
from services.modeling_service import ModelingService
from services.temporal_analytics_service import TemporalAnalyticsService, TemporalBackfillError
from services.yield_service import YieldService
import storage.db as db_module
from storage.db import get_session_factory

logger = get_logger(__name__)


def _update_task_state(
    task,
    meta: dict[str, Any],
    *,
    progress: int,
    stage_label: str,
    stage_code: str | None = None,
    stage_detail: str | None = None,
    stage_detail_code: str | None = None,
    stage_detail_params: dict[str, Any] | None = None,
    estimated_remaining_s: int | None = None,
) -> dict[str, Any]:
    meta.update(
        {
            "status": "running",
            "progress": int(progress),
            "progress_pct": round(float(progress), 2),
            "stage_code": stage_code or stage_label,
            "stage_label": stage_label,
            "stage_detail": stage_detail,
            "stage_detail_code": stage_detail_code,
            "stage_detail_params": dict(stage_detail_params or {}),
            "estimated_remaining_s": estimated_remaining_s,
        }
    )
    task.update_state(state="PROGRESS", meta=meta)
    return meta


async def _reset_task_db_runtime() -> None:
    db_module._engine = None
    db_module._session_factory = None


async def _run_prediction_impl(
    task,
    *,
    field_id: str,
    organization_id: str,
    crop_code: str | None,
    refresh: bool,
) -> dict[str, Any]:
    logs = [f"Старт расчёта прогноза для поля {field_id}."]
    meta = build_job_meta(
        job_type="prediction",
        organization_id=organization_id,
        field_id=field_id,
        status="running",
        progress=5,
        progress_pct=5.0,
        stage_code="prepare",
        stage_label="prepare",
        stage_detail="initializing prediction context",
        stage_detail_code="initializing_context",
        logs=logs,
        estimated_remaining_s=6,
    )
    task.update_state(state="PROGRESS", meta=meta)

    await _reset_task_db_runtime()
    factory = get_session_factory()
    async with factory() as db:
        service = YieldService(db)
        try:
            from services.weekly_profile_service import (
                FEATURE_SCHEMA_VERSION,
                current_season_year,
                ensure_weekly_profile,
            )

            field_uuid = UUID(field_id)
            org_uuid = UUID(organization_id)

            meta = append_job_log(meta, "Проверяю поле и культуру.")
            _update_task_state(
                task,
                meta,
                progress=18,
                stage_label="prepare",
                stage_code="prepare",
                stage_detail="field and crop resolution",
                stage_detail_code="field_crop_resolution",
                estimated_remaining_s=5,
            )
            field = await service._get_field(field_uuid, organization_id=org_uuid)
            crop = await service._resolve_crop(crop_code)

            meta = append_job_log(meta, "Материализую недельный профиль поля.")
            _update_task_state(
                task,
                meta,
                progress=28,
                stage_label="weekly-profile",
                stage_code="weekly_profile",
                stage_detail="canonical weekly feature store",
                stage_detail_code="materializing_weekly_profile",
                estimated_remaining_s=4,
            )
            weekly_rows = await ensure_weekly_profile(
                db,
                organization_id=org_uuid,
                field_id=field_uuid,
                season_year=current_season_year(),
            )
            meta = append_job_log(
                meta,
                f"Недельный профиль готов: {len(weekly_rows)} недель, схема {FEATURE_SCHEMA_VERSION}.",
            )

            if not refresh:
                meta = append_job_log(meta, "Проверяю кэш прогноза.")
                _update_task_state(
                    task,
                    meta,
                    progress=38,
                    stage_label="cache",
                    stage_code="cache",
                    stage_detail="checking latest cached prediction",
                    stage_detail_code="checking_cached_prediction",
                    estimated_remaining_s=4,
                )
                existing = await service._get_latest_prediction(field_uuid, crop.id, organization_id=org_uuid)
                if existing is not None:
                    payload = await service._prediction_to_dict(existing, crop)
                    meta = append_job_log(meta, "Использую уже рассчитанный прогноз за текущую дату.")
                    await db.rollback()
                    return build_job_meta(
                        job_type="prediction",
                        organization_id=organization_id,
                        field_id=field_id,
                        status="done",
                        progress=100,
                        progress_pct=100.0,
                        stage_code="done",
                        stage_label="done",
                        stage_detail="cached prediction ready",
                        stage_detail_code="cached_prediction_ready",
                        started_at=meta.get("started_at"),
                        logs=meta.get("logs"),
                        result={
                            **payload,
                            "feature_schema_version": FEATURE_SCHEMA_VERSION,
                            "weekly_profile_rows": len(weekly_rows),
                        },
                    )

            meta = append_job_log(meta, "Собираю погодные и полевые признаки.")
            _update_task_state(
                task,
                meta,
                progress=52,
                stage_label="features",
                stage_code="features",
                stage_detail="weather and field analytics",
                stage_detail_code="weather_and_field_analytics",
                estimated_remaining_s=4,
            )
            prediction = await service._build_prediction(field, crop, organization_id=org_uuid, persist=True)

            meta = append_job_log(meta, "Сохраняю прогноз и формирую объяснение.")
            _update_task_state(
                task,
                meta,
                progress=86,
                stage_label="persist",
                stage_code="persist",
                stage_detail="writing prediction snapshot",
                stage_detail_code="writing_prediction_snapshot",
                estimated_remaining_s=1,
            )
            db.add(prediction)
            await db.commit()
            await db.refresh(prediction)
            payload = await service._prediction_to_dict(prediction, crop)
            payload["feature_schema_version"] = FEATURE_SCHEMA_VERSION
            payload["weekly_profile_rows"] = len(weekly_rows)
            meta = append_job_log(meta, f"Прогноз готов: {round(float(payload['estimated_yield_kg_ha']), 2)} кг/га.")
            return build_job_meta(
                job_type="prediction",
                organization_id=organization_id,
                field_id=field_id,
                status="done",
                progress=100,
                progress_pct=100.0,
                stage_code="done",
                stage_label="done",
                stage_detail="prediction completed",
                stage_detail_code="prediction_completed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                result=payload,
            )
        except Exception as exc:
            await db.rollback()
            logger.error("prediction_job_failed", field_id=field_id, organization_id=organization_id, error=str(exc), exc_info=True)
            meta = append_job_log(meta, f"Ошибка прогноза: {exc}")
            return build_job_meta(
                job_type="prediction",
                organization_id=organization_id,
                field_id=field_id,
                status="failed",
                progress=max(int(meta.get("progress") or 0), 100),
                progress_pct=float(max(int(meta.get("progress") or 0), 100)),
                stage_code="failed",
                stage_label="failed",
                stage_detail="prediction failed",
                stage_detail_code="prediction_failed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                error_msg=str(exc),
            )


async def _run_modeling_impl(
    task,
    *,
    organization_id: str,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    field_id = str(request_payload["field_id"])
    logs = [f"Старт моделирования сценария для поля {field_id}."]
    meta = build_job_meta(
        job_type="scenario",
        organization_id=organization_id,
        field_id=field_id,
        status="running",
        progress=5,
        progress_pct=5.0,
        stage_code="prepare",
        stage_label="prepare",
        stage_detail="initializing scenario context",
        stage_detail_code="initializing_context",
        logs=logs,
        estimated_remaining_s=8,
    )
    task.update_state(state="PROGRESS", meta=meta)

    await _reset_task_db_runtime()
    factory = get_session_factory()
    async with factory() as db:
        service = ModelingService(db)
        try:
            from services.weekly_profile_service import (
                FEATURE_SCHEMA_VERSION,
                current_season_year,
                ensure_weekly_profile,
            )

            field_uuid = UUID(field_id)
            org_uuid = UUID(organization_id)

            meta = append_job_log(meta, "Проверяю baseline прогноз.")
            _update_task_state(
                task,
                meta,
                progress=18,
                stage_label="baseline",
                stage_code="baseline",
                stage_detail="loading baseline prediction",
                stage_detail_code="loading_baseline_prediction",
                estimated_remaining_s=6,
            )

            meta = append_job_log(meta, "Материализую недельный профиль для сценарного расчёта.")
            _update_task_state(
                task,
                meta,
                progress=34,
                stage_label="weekly-profile",
                stage_code="weekly_profile",
                stage_detail="canonical weekly feature store",
                stage_detail_code="materializing_weekly_profile",
                estimated_remaining_s=5,
            )
            weekly_rows = await ensure_weekly_profile(
                db,
                organization_id=org_uuid,
                field_id=field_uuid,
                season_year=current_season_year(),
            )
            meta = append_job_log(
                meta,
                f"Недельный профиль сценария готов: {len(weekly_rows)} недель, схема {FEATURE_SCHEMA_VERSION}.",
            )

            meta = append_job_log(meta, "Считаю counterfactual response curves.")
            _update_task_state(
                task,
                meta,
                progress=58,
                stage_label="counterfactual",
                stage_code="counterfactual",
                stage_detail="agronomic response evaluation",
                stage_detail_code="agronomic_response_evaluation",
                estimated_remaining_s=4,
            )
            payload = await service.simulate(
                field_uuid,
                organization_id=org_uuid,
                crop_code=request_payload.get("crop_code"),
                scenario_name=request_payload.get("scenario_name"),
                irrigation_pct=float(request_payload.get("irrigation_pct", 0.0)),
                fertilizer_pct=float(request_payload.get("fertilizer_pct", 0.0)),
                expected_rain_mm=float(request_payload.get("expected_rain_mm", 0.0)),
                temperature_delta_c=float(request_payload.get("temperature_delta_c", 0.0)),
                precipitation_factor=(
                    float(request_payload.get("precipitation_factor"))
                    if request_payload.get("precipitation_factor") is not None
                    else None
                ),
                planting_density_pct=float(request_payload.get("planting_density_pct", 0.0)),
                sowing_shift_days=request_payload.get("sowing_shift_days"),
                tillage_type=request_payload.get("tillage_type"),
                pest_pressure=request_payload.get("pest_pressure"),
                soil_compaction=request_payload.get("soil_compaction"),
                irrigation_events=list(request_payload.get("irrigation_events") or []),
                fertilizer_events=list(request_payload.get("fertilizer_events") or []),
                save=True,
            )

            meta = append_job_log(meta, "Сохраняю сценарий и подготавливаю сравнительные метрики.")
            _update_task_state(
                task,
                meta,
                progress=86,
                stage_label="persist",
                stage_code="persist",
                stage_detail="writing scenario snapshot",
                stage_detail_code="writing_scenario_snapshot",
                estimated_remaining_s=1,
            )
            await db.commit()
            payload["feature_schema_version"] = FEATURE_SCHEMA_VERSION
            payload["weekly_profile_rows"] = len(weekly_rows)
            meta = append_job_log(meta, f"Сценарий готов: {round(float(payload['predicted_yield_change_pct']), 2)}%.")
            return build_job_meta(
                job_type="scenario",
                organization_id=organization_id,
                field_id=field_id,
                status="done",
                progress=100,
                progress_pct=100.0,
                stage_code="done",
                stage_label="done",
                stage_detail="scenario completed",
                stage_detail_code="scenario_completed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                result=payload,
            )
        except Exception as exc:
            await db.rollback()
            logger.error("scenario_job_failed", field_id=field_id, organization_id=organization_id, error=str(exc), exc_info=True)
            meta = append_job_log(meta, f"Ошибка сценария: {exc}")
            return build_job_meta(
                job_type="scenario",
                organization_id=organization_id,
                field_id=field_id,
                status="failed",
                progress=max(int(meta.get("progress") or 0), 100),
                progress_pct=float(max(int(meta.get("progress") or 0), 100)),
                stage_code="failed",
                stage_label="failed",
                stage_detail="scenario failed",
                stage_detail_code="scenario_failed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                error_msg=str(exc),
            )


async def _run_temporal_analytics_impl(
    task,
    *,
    field_id: str,
    organization_id: str,
    date_from: str,
    date_to: str,
) -> dict[str, Any]:
    logs = [f"Старт materialization сезонной аналитики для поля {field_id}."]
    meta = build_job_meta(
        job_type="temporal_analytics",
        organization_id=organization_id,
        field_id=field_id,
        status="running",
        progress=5,
        progress_pct=5.0,
        stage_code="prepare",
        stage_label="prepare",
        stage_detail="initializing temporal analytics context",
        stage_detail_code="initializing_context",
        logs=logs,
        estimated_remaining_s=12,
    )
    task.update_state(state="PROGRESS", meta=meta)

    await _reset_task_db_runtime()
    factory = get_session_factory()
    async with factory() as db:
        service = TemporalAnalyticsService(db)

        async def progress_callback(**payload: Any) -> None:
            nonlocal meta
            detail = payload.get("stage_detail")
            params = dict(payload.get("stage_detail_params") or {})
            if detail:
                meta = append_job_log(meta, detail)
            meta = _update_task_state(
                task,
                meta,
                progress=int(payload.get("progress", meta.get("progress") or 0)),
                stage_label=str(payload.get("stage_label") or meta.get("stage_label") or "running"),
                stage_code=payload.get("stage_code"),
                stage_detail=detail,
                stage_detail_code=payload.get("stage_detail_code"),
                stage_detail_params=params,
                estimated_remaining_s=payload.get("estimated_remaining_s"),
            )

        try:
            field_uuid = UUID(field_id)
            org_uuid = UUID(organization_id)
            range_from = date.fromisoformat(date_from)
            range_to = date.fromisoformat(date_to)
            payload = await service.materialize_temporal_range(
                field_uuid,
                organization_id=org_uuid,
                date_from=range_from,
                date_to=range_to,
                progress_callback=progress_callback,
            )
            result_payload = dict(payload.get("result") or {})
            result_payload["materialization"] = {
                "date_from": payload.get("date_from"),
                "date_to": payload.get("date_to"),
                "season_years": payload.get("season_years"),
                "satellite_windows_total": payload.get("satellite_windows_total"),
                "satellite_windows_saved": payload.get("satellite_windows_saved"),
                "weekly_rows_materialized": payload.get("weekly_rows_materialized"),
            }
            meta = append_job_log(meta, "Сезонная аналитика materialized и готова к чтению.")
            return build_job_meta(
                job_type="temporal_analytics",
                organization_id=organization_id,
                field_id=field_id,
                status="done",
                progress=100,
                progress_pct=100.0,
                stage_code="done",
                stage_label="done",
                stage_detail="temporal analytics completed",
                stage_detail_code="temporal_analytics_completed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                result=result_payload,
            )
        except Exception as exc:
            await db.rollback()
            error_code = exc.code if isinstance(exc, TemporalBackfillError) else "temporal_analytics_failed"
            logger.error("temporal_analytics_job_failed", field_id=field_id, organization_id=organization_id, error=str(exc), exc_info=True)
            meta = append_job_log(meta, f"Ошибка сезонной аналитики: {exc}")
            return build_job_meta(
                job_type="temporal_analytics",
                organization_id=organization_id,
                field_id=field_id,
                status="failed",
                progress=max(int(meta.get("progress") or 0), 100),
                progress_pct=float(max(int(meta.get("progress") or 0), 100)),
                stage_code="failed",
                stage_label="failed",
                stage_detail="temporal analytics failed",
                stage_detail_code=error_code,
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                error_msg=str(exc),
                result={"data_status": {"code": error_code, "message_code": error_code}},
            )


@celery.task(name="tasks.analytics.run_prediction", bind=True)
def run_prediction_job(self, field_id: str, organization_id: str, crop_code: str | None = None, refresh: bool = True) -> dict[str, Any]:
    return asyncio.run(
        _run_prediction_impl(
            self,
            field_id=field_id,
            organization_id=organization_id,
            crop_code=crop_code,
            refresh=refresh,
        )
    )


@celery.task(name="tasks.analytics.run_scenario", bind=True)
def run_modeling_job(self, organization_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(
        _run_modeling_impl(
            self,
            organization_id=organization_id,
            request_payload=request_payload,
        )
    )


@celery.task(name="tasks.analytics.run_temporal_analytics", bind=True)
def run_temporal_analytics_job(self, field_id: str, organization_id: str, date_from: str, date_to: str) -> dict[str, Any]:
    return asyncio.run(
        _run_temporal_analytics_impl(
            self,
            field_id=field_id,
            organization_id=organization_id,
            date_from=date_from,
            date_to=date_to,
        )
    )


async def _run_weekly_backfill_impl(
    task,
    *,
    field_id: str,
    organization_id: str,
    season_year: int,
) -> dict[str, Any]:
    from services.weekly_profile_service import ensure_weekly_profile, serialize_weekly_feature_rows

    logs = [f"Старт backfill недельного профиля для поля {field_id}, сезон {season_year}."]
    meta = build_job_meta(
        job_type="weekly_backfill",
        organization_id=organization_id,
        field_id=field_id,
        status="running",
        progress=5,
        progress_pct=5.0,
        stage_code="prepare",
        stage_label="prepare",
        stage_detail="initializing weekly backfill",
        stage_detail_code="initializing",
        logs=logs,
        estimated_remaining_s=60,
    )
    task.update_state(state="PROGRESS", meta=meta)

    await _reset_task_db_runtime()
    factory = get_session_factory()
    async with factory() as db:
        try:
            field_uuid = UUID(field_id)
            org_uuid = UUID(organization_id)
            meta = _update_task_state(
                task, meta,
                progress=20,
                stage_label="backfill",
                stage_code="backfill",
                stage_detail="fetching ERA5 data and computing weekly features",
                stage_detail_code="era5_fetch",
            )
            rows = await ensure_weekly_profile(
                db,
                organization_id=org_uuid,
                field_id=field_uuid,
                season_year=season_year,
                force=True,
            )
            payload_rows = serialize_weekly_feature_rows(rows)
            meta = append_job_log(meta, f"Backfill завершён: {len(payload_rows)} недель записано.")
            return build_job_meta(
                job_type="weekly_backfill",
                organization_id=organization_id,
                field_id=field_id,
                status="done",
                progress=100,
                progress_pct=100.0,
                stage_code="done",
                stage_label="done",
                stage_detail="weekly backfill completed",
                stage_detail_code="weekly_backfill_completed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                result={"weeks_count": len(payload_rows), "season_year": season_year},
            )
        except Exception as exc:
            await db.rollback()
            logger.error("weekly_backfill_job_failed", field_id=field_id, organization_id=organization_id, error=str(exc), exc_info=True)
            meta = append_job_log(meta, f"Ошибка backfill: {exc}")
            return build_job_meta(
                job_type="weekly_backfill",
                organization_id=organization_id,
                field_id=field_id,
                status="failed",
                progress=max(int(meta.get("progress") or 0), 100),
                progress_pct=float(max(int(meta.get("progress") or 0), 100)),
                stage_code="failed",
                stage_label="failed",
                stage_detail="weekly backfill failed",
                stage_detail_code="weekly_backfill_failed",
                started_at=meta.get("started_at"),
                logs=meta.get("logs"),
                error_msg=str(exc),
                result={"error": str(exc)},
            )


@celery.task(name="tasks.analytics.run_weekly_backfill", bind=True)
def run_weekly_backfill_job(self, field_id: str, organization_id: str, season_year: int) -> dict[str, Any]:
    return asyncio.run(
        _run_weekly_backfill_impl(
            self,
            field_id=field_id,
            organization_id=organization_id,
            season_year=season_year,
        )
    )
