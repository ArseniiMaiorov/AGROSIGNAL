"""Helpers for Celery-backed async job status/result payloads."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any
from uuid import UUID

from celery.result import AsyncResult

from core.celery_app import celery


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds(started_at: str | None, updated_at: str | None) -> int | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat((updated_at or started_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((end - start).total_seconds()))


def _normalize_stage_code(stage_label: str | None, status: str | None = None) -> str | None:
    raw = str(stage_label or status or "").strip().lower()
    if not raw:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return normalized or None


def build_job_meta(
    *,
    job_type: str,
    organization_id: UUID | str,
    field_id: UUID | str | None = None,
    status: str,
    progress: int,
    progress_pct: float | None = None,
    stage_code: str | None = None,
    stage_label: str | None = None,
    stage_detail: str | None = None,
    stage_detail_code: str | None = None,
    stage_detail_params: dict[str, Any] | None = None,
    started_at: str | None = None,
    updated_at: str | None = None,
    estimated_remaining_s: int | None = None,
    logs: list[str] | None = None,
    error_msg: str | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = started_at or _iso_now()
    updated_at = updated_at or _iso_now()
    normalized_stage_code = stage_code or _normalize_stage_code(stage_label, status)
    payload = {
        "job_type": job_type,
        "organization_id": str(organization_id),
        "field_id": str(field_id) if field_id else None,
        "status": status,
        "progress": int(progress),
        "progress_pct": round(float(progress if progress_pct is None else progress_pct), 2),
        "stage_code": normalized_stage_code,
        "stage_label": stage_label,
        "stage_detail": stage_detail,
        "stage_detail_code": stage_detail_code,
        "stage_detail_params": dict(stage_detail_params or {}),
        "started_at": started_at,
        "updated_at": updated_at,
        "elapsed_s": _elapsed_seconds(started_at, updated_at),
        "estimated_remaining_s": estimated_remaining_s,
        "logs": list(logs or []),
        "error_msg": error_msg,
        "result_ready": result is not None or status in {"done", "failed", "cancelled", "stale"},
        "result": result,
    }
    return payload


def append_job_log(meta: dict[str, Any], message: str) -> dict[str, Any]:
    logs = list(meta.get("logs") or [])
    logs.append(message)
    meta["logs"] = logs[-24:]
    meta["updated_at"] = _iso_now()
    meta["elapsed_s"] = _elapsed_seconds(meta.get("started_at"), meta.get("updated_at"))
    return meta


def prime_async_job(
    *,
    task_id: str,
    job_type: str,
    organization_id: UUID | str,
    field_id: UUID | str | None = None,
    stage_label: str = "queued",
    stage_detail: str | None = None,
    stage_code: str | None = None,
    stage_detail_code: str | None = None,
    stage_detail_params: dict[str, Any] | None = None,
    logs: list[str] | None = None,
) -> dict[str, Any]:
    meta = build_job_meta(
        job_type=job_type,
        organization_id=organization_id,
        field_id=field_id,
        status="queued",
        progress=0,
        progress_pct=0.0,
        stage_code=stage_code,
        stage_label=stage_label,
        stage_detail=stage_detail,
        stage_detail_code=stage_detail_code,
        stage_detail_params=stage_detail_params,
        logs=logs or [f"Задача {job_type} поставлена в очередь."],
    )
    celery.backend.store_result(task_id, meta, state="PENDING")
    return meta


def build_async_job_submit_payload(meta: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(meta or {})
    progress = int(payload.get("progress", 0) or 0)
    return {
        "status": str(payload.get("status") or "queued"),
        "progress": progress,
        "progress_pct": round(float(payload.get("progress_pct", progress) or 0.0), 2),
        "stage_code": payload.get("stage_code") or _normalize_stage_code(payload.get("stage_label"), payload.get("status")),
        "stage_label": payload.get("stage_label"),
        "stage_detail": payload.get("stage_detail"),
        "stage_detail_code": payload.get("stage_detail_code"),
        "stage_detail_params": dict(payload.get("stage_detail_params") or {}),
        "started_at": payload.get("started_at"),
        "updated_at": payload.get("updated_at"),
        "elapsed_s": payload.get("elapsed_s"),
        "estimated_remaining_s": payload.get("estimated_remaining_s"),
        "logs": list(payload.get("logs") or []),
        "error_msg": payload.get("error_msg"),
    }


def _normalize_meta(task_id: str, state: str, info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        meta = dict(info)
    else:
        meta = {
            "status": "failed" if state == "FAILURE" else "queued",
            "progress": 100 if state == "FAILURE" else 0,
            "progress_pct": 100.0 if state == "FAILURE" else 0.0,
            "error_msg": str(info) if info else None,
            "logs": [],
        }
    meta.setdefault("job_type", "job")
    meta.setdefault("status", "done" if state == "SUCCESS" else "running" if state in {"STARTED", "PROGRESS"} else "queued")
    meta.setdefault("progress", 100 if state == "SUCCESS" else 0)
    meta.setdefault("progress_pct", float(meta.get("progress") or 0.0))
    meta.setdefault("stage_code", _normalize_stage_code(meta.get("stage_label"), meta.get("status")))
    meta.setdefault("stage_detail_code", None)
    meta.setdefault("stage_detail_params", {})
    meta.setdefault("logs", [])
    meta.setdefault("result_ready", state in {"SUCCESS", "FAILURE"})
    meta.setdefault("started_at", meta.get("updated_at") or _iso_now())
    meta.setdefault("updated_at", meta.get("started_at") or _iso_now())
    meta["elapsed_s"] = _elapsed_seconds(meta.get("started_at"), meta.get("updated_at"))
    meta["task_id"] = task_id
    if state == "FAILURE":
        meta["status"] = "failed"
        meta["error_msg"] = meta.get("error_msg") or str(info) or "Задача завершилась ошибкой"
        meta["result_ready"] = True
    return meta


def get_async_job_payload(task_id: str) -> dict[str, Any]:
    result = AsyncResult(task_id, app=celery)
    return _normalize_meta(task_id, result.state, result.info)


def require_job_access(task_payload: dict[str, Any], organization_id: UUID | str, *, job_type: str | None = None) -> None:
    payload_org = str(task_payload.get("organization_id") or "")
    if payload_org and payload_org != str(organization_id):
        raise PermissionError("Задача не найдена")
    if job_type and task_payload.get("job_type") not in {job_type, None, ""}:
        raise PermissionError("Задача не найдена")
