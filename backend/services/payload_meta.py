from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return None


def build_freshness(
    *,
    provider: str | None,
    fetched_at: datetime | str | None = None,
    cache_written_at: datetime | str | None = None,
    source_published_at: datetime | str | None = None,
    ttl_seconds: int | float | None = None,
    stale: bool = False,
    freshness_state: str | None = None,
    model_version: str | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    fetched_iso = iso_or_none(fetched_at)
    cache_written_iso = iso_or_none(cache_written_at)
    source_iso = iso_or_none(source_published_at)
    resolved_freshness_state = freshness_state

    reference_raw = fetched_at or cache_written_at
    reference_dt: datetime | None = None
    if isinstance(reference_raw, datetime):
        reference_dt = reference_raw if reference_raw.tzinfo is not None else reference_raw.replace(tzinfo=timezone.utc)
    elif isinstance(reference_raw, str):
        try:
            reference_dt = datetime.fromisoformat(reference_raw.replace("Z", "+00:00"))
        except ValueError:
            reference_dt = None

    if resolved_freshness_state is not None:
        resolved_freshness_state = str(resolved_freshness_state)
    elif stale:
        resolved_freshness_state = "stale"
    elif reference_dt is not None:
        if ttl_seconds is None:
            resolved_freshness_state = "fresh"
        else:
            age_s = (datetime.now(timezone.utc) - reference_dt).total_seconds()
            resolved_freshness_state = "fresh" if age_s <= float(ttl_seconds) else "stale"
    else:
        resolved_freshness_state = "unknown"

    return {
        "provider": provider,
        "fetched_at": fetched_iso,
        "cache_written_at": cache_written_iso,
        "freshness_state": resolved_freshness_state,
        "source_published_at": source_iso,
        "model_version": model_version,
        "dataset_version": dataset_version,
    }
