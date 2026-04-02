from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import RequestContext, require_permissions
from api.schemas import SatelliteBrowseResponse
from core.logging import get_logger
from providers.sentinelhub.client import SentinelHubClient
from services.payload_meta import build_freshness

logger = get_logger(__name__)

router = APIRouter(prefix="/satellite", tags=["satellite"])

_CLOUD_CLASSES = {3, 8, 9, 10, 11}
_VALID_BROWSE_SCL_CLASSES = {4, 5, 6, 7}
_MIN_VALID_COVERAGE_PCT = 60.0


def _normalize_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, 0.82)
    rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb


def _encode_png_base64(rgb: np.ndarray) -> str:
    encoded_ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not encoded_ok:
        raise RuntimeError("Не удалось закодировать PNG для спутникового превью")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _cloud_cover_from_scl(scl: np.ndarray | None) -> float | None:
    if scl is None:
        return None
    total = int(scl.size)
    if total <= 0:
        return None
    cloudy = int(np.isin(scl.astype(np.uint8), tuple(_CLOUD_CLASSES)).sum())
    return round(cloudy / total * 100.0, 1)


def _valid_coverage_pct(payload: dict[str, np.ndarray]) -> float | None:
    red = payload.get("B4")
    green = payload.get("B3")
    blue = payload.get("B2")
    if red is None or green is None or blue is None:
        return None
    finite = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    scl = payload.get("SCL")
    if scl is not None:
        valid_mask = finite & np.isin(scl.astype(np.uint8), tuple(_VALID_BROWSE_SCL_CLASSES))
    else:
        signal_mask = np.maximum.reduce([np.abs(red), np.abs(green), np.abs(blue)]) > 0.005
        valid_mask = finite & signal_mask
    total = int(valid_mask.size)
    if total <= 0:
        return None
    return round(float(valid_mask.sum()) / total * 100.0, 1)


async def _fetch_scene_payload(
    *,
    sentinel: SentinelHubClient,
    bbox: tuple[float, float, float, float],
    time_from: datetime,
    time_to: datetime,
    width: int,
    height: int,
    max_cloud_pct: int,
) -> dict[str, Any]:
    payload = await sentinel.fetch_tile(
        bbox=bbox,
        time_from=time_from.isoformat().replace("+00:00", "Z"),
        time_to=time_to.isoformat().replace("+00:00", "Z"),
        width=width,
        height=height,
        max_cloud_pct=max_cloud_pct,
    )
    rgb = _normalize_rgb(payload["B4"], payload["B3"], payload["B2"])
    return {
        "payload": payload,
        "rgb": rgb,
        "cloud_cover_pct": _cloud_cover_from_scl(payload.get("SCL")),
        "valid_coverage_pct": _valid_coverage_pct(payload),
        "provider_account": getattr(sentinel, "last_account_alias", "primary"),
        "failover_level": int(getattr(sentinel, "last_failover_level", 0)),
    }


@router.get("/true-color", response_model=SatelliteBrowseResponse)
async def get_true_color_scene(
    minx: float = Query(...),
    miny: float = Query(...),
    maxx: float = Query(...),
    maxy: float = Query(...),
    width: int = Query(960, ge=128, le=1600),
    height: int = Query(960, ge=128, le=1600),
    scene_date: date | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    max_cloud_pct: int = Query(35, ge=0, le=100),
    _ctx: RequestContext = Depends(require_permissions("layers:read")),
) -> SatelliteBrowseResponse:
    bbox = (float(minx), float(miny), float(maxx), float(maxy))
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise HTTPException(status_code=422, detail="bbox must satisfy min < max")

    effective_start = start_date or date(2025, 5, 1)
    effective_end = end_date or date(2025, 8, 31)
    if effective_end <= effective_start:
        raise HTTPException(status_code=422, detail="end_date must be after start_date")

    if scene_date is not None:
        time_from = datetime.combine(scene_date, datetime.min.time(), tzinfo=timezone.utc)
        time_to = time_from + timedelta(days=1) - timedelta(seconds=1)
        requested_window = {
            "start": effective_start.isoformat(),
            "end": effective_end.isoformat(),
        }
        requested_date = scene_date.isoformat()
    else:
        time_from = datetime.combine(effective_start, datetime.min.time(), tzinfo=timezone.utc)
        time_to = datetime.combine(effective_end, datetime.max.time(), tzinfo=timezone.utc)
        requested_window = {
            "start": effective_start.isoformat(),
            "end": effective_end.isoformat(),
        }
        requested_date = None

    sentinel = SentinelHubClient()
    fetched_at = datetime.now(timezone.utc)
    try:
        scene = await _fetch_scene_payload(
            sentinel=sentinel,
            bbox=bbox,
            time_from=time_from,
            time_to=time_to,
            width=width,
            height=height,
            max_cloud_pct=max_cloud_pct,
        )
    except Exception as exc:
        logger.warning("sentinel_browse_fetch_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Sentinel true-color browse unavailable") from exc

    status = "ready"
    resolved_date = requested_date
    fallback_reason: str | None = None

    exact_date_requested = scene_date is not None
    current_coverage = float(scene.get("valid_coverage_pct") or 0.0)
    can_try_window_fallback = exact_date_requested and (
        requested_window["start"] != requested_date or requested_window["end"] != requested_date
    )
    if can_try_window_fallback and current_coverage < _MIN_VALID_COVERAGE_PCT:
        try:
            fallback_scene = await _fetch_scene_payload(
                sentinel=sentinel,
                bbox=bbox,
                time_from=datetime.combine(effective_start, datetime.min.time(), tzinfo=timezone.utc),
                time_to=datetime.combine(effective_end, datetime.max.time(), tzinfo=timezone.utc),
                width=width,
                height=height,
                max_cloud_pct=max_cloud_pct,
            )
            fallback_coverage = float(fallback_scene.get("valid_coverage_pct") or 0.0)
            if fallback_coverage >= _MIN_VALID_COVERAGE_PCT and fallback_coverage > current_coverage:
                scene = fallback_scene
                status = "fallback_window"
                resolved_date = None
                fallback_reason = "insufficient_exact_coverage"
        except Exception as exc:
            logger.warning(
                "sentinel_browse_window_fallback_failed",
                requested_date=requested_date,
                error=str(exc),
            )

    image_base64: str | None = None
    final_coverage = float(scene.get("valid_coverage_pct") or 0.0)
    if final_coverage >= _MIN_VALID_COVERAGE_PCT:
        image_base64 = _encode_png_base64(scene["rgb"])
    else:
        status = "no_data"
        resolved_date = None
        fallback_reason = fallback_reason or "insufficient_valid_coverage"

    if status == "no_data":
        logger.info(
            "sentinel_browse_no_data",
            requested_date=requested_date,
            requested_window=requested_window,
            valid_coverage_pct=scene.get("valid_coverage_pct"),
            fallback_reason=fallback_reason,
        )

    return SatelliteBrowseResponse(
        bbox=list(bbox),
        width=width,
        height=height,
        status=status,
        requested_date=requested_date,
        resolved_date=resolved_date,
        requested_window=requested_window,
        provider="sentinel-2-l2a",
        provider_account=scene.get("provider_account"),
        failover_level=int(scene.get("failover_level", 0)),
        cloud_cover_pct=scene.get("cloud_cover_pct"),
        valid_coverage_pct=scene.get("valid_coverage_pct"),
        image_base64=image_base64,
        freshness=build_freshness(
            provider="sentinel_true_color",
            fetched_at=fetched_at,
            cache_written_at=fetched_at,
        ),
    )
