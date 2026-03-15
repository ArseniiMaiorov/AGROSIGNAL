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

    if scene_date is not None:
        time_from = datetime.combine(scene_date, datetime.min.time(), tzinfo=timezone.utc)
        time_to = time_from + timedelta(days=1) - timedelta(seconds=1)
        requested_window: dict[str, str] | None = None
        requested_date = scene_date.isoformat()
    else:
        effective_start = start_date or date(2025, 5, 1)
        effective_end = end_date or date(2025, 8, 31)
        if effective_end <= effective_start:
            raise HTTPException(status_code=422, detail="end_date must be after start_date")
        time_from = datetime.combine(effective_start, datetime.min.time(), tzinfo=timezone.utc)
        time_to = datetime.combine(effective_end, datetime.max.time(), tzinfo=timezone.utc)
        requested_window = {
            "start": effective_start.isoformat(),
            "end": effective_end.isoformat(),
        }
        requested_date = None

    sentinel = SentinelHubClient()
    try:
        payload = await sentinel.fetch_tile(
            bbox=bbox,
            time_from=time_from.isoformat().replace("+00:00", "Z"),
            time_to=time_to.isoformat().replace("+00:00", "Z"),
            width=width,
            height=height,
            max_cloud_pct=max_cloud_pct,
        )
    except Exception as exc:
        logger.warning("sentinel_browse_fetch_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Sentinel true-color browse unavailable") from exc

    rgb = _normalize_rgb(payload["B4"], payload["B3"], payload["B2"])
    cloud_cover_pct = _cloud_cover_from_scl(payload.get("SCL"))
    image_base64 = _encode_png_base64(rgb)
    fetched_at = datetime.now(timezone.utc)
    return SatelliteBrowseResponse(
        bbox=list(bbox),
        width=width,
        height=height,
        requested_date=requested_date,
        requested_window=requested_window,
        provider="sentinel-2-l2a",
        provider_account=getattr(sentinel, "last_account_alias", "primary"),
        failover_level=int(getattr(sentinel, "last_failover_level", 0)),
        cloud_cover_pct=cloud_cover_pct,
        image_base64=image_base64,
        freshness=build_freshness(
            provider="sentinel_true_color",
            fetched_at=fetched_at,
            cache_written_at=fetched_at,
        ),
    )
