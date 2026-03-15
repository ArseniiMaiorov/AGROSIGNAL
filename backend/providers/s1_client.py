"""Sentinel-1 GRD fetcher via Sentinel Hub Process API."""
from __future__ import annotations

import io
import time

import httpx
import numpy as np
import rasterio

from core.config import get_settings
from core.logging import get_logger
from core.metrics import S1_FETCH_TIME
from providers.s1_evalscripts import S1_EVALSCRIPT
from providers.sentinelhub.client import SentinelHubClient, _extract_api_error

logger = get_logger(__name__)


class SentinelHubS1Client:
    """Minimal Sentinel-1 client reusing Sentinel Hub auth."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.SH_BASE_URL
        self._s2_client = SentinelHubClient()

    def build_request_body(
        self,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
    ) -> dict:
        return {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-1-grd",
                        "dataFilter": {
                            "timeRange": {"from": time_from, "to": time_to},
                            "acquisitionMode": str(self.settings.S1_ACQUISITION_MODE),
                            "polarization": str(self.settings.S1_POLARIZATION),
                        },
                    }
                ],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": S1_EVALSCRIPT,
        }

    async def fetch_tile(
        self,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
    ) -> dict[str, np.ndarray]:
        """Fetch a Sentinel-1 VV/VH tile."""
        body = self.build_request_body(bbox, time_from, time_to, width, height)
        with S1_FETCH_TIME.time():
            async with httpx.AsyncClient() as client:
                t0 = time.time()
                resp, account_alias, failover_level = await self._s2_client.process_request(
                    body,
                    client=client,
                    timeout=120.0,
                )
                latency = time.time() - t0
                logger.info(
                    "sentinel_s1_response",
                    account_alias=account_alias,
                    failover_level=failover_level,
                    status=resp.status_code,
                    latency_s=round(latency, 2),
                )
                if resp.is_error:
                    detail = _extract_api_error(resp)
                    raise httpx.HTTPStatusError(
                        f"Sentinel-1 Process API {resp.status_code}: {detail}",
                        request=resp.request,
                        response=resp,
                    )

        with rasterio.open(io.BytesIO(resp.content)) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32) if src.count >= 2 else np.zeros_like(vv)
        return {"VV": vv, "VH": vh}
