import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

import httpx
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from core.metrics import CACHE_HITS, CACHE_MISSES
from providers.sentinelhub.evalscripts import BANDS_AND_SCL, BANDS_HARMONIZED_AND_SCL

try:
    from providers.sentinelhub.evalscripts import BANDS_V4_REDEDGE
except ImportError:
    BANDS_V4_REDEDGE = None

# Band layout for V4 red-edge evalscript: 19 bands total
# [0-7] = indices: ndvi, ndwi, ndmi, bsi, msi, ndre, cire, re_slope
# [8-17] = bands: b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12
# [18] = SCL
_V4_BAND_NAMES = (
    "NDVI_idx", "NDWI_idx", "NDMI_idx", "BSI_idx", "MSI_idx",
    "NDRE_idx", "CIre_idx", "RE_SLOPE_idx",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "SCL",
)
_HARMONIZED_RAW_KEYS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL")
_HARMONIZED_INDEX_KEYS = (
    "NDVI_idx",
    "NDWI_idx",
    "NDMI_idx",
    "BSI_idx",
    "MSI_idx",
    "NDRE_idx",
    "CIre_idx",
    "RE_SLOPE_idx",
)
_HARMONIZED_RESULT_KEYS = _HARMONIZED_INDEX_KEYS + _HARMONIZED_RAW_KEYS

logger = get_logger(__name__)

_token_cache: dict[str, dict[str, Any]] = {}

_RETRYABLE_STATUS_CODES = {401, 429, 500, 502, 503, 504}
_FAILOVER_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}


def _extract_api_error(resp: httpx.Response) -> str:
    """Return compact message from Sentinel Hub error response."""
    try:
        payload = resp.json()
    except Exception:
        return resp.text[:1000]

    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return json.dumps(payload, ensure_ascii=True)[:1000]

    details = []
    message = error_obj.get("message")
    reason = error_obj.get("reason")
    code = error_obj.get("code")
    if code:
        details.append(f"code={code}")
    if reason:
        details.append(f"reason={reason}")
    if message:
        details.append(f"message={message}")

    errors = error_obj.get("errors")
    if isinstance(errors, dict):
        parameter = errors.get("parameter")
        invalid_value = errors.get("invalidValue")
        description = errors.get("description")
        if parameter:
            details.append(f"parameter={parameter}")
        if invalid_value:
            details.append(f"invalidValue={invalid_value}")
        if description:
            details.append(f"description={description}")

    return "; ".join(details)[:1000]


def _parse_retry_after_seconds(resp: httpx.Response) -> float | None:
    """Parse Retry-After header when available."""
    retry_after = resp.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def _compute_retry_delay(
    resp: httpx.Response,
    attempt: int,
    *,
    base_delay_s: float,
    max_delay_s: float,
) -> float:
    """Return retry delay using Retry-After or bounded exponential backoff."""
    header_delay = _parse_retry_after_seconds(resp)
    if header_delay is not None:
        return min(max_delay_s, header_delay)
    return _compute_backoff_delay(
        attempt,
        base_delay_s=base_delay_s,
        max_delay_s=max_delay_s,
    )


def _compute_backoff_delay(
    attempt: int,
    *,
    base_delay_s: float,
    max_delay_s: float,
) -> float:
    """Return bounded exponential backoff delay with jitter."""
    base = min(max_delay_s, base_delay_s * (2 ** attempt))
    return base * (0.5 + random.random() * 0.5)


class SentinelHubClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.SH_BASE_URL
        self._shared_client: httpx.AsyncClient | None = None
        self._shared_client_loop_id: int | None = None
        self._http2_enabled = importlib.util.find_spec("h2") is not None
        self._token_lock: asyncio.Lock | None = None
        self._token_lock_loop_id: int | None = None
        self.scene_cache_dir = self._resolve_scene_cache_dir()
        self._accounts = self._build_accounts()
        self._last_account_alias = self._accounts[0]["alias"] if self._accounts else "unconfigured"
        self._last_failover_level = 0
        self._account_cooldowns: dict[str, float] = {}

    def _configured_scene_cache_dir(self) -> Path:
        scene_cache_dir = Path(self.settings.SCENE_CACHE_DIR)
        if not scene_cache_dir.is_absolute():
            scene_cache_dir = Path(__file__).resolve().parents[2] / scene_cache_dir
        return scene_cache_dir

    @staticmethod
    def _dir_is_writable(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            with open(probe, "wb") as handle:
                handle.write(b"ok")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _resolve_scene_cache_dir(self) -> Path:
        configured = self._configured_scene_cache_dir()
        if self._dir_is_writable(configured):
            return configured

        fallback_candidates = [
            Path(__file__).resolve().parents[2] / "debug" / "cache" / "sentinel_scenes",
            Path.home() / ".cache" / "autodetect" / "sentinel_scenes",
            Path("/tmp") / f"autodetect_sentinel_scenes_{os.getuid()}",
        ]
        for candidate in fallback_candidates:
            if self._dir_is_writable(candidate):
                logger.warning(
                    "sentinel_scene_cache_fallback_enabled",
                    configured_path=str(configured),
                    fallback_path=str(candidate),
                )
                return candidate
        logger.warning("sentinel_scene_cache_disabled", configured_path=str(configured))
        return configured

    def _build_http_client(self) -> httpx.AsyncClient:
        concurrency = max(1, int(getattr(self.settings, "SENTINEL_CONCURRENT_REQUESTS", 4)))
        limits = httpx.Limits(
            max_connections=max(8, concurrency * 4),
            max_keepalive_connections=max(4, concurrency * 2),
        )
        return httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(120.0),
            http2=self._http2_enabled,
        )

    def _build_accounts(self) -> list[dict[str, str]]:
        accounts: list[dict[str, str]] = []

        def _append(alias: str, client_id: str | None, client_secret: str | None) -> None:
            cid = str(client_id or "").strip()
            secret = str(client_secret or "").strip()
            if cid and secret:
                accounts.append({"alias": alias, "client_id": cid, "client_secret": secret})

        _append("primary", self.settings.SH_CLIENT_ID, self.settings.SH_CLIENT_SECRET)
        if bool(getattr(self.settings, "SH_FAILOVER_ENABLED", True)):
            _append("reserve", getattr(self.settings, "SH_CLIENT_ID_RESERVE", None), getattr(self.settings, "SH_CLIENT_SECRET_RESERVE", None))
            _append(
                "second_reserve",
                getattr(self.settings, "SH_CLIENT_ID_SECOND_RESERVE", None),
                getattr(self.settings, "SH_CLIENT_SECRET_SECOND_RESERVE", None),
            )
        return accounts

    @property
    def last_account_alias(self) -> str:
        return self._last_account_alias

    @property
    def last_failover_level(self) -> int:
        return int(self._last_failover_level)

    @staticmethod
    def _token_entry(alias: str) -> dict[str, Any]:
        return _token_cache.setdefault(alias, {"token": None, "expires_at": 0.0})

    @staticmethod
    def _invalidate_token(alias: str) -> None:
        entry = _token_cache.setdefault(alias, {"token": None, "expires_at": 0.0})
        entry["token"] = None
        entry["expires_at"] = 0.0

    def _ensure_token_lock(self) -> asyncio.Lock:
        """Return a loop-local token lock to avoid cross-loop asyncio reuse."""
        current_loop_id = id(asyncio.get_running_loop())
        if self._token_lock is None or self._token_lock_loop_id != current_loop_id:
            self._token_lock = asyncio.Lock()
            self._token_lock_loop_id = current_loop_id
        return self._token_lock

    async def get_shared_client(self) -> httpx.AsyncClient:
        """Return a persistent shared httpx client (reused across calls)."""
        current_loop_id = id(asyncio.get_running_loop())
        loop_switched = (
            self._shared_client is not None
            and not self._shared_client.is_closed
            and self._shared_client_loop_id is not None
            and self._shared_client_loop_id != current_loop_id
        )
        if loop_switched:
            try:
                await self._shared_client.aclose()
            except RuntimeError as exc:
                if "Event loop is closed" not in str(exc):
                    logger.warning("sentinel_shared_client_recycle_failed", exc_info=True)
            except Exception:
                logger.warning("sentinel_shared_client_recycle_failed", exc_info=True)
            self._shared_client = None
            self._shared_client_loop_id = None
            self._token_lock = None
            self._token_lock_loop_id = None

        if self._shared_client is None or self._shared_client.is_closed:
            self._shared_client = self._build_http_client()
            self._shared_client_loop_id = current_loop_id
            self._ensure_token_lock()
        return self._shared_client

    async def close(self) -> None:
        """Close the shared httpx client if open."""
        if self._shared_client is not None and not self._shared_client.is_closed:
            try:
                await self._shared_client.aclose()
            except RuntimeError as exc:
                # Happens when transports belong to a loop that has already been closed.
                if "Event loop is closed" not in str(exc):
                    raise
            finally:
                self._shared_client = None
                self._shared_client_loop_id = None
                self._token_lock = None
                self._token_lock_loop_id = None

    async def _get_token(self, client: httpx.AsyncClient, account: dict[str, str] | None = None) -> str:
        if account is None:
            if not self._accounts:
                raise RuntimeError("Sentinel Hub credentials are not configured")
            account = self._accounts[0]
        alias = str(account["alias"])
        token_entry = self._token_entry(alias)
        now = time.time()
        if token_entry["token"] and token_entry["expires_at"] > now + 60:
            return str(token_entry["token"])

        async with self._ensure_token_lock():
            now = time.time()
            token_entry = self._token_entry(alias)
            if token_entry["token"] and token_entry["expires_at"] > now + 60:
                return str(token_entry["token"])

            resp = await client.post(
                f"{self.base_url}/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": account["client_id"],
                    "client_secret": account["client_secret"],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            token_entry["token"] = data["access_token"]
            token_entry["expires_at"] = now + data.get("expires_in", 3600)
            return str(token_entry["token"])

    def _build_request_body(
        self,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        evalscript: str = BANDS_AND_SCL,
        max_cloud_pct: int = 40,
    ) -> dict:
        return {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {"from": time_from, "to": time_to},
                            "maxCloudCoverage": max_cloud_pct,
                            "mosaickingOrder": "leastCC",
                        },
                    }
                ],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {"identifier": "default", "format": {"type": "image/tiff"}},
                ],
            },
            "evalscript": evalscript,
        }

    @staticmethod
    def _cache_key(bbox, time_from, time_to, width, height, evalscript_hash, max_cloud_pct: int) -> str:
        data = json.dumps(
            {
                "bbox": list(bbox),
                "time_from": time_from,
                "time_to": time_to,
                "width": width,
                "height": height,
                "evalscript": evalscript_hash,
                "max_cloud_pct": max_cloud_pct,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.scene_cache_dir / f"{cache_key}.npz"

    def _cache_is_fresh(self, path: Path) -> bool:
        ttl_days = max(0, int(getattr(self.settings, "SCENE_CACHE_TTL_DAYS", 30)))
        if ttl_days <= 0:
            return False
        age_seconds = max(0.0, time.time() - path.stat().st_mtime)
        return age_seconds <= ttl_days * 86400

    def _load_cached_scene(self, cache_key: str, *, allow_stale: bool = False) -> dict[str, np.ndarray] | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        if not allow_stale and not self._cache_is_fresh(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return {name: data[name].copy() for name in data.files}
        except Exception:
            logger.warning("sentinel_scene_cache_load_failed", cache_key=cache_key[:12], path=str(path), exc_info=True)
            return None

    def _store_cached_scene(self, cache_key: str, payload: dict[str, np.ndarray]) -> None:
        path = self._cache_path(cache_key)
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as handle:
                np.savez_compressed(handle, **payload)
            tmp_path.replace(path)
        except PermissionError:
            fallback_dir = self._resolve_scene_cache_dir()
            if fallback_dir != self.scene_cache_dir:
                self.scene_cache_dir = fallback_dir
                path = self._cache_path(cache_key)
                tmp_path = path.with_suffix(".tmp")
                try:
                    with open(tmp_path, "wb") as handle:
                        np.savez_compressed(handle, **payload)
                    tmp_path.replace(path)
                    return
                except Exception:
                    logger.warning(
                        "sentinel_scene_cache_store_failed_after_fallback",
                        cache_key=cache_key[:12],
                        path=str(path),
                        exc_info=True,
                    )
                    return
            logger.warning("sentinel_scene_cache_store_failed", cache_key=cache_key[:12], path=str(path), exc_info=True)
        except Exception:
            logger.warning("sentinel_scene_cache_store_failed", cache_key=cache_key[:12], path=str(path), exc_info=True)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                logger.warning("sentinel_scene_cache_tmp_cleanup_failed", path=str(tmp_path), exc_info=True)

    @staticmethod
    def _should_failover_on_exception(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _FAILOVER_STATUS_CODES
        if isinstance(exc, RuntimeError):
            return "failed_external_dependency" in str(exc)
        return False

    def _account_cooldown_remaining_s(self, alias: str) -> float:
        cooldown_until = float(self._account_cooldowns.get(alias, 0.0) or 0.0)
        return max(0.0, cooldown_until - time.time())

    def _mark_account_cooldown(self, alias: str, delay_s: float, *, reason: str) -> None:
        if delay_s <= 0:
            return
        now = time.time()
        cooldown_until = now + delay_s
        existing_until = float(self._account_cooldowns.get(alias, 0.0) or 0.0)
        self._account_cooldowns[alias] = max(existing_until, cooldown_until)
        logger.warning(
            "sentinel_account_cooldown_set",
            account_alias=alias,
            cooldown_s=round(delay_s, 2),
            reason=reason,
        )

    def _extract_account_cooldown(self, exc: Exception) -> tuple[float, str] | None:
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        response = exc.response
        error_detail = _extract_api_error(response)
        if response.status_code == 403 and "ACCESS_INSUFFICIENT_PROCESSING_UNITS" in error_detail:
            return float(getattr(self.settings, "SH_FAILOVER_COOLDOWN_S", 1800)), "insufficient_processing_units"
        if response.status_code == 429:
            header_delay = _parse_retry_after_seconds(response)
            fallback_delay = float(getattr(self.settings, "SH_RATE_LIMIT_COOLDOWN_S", 60))
            return max(header_delay or 0.0, fallback_delay), "rate_limit_exceeded"
        return None

    def _iter_available_accounts(self) -> list[tuple[int, dict[str, str]]]:
        active_accounts: list[tuple[int, dict[str, str]]] = []
        cooled_accounts: list[tuple[int, dict[str, str], float]] = []
        for idx, account in enumerate(self._accounts):
            alias = str(account["alias"])
            cooldown_remaining_s = self._account_cooldown_remaining_s(alias)
            if cooldown_remaining_s > 0:
                cooled_accounts.append((idx, account, cooldown_remaining_s))
            else:
                active_accounts.append((idx, account))

        if active_accounts:
            for _, account, cooldown_remaining_s in cooled_accounts:
                logger.info(
                    "sentinel_account_cooldown_skip",
                    account_alias=str(account["alias"]),
                    cooldown_remaining_s=round(cooldown_remaining_s, 2),
                )
            return active_accounts
        return [(idx, account) for idx, account in enumerate(self._accounts)]

    async def process_request(
        self,
        body: dict[str, Any],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> tuple[httpx.Response, str, int]:
        if client is None:
            client = await self.get_shared_client()
        if not self._accounts:
            raise RuntimeError("Sentinel Hub credentials are not configured")

        last_exc: Exception | None = None
        for failover_level, account in self._iter_available_accounts():
            alias = str(account["alias"])
            try:
                response = await self._process_request_with_account(
                    client=client,
                    body=body,
                    timeout=timeout,
                    account=account,
                )
                self._last_account_alias = alias
                self._last_failover_level = failover_level
                return response, alias, failover_level
            except Exception as exc:
                last_exc = exc
                cooldown = self._extract_account_cooldown(exc)
                if cooldown is not None:
                    self._mark_account_cooldown(alias, cooldown[0], reason=cooldown[1])
                can_failover = failover_level < len(self._accounts) - 1 and self._should_failover_on_exception(exc)
                logger.warning(
                    "sentinel_account_failed",
                    account_alias=alias,
                    failover_level=failover_level,
                    will_failover=can_failover,
                    error=str(exc),
                )
                if not can_failover:
                    raise
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Sentinel Hub request failed before any account was attempted")

    async def _process_request_with_account(
        self,
        *,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        timeout: float,
        account: dict[str, str],
    ) -> httpx.Response:
        alias = str(account["alias"])
        token = await self._get_token(client, account=account)
        resp: httpx.Response | None = None
        latency = 0.0

        max_retries = max(0, int(self.settings.SH_MAX_RETRIES))
        retry_budget = max(0, int(getattr(self.settings, "SH_RETRY_BUDGET", max_retries)))

        def _consume_retry_budget(reason: str) -> None:
            nonlocal retry_budget
            if retry_budget <= 0:
                raise RuntimeError(
                    f"failed_external_dependency: sentinel retry budget exhausted ({reason}, account={alias})"
                )
            retry_budget -= 1

        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                resp = await client.post(
                    f"{self.base_url}/api/v1/process",
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=timeout,
                )
                latency = time.time() - t0
            except httpx.TransportError as exc:
                latency = time.time() - t0
                is_last_attempt = attempt >= max_retries
                logger.warning(
                    "sentinel_transport_error",
                    account_alias=alias,
                    error=str(exc),
                    latency_s=round(latency, 2),
                    attempt=attempt + 1,
                    will_retry=not is_last_attempt,
                )
                if is_last_attempt:
                    raise

                _consume_retry_budget("transport_error")
                delay_s = _compute_backoff_delay(
                    attempt,
                    base_delay_s=float(self.settings.SH_RETRY_BASE_DELAY_S),
                    max_delay_s=float(self.settings.SH_RETRY_MAX_DELAY_S),
                )
                logger.warning(
                    "sentinel_transport_retry",
                    account_alias=alias,
                    attempt=attempt + 1,
                    retry_in_s=round(delay_s, 2),
                )
                await asyncio.sleep(delay_s)
                continue

            logger.info(
                "sentinel_response",
                account_alias=alias,
                status=resp.status_code,
                size=len(resp.content),
                latency_s=round(latency, 2),
                attempt=attempt + 1,
            )

            if not resp.is_error:
                return resp

            error_detail = _extract_api_error(resp)
            logger.error(
                "sentinel_response_error",
                account_alias=alias,
                status=resp.status_code,
                detail=error_detail,
                attempt=attempt + 1,
            )

            should_retry = resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries
            if not should_retry:
                raise httpx.HTTPStatusError(
                    f"Sentinel Process API {resp.status_code}: {error_detail}",
                    request=resp.request,
                    response=resp,
                )

            _consume_retry_budget(f"http_status_{resp.status_code}")
            if resp.status_code == 401:
                self._invalidate_token(alias)
                token = await self._get_token(client, account=account)

            delay_s = _compute_retry_delay(
                resp,
                attempt,
                base_delay_s=float(self.settings.SH_RETRY_BASE_DELAY_S),
                max_delay_s=float(self.settings.SH_RETRY_MAX_DELAY_S),
            )
            logger.warning(
                "sentinel_response_retry",
                account_alias=alias,
                status=resp.status_code,
                attempt=attempt + 1,
                retry_in_s=round(delay_s, 2),
            )
            await asyncio.sleep(delay_s)

        raise RuntimeError(f"Sentinel Hub request exhausted retries for account={alias}")

    async def _fetch_tile_with_client(
        self,
        *,
        client: httpx.AsyncClient,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        max_cloud_pct: int = 40,
    ) -> dict[str, np.ndarray]:
        """Fetch Sentinel-2 tile using an already opened AsyncClient."""
        logger.info(
            "sentinel_fetch",
            bbox=bbox,
            time_from=time_from,
            time_to=time_to,
            width=width,
            height=height,
        )

        body = self._build_request_body(
            bbox, time_from, time_to, width, height,
            max_cloud_pct=max_cloud_pct,
        )

        resp, account_alias, failover_level = await self.process_request(body, client=client, timeout=120.0)
        self._last_account_alias = account_alias
        self._last_failover_level = failover_level

        import io
        import rasterio

        bands_data = {}
        with rasterio.open(io.BytesIO(resp.content)) as src:
            for i, name in enumerate(["B2", "B3", "B4", "B8", "B11", "B12"], 1):
                bands_data[name] = src.read(i).astype(np.float32)
            if src.count >= 7:
                bands_data["SCL"] = src.read(7).astype(np.uint8)

        return bands_data

    async def _fetch_tile_harmonized_with_client(
        self,
        *,
        client: httpx.AsyncClient,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        max_cloud_pct: int = 40,
    ) -> dict[str, np.ndarray]:
        """Fetch Sentinel-2 tile using the raw harmonized band contract."""
        logger.info(
            "sentinel_fetch_harmonized",
            bbox=bbox,
            time_from=time_from,
            time_to=time_to,
            width=width,
            height=height,
        )

        body = self._build_request_body(
            bbox,
            time_from,
            time_to,
            width,
            height,
            evalscript=BANDS_HARMONIZED_AND_SCL,
            max_cloud_pct=max_cloud_pct,
        )

        resp, account_alias, failover_level = await self.process_request(body, client=client, timeout=120.0)
        self._last_account_alias = account_alias
        self._last_failover_level = failover_level

        import io
        import rasterio

        bands_data: dict[str, np.ndarray] = {}
        with rasterio.open(io.BytesIO(resp.content)) as src:
            for band_idx, name in enumerate(_HARMONIZED_RAW_KEYS, start=1):
                if band_idx > src.count:
                    break
                if name == "SCL":
                    bands_data[name] = src.read(band_idx).astype(np.uint8)
                else:
                    bands_data[name] = src.read(band_idx).astype(np.float32)

        return self._normalize_harmonized_payload(bands_data, width=width, height=height)

    @staticmethod
    def _compute_harmonized_indices(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute derived indices from the fixed harmonized raw-band contract."""
        def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = (a - b) / (a + b)
            return np.where(np.isfinite(result), result, np.nan).astype(np.float32, copy=False)

        def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = a / b
            return np.where(np.isfinite(result), result, np.nan).astype(np.float32, copy=False)

        b2 = payload["B2"]
        b3 = payload["B3"]
        b4 = payload["B4"]
        b5 = payload["B5"]
        b7 = payload["B7"]
        b8 = payload["B8"]
        b8a = payload["B8A"]
        b11 = payload["B11"]

        return {
            "NDVI_idx": _norm_diff(b8, b4),
            "NDWI_idx": _norm_diff(b3, b8),
            "NDMI_idx": _norm_diff(b8, b11),
            "BSI_idx": np.where(
                np.isfinite(((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2))),
                (((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2))).astype(np.float32, copy=False),
                np.nan,
            ),
            "MSI_idx": _safe_ratio(b11, b8),
            "NDRE_idx": _norm_diff(b8a, b5),
            "CIre_idx": (_safe_ratio(b8a, b5) - 1.0).astype(np.float32, copy=False),
            "RE_SLOPE_idx": _norm_diff(b7, b5),
        }

    def _normalize_harmonized_payload(
        self,
        payload: dict[str, np.ndarray],
        *,
        width: int,
        height: int,
    ) -> dict[str, np.ndarray]:
        """Normalize any tile payload to the fixed harmonized schema."""
        normalized: dict[str, np.ndarray] = {}
        for key in _HARMONIZED_RAW_KEYS:
            value = payload.get(key)
            if value is None:
                dtype = np.uint8 if key == "SCL" else np.float32
                fill = 0 if key == "SCL" else np.nan
                normalized[key] = np.full((height, width), fill, dtype=dtype)
            else:
                normalized[key] = value.astype(np.uint8 if key == "SCL" else np.float32, copy=False)

        derived = self._compute_harmonized_indices(normalized)
        for key in _HARMONIZED_INDEX_KEYS:
            if key in payload:
                normalized[key] = payload[key].astype(np.float32, copy=False)
            else:
                normalized[key] = derived[key]
        return normalized

    def _empty_harmonized_window(self, *, width: int, height: int) -> dict[str, np.ndarray]:
        """Return an empty fixed-schema window for failed fetches."""
        payload: dict[str, np.ndarray] = {}
        for key in _HARMONIZED_RESULT_KEYS:
            dtype = np.uint8 if key == "SCL" else np.float32
            fill = 0 if key == "SCL" else np.nan
            payload[key] = np.full((height, width), fill, dtype=dtype)
        return payload

    async def _fetch_tile_v4_with_client(
        self,
        *,
        client: httpx.AsyncClient,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        max_cloud_pct: int = 40,
    ) -> dict[str, np.ndarray]:
        """Fetch Sentinel-2 tile with V4 red-edge evalscript."""
        if BANDS_V4_REDEDGE is None:
            raise RuntimeError("V4 red-edge evalscript not available")

        logger.info(
            "sentinel_fetch_v4",
            bbox=bbox,
            time_from=time_from,
            time_to=time_to,
            width=width,
            height=height,
        )

        body = self._build_request_body(
            bbox, time_from, time_to, width, height,
            evalscript=BANDS_V4_REDEDGE,
            max_cloud_pct=max_cloud_pct,
        )

        resp, account_alias, failover_level = await self.process_request(body, client=client, timeout=120.0)
        self._last_account_alias = account_alias
        self._last_failover_level = failover_level

        import io
        import rasterio

        bands_data: dict[str, np.ndarray] = {}
        with rasterio.open(io.BytesIO(resp.content)) as src:
            n_bands = src.count
            for band_idx in range(min(n_bands, len(_V4_BAND_NAMES))):
                name = _V4_BAND_NAMES[band_idx]
                if name == "SCL":
                    bands_data[name] = src.read(band_idx + 1).astype(np.uint8)
                else:
                    bands_data[name] = src.read(band_idx + 1).astype(np.float32)

        return self._normalize_harmonized_payload(bands_data, width=width, height=height)

    async def fetch_tile(
        self,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        max_cloud_pct: int = 40,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, np.ndarray]:
        """Fetch Sentinel-2 bands and SCL for a tile.

        Returns dict with keys: B2, B3, B4, B8, B11, B12, SCL
        Each value is shape (H, W).
        """
        if client is None:
            client = await self.get_shared_client()
        evalscript_hash = hashlib.sha1(BANDS_AND_SCL.encode("utf-8")).hexdigest()
        cache_key = self._cache_key(
            bbox,
            time_from,
            time_to,
            width,
            height,
            evalscript_hash,
            max_cloud_pct,
        )
        cached = self._load_cached_scene(cache_key)
        if cached is not None:
            CACHE_HITS.inc()
            return cached

        CACHE_MISSES.inc()
        try:
            payload = await self._fetch_tile_with_client(
                client=client,
                bbox=bbox,
                time_from=time_from,
                time_to=time_to,
                width=width,
                height=height,
                max_cloud_pct=max_cloud_pct,
            )
        except Exception as exc:
            stale = self._load_cached_scene(cache_key, allow_stale=True)
            if stale is not None:
                logger.warning(
                    "sentinel_scene_cache_stale_fallback",
                    cache_key=cache_key[:12],
                    error=str(exc),
                    provider_state="degraded_external_dependency",
                )
                return stale
            raise
        self._store_cached_scene(cache_key, payload)
        return payload

    async def fetch_tile_harmonized(
        self,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        max_cloud_pct: int = 40,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, np.ndarray]:
        """Fetch harmonized raw-band contract with computed indices."""
        if client is None:
            client = await self.get_shared_client()
        evalscript_hash = hashlib.sha1(BANDS_HARMONIZED_AND_SCL.encode("utf-8")).hexdigest()
        cache_key = self._cache_key(
            bbox,
            time_from,
            time_to,
            width,
            height,
            evalscript_hash,
            max_cloud_pct,
        )
        cached = self._load_cached_scene(cache_key)
        if cached is not None:
            CACHE_HITS.inc()
            return self._normalize_harmonized_payload(cached, width=width, height=height)

        CACHE_MISSES.inc()
        try:
            payload = await self._fetch_tile_harmonized_with_client(
                client=client,
                bbox=bbox,
                time_from=time_from,
                time_to=time_to,
                width=width,
                height=height,
                max_cloud_pct=max_cloud_pct,
            )
        except Exception as exc:
            stale = self._load_cached_scene(cache_key, allow_stale=True)
            if stale is not None:
                logger.warning(
                    "sentinel_harmonized_cache_stale_fallback",
                    cache_key=cache_key[:12],
                    error=str(exc),
                    provider_state="degraded_external_dependency",
                )
                return self._normalize_harmonized_payload(stale, width=width, height=height)
            raise
        self._store_cached_scene(cache_key, payload)
        return payload

    async def fetch_tile_v4(
        self,
        bbox: tuple[float, float, float, float],
        time_from: str,
        time_to: str,
        width: int,
        height: int,
        max_cloud_pct: int = 40,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, np.ndarray]:
        """Fetch V4 tile with fixed harmonized schema and safe fallback."""
        if BANDS_V4_REDEDGE is None:
            return await self.fetch_tile_harmonized(
                bbox,
                time_from,
                time_to,
                width,
                height,
                max_cloud_pct,
                client=client,
            )

        if client is None:
            client = await self.get_shared_client()
        evalscript_hash = hashlib.sha1(BANDS_V4_REDEDGE.encode("utf-8")).hexdigest()
        cache_key = self._cache_key(bbox, time_from, time_to, width, height, evalscript_hash, max_cloud_pct)
        cached = self._load_cached_scene(cache_key)
        if cached is not None:
            CACHE_HITS.inc()
            return cached

        CACHE_MISSES.inc()
        try:
            payload = await self._fetch_tile_v4_with_client(
                client=client, bbox=bbox, time_from=time_from, time_to=time_to,
                width=width, height=height, max_cloud_pct=max_cloud_pct,
            )
        except Exception as exc:
            logger.warning("sentinel_v4_fallback_to_harmonized", error=str(exc))
            return await self.fetch_tile_harmonized(
                bbox,
                time_from,
                time_to,
                width,
                height,
                max_cloud_pct,
                client=client,
            )
        self._store_cached_scene(cache_key, payload)
        return payload

    async def fetch_multitemporal(
        self,
        bbox: tuple[float, float, float, float],
        dates: list[tuple[str, str]],
        width: int,
        height: int,
        max_cloud_pct: int = 40,
        *,
        client: httpx.AsyncClient | None = None,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Fetch multi-temporal stack.

        Returns dict with keys: B2, B3, B4, B8, B11, B12, SCL
        Each value is shape (T, H, W).
        """
        if not dates:
            raise ValueError("dates must not be empty")
        stacks = {k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]}
        concurrency = max(1, int(getattr(self.settings, "SENTINEL_CONCURRENT_REQUESTS", 4)))
        semaphore = asyncio.Semaphore(concurrency)
        active_client = client if client is not None else await self.get_shared_client()
        total_windows = len(dates)
        completed_windows = 0
        active_window_hint = {
            "time_from": dates[0][0],
            "time_to": dates[0][1],
        }
        keepalive_stop = asyncio.Event()
        keepalive_interval_s = max(
            0.1,
            float(getattr(self.settings, "SENTINEL_FETCH_KEEPALIVE_S", 15.0)),
        )

        def _notify_progress(completed: int, time_from: str, time_to: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(int(completed), int(total_windows), time_from, time_to)
            except Exception as exc:
                logger.debug(
                    "sentinel_fetch_progress_callback_failed",
                    completed=completed,
                    total=total_windows,
                    time_from=time_from,
                    time_to=time_to,
                    error=str(exc),
                )

        async def _emit_keepalive() -> None:
            if progress_callback is None:
                return
            while True:
                try:
                    await asyncio.wait_for(keepalive_stop.wait(), timeout=keepalive_interval_s)
                    return
                except asyncio.TimeoutError:
                    if completed_windows >= total_windows:
                        return
                    _notify_progress(
                        completed_windows,
                        str(active_window_hint["time_from"]),
                        str(active_window_hint["time_to"]),
                    )

        async def _fetch_window(time_from: str, time_to: str) -> dict[str, np.ndarray]:
            nonlocal completed_windows
            async with semaphore:
                active_window_hint["time_from"] = time_from
                active_window_hint["time_to"] = time_to
                try:
                    return await self.fetch_tile(
                        bbox,
                        time_from,
                        time_to,
                        width,
                        height,
                        max_cloud_pct=max_cloud_pct,
                        client=active_client,
                    )
                finally:
                    completed_windows += 1
                    _notify_progress(completed_windows, time_from, time_to)

        tasks = [_fetch_window(time_from, time_to) for time_from, time_to in dates]
        keepalive_task = asyncio.create_task(_emit_keepalive())
        try:
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            keepalive_stop.set()
            await keepalive_task

        for (time_from, time_to), tile_data in zip(dates, fetched):
            if isinstance(tile_data, Exception):
                logger.warning(
                    "sentinel_window_failed",
                    time_from=time_from,
                    time_to=time_to,
                    error=str(tile_data),
                )
                tile_data = {
                    "B2": np.full((height, width), np.nan, dtype=np.float32),
                    "B3": np.full((height, width), np.nan, dtype=np.float32),
                    "B4": np.full((height, width), np.nan, dtype=np.float32),
                    "B8": np.full((height, width), np.nan, dtype=np.float32),
                    "B11": np.full((height, width), np.nan, dtype=np.float32),
                    "B12": np.full((height, width), np.nan, dtype=np.float32),
                    "SCL": np.zeros((height, width), dtype=np.uint8),
                }
            for key in stacks:
                if key in tile_data:
                    stacks[key].append(tile_data[key])

        result = {}
        for k, arrays in stacks.items():
            if arrays:
                result[k] = np.stack(arrays, axis=0)

        return result

    async def fetch_multitemporal_v4(
        self,
        bbox: tuple[float, float, float, float],
        dates: list[tuple[str, str]],
        width: int,
        height: int,
        max_cloud_pct: int = 40,
        *,
        client: httpx.AsyncClient | None = None,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Backward-compatible alias for the fixed-schema harmonized fetch."""
        return await self.fetch_multitemporal_harmonized(
            bbox,
            dates,
            width,
            height,
            max_cloud_pct=max_cloud_pct,
            client=client,
            progress_callback=progress_callback,
        )

    async def fetch_multitemporal_harmonized(
        self,
        bbox: tuple[float, float, float, float],
        dates: list[tuple[str, str]],
        width: int,
        height: int,
        max_cloud_pct: int = 40,
        *,
        client: httpx.AsyncClient | None = None,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
        prefer_v4: bool = True,
    ) -> dict[str, np.ndarray]:
        """Fetch a fixed-schema multi-temporal stack for autodetect runtime."""
        if not dates:
            raise ValueError("dates must not be empty")

        stacks: dict[str, list[np.ndarray]] = {k: [] for k in _HARMONIZED_RESULT_KEYS}
        concurrency = max(1, int(getattr(self.settings, "SENTINEL_CONCURRENT_REQUESTS", 4)))
        semaphore = asyncio.Semaphore(concurrency)
        active_client = client if client is not None else await self.get_shared_client()
        total_windows = len(dates)
        completed_windows = 0
        active_window_hint = {
            "time_from": dates[0][0],
            "time_to": dates[0][1],
        }
        keepalive_stop = asyncio.Event()
        keepalive_interval_s = max(
            0.1,
            float(getattr(self.settings, "SENTINEL_FETCH_KEEPALIVE_S", 15.0)),
        )

        def _notify_progress(completed: int, time_from: str, time_to: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(int(completed), int(total_windows), time_from, time_to)
            except Exception as exc:
                logger.debug(
                    "sentinel_harmonized_progress_callback_failed",
                    completed=completed,
                    total=total_windows,
                    time_from=time_from,
                    time_to=time_to,
                    error=str(exc),
                )

        async def _emit_keepalive() -> None:
            if progress_callback is None:
                return
            while True:
                try:
                    await asyncio.wait_for(keepalive_stop.wait(), timeout=keepalive_interval_s)
                    return
                except asyncio.TimeoutError:
                    if completed_windows >= total_windows:
                        return
                    _notify_progress(
                        completed_windows,
                        str(active_window_hint["time_from"]),
                        str(active_window_hint["time_to"]),
                    )

        async def _fetch_window(time_from: str, time_to: str) -> dict[str, np.ndarray]:
            nonlocal completed_windows
            async with semaphore:
                active_window_hint["time_from"] = time_from
                active_window_hint["time_to"] = time_to
                try:
                    if prefer_v4:
                        return await self.fetch_tile_v4(
                            bbox,
                            time_from,
                            time_to,
                            width,
                            height,
                            max_cloud_pct=max_cloud_pct,
                            client=active_client,
                        )
                    return await self.fetch_tile_harmonized(
                        bbox,
                        time_from,
                        time_to,
                        width,
                        height,
                        max_cloud_pct=max_cloud_pct,
                        client=active_client,
                    )
                finally:
                    completed_windows += 1
                    _notify_progress(completed_windows, time_from, time_to)

        tasks = [_fetch_window(time_from, time_to) for time_from, time_to in dates]
        keepalive_task = asyncio.create_task(_emit_keepalive())
        try:
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            keepalive_stop.set()
            await keepalive_task
        empty_window = self._empty_harmonized_window(width=width, height=height)

        for (time_from, time_to), tile_data in zip(dates, fetched):
            if isinstance(tile_data, Exception):
                logger.warning(
                    "sentinel_harmonized_window_failed",
                    time_from=time_from,
                    time_to=time_to,
                    error=str(tile_data),
                )
                tile_data = empty_window
            else:
                tile_data = self._normalize_harmonized_payload(tile_data, width=width, height=height)
            for key in _HARMONIZED_RESULT_KEYS:
                stacks[key].append(tile_data[key])

        return {key: np.stack(arrays, axis=0) for key, arrays in stacks.items() if arrays}
