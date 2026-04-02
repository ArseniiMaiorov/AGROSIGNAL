"""Tests for Sentinel Hub client retry utilities."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np
import pytest

from providers.sentinelhub.client import (
    SentinelHubClient,
    _HARMONIZED_RESULT_KEYS,
    _compute_backoff_delay,
    _compute_retry_delay,
    _parse_retry_after_seconds,
    _token_cache,
)
from providers.sentinelhub.evalscripts import BANDS_V4_REDEDGE


def _make_response(status_code: int = 429, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://services.sentinel-hub.com/api/v1/process")
    return httpx.Response(status_code=status_code, headers=headers or {}, request=request)


def _make_error_response(status_code: int, code: str, reason: str, message: str) -> httpx.Response:
    request = httpx.Request("POST", "https://services.sentinel-hub.com/api/v1/process")
    return httpx.Response(
        status_code,
        json={"error": {"code": code, "reason": reason, "message": message}},
        request=request,
    )


class _FakeAsyncClient:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)

    async def post(self, url: str, **_kwargs) -> httpx.Response:
        if not self._responses:
            raise AssertionError(f"Unexpected POST without prepared response: {url}")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_parse_retry_after_seconds():
    resp = _make_response(headers={"Retry-After": "7"})
    assert _parse_retry_after_seconds(resp) == pytest.approx(7.0)


def test_compute_retry_delay_uses_header():
    resp = _make_response(headers={"Retry-After": "11"})
    delay = _compute_retry_delay(resp, attempt=0, base_delay_s=2.0, max_delay_s=30.0)
    assert delay == pytest.approx(11.0)


def test_compute_retry_delay_falls_back_to_exponential_backoff():
    resp = _make_response(headers={})
    delay = _compute_retry_delay(resp, attempt=2, base_delay_s=2.0, max_delay_s=30.0)
    # base=8.0 with jitter in [0.5, 1.0) range
    assert 4.0 <= delay <= 8.0


def test_compute_retry_delay_is_capped():
    resp = _make_response(headers={"Retry-After": "120"})
    delay = _compute_retry_delay(resp, attempt=5, base_delay_s=2.0, max_delay_s=30.0)
    assert delay == pytest.approx(30.0)


def test_compute_backoff_delay():
    delay = _compute_backoff_delay(attempt=3, base_delay_s=2.0, max_delay_s=30.0)
    # base=16.0 with jitter in [0.5, 1.0) range
    assert 8.0 <= delay <= 16.0


def test_fetch_multitemporal_tolerates_partial_failure(monkeypatch):
    client = SentinelHubClient()

    async def _fake_fetch_tile(
        bbox,
        time_from,
        time_to,
        width,
        height,
        max_cloud_pct=40,
        *,
        client=None,
    ):
        if time_from.startswith("2025-05-08"):
            raise RuntimeError("temporary error")
        base = np.ones((height, width), dtype=np.float32)
        return {
            "B2": base.copy(),
            "B3": base.copy(),
            "B4": base.copy(),
            "B8": base.copy(),
            "B11": base.copy(),
            "B12": base.copy(),
            "SCL": np.full((height, width), 4, dtype=np.uint8),
        }

    monkeypatch.setattr(client, "fetch_tile", _fake_fetch_tile)

    result = asyncio.run(
        client.fetch_multitemporal(
            (29.0, 58.0, 30.0, 59.0),
            [
                ("2025-05-01T00:00:00Z", "2025-05-07T23:59:59Z"),
                ("2025-05-08T00:00:00Z", "2025-05-14T23:59:59Z"),
            ],
            width=4,
            height=3,
            client=object(),
        )
    )

    assert result["B2"].shape == (2, 3, 4)
    assert np.isnan(result["B2"][1]).all()
    assert np.count_nonzero(result["SCL"][1]) == 0


def test_fetch_tile_uses_persistent_cache(monkeypatch, tmp_path: Path):
    client = SentinelHubClient()
    client.scene_cache_dir = tmp_path
    client.settings = client.settings.model_copy(update={"SCENE_CACHE_TTL_DAYS": 30})

    calls = {"count": 0}

    async def _fake_fetch_tile_with_client(**_kwargs):
        calls["count"] += 1
        arr = np.ones((3, 4), dtype=np.float32)
        return {
            "B2": arr.copy(),
            "B3": arr.copy(),
            "B4": arr.copy(),
            "B8": arr.copy(),
            "B11": arr.copy(),
            "B12": arr.copy(),
            "SCL": np.full((3, 4), 4, dtype=np.uint8),
        }

    monkeypatch.setattr(client, "_fetch_tile_with_client", _fake_fetch_tile_with_client)

    first = asyncio.run(
        client.fetch_tile(
            (29.0, 58.0, 30.0, 59.0),
            "2025-05-01T00:00:00Z",
            "2025-05-07T23:59:59Z",
            width=4,
            height=3,
            client=object(),
        )
    )
    second = asyncio.run(
        client.fetch_tile(
            (29.0, 58.0, 30.0, 59.0),
            "2025-05-01T00:00:00Z",
            "2025-05-07T23:59:59Z",
            width=4,
            height=3,
            client=object(),
        )
    )

    assert calls["count"] == 1
    assert np.array_equal(first["B2"], second["B2"])


def test_fetch_tile_uses_stale_cache_when_provider_fails(monkeypatch, tmp_path: Path):
    client = SentinelHubClient()
    client.scene_cache_dir = tmp_path
    client.settings = client.settings.model_copy(update={"SCENE_CACHE_TTL_DAYS": 0})

    async def _seed_fetch(**_kwargs):
        arr = np.ones((3, 4), dtype=np.float32)
        return {
            "B2": arr.copy(),
            "B3": arr.copy(),
            "B4": arr.copy(),
            "B8": arr.copy(),
            "B11": arr.copy(),
            "B12": arr.copy(),
            "SCL": np.full((3, 4), 4, dtype=np.uint8),
        }

    monkeypatch.setattr(client, "_fetch_tile_with_client", _seed_fetch)
    seeded = asyncio.run(
        client.fetch_tile(
            (29.0, 58.0, 30.0, 59.0),
            "2025-05-01T00:00:00Z",
            "2025-05-07T23:59:59Z",
            width=4,
            height=3,
            client=object(),
        )
    )
    assert np.all(seeded["B2"] == 1.0)

    async def _failing_fetch(**_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(client, "_fetch_tile_with_client", _failing_fetch)
    cached = asyncio.run(
        client.fetch_tile(
            (29.0, 58.0, 30.0, 59.0),
            "2025-05-01T00:00:00Z",
            "2025-05-07T23:59:59Z",
            width=4,
            height=3,
            client=object(),
        )
    )

    assert np.array_equal(cached["B2"], seeded["B2"])


def test_normalize_harmonized_payload_returns_fixed_schema():
    client = SentinelHubClient()
    payload = {
        "B2": np.ones((3, 4), dtype=np.float32),
        "B3": np.full((3, 4), 2.0, dtype=np.float32),
        "B4": np.full((3, 4), 3.0, dtype=np.float32),
        "B5": np.full((3, 4), 4.0, dtype=np.float32),
        "B6": np.full((3, 4), 5.0, dtype=np.float32),
        "B7": np.full((3, 4), 6.0, dtype=np.float32),
        "B8": np.full((3, 4), 7.0, dtype=np.float32),
        "B8A": np.full((3, 4), 8.0, dtype=np.float32),
        "B11": np.full((3, 4), 9.0, dtype=np.float32),
        "B12": np.full((3, 4), 10.0, dtype=np.float32),
        "SCL": np.full((3, 4), 4, dtype=np.uint8),
    }

    normalized = client._normalize_harmonized_payload(payload, width=4, height=3)

    assert set(normalized.keys()) == set(_HARMONIZED_RESULT_KEYS)
    assert normalized["SCL"].dtype == np.uint8
    assert normalized["NDVI_idx"].shape == (3, 4)
    assert np.isfinite(normalized["NDMI_idx"]).all()


def test_fetch_tile_v4_fallback_returns_harmonized_contract(monkeypatch):
    client = SentinelHubClient()

    async def _failing_fetch_tile_v4_with_client(**_kwargs):
        raise RuntimeError("v4 unavailable")

    async def _fake_fetch_tile_harmonized(
        bbox,
        time_from,
        time_to,
        width,
        height,
        max_cloud_pct=40,
        *,
        client=None,
    ):
        arr = np.ones((height, width), dtype=np.float32)
        return client_ref._normalize_harmonized_payload(
            {
                "B2": arr.copy(),
                "B3": arr.copy(),
                "B4": arr.copy(),
                "B5": arr.copy(),
                "B6": arr.copy(),
                "B7": arr.copy(),
                "B8": arr.copy(),
                "B8A": arr.copy(),
                "B11": arr.copy(),
                "B12": arr.copy(),
                "SCL": np.full((height, width), 4, dtype=np.uint8),
            },
            width=width,
            height=height,
        )

    client_ref = client
    monkeypatch.setattr(client, "_fetch_tile_v4_with_client", _failing_fetch_tile_v4_with_client)
    monkeypatch.setattr(client, "fetch_tile_harmonized", _fake_fetch_tile_harmonized)

    payload = asyncio.run(
        client.fetch_tile_v4(
            (29.0, 58.0, 30.0, 59.0),
            "2025-05-01T00:00:00Z",
            "2025-05-07T23:59:59Z",
            width=4,
            height=3,
            client=object(),
        )
    )

    assert set(payload.keys()) == set(_HARMONIZED_RESULT_KEYS)
    assert payload["B11"].shape == (3, 4)
    assert payload["B12"].shape == (3, 4)


def test_fetch_multitemporal_harmonized_prefer_v4_keeps_fixed_schema(monkeypatch):
    client = SentinelHubClient()

    async def _fake_fetch_tile_v4(
        bbox,
        time_from,
        time_to,
        width,
        height,
        max_cloud_pct=40,
        *,
        client=None,
    ):
        arr = np.ones((height, width), dtype=np.float32)
        return {
            "B2": arr.copy(),
            "B3": arr.copy(),
            "B4": arr.copy(),
            "B5": arr.copy(),
            "B6": arr.copy(),
            "B7": arr.copy(),
            "B8": arr.copy(),
            "B8A": arr.copy(),
            "B11": arr.copy(),
            "B12": arr.copy(),
            "SCL": np.full((height, width), 4, dtype=np.uint8),
        }

    monkeypatch.setattr(client, "fetch_tile_v4", _fake_fetch_tile_v4)

    result = asyncio.run(
        client.fetch_multitemporal_harmonized(
            (29.0, 58.0, 30.0, 59.0),
            [
                ("2025-05-01T00:00:00Z", "2025-05-07T23:59:59Z"),
                ("2025-05-08T00:00:00Z", "2025-05-14T23:59:59Z"),
            ],
            width=4,
            height=3,
            client=object(),
            prefer_v4=True,
        )
    )

    assert set(result.keys()) == set(_HARMONIZED_RESULT_KEYS)
    assert result["B11"].shape == (2, 3, 4)
    assert result["B12"].shape == (2, 3, 4)
    assert result["NDMI_idx"].shape == (2, 3, 4)


def test_fetch_multitemporal_harmonized_emits_keepalive_progress(monkeypatch):
    client = SentinelHubClient()
    client.settings = client.settings.model_copy(update={"SENTINEL_FETCH_KEEPALIVE_S": 0.01})
    progress_events: list[tuple[int, int, str, str]] = []
    release_fetch = asyncio.Event()

    async def _slow_fetch_tile_v4(
        bbox,
        time_from,
        time_to,
        width,
        height,
        max_cloud_pct=40,
        *,
        client=None,
    ):
        await release_fetch.wait()
        arr = np.ones((height, width), dtype=np.float32)
        return {
            "B2": arr.copy(),
            "B3": arr.copy(),
            "B4": arr.copy(),
            "B5": arr.copy(),
            "B6": arr.copy(),
            "B7": arr.copy(),
            "B8": arr.copy(),
            "B8A": arr.copy(),
            "B11": arr.copy(),
            "B12": arr.copy(),
            "SCL": np.full((height, width), 4, dtype=np.uint8),
        }

    monkeypatch.setattr(client, "fetch_tile_v4", _slow_fetch_tile_v4)

    async def _run_test() -> None:
        task = asyncio.create_task(
            client.fetch_multitemporal_harmonized(
                (29.0, 58.0, 30.0, 59.0),
                [("2025-05-01T00:00:00Z", "2025-05-07T23:59:59Z")],
                width=4,
                height=3,
                client=object(),
                progress_callback=lambda completed, total, time_from, time_to: progress_events.append(
                    (completed, total, time_from, time_to)
                ),
                prefer_v4=True,
            )
        )
        await asyncio.sleep(0.18)
        assert progress_events
        assert any(event[0] == 0 and event[1] == 1 for event in progress_events)
        release_fetch.set()
        await task

    asyncio.run(_run_test())

    assert progress_events[-1][0] == 1


def test_v4_evalscript_uses_single_default_output_contract():
    assert 'id: "default"' in BANDS_V4_REDEDGE
    assert "bands: 19" in BANDS_V4_REDEDGE
    assert "indices:" not in BANDS_V4_REDEDGE


def test_process_request_fails_over_to_reserve_account():
    _token_cache.clear()
    client = SentinelHubClient()
    client.settings = client.settings.model_copy(
        update={
            "SH_CLIENT_ID": "primary-id",
            "SH_CLIENT_SECRET": "primary-secret",
            "SH_CLIENT_ID_RESERVE": "reserve-id",
            "SH_CLIENT_SECRET_RESERVE": "reserve-secret",
            "SH_FAILOVER_ENABLED": True,
            "SH_MAX_RETRIES": 0,
        }
    )
    client._accounts = client._build_accounts()

    oauth_primary = httpx.Response(
        200,
        json={"access_token": "primary-token", "expires_in": 3600},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/oauth/token"),
    )
    process_primary = _make_response(status_code=429)
    oauth_reserve = httpx.Response(
        200,
        json={"access_token": "reserve-token", "expires_in": 3600},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/oauth/token"),
    )
    process_reserve = httpx.Response(
        200,
        json={"ok": True},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/api/v1/process"),
    )

    fake_client = _FakeAsyncClient([oauth_primary, process_primary, oauth_reserve, process_reserve])
    response, account_alias, failover_level = asyncio.run(
        client.process_request({"input": {}, "output": {}, "evalscript": ""}, client=fake_client)
    )

    assert response.status_code == 200
    assert account_alias == "reserve"
    assert failover_level == 1


def test_process_request_puts_exhausted_account_on_cooldown_and_skips_it():
    _token_cache.clear()
    client = SentinelHubClient()
    client.settings = client.settings.model_copy(
        update={
            "SH_CLIENT_ID": "primary-id",
            "SH_CLIENT_SECRET": "primary-secret",
            "SH_CLIENT_ID_RESERVE": "reserve-id",
            "SH_CLIENT_SECRET_RESERVE": "reserve-secret",
            "SH_FAILOVER_ENABLED": True,
            "SH_MAX_RETRIES": 0,
            "SH_FAILOVER_COOLDOWN_S": 900,
        }
    )
    client._accounts = client._build_accounts()

    oauth_primary = httpx.Response(
        200,
        json={"access_token": "primary-token", "expires_in": 3600},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/oauth/token"),
    )
    process_primary = _make_error_response(
        403,
        "ACCESS_INSUFFICIENT_PROCESSING_UNITS",
        "Forbidden",
        "Insufficient processing units or requests available in your account.",
    )
    oauth_reserve = httpx.Response(
        200,
        json={"access_token": "reserve-token", "expires_in": 3600},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/oauth/token"),
    )
    process_reserve = httpx.Response(
        200,
        json={"ok": True},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/api/v1/process"),
    )

    first_client = _FakeAsyncClient([oauth_primary, process_primary, oauth_reserve, process_reserve])
    response, account_alias, failover_level = asyncio.run(
        client.process_request({"input": {}, "output": {}, "evalscript": ""}, client=first_client)
    )

    assert response.status_code == 200
    assert account_alias == "reserv"
    assert failover_level == 1
    assert client._account_cooldown_remaining_s("primary") > 0

    second_process_reserve = httpx.Response(
        200,
        json={"ok": True},
        request=httpx.Request("POST", "https://services.sentinel-hub.com/api/v1/process"),
    )
    second_client = _FakeAsyncClient([second_process_reserve])
    response, account_alias, failover_level = asyncio.run(
        client.process_request({"input": {}, "output": {}, "evalscript": ""}, client=second_client)
    )

    assert response.status_code == 200
    assert account_alias == "reserv"
    assert failover_level == 1
