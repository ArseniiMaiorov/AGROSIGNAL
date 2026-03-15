"""Tests for v3 Sentinel-1 client and preprocessing utilities."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
from rasterio.io import MemoryFile

from processing.fields.s1_preprocess import preprocess_s1
from providers.s1_client import SentinelHubS1Client


def _make_tiff_bytes() -> bytes:
    data = np.zeros((2, 4, 4), dtype=np.float32)
    data[0, :, :] = 2.0
    data[1, :, :] = 1.0
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 2,
        "dtype": "float32",
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
        return memfile.read()


def test_s1_client_build_request_body_contains_expected_payload():
    client = SentinelHubS1Client()

    body = client.build_request_body(
        bbox=(10.0, 20.0, 11.0, 21.0),
        time_from="2025-05-01T00:00:00Z",
        time_to="2025-05-31T23:59:59Z",
        width=256,
        height=256,
    )

    assert body["input"]["data"][0]["type"] == "sentinel-1-grd"
    assert body["output"]["width"] == 256
    assert "VV" in body["evalscript"]
    assert "VH" in body["evalscript"]


def test_s1_client_fetch_tile_reads_vv_vh(monkeypatch):
    client = SentinelHubS1Client()

    tiff_bytes = _make_tiff_bytes()

    class _StubResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.content = tiff_bytes
            self.is_error = False
            self.request = httpx.Request("POST", "https://example.test")

    async def _fake_process_request(_body, *, client=None, timeout=120.0):
        del client, timeout
        return _StubResponse(), "primary", 0

    client._s2_client.process_request = _fake_process_request

    result = asyncio.run(
        client.fetch_tile(
            bbox=(10.0, 20.0, 11.0, 21.0),
            time_from="2025-05-01T00:00:00Z",
            time_to="2025-05-31T23:59:59Z",
            width=4,
            height=4,
        )
    )

    assert result["VV"].shape == (4, 4)
    assert result["VH"].shape == (4, 4)
    assert np.allclose(result["VV"], 2.0)
    assert np.allclose(result["VH"], 1.0)


def test_preprocess_s1_returns_expected_feature_layers():
    vv = np.full((8, 8), 2.0, dtype=np.float32)
    vh = np.full((8, 8), 1.0, dtype=np.float32)

    result = preprocess_s1(vv, vh, cfg=SimpleNamespace(S1_LEE_FILTER_ENABLE=True, S1_LEE_WINDOW_SIZE=5))

    assert result["VV_edge"].shape == (8, 8)
    assert result["VHVV_ratio"].shape == (8, 8)
    assert result["VV_edge"].dtype == np.float32
    assert result["VHVV_ratio"].dtype == np.float32
