from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.weather_service import CurrentWeatherPayload, WeatherService


class _FakeExecuteResult:
    def __init__(self, scalar):
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar


class _FakeDb:
    def __init__(self) -> None:
        self.rows = []
        self._cached = None
        self.execute = AsyncMock(side_effect=self._execute)

    async def _execute(self, _stmt):
        return _FakeExecuteResult(self._cached)

    def add(self, row):
        self.rows.append(row)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_get_current_weather_uses_cache(monkeypatch):
    db = _FakeDb()
    WeatherService._current_cache.clear()
    cached = SimpleNamespace(
        latitude=59.0,
        longitude=30.0,
        observed_at=datetime(2026, 3, 7, tzinfo=timezone.utc),
        provider='openmeteo',
        temperature_c=12.0,
        apparent_temperature_c=11.0,
        precipitation_mm=0.0,
        wind_speed_m_s=3.0,
        humidity_pct=60.0,
        cloud_cover_pct=20.0,
        pressure_hpa=1005.0,
        soil_moisture=0.22,
    )
    db._cached = cached
    service = WeatherService(db)
    monkeypatch.setattr(service, '_fetch_current_weather', AsyncMock())

    payload = await service.get_current_weather(59.0, 30.0)

    assert payload['cached'] is True
    assert payload['provider'] == 'openmeteo'
    service._fetch_current_weather.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_current_weather_saves_fetched_payload(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._current_cache.clear()
    monkeypatch.setattr(
        service,
        '_fetch_current_weather',
        AsyncMock(
            return_value=CurrentWeatherPayload(
                latitude=45.0,
                longitude=39.0,
                observed_at=datetime(2026, 3, 7, tzinfo=timezone.utc),
                provider='openmeteo',
                temperature_c=16.2,
                apparent_temperature_c=15.5,
                precipitation_mm=0.8,
                wind_speed_m_s=4.1,
                humidity_pct=58.0,
                cloud_cover_pct=18.0,
                pressure_hpa=1008.0,
                soil_moisture=0.19,
                payload={'ok': True},
            )
        ),
    )

    payload = await service.get_current_weather(45.0, 39.0)

    assert payload['cached'] is False
    assert payload['temperature_c'] == 16.2
    assert len(db.rows) == 1
    assert db.rows[0].provider == 'openmeteo'


@pytest.mark.asyncio
async def test_get_current_weather_uses_fresh_in_memory_cache(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._current_cache.clear()
    cached_payload = CurrentWeatherPayload(
        latitude=45.0,
        longitude=39.0,
        observed_at=datetime.now(timezone.utc),
        provider="openmeteo",
        temperature_c=14.1,
        apparent_temperature_c=13.6,
        precipitation_mm=0.1,
        wind_speed_m_s=3.2,
        humidity_pct=63.0,
        cloud_cover_pct=25.0,
        pressure_hpa=1007.0,
        soil_moisture=0.21,
        payload={"ok": True},
    )
    WeatherService._current_cache[(45.0, 39.0)] = cached_payload
    monkeypatch.setattr(service, "_fetch_current_weather", AsyncMock())

    payload = await service.get_current_weather(45.0, 39.0)

    assert payload["provider"] == "openmeteo"
    assert payload["cached"] is True
    assert payload["temperature_c"] == pytest.approx(14.1)
    service._fetch_current_weather.assert_not_awaited()


@pytest.mark.asyncio
async def test_forecast_gracefully_degrades_on_provider_error(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._forecast_cache.clear()
    WeatherService._current_cache.clear()
    monkeypatch.setattr(service, '_fetch_openmeteo_forecast', AsyncMock(side_effect=RuntimeError('boom')))

    payload = await service.get_forecast(58.0, 30.0, days=5)

    assert payload['provider'] == 'fallback'
    assert payload['forecast'] == []
    assert payload['error'] == 'Прогноз временно недоступен'


@pytest.mark.asyncio
async def test_current_weather_gracefully_degrades_without_cache(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._current_cache.clear()
    monkeypatch.setattr(service, '_fetch_current_weather', AsyncMock(side_effect=RuntimeError('boom')))

    payload = await service.get_current_weather(58.0, 30.0)

    assert payload['provider'] == 'fallback'
    assert payload['cached'] is False
    assert payload['error'] == 'Текущая погода временно недоступна'


@pytest.mark.asyncio
async def test_forecast_uses_fresh_in_memory_cache(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._forecast_cache.clear()
    WeatherService._current_cache.clear()
    WeatherService._forecast_cache[(58.0, 30.0, 5)] = (
        datetime.now(timezone.utc),
        [{"date": "2026-03-11", "temp_max_c": 12.0}],
    )
    monkeypatch.setattr(service, '_fetch_openmeteo_forecast', AsyncMock())

    payload = await service.get_forecast(58.0, 30.0, days=5)

    assert payload['provider'] == 'openmeteo'
    assert payload['forecast'] == [{"date": "2026-03-11", "temp_max_c": 12.0}]
    service._fetch_openmeteo_forecast.assert_not_awaited()


@pytest.mark.asyncio
async def test_forecast_uses_stale_cache_on_provider_error(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._forecast_cache.clear()
    WeatherService._current_cache.clear()
    stale_time = datetime(2026, 3, 7, tzinfo=timezone.utc)
    WeatherService._forecast_cache[(58.0, 30.0, 5)] = (
        stale_time,
        [{"date": "2026-03-07", "temp_max_c": 10.0}],
    )
    monkeypatch.setattr(service, '_fetch_openmeteo_forecast', AsyncMock(side_effect=RuntimeError('boom')))

    payload = await service.get_forecast(58.0, 30.0, days=5)

    assert payload['provider'] == 'openmeteo'
    assert payload['forecast'] == [{"date": "2026-03-07", "temp_max_c": 10.0}]
    assert payload['freshness']['freshness_state'] in {'stale', 'aging'}


@pytest.mark.asyncio
async def test_openmeteo_forecast_includes_temp_mean_c(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    monkeypatch.setattr(
        service,
        '_fetch_json_with_retries',
        AsyncMock(
            return_value={
                "daily": {
                    "time": ["2026-03-11"],
                    "temperature_2m_max": [12.0],
                    "temperature_2m_min": [4.0],
                    "precipitation_sum": [1.5],
                    "wind_speed_10m_max": [6.0],
                    "cloud_cover_mean": [32.0],
                },
            }
        ),
    )

    payload = await service._fetch_openmeteo_forecast(58.0, 30.0, 1)

    assert payload[0]["temp_mean_c"] == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_current_weather_uses_stale_in_memory_cache_on_provider_error(monkeypatch):
    db = _FakeDb()
    service = WeatherService(db)
    WeatherService._current_cache.clear()
    cached_payload = CurrentWeatherPayload(
        latitude=58.0,
        longitude=30.0,
        observed_at=datetime(2026, 3, 7, tzinfo=timezone.utc),
        provider="openmeteo",
        temperature_c=9.5,
        apparent_temperature_c=8.7,
        precipitation_mm=0.2,
        wind_speed_m_s=4.3,
        humidity_pct=71.0,
        cloud_cover_pct=42.0,
        pressure_hpa=1003.0,
        soil_moisture=0.25,
        payload={"ok": True},
    )
    WeatherService._current_cache[(58.0, 30.0)] = cached_payload
    monkeypatch.setattr(service, "_fetch_current_weather", AsyncMock(side_effect=RuntimeError("boom")))

    payload = await service.get_current_weather(58.0, 30.0)

    assert payload["provider"] == "openmeteo"
    assert payload["cached"] is True
    assert payload["stale"] is True
    assert payload["temperature_c"] == pytest.approx(9.5)
