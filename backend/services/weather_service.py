"""Сервис получения погодных данных.

Сервис сначала пытается использовать настроенный источник, затем мягко
деградирует на Open-Meteo без API-ключа.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from services.payload_meta import build_freshness
from storage.db import WeatherData

logger = get_logger(__name__)

_FORECAST_CACHE: dict[tuple[float, float, int], tuple[datetime, list[dict[str, Any]]]] = {}
_CURRENT_CACHE: dict[tuple[float, float], CurrentWeatherPayload] = {}


@dataclass(slots=True)
class CurrentWeatherPayload:
    latitude: float
    longitude: float
    observed_at: datetime
    provider: str
    temperature_c: float | None
    apparent_temperature_c: float | None
    precipitation_mm: float | None
    wind_speed_m_s: float | None
    u_wind_10m: float | None = None
    v_wind_10m: float | None = None
    wind_direction_deg: float | None = None
    humidity_pct: float | None = None
    cloud_cover_pct: float | None = None
    pressure_hpa: float | None = None
    soil_moisture: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class WeatherService:
    """Работа с текущей погодой и краткосрочным прогнозом."""

    _forecast_cache = _FORECAST_CACHE
    _current_cache = _CURRENT_CACHE

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    @staticmethod
    def _round_coord(value: float) -> float:
        return round(float(value), 4)

    async def get_current_weather(self, lat: float, lon: float, *, organization_id: UUID | None = None) -> dict[str, Any]:
        """Вернуть текущую погоду с кэшем в базе."""
        lat_key = self._round_coord(lat)
        lon_key = self._round_coord(lon)
        cached = await self._get_cached_weather(lat_key, lon_key, organization_id=organization_id)
        if cached is not None:
            logger.info("weather_cache_hit", latitude=lat_key, longitude=lon_key, provider=cached.provider)
            return self._record_to_response(cached, cached=True)
        current_cached = self._get_cached_current(lat_key, lon_key)
        if current_cached is not None:
            logger.info("weather_current_memory_cache_hit", latitude=lat_key, longitude=lon_key, provider=current_cached.provider)
            return self._payload_to_response(current_cached, cached=True)

        try:
            payload = await self._fetch_current_weather(lat_key, lon_key)
            self._store_cached_current(lat_key, lon_key, payload)
            record = WeatherData(
                organization_id=organization_id,
                latitude=lat_key,
                longitude=lon_key,
                observed_at=payload.observed_at,
                provider=payload.provider,
                temperature_c=payload.temperature_c,
                apparent_temperature_c=payload.apparent_temperature_c,
                precipitation_mm=payload.precipitation_mm,
                wind_speed_m_s=payload.wind_speed_m_s,
                humidity_pct=payload.humidity_pct,
                cloud_cover_pct=payload.cloud_cover_pct,
                pressure_hpa=payload.pressure_hpa,
                soil_moisture=payload.soil_moisture,
                payload=payload.payload,
            )
            self.db.add(record)
            await self.db.flush()
            logger.info("weather_current_saved", latitude=lat_key, longitude=lon_key, provider=payload.provider)
            return self._record_to_response(record, cached=False)
        except Exception:
            if not bool(self.settings.WEATHER_ALLOW_STALE_ON_FAILURE):
                logger.error("weather_current_failed_no_stale_allowed", latitude=lat_key, longitude=lon_key, exc_info=True)
                return self._fallback_current_response(lat_key, lon_key)
            stale = await self._get_latest_cached_weather(lat_key, lon_key, organization_id=organization_id)
            if stale is not None:
                logger.warning("weather_provider_failed_using_stale_cache", latitude=lat_key, longitude=lon_key)
                response = self._record_to_response(stale, cached=True, stale=True)
                response["stale"] = True
                return response
            in_memory = self._get_cached_current(lat_key, lon_key, allow_stale=True)
            if in_memory is not None:
                logger.warning("weather_provider_failed_using_in_memory_cache", latitude=lat_key, longitude=lon_key)
                return self._payload_to_response(in_memory, cached=True, stale=True)
            logger.error("weather_current_failed_no_cache", latitude=lat_key, longitude=lon_key, exc_info=True)
            return self._fallback_current_response(lat_key, lon_key)

    async def get_forecast(self, lat: float, lon: float, days: int = 5, *, organization_id: UUID | None = None) -> dict[str, Any]:
        """Вернуть прогноз погоды на несколько дней."""
        days = max(1, min(int(days), 10))
        lat_key = self._round_coord(lat)
        lon_key = self._round_coord(lon)
        cached = self._get_cached_forecast(lat_key, lon_key, days)
        if cached is not None:
            cached_at, data = cached
            logger.info("weather_forecast_cache_hit", latitude=lat_key, longitude=lon_key, days=days)
            return self._forecast_response(
                lat_key,
                lon_key,
                days,
                data,
                provider="openmeteo",
                fetched_at=cached_at,
                cache_written_at=cached_at,
            )
        try:
            data = await self._fetch_openmeteo_forecast(lat_key, lon_key, days)
            cached_at = datetime.now(timezone.utc)
            self._store_cached_forecast(lat_key, lon_key, days, data, cached_at=cached_at)
            logger.info("weather_forecast_success", latitude=lat_key, longitude=lon_key, days=days)
            return self._forecast_response(
                lat_key,
                lon_key,
                days,
                data,
                provider="openmeteo",
                fetched_at=cached_at,
                cache_written_at=cached_at,
            )
        except Exception as exc:
            logger.error("weather_forecast_failed", latitude=lat_key, longitude=lon_key, days=days, error=str(exc), exc_info=True)
            if bool(self.settings.WEATHER_ALLOW_STALE_ON_FAILURE):
                stale = self._get_cached_forecast(lat_key, lon_key, days, allow_stale=True)
                if stale is not None:
                    cached_at, data = stale
                    logger.warning("weather_forecast_failed_using_stale_cache", latitude=lat_key, longitude=lon_key, days=days)
                    return self._forecast_response(
                        lat_key,
                        lon_key,
                        days,
                        data,
                        provider="openmeteo",
                        fetched_at=cached_at,
                        cache_written_at=cached_at,
                        stale=True,
                    )
            return {
                "latitude": lat_key,
                "longitude": lon_key,
                "provider": "fallback",
                "days": days,
                "forecast": [],
                "error": "Прогноз временно недоступен",
                "freshness": build_freshness(
                    provider="fallback",
                    fetched_at=datetime.now(timezone.utc),
                    freshness_state="unknown",
                ),
            }

    def _weather_cache_ttl(self) -> timedelta:
        return timedelta(minutes=max(1, int(self.settings.WEATHER_CACHE_TTL_MINUTES)))

    def _get_cached_current(
        self,
        lat: float,
        lon: float,
        *,
        allow_stale: bool = False,
    ) -> CurrentWeatherPayload | None:
        payload = self._current_cache.get((lat, lon))
        if payload is None:
            return None
        if allow_stale:
            return payload
        if datetime.now(timezone.utc) - payload.observed_at > self._weather_cache_ttl():
            return None
        return payload

    def _store_cached_current(
        self,
        lat: float,
        lon: float,
        payload: CurrentWeatherPayload,
    ) -> None:
        self._current_cache[(lat, lon)] = payload

    def _get_cached_forecast(
        self,
        lat: float,
        lon: float,
        days: int,
        *,
        allow_stale: bool = False,
    ) -> tuple[datetime, list[dict[str, Any]]] | None:
        key = (lat, lon, days)
        cached = self._forecast_cache.get(key)
        if cached is None:
            return None
        cached_at, payload = cached
        if allow_stale:
            return cached_at, list(payload)
        if datetime.now(timezone.utc) - cached_at > self._weather_cache_ttl():
            return None
        return cached_at, list(payload)

    def _store_cached_forecast(
        self,
        lat: float,
        lon: float,
        days: int,
        payload: list[dict[str, Any]],
        *,
        cached_at: datetime,
    ) -> None:
        self._forecast_cache[(lat, lon, days)] = (cached_at, list(payload))

    def _forecast_response(
        self,
        lat: float,
        lon: float,
        days: int,
        forecast: list[dict[str, Any]],
        *,
        provider: str,
        fetched_at: datetime,
        cache_written_at: datetime | None = None,
        stale: bool = False,
    ) -> dict[str, Any]:
        return {
            "latitude": lat,
            "longitude": lon,
            "provider": provider,
            "days": days,
            "forecast": forecast,
            "freshness": build_freshness(
                provider=provider,
                fetched_at=fetched_at,
                cache_written_at=cache_written_at,
                stale=stale,
                ttl_seconds=int(self.settings.WEATHER_CACHE_TTL_MINUTES) * 60,
            ),
        }

    async def _get_cached_weather(self, lat: float, lon: float, *, organization_id: UUID | None = None) -> WeatherData | None:
        ttl = timedelta(minutes=int(self.settings.WEATHER_CACHE_TTL_MINUTES))
        threshold = datetime.now(timezone.utc) - ttl
        stmt = (
            select(WeatherData)
            .where(WeatherData.organization_id == organization_id)
            .where(WeatherData.latitude == lat)
            .where(WeatherData.longitude == lon)
            .where(WeatherData.observed_at >= threshold)
            .order_by(desc(WeatherData.observed_at))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_latest_cached_weather(self, lat: float, lon: float, *, organization_id: UUID | None = None) -> WeatherData | None:
        stmt = (
            select(WeatherData)
            .where(WeatherData.organization_id == organization_id)
            .where(WeatherData.latitude == lat)
            .where(WeatherData.longitude == lon)
            .order_by(desc(WeatherData.observed_at))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _fetch_current_weather(self, lat: float, lon: float) -> CurrentWeatherPayload:
        preferred = str(self.settings.WEATHER_PROVIDER).strip().lower()
        if preferred == "openweather" and self.settings.OPENWEATHER_API_KEY:
            try:
                return await self._fetch_openweather_current(lat, lon)
            except Exception as exc:
                logger.warning("weather_primary_failed", provider="openweather", error=str(exc), fallback="openmeteo")
        return await self._fetch_openmeteo_current(lat, lon)

    async def _fetch_openweather_current(self, lat: float, lon: float) -> CurrentWeatherPayload:
        url = f"{self.settings.OPENWEATHER_BASE_URL}/onecall"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.settings.OPENWEATHER_API_KEY,
            "units": "metric",
            "exclude": "minutely,hourly,daily,alerts",
        }
        data = await self._fetch_json_with_retries(url, params=params, provider="openweather", purpose="current")
        current = data.get("current", {})
        observed_at = datetime.fromtimestamp(int(current.get("dt", datetime.now(timezone.utc).timestamp())), tz=timezone.utc)
        wind_speed = current.get("wind_speed")
        wind_direction = current.get("wind_deg")
        u_wind, v_wind = _wind_vector_from_speed_direction(wind_speed, wind_direction)
        return CurrentWeatherPayload(
            latitude=lat,
            longitude=lon,
            observed_at=observed_at,
            provider="openweather",
            temperature_c=current.get("temp"),
            apparent_temperature_c=current.get("feels_like"),
            precipitation_mm=(current.get("rain") or {}).get("1h", 0.0),
            wind_speed_m_s=wind_speed,
            u_wind_10m=u_wind,
            v_wind_10m=v_wind,
            wind_direction_deg=wind_direction,
            humidity_pct=current.get("humidity"),
            cloud_cover_pct=current.get("clouds"),
            pressure_hpa=current.get("pressure"),
            soil_moisture=None,
            payload={
                **data,
                "derived": {
                    "u_wind_10m": u_wind,
                    "v_wind_10m": v_wind,
                    "wind_direction_deg": wind_direction,
                },
            },
        )

    async def _fetch_openmeteo_current(self, lat: float, lon: float) -> CurrentWeatherPayload:
        url = f"{self.settings.OPENMETEO_BASE_URL}/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "pressure_msl",
                "cloud_cover",
                "soil_moisture_0_to_1cm",
            ]),
            "timezone": "auto",
            "forecast_days": 1,
        }
        data = await self._fetch_json_with_retries(url, params=params, provider="openmeteo", purpose="current")
        current = data.get("current", {})
        observed_at = datetime.fromisoformat(str(current.get("time", datetime.now(timezone.utc).isoformat())).replace("Z", "+00:00"))
        wind_speed = current.get("wind_speed_10m")
        wind_direction = current.get("wind_direction_10m")
        u_wind, v_wind = _wind_vector_from_speed_direction(wind_speed, wind_direction)
        return CurrentWeatherPayload(
            latitude=lat,
            longitude=lon,
            observed_at=observed_at,
            provider="openmeteo",
            temperature_c=current.get("temperature_2m"),
            apparent_temperature_c=current.get("apparent_temperature"),
            precipitation_mm=current.get("precipitation"),
            wind_speed_m_s=wind_speed,
            u_wind_10m=u_wind,
            v_wind_10m=v_wind,
            wind_direction_deg=wind_direction,
            humidity_pct=current.get("relative_humidity_2m"),
            cloud_cover_pct=current.get("cloud_cover"),
            pressure_hpa=current.get("pressure_msl"),
            soil_moisture=current.get("soil_moisture_0_to_1cm"),
            payload={
                **data,
                "derived": {
                    "u_wind_10m": u_wind,
                    "v_wind_10m": v_wind,
                    "wind_direction_deg": wind_direction,
                },
            },
        )

    async def _fetch_openmeteo_forecast(self, lat: float, lon: float, days: int) -> list[dict[str, Any]]:
        url = f"{self.settings.OPENMETEO_BASE_URL}/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
                "cloud_cover_mean",
            ]),
            "timezone": "auto",
            "forecast_days": days,
        }
        data = await self._fetch_json_with_retries(url, params=params, provider="openmeteo", purpose="forecast")
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        result: list[dict[str, Any]] = []
        for index, day in enumerate(dates):
            temp_max = _safe_index(daily.get("temperature_2m_max", []), index)
            temp_min = _safe_index(daily.get("temperature_2m_min", []), index)
            temp_mean = None
            if temp_max is not None and temp_min is not None:
                temp_mean = round((float(temp_max) + float(temp_min)) / 2.0, 2)
            elif temp_max is not None:
                temp_mean = temp_max
            elif temp_min is not None:
                temp_mean = temp_min
            result.append(
                {
                    "date": day,
                    "temp_max_c": temp_max,
                    "temp_min_c": temp_min,
                    "temp_mean_c": temp_mean,
                    "precipitation_mm": _safe_index(daily.get("precipitation_sum", []), index),
                    "wind_speed_m_s": _safe_index(daily.get("wind_speed_10m_max", []), index),
                    "cloud_cover_pct": _safe_index(daily.get("cloud_cover_mean", []), index),
                }
            )
        return result

    async def _fetch_json_with_retries(
        self,
        url: str,
        *,
        params: dict[str, Any],
        provider: str,
        purpose: str,
    ) -> dict[str, Any]:
        attempts = max(1, int(self.settings.WEATHER_HTTP_RETRIES))
        timeout = httpx.Timeout(
            timeout=float(self.settings.WEATHER_HTTP_TIMEOUT_S),
            connect=float(self.settings.WEATHER_HTTP_CONNECT_TIMEOUT_S),
        )
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                delay_s = float(self.settings.WEATHER_HTTP_RETRY_BACKOFF_S) * attempt
                logger.warning(
                    "weather_http_retry",
                    provider=provider,
                    purpose=purpose,
                    attempt=attempt,
                    retry_in_s=round(delay_s, 2),
                    error=str(exc),
                )
                await asyncio.sleep(delay_s)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _fallback_current_response(lat: float, lon: float) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "latitude": lat,
            "longitude": lon,
            "observed_at": now.isoformat(),
            "provider": "fallback",
            "cached": False,
            "error": "Текущая погода временно недоступна",
            "temperature_c": None,
            "apparent_temperature_c": None,
            "precipitation_mm": None,
            "wind_speed_m_s": None,
            "u_wind_10m": None,
            "v_wind_10m": None,
            "wind_direction_deg": None,
            "humidity_pct": None,
            "cloud_cover_pct": None,
            "pressure_hpa": None,
            "soil_moisture": None,
            "freshness": build_freshness(
                provider="fallback",
                fetched_at=now,
                freshness_state="unknown",
            ),
        }

    @staticmethod
    def _payload_to_response(
        payload: CurrentWeatherPayload,
        *,
        cached: bool,
        stale: bool = False,
    ) -> dict[str, Any]:
        return {
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "observed_at": payload.observed_at.isoformat(),
            "provider": payload.provider,
            "cached": cached,
            "stale": stale,
            "temperature_c": payload.temperature_c,
            "apparent_temperature_c": payload.apparent_temperature_c,
            "precipitation_mm": payload.precipitation_mm,
            "wind_speed_m_s": payload.wind_speed_m_s,
            "u_wind_10m": payload.u_wind_10m,
            "v_wind_10m": payload.v_wind_10m,
            "wind_direction_deg": payload.wind_direction_deg,
            "humidity_pct": payload.humidity_pct,
            "cloud_cover_pct": payload.cloud_cover_pct,
            "pressure_hpa": payload.pressure_hpa,
            "soil_moisture": payload.soil_moisture,
            "freshness": build_freshness(
                provider=payload.provider,
                fetched_at=payload.observed_at,
                cache_written_at=payload.observed_at,
                stale=stale,
                ttl_seconds=int(get_settings().WEATHER_CACHE_TTL_MINUTES) * 60,
            ),
        }

    @staticmethod
    def _record_to_response(record: WeatherData, *, cached: bool, stale: bool = False) -> dict[str, Any]:
        return {
            "latitude": record.latitude,
            "longitude": record.longitude,
            "observed_at": record.observed_at.isoformat(),
            "provider": record.provider,
            "cached": cached,
            "temperature_c": record.temperature_c,
            "apparent_temperature_c": record.apparent_temperature_c,
            "precipitation_mm": record.precipitation_mm,
            "wind_speed_m_s": record.wind_speed_m_s,
            "u_wind_10m": _payload_value(getattr(record, "payload", None), "u_wind_10m"),
            "v_wind_10m": _payload_value(getattr(record, "payload", None), "v_wind_10m"),
            "wind_direction_deg": _payload_value(getattr(record, "payload", None), "wind_direction_deg"),
            "humidity_pct": record.humidity_pct,
            "cloud_cover_pct": record.cloud_cover_pct,
            "pressure_hpa": record.pressure_hpa,
            "soil_moisture": record.soil_moisture,
            "freshness": build_freshness(
                provider=record.provider,
                fetched_at=record.observed_at,
                cache_written_at=getattr(record, "created_at", None),
                ttl_seconds=int(get_settings().WEATHER_CACHE_TTL_MINUTES) * 60,
                stale=stale,
            ),
        }


def _safe_index(items: list[Any], index: int) -> Any:
    if 0 <= index < len(items):
        return items[index]
    return None


def _wind_vector_from_speed_direction(
    speed_m_s: float | None,
    direction_deg: float | None,
) -> tuple[float | None, float | None]:
    if speed_m_s is None or direction_deg is None:
        return None, None
    try:
        speed = float(speed_m_s)
        direction = math.radians(float(direction_deg))
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "wind_vector_conversion_failed: speed=%s direction=%s error=%s",
            speed_m_s, direction_deg, exc,
        )
        return None, None
    return float(-speed * math.sin(direction)), float(-speed * math.cos(direction))


def _payload_value(payload: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    derived = payload.get("derived")
    if isinstance(derived, dict) and key in derived:
        return derived.get(key)
    current = payload.get("current")
    if isinstance(current, dict) and key in current:
        return current.get(key)
    return payload.get(key)
