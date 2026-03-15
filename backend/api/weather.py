"""API погодных данных."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import WeatherCurrentResponse, WeatherForecastResponse
from services.weather_service import WeatherService
from storage.db import get_db

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/current", response_model=WeatherCurrentResponse)
async def get_current_weather(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    ctx: RequestContext = Depends(require_permissions("weather:read")),
    db: AsyncSession = Depends(get_db),
) -> WeatherCurrentResponse:
    service = WeatherService(db)
    payload = await service.get_current_weather(lat, lon, organization_id=ctx.organization_id)
    return WeatherCurrentResponse(**payload)


@router.get("/forecast", response_model=WeatherForecastResponse)
async def get_weather_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: int = Query(5, ge=1, le=10),
    ctx: RequestContext = Depends(require_permissions("weather:read")),
    db: AsyncSession = Depends(get_db),
) -> WeatherForecastResponse:
    service = WeatherService(db)
    payload = await service.get_forecast(lat, lon, days=days, organization_id=ctx.organization_id)
    return WeatherForecastResponse(**payload)
