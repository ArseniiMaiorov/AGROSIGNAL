#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any


def _serialize_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric != numeric:
        return None
    return round(numeric, 6)


async def _build_rows(limit: int | None = None) -> list[dict[str, Any]]:
    from geoalchemy2.shape import to_shape
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from storage.db import CropAssignment, FieldSeason, SoilProfile, get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(FieldSeason)
            .options(
                selectinload(FieldSeason.field),
                selectinload(FieldSeason.crop_assignments).selectinload(CropAssignment.crop),
                selectinload(FieldSeason.yield_observations),
                selectinload(FieldSeason.management_events),
                selectinload(FieldSeason.weather_daily),
            )
            .order_by(FieldSeason.season_year.desc(), FieldSeason.id.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        seasons = (await session.execute(stmt)).scalars().all()
        rows: list[dict[str, Any]] = []
        for season in seasons:
            if not season.yield_observations:
                continue
            if not season.crop_assignments:
                continue
            field = season.field
            if field is None:
                continue
            geom = to_shape(field.geom)
            centroid = geom.centroid
            soil_stmt = (
                select(SoilProfile)
                .where(SoilProfile.field_id == field.id)
                .order_by(SoilProfile.sampled_at.desc())
                .limit(1)
            )
            soil = (await session.execute(soil_stmt)).scalar_one_or_none()
            crop_assignment = season.crop_assignments[0]
            crop = crop_assignment.crop
            management_events = list(season.management_events or [])
            weather_daily = list(season.weather_daily or [])
            yields = [float(item.yield_kg_ha) for item in season.yield_observations if item.yield_kg_ha is not None]
            if not yields:
                continue
            precip_values = [float(item.precipitation_mm) for item in weather_daily if item.precipitation_mm is not None]
            gdd_values = [float(item.gdd) for item in weather_daily if item.gdd is not None]
            vpd_values = [float(item.vpd) for item in weather_daily if item.vpd is not None]
            temp_values = [float(item.temperature_mean_c) for item in weather_daily if item.temperature_mean_c is not None]
            management_total = sum(float(item.amount or 0.0) for item in management_events)
            event_types = sorted({str(item.event_type) for item in management_events if item.event_type})
            rows.append(
                {
                    "field_id": str(field.id),
                    "season_year": int(season.season_year),
                    "crop_code": crop_assignment.crop_code,
                    "crop_name": crop.name if crop is not None else crop_assignment.crop_code,
                    "yield_kg_ha": round(mean(yields), 4),
                    "latitude": round(float(centroid.y), 6),
                    "longitude": round(float(centroid.x), 6),
                    "field_area_ha": round(float(field.area_m2 or 0.0) / 10000.0, 4),
                    "field_perimeter_m": round(float(field.perimeter_m or 0.0), 4),
                    "soil_texture_class": soil.texture_class if soil else None,
                    "soil_organic_matter_pct": _serialize_float(soil.organic_matter_pct if soil else None),
                    "soil_ph": _serialize_float(soil.ph if soil else None),
                    "soil_n_ppm": _serialize_float(soil.n_ppm if soil else None),
                    "soil_p_ppm": _serialize_float(soil.p_ppm if soil else None),
                    "soil_k_ppm": _serialize_float(soil.k_ppm if soil else None),
                    "management_total_amount": round(float(management_total), 4),
                    "management_event_count": len(management_events),
                    "management_event_types": event_types,
                    "season_precipitation_mm": round(sum(precip_values), 4) if precip_values else None,
                    "season_gdd_sum": round(sum(gdd_values), 4) if gdd_values else None,
                    "season_vpd_mean": round(mean(vpd_values), 4) if vpd_values else None,
                    "season_temperature_mean_c": round(mean(temp_values), 4) if temp_values else None,
                    "weather_day_count": len(weather_daily),
                    "observed_on_min": min((item.observed_on for item in weather_daily), default=None),
                    "observed_on_max": max((item.observed_on for item in weather_daily), default=None),
                }
            )
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare tenant/public yield corpus from DB tables")
    parser.add_argument("--output", type=Path, default=Path("backend/debug/runs/yield_corpus_v2.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{args.output.resolve()} (dry-run)")
        return 0

    from core.settings import get_settings

    _ = get_settings()  # ensure env is readable before opening DB
    rows = asyncio.run(_build_rows(limit=args.limit or None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = dict(row)
            for key in ("observed_on_min", "observed_on_max"):
                value = payload.get(key)
                if isinstance(value, date):
                    payload[key] = value.isoformat()
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    print(f"{args.output.resolve()} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
