"""Region routing helpers shared across runtime and date selection."""
from __future__ import annotations


def resolve_region_band(
    lat: float,
    *,
    south_max: float = 48.0,
    north_min: float = 57.0,
) -> str:
    value = float(lat)
    if value < float(south_max):
        return "south"
    if value >= float(north_min):
        return "north"
    return "central"


def resolve_region_boundary_profile(lat: float, cfg) -> str:
    band = resolve_region_band(
        lat,
        south_max=float(getattr(cfg, "REGION_LAT_SOUTH_MAX", 48.0)),
        north_min=float(getattr(cfg, "REGION_LAT_NORTH_MIN", 57.0)),
    )
    if band == "south" and bool(getattr(cfg, "SOUTH_PROFILE_ENABLED", True)):
        return "south_recall"
    if band == "north" and bool(getattr(cfg, "NORTH_PROFILE_ENABLED", True)):
        return "north_boundary"
    return "balanced"
