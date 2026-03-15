#!/usr/bin/env python3
"""
Fetch Sentinel-2 composite for a single area in two seasons:
  - cold  : 2024-10-01 .. 2025-03-31  (вечнозелёное выделяется)
  - warm  : 2025-04-01 .. 2025-10-31  (поля с высоким NDVI выделяются)

Output: backend/debug/runs/single_area/<tile_id>.npz
"""
from __future__ import annotations

import os
import sys
import asyncio
import inspect
from math import cos, radians
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")


# ---------------------------------------------------------------------------
# helpers: settings / client
# ---------------------------------------------------------------------------
def _get_settings():
    try:
        from core.config import getsettings  # type: ignore
        return getsettings()
    except Exception:
        from core.config import get_settings  # type: ignore
        return get_settings()


def _get_client():
    from providers.sentinelhub.client import SentinelHubClient  # type: ignore
    return SentinelHubClient()


def import_callable(module_path: str, *names: str):
    mod = __import__(module_path, fromlist=["*"])
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise ImportError(f"None of {names} found in {module_path}")


def kget(d: dict, *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Missing keys {keys}; got {list(d.keys())}")


# ---------------------------------------------------------------------------
# lazy imports (same pattern as fetch_real_tiles.py)
# ---------------------------------------------------------------------------
settings = _get_settings()
client = _get_client()

compute_all_indices = import_callable(
    "processing.fields.indices", "compute_all_indices", "computeallindices"
)
build_valid_mask_from_scl = import_callable(
    "processing.fields.composite", "build_valid_mask_from_scl", "buildvalidmaskfromscl"
)
select_dates_by_coverage = import_callable(
    "processing.fields.composite", "select_dates_by_coverage", "selectdatesbycoverage"
)
try:
    build_multiyear_composite = import_callable(
        "processing.fields.temporal_composite", "build_multiyear_composite"
    )
except Exception:
    build_multiyear_composite = import_callable(
        "processing.fields.temporalcomposite",
        "build_multiyear_composite", "buildmultiyearcomposite",
    )


# ---------------------------------------------------------------------------
# area + time windows
# ---------------------------------------------------------------------------
def bbox_from_center(
    lat: float, lon: float, radius_km: float
) -> tuple[float, float, float, float]:
    dlat = radius_km / 110.574
    dlon = radius_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def build_time_windows(
    start: date, end: date, window_days: int = 30
) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        wend = min(end, cur + timedelta(days=window_days - 1))
        windows.append(
            (f"{cur.isoformat()}T00:00:00Z", f"{wend.isoformat()}T23:59:59Z")
        )
        cur = wend + timedelta(days=1)
    return windows


# одна точка, радиус 1 км
LAT, LON = 59.839270, 56.578116
BBOX = bbox_from_center(LAT, LON, radius_km=1.0)
W, H = 256, 256

SEASONS: dict[str, tuple[date, date]] = {
    "cold": (date(2024, 10, 1), date(2025, 3, 31)),   # осень + зима
    "warm": (date(2025, 4, 1), date(2025, 10, 31)),   # сезон 2025
}

OUT_DIR = PROJECT_ROOT / "backend/debug/runs/single_area"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# thin wrappers (inspect-based, same as fetch_real_tiles.py)
# ---------------------------------------------------------------------------
def _call_select_dates(valid_mask: np.ndarray, n_dates: int):
    params = set(inspect.signature(select_dates_by_coverage).parameters.keys())
    kwargs: dict[str, Any] = {}

    kwargs["min_valid_pct" if "min_valid_pct" in params else "minvalidpct"] = 0.30
    kwargs["n_dates" if "n_dates" in params else "ndates"] = n_dates
    kwargs["min_good_dates" if "min_good_dates" in params else "mingooddates"] = 2
    if "return_metadata" in params:
        kwargs["return_metadata"] = True
    elif "returnmetadata" in params:
        kwargs["returnmetadata"] = True

    out = select_dates_by_coverage(valid_mask, **kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def _call_build_composite(
    ndvi_sel: np.ndarray,
    valid_sel: np.ndarray,
    edge_bands: dict[str, np.ndarray],
) -> dict[str, Any]:
    params = set(inspect.signature(build_multiyear_composite).parameters.keys())
    kwargs: dict[str, Any] = {}

    kwargs["ndvi_stack" if "ndvi_stack" in params else "ndvistack"] = ndvi_sel
    kwargs["valid_mask" if "valid_mask" in params else "validmask"] = valid_sel
    kwargs["edge_bands" if "edge_bands" in params else "edgebands"] = edge_bands
    kwargs["cfg"] = settings

    return build_multiyear_composite(**kwargs)


# ---------------------------------------------------------------------------
# fetch + build composite for one season
# ---------------------------------------------------------------------------
async def _fetch_tile(bbox, time_from, time_to, w, h, max_cloud_pct):
    if hasattr(client, "fetchtile"):
        return await client.fetchtile(bbox, time_from, time_to, w, h, max_cloud_pct)  # type: ignore
    return await client.fetch_tile(bbox, time_from, time_to, w, h, max_cloud_pct=max_cloud_pct)  # type: ignore


async def fetch_season(season_name: str, start: date, end: date) -> bool:
    tile_id = f"single_area_{season_name}"
    windows = build_time_windows(start, end, window_days=30)

    print(f"\n📡 {tile_id}  bbox={BBOX}")
    print(f"   season: {start} .. {end}  ({len(windows)} windows)")

    band_lists: dict[str, list[np.ndarray]] = {
        k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12"]
    }
    scl_list: list[np.ndarray] = []

    for tf, tt in windows:
        try:
            result: dict[str, Any] = await _fetch_tile(
                BBOX, tf, tt, W, H, max_cloud_pct=60
            )
            for k in band_lists:
                if k not in result:
                    raise KeyError(f"Missing band {k}")
                band_lists[k].append(np.asarray(result[k], dtype=np.float32))
            scl = result.get("SCL")
            if scl is None:
                scl = np.full((H, W), 4, dtype=np.uint8)
            scl_list.append(np.asarray(scl, dtype=np.uint8))
            print(f"  ✅ {tf[:10]} OK")
        except Exception as e:
            print(f"  ⚠️  {tf[:10]} skip: {str(e)[:180]}")

    if len(scl_list) < 2:
        print(f"  ❌ {tile_id}: too few scenes ({len(scl_list)})")
        return False

    bands = {k: np.stack(v, axis=0).astype(np.float32) for k, v in band_lists.items() if v}
    scl = np.stack(scl_list, axis=0).astype(np.uint8)

    valid_mask = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)
    indices = compute_all_indices(bands)
    ndvi = indices["NDVI"]

    selected, meta = _call_select_dates(valid_mask, n_dates=min(8, valid_mask.shape[0]))
    selected = np.asarray(selected) if selected is not None else np.asarray([], dtype=int)
    if selected.size == 0:
        print(f"  ❌ {tile_id}: no selected dates, meta={meta}")
        return False

    ndvi_sel = np.asarray(ndvi[selected], dtype=np.float32)
    valid_sel = np.asarray(valid_mask[selected], dtype=bool)

    edge_bands = {k: np.asarray(bands[k][selected], dtype=np.float32) for k in ["B2", "B3", "B4", "B8"]}
    edge_bands["ndvi"] = ndvi_sel

    comp = _call_build_composite(ndvi_sel, valid_sel, edge_bands)

    edge     = np.asarray(kget(comp, "edge_composite", "edgecomposite"), dtype=np.float32)
    maxndvi  = np.asarray(kget(comp, "max_ndvi",       "maxndvi"),       dtype=np.float32)
    meanndvi = np.asarray(kget(comp, "mean_ndvi",      "meanndvi"),      dtype=np.float32)
    ndvistd  = np.asarray(kget(comp, "ndvi_std",       "ndvistd"),       dtype=np.float32)

    out_path = OUT_DIR / f"{tile_id}.npz"
    np.savez_compressed(
        out_path,
        edgecomposite=edge,
        maxndvi=maxndvi,
        meanndvi=meanndvi,
        ndvistd=ndvistd,
        n_valid_scenes=np.int32(int(comp.get("n_valid_scenes", int(selected.size)))),
        scl_median=np.median(scl, axis=0).astype(np.uint8),
        bbox=np.array(BBOX, dtype=np.float64),
        season=np.bytes_(season_name),
    )
    print(f"  💾 saved → {out_path}")
    return True


# ---------------------------------------------------------------------------
async def main():
    ok = 0
    for season_name, (start, end) in SEASONS.items():
        try:
            if await fetch_season(season_name, start, end):
                ok += 1
        except Exception as e:
            import traceback
            print(f"  💥 {season_name} FATAL: {e}")
            traceback.print_exc()

    print(f"\n{'✅' if ok > 0 else '❌'} DONE: {ok}/{len(SEASONS)} seasons saved")
    print(f"   → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
