#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import asyncio
import inspect
from math import cos, radians
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

# чтобы импорты Settings/SQLAlchemy не падали
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

LAT, LON = 59.839270, 56.578116
RADIUS_KM = 1.0  # радиус 1 км -> bbox +/-1 км
W, H = 256, 256
MAX_CLOUD_PCT = 60
TILE_ID = "single_2025_59.839270_56.578116"

# один сезон (как раньше, окна 15-е число каждого месяца)
TIME_WINDOWS = [
    ("2025-04-15T00:00:00Z", "2025-05-15T23:59:59Z"),
    ("2025-05-15T00:00:00Z", "2025-06-15T23:59:59Z"),
    ("2025-06-15T00:00:00Z", "2025-07-15T23:59:59Z"),
    ("2025-07-15T00:00:00Z", "2025-08-15T23:59:59Z"),
    ("2025-08-15T00:00:00Z", "2025-09-15T23:59:59Z"),
    ("2025-09-15T00:00:00Z", "2025-10-15T23:59:59Z"),
]

OUT_DIR = PROJECT_ROOT / "backend/debug/runs/single_tile_2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def bbox_from_center(lat: float, lon: float, half_size_km: float) -> tuple[float, float, float, float]:
    dlat = half_size_km / 110.574
    dlon = half_size_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def import_any(module_path: str, *names: str):
    mod = __import__(module_path, fromlist=["*"])
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise ImportError(f"None of {names} found in {module_path}")

def kget(d: dict[str, Any], *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Missing keys {keys}; got {list(d.keys())}")

def get_settings():
    try:
        from core.config import getsettings  # type: ignore
        return getsettings()
    except Exception:
        from core.config import get_settings  # type: ignore
        return get_settings()

settings = get_settings()

SentinelHubClient = import_any("providers.sentinelhub.client", "SentinelHubClient")
client = SentinelHubClient()

compute_all_indices = import_any("processing.fields.indices", "compute_all_indices", "computeallindices")
build_valid_mask_from_scl = import_any("processing.fields.composite", "build_valid_mask_from_scl", "buildvalidmaskfromscl")
select_dates_by_coverage = import_any("processing.fields.composite", "select_dates_by_coverage", "selectdatesbycoverage")
try:
    build_multiyear_composite = import_any("processing.fields.temporal_composite", "build_multiyear_composite")
except Exception:
    build_multiyear_composite = import_any("processing.fields.temporalcomposite", "build_multiyear_composite", "buildmultiyearcomposite")

async def _fetch_tile(bbox, time_from, time_to, w, h, max_cloud_pct):
    # в проекте есть fetchtile(bbox, timefrom, timeto, width, height, maxcloudpct=...) [file:2]
    if hasattr(client, "fetchtile"):
        return await client.fetchtile(bbox, time_from, time_to, w, h, maxcloudpct=max_cloud_pct)  # type: ignore
    return await client.fetch_tile(bbox, time_from, time_to, w, h, max_cloud_pct=max_cloud_pct)  # type: ignore

def _call_select_dates(valid_mask: np.ndarray, n_dates: int):
    params = set(inspect.signature(select_dates_by_coverage).parameters.keys())
    kwargs: dict[str, Any] = {}

    if "min_valid_pct" in params:
        kwargs["min_valid_pct"] = 0.30
    elif "minvalidpct" in params:
        kwargs["minvalidpct"] = 0.30

    if "n_dates" in params:
        kwargs["n_dates"] = n_dates
    elif "ndates" in params:
        kwargs["ndates"] = n_dates

    if "min_good_dates" in params:
        kwargs["min_good_dates"] = 2
    elif "mingooddates" in params:
        kwargs["mingooddates"] = 2

    if "return_metadata" in params:
        kwargs["return_metadata"] = True
    elif "returnmetadata" in params:
        kwargs["returnmetadata"] = True

    out = select_dates_by_coverage(valid_mask, **kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}

def _call_build_multiyear(ndvi_sel: np.ndarray, valid_sel: np.ndarray, edge_bands: dict[str, np.ndarray]):
    params = set(inspect.signature(build_multiyear_composite).parameters.keys())
    kwargs: dict[str, Any] = {}

    if "ndvi_stack" in params:
        kwargs["ndvi_stack"] = ndvi_sel
    elif "ndvistack" in params:
        kwargs["ndvistack"] = ndvi_sel

    if "valid_mask" in params:
        kwargs["valid_mask"] = valid_sel
    elif "validmask" in params:
        kwargs["validmask"] = valid_sel

    if "edge_bands" in params:
        kwargs["edge_bands"] = edge_bands
    elif "edgebands" in params:
        kwargs["edgebands"] = edge_bands

    kwargs["cfg"] = settings
    return build_multiyear_composite(**kwargs)

async def main():
    bbox = bbox_from_center(LAT, LON, RADIUS_KM)
    print(f"📡 {TILE_ID} bbox={bbox}")
    print(f"   windows: {len(TIME_WINDOWS)} (2025-04-15 .. 2025-10-15)")

    band_lists: dict[str, list[np.ndarray]] = {k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12"]}
    scl_list: list[np.ndarray] = []

    for tf, tt in TIME_WINDOWS:
        try:
            result: dict[str, Any] = await _fetch_tile(bbox, tf, tt, W, H, MAX_CLOUD_PCT)

            for k in band_lists:
                if k not in result:
                    raise KeyError(f"Missing band {k}")
                band_lists[k].append(np.asarray(result[k], dtype=np.float32))

            scl = result.get("SCL", None)
            if scl is None:
                scl = np.full((H, W), 4, dtype=np.uint8)
            scl_list.append(np.asarray(scl, dtype=np.uint8))

            print(f"  ✅ {tf[:10]} OK")
        except Exception as e:
            print(f"  ⚠️  {tf[:10]} skip: {str(e)[:180]}")

    if len(scl_list) < 2:
        raise RuntimeError("too few valid scenes")

    bands = {k: np.stack(v, axis=0).astype(np.float32) for k, v in band_lists.items() if v}
    scl = np.stack(scl_list, axis=0).astype(np.uint8)

    valid_mask = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)  # (T,H,W)
    indices = compute_all_indices(bands)
    ndvi = indices["NDVI"]

    selected, meta = _call_select_dates(valid_mask, n_dates=min(6, valid_mask.shape[0]))
    selected = np.asarray(selected) if selected is not None else np.asarray([], dtype=int)
    if selected.size == 0:
        raise RuntimeError(f"no selected dates, meta={meta}")

    ndvi_sel = np.asarray(ndvi[selected], dtype=np.float32)
    valid_sel = np.asarray(valid_mask[selected], dtype=bool)

    edge_bands = {k: np.asarray(bands[k][selected], dtype=np.float32) for k in ["B2", "B3", "B4", "B8"]}
    edge_bands["ndvi"] = ndvi_sel

    comp = _call_build_multiyear(ndvi_sel, valid_sel, edge_bands)

    edge = np.asarray(kget(comp, "edge_composite", "edgecomposite"), dtype=np.float32)
    maxndvi = np.asarray(kget(comp, "max_ndvi", "maxndvi"), dtype=np.float32)
    meanndvi = np.asarray(kget(comp, "mean_ndvi", "meanndvi"), dtype=np.float32)
    ndvistd = np.asarray(kget(comp, "ndvi_std", "ndvistd"), dtype=np.float32)

    out_path = OUT_DIR / f"{TILE_ID}.npz"
    np.savez_compressed(
        out_path,
        edgecomposite=edge,
        maxndvi=maxndvi,
        meanndvi=meanndvi,
        ndvistd=ndvistd,
        n_valid_scenes=np.int32(int(comp.get("n_valid_scenes", int(selected.size)))),
        scl_median=np.median(scl, axis=0).astype(np.uint8),
        bbox=np.array(bbox, dtype=np.float64),
    )

    print(f"\n💾 saved {out_path}")
    print(f"✅ DONE: 1/1 tiles saved")
    print(f"   → {OUT_DIR.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
