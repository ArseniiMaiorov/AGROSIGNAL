#!/usr/bin/env python3
"""
Fetch one tile for point (59.839270, 56.578116), radius ~1 km.
Season: 2025-04-15 .. 2025-10-14 (30-day windows).
Saves .npz, GeoTIFF, PNG quicklook.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from datetime import date, timedelta
from math import cos, radians
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
load_dotenv(PROJECT_ROOT / ".env")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

# target point
LAT, LON = 59.839270, 56.578116
HALF_KM = 1.0
W, H = 256, 256
WINDOW_DAYS = 30
MAX_CLOUD_PCT = 60

RANGE_START = date(2025, 4, 15)
RANGE_END = date(2025, 10, 14)
TILE_ID = "perm_single_2025"

OUT_DIR = PROJECT_ROOT / "backend/debug/runs/single_tile"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bbox_from_center(lat: float, lon: float, half_km: float) -> tuple[float, float, float, float]:
    dlat = half_km / 110.574
    dlon = half_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def build_time_windows(start: date, end: date, days: int) -> list[tuple[str, str]]:
    wins: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        wend = min(end, cur + timedelta(days=days - 1))
        wins.append((f"{cur.isoformat()}T00:00:00Z", f"{wend.isoformat()}T23:59:59Z"))
        cur = wend + timedelta(days=1)
    return wins


def import_any(module: str, *names: str):
    mod = __import__(module, fromlist=["*"])
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise ImportError(f"None of {names} in {module}")


def kget(d: dict[str, Any], *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"keys {keys} not found; got {list(d.keys())}")


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


settings = _get_settings()

compute_all_indices = import_any("processing.fields.indices", "compute_all_indices", "computeallindices")
build_valid_mask_from_scl = import_any("processing.fields.composite", "build_valid_mask_from_scl", "buildvalidmaskfromscl")
select_dates_by_coverage = import_any("processing.fields.composite", "select_dates_by_coverage", "selectdatesbycoverage")

try:
    build_multiyear_composite = import_any("processing.fields.temporal_composite", "build_multiyear_composite")
except Exception:
    build_multiyear_composite = import_any(
        "processing.fields.temporalcomposite",
        "build_multiyear_composite",
        "buildmultiyearcomposite",
    )


def _call_select_dates(valid_mask: np.ndarray, n_dates: int):
    p = set(inspect.signature(select_dates_by_coverage).parameters)

    kw: dict[str, Any] = {}
    kw["min_valid_pct" if "min_valid_pct" in p else "minvalidpct"] = 0.30
    kw["n_dates" if "n_dates" in p else "ndates"] = n_dates
    kw["min_good_dates" if "min_good_dates" in p else "mingooddates"] = 2
    kw["return_metadata" if "return_metadata" in p else "returnmetadata"] = True

    out = select_dates_by_coverage(valid_mask, **kw)
    return (out[0], out[1]) if isinstance(out, tuple) and len(out) == 2 else (out, {})


def _call_build_multiyear(ndvi_sel: np.ndarray, valid_sel: np.ndarray, edge_bands: dict[str, np.ndarray]):
    p = set(inspect.signature(build_multiyear_composite).parameters)

    kw: dict[str, Any] = {}
    kw["ndvi_stack" if "ndvi_stack" in p else "ndvistack"] = ndvi_sel
    kw["valid_mask" if "valid_mask" in p else "validmask"] = valid_sel
    kw["edge_bands" if "edge_bands" in p else "edgebands"] = edge_bands
    kw["cfg"] = settings

    return build_multiyear_composite(**kw)


async def fetch_and_save():
    client = _get_client()
    bbox = bbox_from_center(LAT, LON, HALF_KM)
    wins = build_time_windows(RANGE_START, RANGE_END, WINDOW_DAYS)

    print(f"📡 {TILE_ID}  bbox={tuple(round(v,6) for v in bbox)}")
    print(f"   windows: {len(wins)}  ({RANGE_START} .. {RANGE_END})")

    band_lists: dict[str, list[np.ndarray]] = {k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12"]}
    scl_list: list[np.ndarray] = []

    for tf, tt in wins:
        try:
            if hasattr(client, "fetchtile"):
                result: dict[str, Any] = await client.fetchtile(bbox, tf, tt, W, H, MAX_CLOUD_PCT)  # type: ignore
            else:
                result = await client.fetch_tile(bbox, tf, tt, W, H, max_cloud_pct=MAX_CLOUD_PCT)  # type: ignore

            for k in band_lists:
                if k not in result:
                    raise KeyError(f"missing band {k}; got keys={list(result.keys())}")
                band_lists[k].append(np.asarray(result[k], dtype=np.float32))

            # FIX: numpy array нельзя использовать в `or`
            scl = result.get("SCL", None)
            if scl is None:
                scl = np.full((H, W), 4, dtype=np.uint8)
            scl_list.append(np.asarray(scl, dtype=np.uint8))

            print(f"  ✅ {tf[:10]}")
        except Exception as e:
            print(f"  ⚠️  {tf[:10]} skip: {str(e)[:220]}")

    if len(scl_list) < 2:
        raise RuntimeError(f"too few valid scenes: {len(scl_list)}/{len(wins)}")

    bands = {k: np.stack(v, axis=0).astype(np.float32) for k, v in band_lists.items()}
    scl = np.stack(scl_list, axis=0).astype(np.uint8)

    valid_mask = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)
    indices = compute_all_indices(bands)
    ndvi = indices["NDVI"]

    selected, meta = _call_select_dates(valid_mask, n_dates=min(8, valid_mask.shape[0]))
    selected = np.asarray(selected) if selected is not None else np.array([], dtype=int)
    if selected.size == 0:
        raise RuntimeError(f"no dates selected: meta={meta}")
    print(f"  🗓 selected {selected.size} of {valid_mask.shape[0]} scenes")

    ndvi_sel = np.asarray(ndvi[selected], dtype=np.float32)
    valid_sel = np.asarray(valid_mask[selected], dtype=bool)

    edge_bands = {k: np.asarray(bands[k][selected], dtype=np.float32) for k in ["B2", "B3", "B4", "B8"]}
    edge_bands["ndvi"] = ndvi_sel

    comp = _call_build_multiyear(ndvi_sel, valid_sel, edge_bands)

    edge = np.asarray(kget(comp, "edge_composite", "edgecomposite"), dtype=np.float32)
    maxndvi = np.asarray(kget(comp, "max_ndvi", "maxndvi"), dtype=np.float32)
    meanndvi = np.asarray(kget(comp, "mean_ndvi", "meanndvi"), dtype=np.float32)
    ndvistd = np.asarray(kget(comp, "ndvi_std", "ndvistd"), dtype=np.float32)

    # ── npz
    npz_path = OUT_DIR / f"{TILE_ID}.npz"
    np.savez_compressed(
        npz_path,
        edgecomposite=edge,
        maxndvi=maxndvi,
        meanndvi=meanndvi,
        ndvistd=ndvistd,
        n_valid_scenes=np.int32(int(comp.get("n_valid_scenes", int(selected.size)))),
        scl_median=np.median(scl, axis=0).astype(np.uint8),
        bbox=np.array(bbox, dtype=np.float64),
    )
    print(f"\n  💾 .npz  → {npz_path}")

    # ── GeoTIFF
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.stack([edge, maxndvi, meanndvi, ndvistd], axis=0).astype(np.float32)
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)
    tif_path = OUT_DIR / f"{TILE_ID}_composite.tif"

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=0.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as ds:
        for i, name in enumerate(["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"], start=1):
            ds.write(arr[i - 1], i)
            ds.set_band_description(i, name)
    print(f"  🗺  .tif  → {tif_path}")

    scl_med = np.median(scl, axis=0).astype(np.uint8)
    scl_path = OUT_DIR / f"{TILE_ID}_scl_median.tif"
    with rasterio.open(
        scl_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=255,
        compress="deflate",
        predictor=1,
    ) as ds:
        ds.write(scl_med, 1)
        ds.set_band_description(1, "scl_median")
    print(f"  🗺  .tif  → {scl_path}")

    # ── PNG quicklook
    import matplotlib.pyplot as plt

    names = ["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"]
    cmaps = ["hot", "RdYlGn", "RdYlGn", "plasma"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    for i in range(4):
        img = arr[i]
        vmin = float(np.percentile(img, 2))
        vmax = float(np.percentile(img, 98))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        axes[i].imshow(img, cmap=cmaps[i], vmin=vmin, vmax=vmax)
        axes[i].set_title(names[i])
        axes[i].axis("off")

    fig.suptitle(f"{TILE_ID}\n{LAT:.6f}, {LON:.6f} | {RANGE_START}..{RANGE_END}", fontsize=9)
    fig.tight_layout()

    png_path = OUT_DIR / f"{TILE_ID}_quicklook.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"  🖼  .png  → {png_path}")

    print("\n✅ ALL DONE")


if __name__ == "__main__":
    asyncio.run(fetch_and_save())
