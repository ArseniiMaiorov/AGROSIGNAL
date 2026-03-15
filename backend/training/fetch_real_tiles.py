#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import asyncio
import inspect
import argparse
from math import cos, radians
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
BACKEND_DIR = PROJECT_ROOT / "backend"

load_dotenv(PROJECT_ROOT / ".env")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# чтобы импорты Settings/SQLAlchemy не падали в "training" контексте
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")


def _get_settings():
    # в проекте встречаются оба варианта
    try:
        from core.config import getsettings  # type: ignore
        return getsettings()
    except Exception:
        from core.config import get_settings  # type: ignore
        return get_settings()


def _get_sentinel_client():
    from providers.sentinelhub.client import SentinelHubClient  # type: ignore
    return SentinelHubClient()


def bbox_from_center(lat: float, lon: float, half_size_km: float) -> tuple[float, float, float, float]:
    dlat = half_size_km / 110.574
    dlon = half_size_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def utc_start(d: date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def utc_end(d: date) -> str:
    return f"{d.isoformat()}T23:59:59Z"


def build_time_windows(start: date, end: date, window_days: int) -> list[tuple[str, str]]:
    if end < start:
        raise ValueError("end must be >= start")
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    windows: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        wend = min(end, cur + timedelta(days=window_days - 1))
        windows.append((utc_start(cur), utc_end(wend)))
        cur = wend + timedelta(days=1)
    return windows


def import_callable(module_path: str, *names: str):
    mod = __import__(module_path, fromlist=["*"])
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise ImportError(f"None of {names} found in {module_path}")


def pick_kwargs(func, candidates: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    params = sig.parameters
    out: dict[str, Any] = {}
    for k, v in candidates.items():
        if v is None:
            continue
        if k in params:
            out[k] = v
    return out


def kget(d: dict[str, Any], *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Missing keys {keys}; got {list(d.keys())}")


settings: Any | None = None
client: Any | None = None
compute_all_indices: Any | None = None
build_valid_mask_from_scl: Any | None = None
select_dates_by_coverage: Any | None = None
build_multiyear_composite: Any | None = None


HALF_SIZE_KM = 10.0

# (tile_id, lat, lon) — расширенный список v5 (64 тайла, 6+ геокластеров)
REGION_CENTERS: list[tuple[str, float, float]] = [
    # Cluster A: South
    ("krasnodar_01", 45.045, 39.010),
    ("krasnodar_02", 45.235, 38.850),
    ("stavropol_01", 45.535, 43.150),
    ("stavropol_02", 45.335, 43.550),
    ("rostov_01", 47.230, 39.720),
    ("krasnodar_03", 45.350, 39.200),
    ("krasnodar_04", 45.600, 38.500),
    ("adygea_01", 44.880, 39.800),
    ("rostov_02", 47.500, 40.100),
    ("rostov_03", 47.800, 39.300),
    ("volgograd_01", 48.700, 44.500),
    ("volgograd_02", 49.100, 43.800),
    ("dagestan_01", 43.500, 47.000),
    # Cluster B: Central Chernozem
    ("belgorod_01", 50.585, 37.550),
    ("belgorod_02", 50.635, 37.750),
    ("voronezh_01", 51.535, 40.950),
    ("voronezh_02", 51.435, 40.750),
    ("kursk_01", 51.735, 36.150),
    ("kursk_02", 51.635, 36.350),
    ("saratov_01", 51.530, 46.000),
    ("tambov_01", 52.700, 41.400),
    ("tambov_02", 52.400, 41.800),
    ("lipetsk_01", 52.600, 39.600),
    ("orel_01", 52.970, 36.060),
    ("penza_01", 53.200, 45.000),
    ("saratov_02", 51.800, 46.500),
    ("saratov_03", 52.100, 45.500),
    # Cluster C: Volga / Ural
    ("samara_01", 53.200, 50.100),
    ("samara_02", 53.500, 50.600),
    ("orenburg_01", 51.770, 55.100),
    ("orenburg_02", 52.300, 54.500),
    ("bashkortostan_01", 54.300, 56.000),
    ("bashkortostan_02", 54.700, 55.500),
    ("chelyabinsk_01", 54.500, 61.400),
    ("tatarstan_01", 55.800, 49.100),
    # Cluster D: Non-chernozem
    ("moscow_01", 55.500, 37.600),
    ("tver_01", 56.850, 35.900),
    ("smolensk_01", 54.780, 32.050),
    ("kaluga_01", 54.500, 36.250),
    ("yaroslavl_01", 57.600, 39.850),
    ("kostroma_01", 57.770, 40.950),
    ("vologda_01", 59.220, 39.880),
    ("nizhny_01", 56.300, 44.000),
    ("kirov_01", 58.600, 49.700),
    # Cluster E: North-West
    ("lenoblast_01", 58.691208, 29.893892),
    ("lenoblast_02", 59.500, 30.200),
    ("lenoblast_03", 59.100, 30.500),
    ("permkrai_01", 57.430, 56.950),
    ("pskov_01", 57.800, 28.300),
    ("pskov_02", 57.500, 29.900),
    ("lenoblast_04", 59.800, 29.800),
    ("lenoblast_05", 58.700, 30.800),
    ("novgorod_01", 58.520, 31.280),
    ("novgorod_02", 58.100, 32.500),
    ("pskov_03", 57.000, 29.300),
    ("tver_02", 56.500, 36.400),
    ("karelia_01", 61.800, 34.300),
    ("permkrai_02", 58.000, 56.400),
    # Cluster F: West Siberia
    ("tyumen_01", 57.150, 68.250),
    ("omsk_01", 54.970, 73.370),
    ("novosibirsk_01", 55.030, 82.920),
    ("altai_01", 53.350, 83.760),
    ("altai_02", 52.500, 84.900),
    ("kurgan_01", 55.450, 65.300),
]

REGIONS: list[tuple[str, tuple[float, float, float, float]]] = [
    (
        tile_id,
        bbox_from_center(lat=lat, lon=lon, half_size_km=HALF_SIZE_KM),
    )
    for tile_id, lat, lon in REGION_CENTERS
]

W, H = 1024, 1024

# 2 года непрерывно
RANGE_START = date(2023, 4, 15)
RANGE_END = date(2025, 4, 14)
WINDOW_DAYS = 30  # 24 окна примерно (быстрее/медленнее меняй тут)

TIME_WINDOWS: list[tuple[str, str]] = build_time_windows(RANGE_START, RANGE_END, WINDOW_DAYS)

OUT_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch real Sentinel-2 tiles for BoundaryUNet training")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root (defaults to repository root inferred from script path)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile width/height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--min-scenes",
        type=int,
        default=4,
        help="Minimum number of valid scenes required to save a tile (default: 4)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tiles that already exist in output directory (default: enabled)",
    )
    return parser.parse_args()


def _configure_runtime(project_root: Path, tile_size: int) -> None:
    global PROJECT_ROOT, BACKEND_DIR, settings, client
    global compute_all_indices, build_valid_mask_from_scl, select_dates_by_coverage, build_multiyear_composite
    global W, H, OUT_DIR, TIME_WINDOWS

    PROJECT_ROOT = project_root.resolve()
    BACKEND_DIR = PROJECT_ROOT / "backend"
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    W = int(tile_size)
    H = int(tile_size)
    if W <= 0 or H <= 0:
        raise ValueError(f"tile-size must be positive, got {tile_size}")

    OUT_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TIME_WINDOWS = build_time_windows(RANGE_START, RANGE_END, WINDOW_DAYS)

    settings = _get_settings()
    client = _get_sentinel_client()
    compute_all_indices = import_callable("processing.fields.indices", "compute_all_indices", "computeallindices")
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
            "processing.fields.temporalcomposite", "build_multiyear_composite", "buildmultiyearcomposite"
        )


async def _fetch_tile(bbox, time_from, time_to, w, h, max_cloud_pct):
    if client is None:
        raise RuntimeError("Sentinel client is not initialized")
    # В проекте точно есть fetchtile(...) [file:2], но держим fallback
    if hasattr(client, "fetchtile"):
        return await client.fetchtile(bbox, time_from, time_to, w, h, max_cloud_pct)  # type: ignore
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

    if "cfg" in params:
        kwargs["cfg"] = settings
    else:
        # на всякий случай, но обычно cfg есть
        kwargs["cfg"] = settings

    return build_multiyear_composite(**kwargs)


async def fetch_region(
    tile_id: str,
    bbox: tuple[float, float, float, float],
    *,
    min_scenes: int,
    skip_existing: bool,
) -> bool:
    out_path = OUT_DIR / f"{tile_id}.npz"
    if skip_existing and out_path.exists():
        print(f"\n⏭️  {tile_id}: already exists, skip ({out_path.name})")
        return True

    print(f"\n📡 {tile_id} bbox={bbox} windows={len(TIME_WINDOWS)} ({RANGE_START}..{RANGE_END})")

    band_lists: dict[str, list[np.ndarray]] = {k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12"]}
    scl_list: list[np.ndarray] = []
    concurrency = max(1, int(getattr(settings, "SENTINEL_CONCURRENT_REQUESTS", 4)))
    semaphore = asyncio.Semaphore(concurrency)

    async def _fetch_window(tf: str, tt: str):
        async with semaphore:
            try:
                return tf, tt, await _fetch_tile(bbox, tf, tt, W, H, max_cloud_pct=60)
            except Exception as e:
                print(f"  ⚠️  {tf[:10]} skip: {str(e)[:180]}")
                return None

    results = await asyncio.gather(*[_fetch_window(tf, tt) for tf, tt in TIME_WINDOWS])

    for item in results:
        if item is None:
            continue
        _, _, result = item
        try:
            for k in band_lists:
                if k not in result:
                    raise KeyError(f"Missing band {k}")
                band_lists[k].append(np.asarray(result[k], dtype=np.float32))

            scl = result.get("SCL")
            if scl is None:
                scl = np.full((H, W), 4, dtype=np.uint8)
            scl_list.append(np.asarray(scl, dtype=np.uint8))
        except Exception as e:
            print(f"  ⚠️  malformed scene skipped: {str(e)[:180]}")

    if len(scl_list) < max(2, int(min_scenes)):
        print(f"  ❌ {tile_id}: too few scenes ({len(scl_list)} < {max(2, int(min_scenes))})")
        return False

    bands = {k: np.stack(v, axis=0).astype(np.float32) for k, v in band_lists.items() if v}
    scl = np.stack(scl_list, axis=0).astype(np.uint8)

    valid_mask = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)  # (T,H,W)
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

    comp = _call_build_multiyear(ndvi_sel, valid_sel, edge_bands)

    edge = np.asarray(kget(comp, "edge_composite", "edgecomposite"), dtype=np.float32)
    maxndvi = np.asarray(kget(comp, "max_ndvi", "maxndvi"), dtype=np.float32)
    meanndvi = np.asarray(kget(comp, "mean_ndvi", "meanndvi"), dtype=np.float32)
    ndvistd = np.asarray(kget(comp, "ndvi_std", "ndvistd"), dtype=np.float32)

    # --- real spectral composites from selected dates ---
    sel = selected  # index array into (T, H, W)
    valid_sel_f = valid_sel.astype(np.float32)  # (Tsel, H, W)

    def _masked_nanstat(arr, stat="mean"):
        """Compute stat over selected dates, masking invalid pixels."""
        a = np.where(valid_sel, arr, np.nan)
        with np.errstate(all="ignore"):
            if stat == "mean":
                return np.nanmean(a, axis=0).astype(np.float32)
            elif stat == "median":
                return np.nanmedian(a, axis=0).astype(np.float32)
            elif stat == "max":
                return np.nanmax(a, axis=0).astype(np.float32)
        return np.nanmean(a, axis=0).astype(np.float32)

    # NDWI from real B3, B8
    ndwi_sel = indices["NDWI"][sel]
    ndwi_mean = _masked_nanstat(ndwi_sel, "mean")
    ndwi_median = _masked_nanstat(ndwi_sel, "median")

    # MNDWI from real B3, B11
    mndwi_sel = indices["MNDWI"][sel]
    mndwi_max = _masked_nanstat(mndwi_sel, "max")

    # BSI from real B2, B4, B8, B11
    bsi_sel = indices["BSI"][sel]
    bsi_mean = _masked_nanstat(bsi_sel, "mean")

    # NDMI from real B8, B11
    ndmi_sel = indices["NDMI"][sel]
    ndmi_mean = _masked_nanstat(ndmi_sel, "mean")

    # Real band medians for RGB channels (selected dates)
    nir_median = _masked_nanstat(bands["B8"][sel], "median")
    red_median = _masked_nanstat(bands["B4"][sel], "median")
    green_median = _masked_nanstat(bands["B3"][sel], "median")
    blue_median = _masked_nanstat(bands["B2"][sel], "median")
    swir_median = _masked_nanstat(bands["B11"][sel], "median")

    # SCL valid fraction per pixel
    scl_valid_fraction = np.mean(valid_mask, axis=0).astype(np.float32)  # (H, W)

    # NDVI temporal entropy (Shannon entropy of NDVI time series per pixel)
    ndvi_clipped = np.clip(np.where(valid_sel, ndvi_sel, np.nan), 0.0, 1.0)
    n_bins = 10
    h, w = ndvi_clipped.shape[1], ndvi_clipped.shape[2]
    ndvi_entropy = np.zeros((h, w), dtype=np.float32)
    # Vectorized: bin each pixel's temporal NDVI, compute entropy
    bin_indices = np.clip((ndvi_clipped * n_bins).astype(np.int32), 0, n_bins - 1)  # (T, H, W)
    valid_finite = np.isfinite(ndvi_clipped)
    valid_count_ent = valid_finite.sum(axis=0)  # (H, W)
    for b in range(n_bins):
        count_b = ((bin_indices == b) & valid_finite).sum(axis=0).astype(np.float32)
        with np.errstate(all="ignore"):
            prob = count_b / np.maximum(valid_count_ent.astype(np.float32), 1.0)
            contrib = np.where(prob > 0, -prob * np.log2(prob), 0.0)
        ndvi_entropy += contrib.astype(np.float32)
    ndvi_entropy[valid_count_ent < 2] = 0.0

    np.savez_compressed(
        out_path,
        # --- original channels ---
        edgecomposite=edge,
        maxndvi=maxndvi,
        meanndvi=meanndvi,
        ndvistd=ndvistd,
        n_valid_scenes=np.int32(int(comp.get("n_valid_scenes", int(selected.size)))),
        scl_median=np.median(scl, axis=0).astype(np.uint8),
        bbox=np.array(bbox, dtype=np.float64),
        train_data_version=np.array(str(getattr(settings, "TRAIN_DATA_VERSION", "real_tiles_v5"))),
        feature_stack_version=np.array(str(getattr(settings, "FEATURE_STACK_VERSION", "v5_16ch"))),
        # --- real spectral composites ---
        ndwi_mean=ndwi_mean,
        ndwi_median=ndwi_median,
        mndwi_max=mndwi_max,
        bsi_mean=bsi_mean,
        ndmi_mean=ndmi_mean,
        nir_median=nir_median,
        red_median=red_median,
        green_median=green_median,
        blue_median=blue_median,
        swir_median=swir_median,
        scl_valid_fraction=scl_valid_fraction,
        ndvi_entropy=ndvi_entropy,
    )
    print(f"  💾 saved {out_path}  ({len(sel)} dates, "
          f"ndwi_mean range=[{np.nanmin(ndwi_mean):.3f},{np.nanmax(ndwi_mean):.3f}], "
          f"bsi_mean range=[{np.nanmin(bsi_mean):.3f},{np.nanmax(bsi_mean):.3f}])")
    return True


async def main():
    args = _parse_args()
    _configure_runtime(args.project_root, args.tile_size)
    print(f"📦 project_root={PROJECT_ROOT}")
    print(f"🧱 tile_size={W}x{H}")
    print(f"🧰 min_scenes={max(2, int(args.min_scenes))} skip_existing={bool(args.skip_existing)}")

    ok = 0
    try:
        for tile_id, bbox in REGIONS:
            try:
                if await fetch_region(
                    tile_id,
                    bbox,
                    min_scenes=max(2, int(args.min_scenes)),
                    skip_existing=bool(args.skip_existing),
                ):
                    ok += 1
            except Exception as e:
                print(f"  💥 {tile_id} FATAL: {e}")
    finally:
        if client is not None and hasattr(client, "close"):
            await client.close()

    print(f"\n✅ DONE: {ok}/{len(REGIONS)} tiles saved")
    print(f"   → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
