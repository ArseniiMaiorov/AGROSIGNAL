#!/usr/bin/env python3
"""
Regenerate ekaterinburg_feature_maps.png for the thesis figures.
Fetches Sentinel-2 data for Ekaterinburg (Sverdlovsk Oblast) and renders
a 4-panel feature map (edge_composite, max_ndvi, mean_ndvi, ndvi_std).

Run from the project root:
  cd /home/arsenii-maiorov/Documents/SUAI/Диплом/AutoDetect_v2.0
  python backend/training/regen_ekb_feature_maps.py
"""
from __future__ import annotations

import os
import sys
import asyncio
import inspect
from math import cos, radians
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR  = PROJECT_ROOT / "backend"
ENV_PATH     = PROJECT_ROOT / ".env"
OUT_PNG      = PROJECT_ROOT / "ДИПЛОМ/figures/ekaterinburg_feature_maps.png"

load_dotenv(ENV_PATH)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

# Ekaterinburg city centre
LAT, LON    = 56.8386, 60.6057
RADIUS_KM   = 35.0        # cover the whole agglomeration
W, H        = 1024, 1024
MAX_CLOUD   = 70
TILE_ID     = "ekaterinburg_01"
DATE_FROM   = "2025-04-15"
DATE_TO     = "2025-09-30"

TIME_WINDOWS = [
    ("2025-04-15T00:00:00Z", "2025-05-15T23:59:59Z"),
    ("2025-05-15T00:00:00Z", "2025-06-15T23:59:59Z"),
    ("2025-06-15T00:00:00Z", "2025-07-15T23:59:59Z"),
    ("2025-07-15T00:00:00Z", "2025-08-15T23:59:59Z"),
    ("2025-08-15T00:00:00Z", "2025-09-15T23:59:59Z"),
    ("2025-09-15T00:00:00Z", "2025-10-15T23:59:59Z"),
]

BANDS_DISPLAY = ["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"]
CMAPS_DISPLAY = ["viridis",        "viridis",  "viridis",   "viridis"]


def bbox_from_center(lat, lon, half_km):
    dlat = half_km / 110.574
    dlon = half_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def import_any(module_path, *names):
    mod = __import__(module_path, fromlist=["*"])
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise ImportError(f"None of {names} found in {module_path}")

def kget(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Missing keys {keys}")


def _get_settings():
    try:
        from core.config import getsettings
        return getsettings()
    except Exception:
        from core.config import get_settings
        return get_settings()


settings = _get_settings()

SentinelHubClient      = import_any("providers.sentinelhub.client", "SentinelHubClient")
client                 = SentinelHubClient()
compute_all_indices    = import_any("processing.fields.indices", "compute_all_indices", "computeallindices")
build_valid_mask_from_scl = import_any("processing.fields.composite",
                                        "build_valid_mask_from_scl", "buildvalidmaskfromscl")
select_dates_by_coverage  = import_any("processing.fields.composite",
                                        "select_dates_by_coverage", "selectdatesbycoverage")
try:
    build_multiyear_composite = import_any("processing.fields.temporal_composite",
                                            "build_multiyear_composite")
except Exception:
    build_multiyear_composite = import_any("processing.fields.temporalcomposite",
                                            "build_multiyear_composite", "buildmultiyearcomposite")


async def _fetch(bbox, tf, tt):
    if hasattr(client, "fetchtile"):
        return await client.fetchtile(bbox, tf, tt, W, H, maxcloudpct=MAX_CLOUD)
    return await client.fetch_tile(bbox, tf, tt, W, H, max_cloud_pct=MAX_CLOUD)


def _select_dates(valid_mask):
    params = set(inspect.signature(select_dates_by_coverage).parameters.keys())
    kw: dict[str, Any] = {}
    for k1, k2 in [("min_valid_pct", "minvalidpct"), ("n_dates", "ndates"),
                   ("min_good_dates", "mingooddates"), ("return_metadata", "returnmetadata")]:
        if k1 in params:   kw[k1] = {"min_valid_pct": 0.30, "n_dates": 6,
                                      "min_good_dates": 2, "return_metadata": True}[k1]
        elif k2 in params: kw[k2] = {"minvalidpct": 0.30, "ndates": 6,
                                      "mingooddates": 2, "returnmetadata": True}[k2]
    out = select_dates_by_coverage(valid_mask, **kw)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def _build_composite(ndvi_sel, valid_sel, edge_bands):
    params = set(inspect.signature(build_multiyear_composite).parameters.keys())
    kw: dict[str, Any] = {}
    for k1, k2, v in [("ndvi_stack", "ndvistack", ndvi_sel),
                       ("valid_mask", "validmask", valid_sel),
                       ("edge_bands", "edgebands", edge_bands)]:
        if k1 in params:   kw[k1] = v
        elif k2 in params: kw[k2] = v
    kw["cfg"] = settings
    return build_multiyear_composite(**kw)


def render_and_save(edge, maxndvi, meanndvi, ndvistd):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    data_bands = [edge, maxndvi, meanndvi, ndvistd]

    for ax, band, title, cmap in zip(axes.ravel(), data_bands, BANDS_DISPLAY, CMAPS_DISPLAY):
        valid = band[np.isfinite(band)]
        if valid.size > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(band, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{TILE_ID}  ({DATE_FROM} \u2026 {DATE_TO})", fontsize=15, fontweight="bold")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Saved  →  {OUT_PNG}")


async def main():
    bbox = bbox_from_center(LAT, LON, RADIUS_KM)
    print(f"📡  Fetching  {TILE_ID}  bbox={[round(x, 4) for x in bbox]}")

    band_lists: dict[str, list] = {k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12"]}
    scl_list: list = []

    for tf, tt in TIME_WINDOWS:
        try:
            result = await _fetch(bbox, tf, tt)
            for k in band_lists:
                if k not in result:
                    raise KeyError(f"Missing band {k}")
                band_lists[k].append(np.asarray(result[k], dtype=np.float32))
            scl = result.get("SCL", None)
            scl_list.append(np.asarray(scl if scl is not None else np.full((H, W), 4, dtype=np.uint8),
                                        dtype=np.uint8))
            print(f"  ✅  {tf[:10]}")
        except Exception as e:
            print(f"  ⚠️   {tf[:10]}  skip: {str(e)[:160]}")

    if len(scl_list) < 2:
        raise RuntimeError("Too few valid scenes fetched.")

    bands    = {k: np.stack(v, 0).astype(np.float32) for k, v in band_lists.items() if v}
    scl      = np.stack(scl_list, 0).astype(np.uint8)
    valid    = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)
    indices  = compute_all_indices(bands)
    ndvi     = indices["NDVI"]

    selected, _ = _select_dates(valid)
    selected    = np.asarray(selected)
    if selected.size == 0:
        raise RuntimeError("No dates selected after cloud filtering.")

    ndvi_sel  = ndvi[selected].astype(np.float32)
    valid_sel = valid[selected]
    edge_b    = {k: bands[k][selected].astype(np.float32) for k in ["B2", "B3", "B4", "B8"]}
    edge_b["ndvi"] = ndvi_sel

    comp = _build_composite(ndvi_sel, valid_sel, edge_b)

    edge    = np.asarray(kget(comp, "edge_composite", "edgecomposite"), dtype=np.float32)
    maxndvi = np.asarray(kget(comp, "max_ndvi",       "maxndvi"),       dtype=np.float32)
    meandvi = np.asarray(kget(comp, "mean_ndvi",      "meanndvi"),      dtype=np.float32)
    std     = np.asarray(kget(comp, "ndvi_std",       "ndvistd"),       dtype=np.float32)

    render_and_save(edge, maxndvi, meandvi, std)


if __name__ == "__main__":
    asyncio.run(main())
