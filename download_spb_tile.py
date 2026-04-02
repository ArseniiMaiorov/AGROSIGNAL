#!/usr/bin/env python3
"""
Download Sentinel-2 L2A composite for St. Petersburg (30km radius).
Produces lenoblast_spb.npz with the same keys as existing real_tiles/*.npz.

Center: 59.928622°N, 30.306225°E
Bbox (30km radius): lon [29.767, 30.846], lat [59.658, 60.199]
Resolution: ~58m/px at 1024×512 px  (matches wallpaper aspect needs)
"""
from __future__ import annotations
import io, json, math, sys, time
from pathlib import Path

import numpy as np
import requests
import tifffile
from scipy import ndimage

# ── Credentials (from .env) ────────────────────────────────────────────────────
CLIENT_ID     = "40102487-48ea-42b9-823b-ecc868ac3e0c"
CLIENT_SECRET = "lXxDYYr8L87XonLWzJQbcO7GjjT9DKtx"
BASE_URL      = "https://services.sentinel-hub.com"

OUT_PATH = Path(__file__).parent / "backend/debug/runs/real_tiles/lenoblast_spb.npz"

# SPB center 59.928622N 30.306225E, radius 30km
LAT, LON = 59.928622, 30.306225
R_KM     = 50
dlat     = R_KM / 111.0
dlon     = R_KM / (111.0 * math.cos(math.radians(LAT)))
BBOX     = [LON - dlon, LAT - dlat, LON + dlon, LAT + dlat]  # W S E N
W_PX, H_PX = 2048, 2048

print(f"BBox: {[round(v,4) for v in BBOX]}")
print(f"Output: {W_PX}×{H_PX} px  ({OUT_PATH.name})")

# ── Evalscript — returns B02 B03 B04 B08 B11 SCL ─────────────────────────────
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "B11", "SCL"],
      units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN"]
    }],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.SCL];
}
"""

# ── Auth ──────────────────────────────────────────────────────────────────────
def get_token() -> str:
    r = requests.post(
        f"{BASE_URL}/auth/realms/main/protocol/openid-connect/token",
        data={"grant_type": "client_credentials",
              "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

# ── One Process-API request → float32 array (H, W, 6) ────────────────────────
def fetch_scene(token: str, date_from: str, date_to: str) -> np.ndarray | None:
    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_from}T00:00:00Z",
                        "to":   f"{date_to}T23:59:59Z"
                    },
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": 80,
                }
            }]
        },
        "output": {
            "width":  W_PX,
            "height": H_PX,
            "responses": [{"identifier": "default",
                           "format": {"type": "image/tiff"}}]
        },
        "evalscript": EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json",
               "Accept": "image/tiff"}
    r = requests.post(f"{BASE_URL}/api/v1/process",
                      headers=headers, json=payload, timeout=120)
    if r.status_code == 429:
        print("  rate-limit, sleeping 30s …")
        time.sleep(30)
        return fetch_scene(token, date_from, date_to)
    if r.status_code != 200:
        print(f"  !! HTTP {r.status_code}: {r.text[:300]}")
        return None
    arr = tifffile.imread(io.BytesIO(r.content))   # (H, W, 6) float32
    if arr.ndim == 2:   # single-band edge case
        return None
    return arr.astype(np.float32)

# ── Monthly windows 2021–2023 (clear-sky months: Apr-Sep) ────────────────────
MONTHS = [
    # (date_from, date_to, label)
    ("2021-05-01","2021-05-31","2021-05"),
    ("2021-07-01","2021-07-31","2021-07"),
    ("2021-08-01","2021-08-31","2021-08"),
    ("2022-05-01","2022-05-31","2022-05"),
    ("2022-06-01","2022-06-30","2022-06"),
    ("2022-07-01","2022-07-31","2022-07"),
    ("2022-08-01","2022-08-31","2022-08"),
    ("2022-09-01","2022-09-30","2022-09"),
    ("2023-05-01","2023-05-31","2023-05"),
    ("2023-06-01","2023-06-30","2023-06"),
    ("2023-07-01","2023-07-31","2023-07"),
    ("2023-08-01","2023-08-31","2023-08"),
]

SCL_VALID = {4, 5, 6, 7}   # vegetation, non-veg, water, unclassified (not cloud/shadow)

def main() -> None:
    token = get_token()
    print("Token OK")

    scenes_b4  = []   # (H, W) red
    scenes_b8  = []   # (H, W) NIR
    scenes_b3  = []   # (H, W) green
    scenes_b11 = []   # (H, W) SWIR
    scenes_b2  = []   # (H, W) blue
    valid_masks= []   # (H, W) bool

    for i, (df, dt, lbl) in enumerate(MONTHS, 1):
        print(f"[{i}/{len(MONTHS)}] {lbl} …", end=" ", flush=True)
        arr = fetch_scene(token, df, dt)
        if arr is None:
            print("skip (no data)")
            continue
        # band order: B02 B03 B04 B08 B11 SCL (0-indexed)
        b02, b03, b04, b08, b11, scl = [arr[..., k] for k in range(6)]
        scl_int = scl.astype(np.uint8)
        valid   = np.isin(scl_int, list(SCL_VALID))
        valid_frac = valid.mean()
        print(f"valid={valid_frac:.2f}")
        if valid_frac < 0.10:
            print("  skip (too cloudy)")
            continue
        scenes_b4 .append(np.where(valid, b04,  np.nan).astype(np.float32))
        scenes_b8 .append(np.where(valid, b08,  np.nan).astype(np.float32))
        scenes_b3 .append(np.where(valid, b03,  np.nan).astype(np.float32))
        scenes_b11.append(np.where(valid, b11,  np.nan).astype(np.float32))
        scenes_b2 .append(np.where(valid, b02,  np.nan).astype(np.float32))
        valid_masks.append(valid)
        time.sleep(0.5)

    if not scenes_b4:
        print("No valid scenes! Aborting.")
        sys.exit(1)

    print(f"\nCompositing {len(scenes_b4)} scenes …")
    stk_b4  = np.stack(scenes_b4,  axis=0)   # (N, H, W)
    stk_b8  = np.stack(scenes_b8,  axis=0)
    stk_b3  = np.stack(scenes_b3,  axis=0)
    stk_b11 = np.stack(scenes_b11, axis=0)
    stk_b2  = np.stack(scenes_b2,  axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_stk = (stk_b8 - stk_b4) / (stk_b8 + stk_b4 + 1e-9)
        ndwi_stk = (stk_b3 - stk_b8) / (stk_b3 + stk_b8 + 1e-9)

    maxndvi  = np.nanmax(ndvi_stk,  axis=0)
    meanndvi = np.nanmean(ndvi_stk, axis=0)
    ndvistd  = np.nanstd(ndvi_stk,  axis=0)
    ndwi_mean= np.nanmean(ndwi_stk, axis=0)

    # Best-pixel composite (max NDVI scene) for band medians
    best_idx    = np.nanargmax(ndvi_stk, axis=0)
    flat_idx    = best_idx.ravel()
    flat_b4     = stk_b4 .reshape(len(scenes_b4), -1)
    flat_b3     = stk_b3 .reshape(len(scenes_b3), -1)
    flat_b8     = stk_b8 .reshape(len(scenes_b8), -1)
    flat_b11    = stk_b11.reshape(len(scenes_b11), -1)
    n_px        = flat_b4.shape[1]
    row_idx     = np.arange(n_px)
    red_median  = flat_b4 [flat_idx, row_idx].reshape(H_PX, W_PX)
    green_median= flat_b3 [flat_idx, row_idx].reshape(H_PX, W_PX)
    nir_median  = flat_b8 [flat_idx, row_idx].reshape(H_PX, W_PX)
    swir_median = flat_b11[flat_idx, row_idx].reshape(H_PX, W_PX)

    # Edge composite — Sobel on max-NDVI image (same as existing pipeline)
    ndvi_best = maxndvi.copy()
    ndvi_best = np.nan_to_num(ndvi_best, nan=0.0)
    from scipy.ndimage import sobel, gaussian_filter
    smoothed  = gaussian_filter(ndvi_best, sigma=1.0)
    sx = sobel(smoothed, axis=1).astype(np.float32)
    sy = sobel(smoothed, axis=0).astype(np.float32)
    edgecomposite = np.hypot(sx, sy).astype(np.float32)
    # Normalize
    p99 = np.percentile(edgecomposite, 99)
    if p99 > 0:
        edgecomposite = np.clip(edgecomposite / p99, 0, 1)

    bbox_arr = np.array(BBOX, dtype=np.float64)

    print("Saving …")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(OUT_PATH),
        edgecomposite = edgecomposite.astype(np.float32),
        maxndvi       = maxndvi      .astype(np.float32),
        meanndvi      = meanndvi     .astype(np.float32),
        ndvistd       = ndvistd      .astype(np.float32),
        ndwi_mean     = ndwi_mean    .astype(np.float32),
        mndwi_max     = ndwi_mean    .astype(np.float32),  # stub
        red_median    = red_median   .astype(np.float32),
        green_median  = green_median .astype(np.float32),
        nir_median    = nir_median   .astype(np.float32),
        swir_median   = swir_median  .astype(np.float32),
        bbox          = bbox_arr,
    )
    print(f"Saved → {OUT_PATH}  ({OUT_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"  maxndvi mean={np.nanmean(maxndvi):.3f}  ndvistd mean={np.nanmean(ndvistd):.3f}")
    print(f"  edgecomposite mean={edgecomposite.mean():.3f}")


if __name__ == "__main__":
    main()
