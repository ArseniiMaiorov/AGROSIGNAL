#!/usr/bin/env python3
"""Download Sentinel-2 composite for Murmansk (30km radius, summer months only)."""
from __future__ import annotations
import io, math, sys, time
from pathlib import Path

import numpy as np
import requests
import tifffile
from scipy.ndimage import sobel, gaussian_filter

CLIENT_ID     = "40102487-48ea-42b9-823b-ecc868ac3e0c"
CLIENT_SECRET = "lXxDYYr8L87XonLWzJQbcO7GjjT9DKtx"
BASE_URL      = "https://services.sentinel-hub.com"
OUT_PATH      = Path(__file__).parent / "backend/debug/runs/real_tiles/murmansk_51.npz"

LAT, LON = 68.9585, 33.0827
R_KM     = 20
dlat     = R_KM / 111.0
dlon     = R_KM / (111.0 * math.cos(math.radians(LAT)))
BBOX     = [LON - dlon, LAT - dlat, LON + dlon, LAT + dlat]
W_PX, H_PX = 1024, 1024

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B08","B11","SCL"],
      units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN"]
    }],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.SCL];
}
"""

# Summer only (June–August) — above Arctic Circle, no data in winter
MONTHS = [
    ("2021-06-01","2021-06-30","2021-06"),
    ("2021-07-01","2021-07-31","2021-07"),
    ("2021-08-01","2021-08-31","2021-08"),
    ("2022-06-01","2022-06-30","2022-06"),
    ("2022-07-01","2022-07-31","2022-07"),
    ("2022-08-01","2022-08-31","2022-08"),
    ("2023-06-01","2023-06-30","2023-06"),
    ("2023-07-01","2023-07-31","2023-07"),
    ("2023-08-01","2023-08-31","2023-08"),
]
SCL_VALID = {4, 5, 6, 7}

def get_token():
    r = requests.post(
        f"{BASE_URL}/auth/realms/main/protocol/openid-connect/token",
        data={"grant_type":"client_credentials",
              "client_id":CLIENT_ID,"client_secret":CLIENT_SECRET}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def fetch_scene(token, df, dt):
    payload = {
        "input": {
            "bounds": {"bbox": BBOX,
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{df}T00:00:00Z",
                                                   "to":   f"{dt}T23:59:59Z"},
                                     "mosaickingOrder": "leastCC",
                                     "maxCloudCoverage": 85}}]
        },
        "output": {"width": W_PX, "height": H_PX,
                   "responses": [{"identifier":"default",
                                  "format":{"type":"image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json", "Accept": "image/tiff"}
    r = requests.post(f"{BASE_URL}/api/v1/process",
                      headers=headers, json=payload, timeout=120)
    if r.status_code == 429:
        print("  rate-limit, sleeping 30s …"); time.sleep(30)
        return fetch_scene(token, df, dt)
    if r.status_code != 200:
        print(f"  !! HTTP {r.status_code}: {r.text[:200]}"); return None
    arr = tifffile.imread(io.BytesIO(r.content))
    return arr.astype(np.float32) if arr.ndim >= 3 else None

def main():
    print(f"BBox: {[round(v,3) for v in BBOX]}")
    token = get_token(); print("Token OK")

    stacks = {k: [] for k in ("b4","b8","b3","b11","b2")}
    valid_list = []

    for i,(df,dt,lbl) in enumerate(MONTHS,1):
        print(f"[{i}/{len(MONTHS)}] {lbl} …", end=" ", flush=True)
        arr = fetch_scene(token, df, dt)
        if arr is None: print("skip"); continue
        b02,b03,b04,b08,b11,scl = [arr[...,k] for k in range(6)]
        valid = np.isin(scl.astype(np.uint8), list(SCL_VALID))
        vf = valid.mean()
        print(f"valid={vf:.2f}")
        if vf < 0.08: print("  too cloudy"); continue
        stacks["b4"] .append(np.where(valid,b04,np.nan).astype(np.float32))
        stacks["b8"] .append(np.where(valid,b08,np.nan).astype(np.float32))
        stacks["b3"] .append(np.where(valid,b03,np.nan).astype(np.float32))
        stacks["b11"].append(np.where(valid,b11,np.nan).astype(np.float32))
        stacks["b2"] .append(np.where(valid,b02,np.nan).astype(np.float32))
        time.sleep(0.4)

    if not stacks["b4"]:
        print("No valid scenes!"); sys.exit(1)

    print(f"\nCompositing {len(stacks['b4'])} scenes …")
    stk = {k: np.stack(v,0) for k,v in stacks.items()}
    with np.errstate(invalid="ignore",divide="ignore"):
        ndvi_stk = (stk["b8"]-stk["b4"])/(stk["b8"]+stk["b4"]+1e-9)
        ndwi_stk = (stk["b3"]-stk["b8"])/(stk["b3"]+stk["b8"]+1e-9)

    maxndvi  = np.nanmax(ndvi_stk,  0)
    meanndvi = np.nanmean(ndvi_stk, 0)
    ndvistd  = np.nanstd(ndvi_stk,  0)
    ndwi_mean= np.nanmean(ndwi_stk, 0)

    n = len(stacks["b4"])
    ndvi_safe = np.where(np.isnan(ndvi_stk), -1.0, ndvi_stk)
    best_idx = np.argmax(ndvi_safe, 0).ravel()
    px = stk["b4"].reshape(n,-1).shape[1]
    ri = np.arange(px)
    red_median  = stk["b4"] .reshape(n,-1)[best_idx,ri].reshape(H_PX,W_PX)
    green_median= stk["b3"] .reshape(n,-1)[best_idx,ri].reshape(H_PX,W_PX)
    nir_median  = stk["b8"] .reshape(n,-1)[best_idx,ri].reshape(H_PX,W_PX)
    swir_median = stk["b11"].reshape(n,-1)[best_idx,ri].reshape(H_PX,W_PX)

    ndvi_best = np.nan_to_num(maxndvi, nan=0.0)
    ec = np.hypot(sobel(gaussian_filter(ndvi_best,1),1),
                  sobel(gaussian_filter(ndvi_best,1),0)).astype(np.float32)
    p99 = np.percentile(ec,99); ec = np.clip(ec/p99,0,1) if p99>0 else ec

    print("Saving …")
    np.savez_compressed(str(OUT_PATH),
        edgecomposite=ec, maxndvi=maxndvi.astype(np.float32),
        meanndvi=meanndvi.astype(np.float32), ndvistd=ndvistd.astype(np.float32),
        ndwi_mean=ndwi_mean.astype(np.float32), mndwi_max=ndwi_mean.astype(np.float32),
        red_median=red_median, green_median=green_median,
        nir_median=nir_median, swir_median=swir_median,
        bbox=np.array(BBOX,dtype=np.float64))
    print(f"Saved → {OUT_PATH.name}  ({OUT_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"  maxndvi={np.nanmean(maxndvi):.3f}  ec={ec.mean():.3f}")

if __name__ == "__main__":
    main()
