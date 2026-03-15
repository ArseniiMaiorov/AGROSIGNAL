#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

IN_DIR = Path("backend/debug/runs/single_tile_2025")
OUT_DIR = Path("backend/debug/runs/single_tile_2025_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tile_npz = next(iter(sorted(IN_DIR.glob("*.npz"))), None)
if tile_npz is None:
    raise SystemExit(f"No .npz found in {IN_DIR.resolve()}")

z = np.load(tile_npz)

tile_id = tile_npz.stem
bbox = tuple(map(float, z["bbox"].tolist()))

edge = z["edgecomposite"].astype(np.float32)
maxndvi = z["maxndvi"].astype(np.float32)
meanndvi = z["meanndvi"].astype(np.float32)
ndvistd = z["ndvistd"].astype(np.float32)

H, W = edge.shape
transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)

# composite GeoTIFF (4 bands)
comp = np.stack([edge, maxndvi, meanndvi, ndvistd], axis=0)
tif_path = OUT_DIR / f"{tile_id}_composite.tif"

with rasterio.open(
    tif_path, "w",
    driver="GTiff",
    height=H, width=W,
    count=4,
    dtype="float32",
    crs="EPSG:4326",
    transform=transform,
    nodata=0.0,
    compress="deflate",
    predictor=2,
    tiled=True,
    blockxsize=256 if W >= 256 else W,
    blockysize=256 if H >= 256 else H,
) as ds:
    names = ["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"]
    for i, name in enumerate(names, start=1):
        ds.write(comp[i-1], i)
        ds.set_band_description(i, name)

# SCL median (если есть)
if "scl_median" in z:
    scl = z["scl_median"].astype(np.uint8)
    scl_path = OUT_DIR / f"{tile_id}_scl_median.tif"
    with rasterio.open(
        scl_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=255,
        compress="deflate",
        predictor=1,
    ) as ds:
        ds.write(scl, 1)
        ds.set_band_description(1, "scl_median")

# PNG quicklook (2x2, как мы уже делали)
png_path = OUT_DIR / f"{tile_id}_quicklook.png"
names = ["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"]
cmaps = ["hot", "RdYlGn", "RdYlGn", "plasma"]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()
for i in range(4):
    img = comp[i]
    vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
    if abs(float(vmax) - float(vmin)) < 1e-6:
        vmax = vmin + 1e-6
    axes[i].imshow(img, cmap=cmaps[i], vmin=vmin, vmax=vmax)
    axes[i].set_title(names[i])
    axes[i].axis("off")

fig.suptitle(tile_id, fontsize=10)
fig.tight_layout()
fig.savefig(png_path, dpi=180)
plt.close(fig)

print("✅ Export done:")
print("  NPZ:", tile_npz.resolve())
print("  TIF:", tif_path.resolve())
print("  PNG:", png_path.resolve())
