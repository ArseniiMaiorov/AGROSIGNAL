#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

IN_DIR = Path("backend/debug/runs/real_tiles")
OUT_DIR = Path("backend/debug/runs/real_tiles_geotiff")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_geotiff(path: Path, arr: np.ndarray, bbox: tuple[float, float, float, float], crs: str, nodata=None, descriptions=None):
    h, w = arr.shape[-2], arr.shape[-1]
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    if arr.ndim == 2:
        data = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        data = arr
    else:
        raise ValueError(f"Unexpected array dims: {arr.shape}")

    count = data.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=2 if np.issubdtype(data.dtype, np.floating) else 1,
        tiled=True,
        blockxsize=256 if w >= 256 else w,
        blockysize=256 if h >= 256 else h,
    ) as dst:
        for i in range(count):
            dst.write(data[i], i + 1)
            if descriptions and i < len(descriptions) and descriptions[i]:
                dst.set_band_description(i + 1, descriptions[i])

rows = []
crs = "EPSG:4326"

for npz_path in sorted(IN_DIR.glob("*.npz")):
    tile_id = npz_path.stem
    z = np.load(npz_path)

    bbox = tuple(map(float, z["bbox"].tolist())) if "bbox" in z else None
    if bbox is None:
        raise RuntimeError(f"{npz_path}: missing bbox")

    edge = z["edgecomposite"].astype(np.float32)
    maxndvi = z["maxndvi"].astype(np.float32)
    meanndvi = z["meanndvi"].astype(np.float32)
    ndvistd = z["ndvistd"].astype(np.float32)
    scl = z["scl_median"].astype(np.uint8) if "scl_median" in z else None
    n_valid = int(z["n_valid_scenes"]) if "n_valid_scenes" in z else None

    comp = np.stack([edge, maxndvi, meanndvi, ndvistd], axis=0)

    write_geotiff(
        OUT_DIR / f"{tile_id}_composite.tif",
        comp,
        bbox=bbox,
        crs=crs,
        nodata=0.0,
        descriptions=["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"],
    )

    if scl is not None:
        write_geotiff(
            OUT_DIR / f"{tile_id}_scl_median.tif",
            scl,
            bbox=bbox,
            crs=crs,
            nodata=255,
            descriptions=["scl_median"],
        )

    rows.append({
        "tile_id": tile_id,
        "path_npz": str(npz_path),
        "path_composite_tif": str(OUT_DIR / f"{tile_id}_composite.tif"),
        "path_scl_tif": str(OUT_DIR / f"{tile_id}_scl_median.tif") if scl is not None else "",
        "bbox": bbox,
        "n_valid_scenes": n_valid,
        "edge_mean": float(edge.mean()),
        "edge_max": float(edge.max()),
        "maxndvi_mean": float(maxndvi.mean()),
        "maxndvi_max": float(maxndvi.max()),
        "ndvistd_mean": float(ndvistd.mean()),
        "ndvistd_max": float(ndvistd.max()),
    })

df = pd.DataFrame(rows).sort_values("tile_id")
df.to_csv(OUT_DIR / "real_tiles_qc.csv", index=False)
print(f"✅ Wrote GeoTIFFs + QC CSV to: {OUT_DIR.resolve()}")
