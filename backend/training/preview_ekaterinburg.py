#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

IN_DIR  = Path("backend/debug/runs/ekaterinburg_geotiff")
OUT_DIR = IN_DIR / "preview_png"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["edge_composite", "max_ndvi", "mean_ndvi", "ndvi_std"]
CMAPS = ["hot",            "RdYlGn",   "YlGn",      "plasma"]

for tif in sorted(IN_DIR.glob("*_composite.tif")):
    tile_id = tif.stem.replace("_composite", "")
    with rasterio.open(tif) as ds:
        arr = ds.read()                      # (4, H, W)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        band = arr[i]
        vmin, vmax = np.percentile(band[band > 0], [2, 98]) if (band > 0).any() else (0, 1)
        im = ax.imshow(band, cmap=CMAPS[i], vmin=vmin, vmax=vmax)
        ax.set_title(BANDS[i], fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(tile_id, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_png = OUT_DIR / f"{tile_id}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_png.name}")

print(f"\n🖼  previews → {OUT_DIR.resolve()}")
