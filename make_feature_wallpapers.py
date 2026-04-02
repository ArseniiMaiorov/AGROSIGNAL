#!/usr/bin/env python3
"""
Feature-map wallpapers for Perm and Solikamsk.

Uses the TerraINFO application's own processing pipeline:
  SentinelHub client → temporal composite → edge_composite → render

Output: wallpaper_feature_perm.png, wallpaper_feature_solikamsk.png
        Samsung Galaxy S25 Ultra — 1440 × 3088 px
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from math import cos, radians
from pathlib import Path
from typing import Any

import numpy as np

# ── Project bootstrap ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR  = PROJECT_ROOT / "backend"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Prevent SQLAlchemy from requiring a real DB at import time
os.environ.setdefault("DATABASE_URL",      "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

# ── PIL / matplotlib ─────────────────────────────────────────────────────────
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ── Wallpaper dimensions ─────────────────────────────────────────────────────
WP_W  = 1440
WP_H  = 3088
# Fetch area: 1440 × 2400 px at 10m/px → 14.4 km × 24 km (within SH 2500px limit)
FETCH_W = 1440
FETCH_H = 2400

# ── City definitions ─────────────────────────────────────────────────────────
CITIES = {
    "perm": {
        "name_ru":  "Пермь",
        "name_en":  "Perm",
        "lat":      58.010,
        "lon":      56.230,
        "region":   "Пермский край",
        "zoom_note": "масштаб ~14×24 км · Sentinel-2 · 10 м/пкс",
    },
    "solikamsk": {
        "name_ru":  "Соликамск",
        "name_en":  "Solikamsk",
        "lat":      59.640,
        "lon":      56.774,
        "region":   "Пермский край",
        "zoom_note": "масштаб ~14×24 км · Sentinel-2 · 10 м/пкс",
    },
}

# Seasonal time windows (2025) — same pattern as the training pipeline
TIME_WINDOWS = [
    ("2025-04-15T00:00:00Z", "2025-05-15T23:59:59Z"),
    ("2025-05-15T00:00:00Z", "2025-06-20T23:59:59Z"),
    ("2025-06-20T00:00:00Z", "2025-07-20T23:59:59Z"),
    ("2025-07-20T00:00:00Z", "2025-08-20T23:59:59Z"),
    ("2025-08-20T00:00:00Z", "2025-09-20T23:59:59Z"),
    ("2025-09-20T00:00:00Z", "2025-10-15T23:59:59Z"),
]
MAX_CLOUD_PCT = 60


# ── Helpers ──────────────────────────────────────────────────────────────────

def bbox_from_center_portrait(lat: float, lon: float) -> tuple[float, float, float, float]:
    """Compute bbox for FETCH_W × FETCH_H at 10m/px centred on (lat, lon)."""
    half_w_km = FETCH_W * 0.010 / 2           # px → km
    half_h_km = FETCH_H * 0.010 / 2
    dlat = half_h_km / 110.574
    dlon = half_w_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def kget(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of {keys} in {list(d.keys())}")


# ── Import project modules ───────────────────────────────────────────────────

def _import_project():
    from core.config import get_settings
    settings = get_settings()

    from providers.sentinelhub.client import SentinelHubClient
    client = SentinelHubClient()

    from processing.fields.indices   import compute_all_indices
    from processing.fields.composite import build_valid_mask_from_scl, select_dates_by_coverage
    from processing.fields.temporal_composite import build_multiyear_composite

    return settings, client, compute_all_indices, build_valid_mask_from_scl, \
           select_dates_by_coverage, build_multiyear_composite


# ── SentinelHub fetch ────────────────────────────────────────────────────────

async def fetch_scene(client, bbox, time_from: str, time_to: str) -> dict[str, Any] | None:
    """Fetch one S2 scene; return dict with band arrays or None on failure."""
    fetch_fn = getattr(client, "fetch_tile", None) or getattr(client, "fetchtile")
    try:
        result = await fetch_fn(bbox, time_from, time_to, FETCH_W, FETCH_H,
                                max_cloud_pct=MAX_CLOUD_PCT)
        return result
    except Exception as exc:
        print(f"    skip {time_from[:10]}: {str(exc)[:120]}")
        return None


async def fetch_all_scenes(client, bbox) -> tuple[dict[str, list], list]:
    """Fetch all seasonal windows, return (band_lists, scl_list)."""
    band_keys  = ["B2", "B3", "B4", "B8", "B11", "B12"]
    band_lists = {k: [] for k in band_keys}
    scl_list   = []

    for i, (tf, tt) in enumerate(TIME_WINDOWS, 1):
        print(f"  [{i}/{len(TIME_WINDOWS)}] fetching {tf[:10]} → {tt[:10]} …")
        result = await fetch_scene(client, bbox, tf, tt)
        if result is None:
            continue

        ok = True
        for k in band_keys:
            if k not in result:
                print(f"    missing band {k}, skip window")
                ok = False
                break
        if not ok:
            continue

        for k in band_keys:
            band_lists[k].append(np.asarray(result[k], dtype=np.float32))

        scl = result.get("SCL", None)
        if scl is None:
            scl = np.full((FETCH_H, FETCH_W), 4, dtype=np.uint8)
        scl_list.append(np.asarray(scl, dtype=np.uint8))
        print(f"    ✓ scene {tf[:10]} OK  (valid_scenes={len(scl_list)})")

    return band_lists, scl_list


# ── Processing ───────────────────────────────────────────────────────────────

def compute_features(
    band_lists, scl_list,
    compute_all_indices,
    build_valid_mask_from_scl,
    select_dates_by_coverage,
    build_multiyear_composite,
    settings,
) -> dict[str, np.ndarray]:
    """Run the application's temporal composite pipeline → feature maps."""
    bands = {k: np.stack(v, axis=0) for k, v in band_lists.items() if v}
    scl   = np.stack(scl_list, axis=0).astype(np.uint8)

    valid_mask = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)  # (T,H,W)
    indices    = compute_all_indices(bands)
    ndvi       = np.asarray(indices["NDVI"], dtype=np.float32)

    # Select best dates by coverage
    sel_out = select_dates_by_coverage(valid_mask, n_dates=min(6, valid_mask.shape[0]),
                                       min_valid_pct=0.30, min_good_dates=2,
                                       return_metadata=True)
    if isinstance(sel_out, tuple):
        selected, _ = sel_out
    else:
        selected = sel_out
    selected = np.asarray(selected)
    if selected.size == 0:
        selected = np.arange(len(scl_list))

    ndvi_sel  = ndvi[selected]
    valid_sel = valid_mask[selected]
    edge_bands = {k: bands[k][selected] for k in ["B2", "B3", "B4", "B8"]}
    edge_bands["ndvi"] = ndvi_sel

    comp = build_multiyear_composite(
        ndvi_stack=ndvi_sel,
        valid_mask=valid_sel,
        edge_bands=edge_bands,
        cfg=settings,
    )

    edge     = np.nan_to_num(np.asarray(kget(comp, "edge_composite",   "edgecomposite"),  dtype=np.float32))
    maxndvi  = np.nan_to_num(np.asarray(kget(comp, "max_ndvi",         "maxndvi"),         dtype=np.float32))
    meanndvi = np.nan_to_num(np.asarray(kget(comp, "mean_ndvi",        "meanndvi"),        dtype=np.float32))
    ndvistd  = np.nan_to_num(np.asarray(kget(comp, "ndvi_std",         "ndvistd"),         dtype=np.float32))

    # Also keep raw RGB for background
    r = np.nan_to_num(np.nanmedian(bands.get("B4", np.zeros((1,FETCH_H,FETCH_W))), axis=0))
    g = np.nan_to_num(np.nanmedian(bands.get("B3", np.zeros((1,FETCH_H,FETCH_W))), axis=0))
    b = np.nan_to_num(np.nanmedian(bands.get("B2", np.zeros((1,FETCH_H,FETCH_W))), axis=0))
    nir  = np.nan_to_num(np.nanmedian(bands.get("B8",  np.zeros((1,FETCH_H,FETCH_W))), axis=0))
    swir = np.nan_to_num(np.nanmedian(bands.get("B11", np.zeros((1,FETCH_H,FETCH_W))), axis=0))

    return {
        "edge_composite": edge,
        "max_ndvi":       maxndvi,
        "mean_ndvi":      meanndvi,
        "ndvi_std":       ndvistd,
        "rgb_r": r, "rgb_g": g, "rgb_b": b,
        "nir": nir, "swir": swir,
    }


# ── Render ───────────────────────────────────────────────────────────────────

def _clip_normalize(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


# Custom TerraINFO palette: dark-blue → teal → yellow (similar to viridis/plasma)
_AGRO_COLORS = [
    (0.02, 0.02, 0.10),   # deep navy (background / no-edge)
    (0.04, 0.12, 0.30),   # dark blue
    (0.05, 0.30, 0.55),   # medium blue
    (0.05, 0.55, 0.65),   # teal
    (0.20, 0.75, 0.50),   # green-teal
    (0.70, 0.90, 0.20),   # yellow-green
    (1.00, 0.95, 0.10),   # bright yellow (strong edges)
]
AGRO_CMAP = LinearSegmentedColormap.from_list("terrainfo", _AGRO_COLORS, N=512)


def render_feature_map(feats: dict[str, np.ndarray]) -> np.ndarray:
    """
    Render a beautiful feature-map composite image at FETCH_W × FETCH_H.

    Layers (bottom → top):
      1. False-colour background: SWIR-NIR-R composite (shows field structure)
      2. Edge composite overlay with TerraINFO custom colormap
      3. NDVI vibrancy mask to boost vegetated areas
    """
    H, W = FETCH_H, FETCH_W
    edge = feats["edge_composite"]
    ndvi = feats["max_ndvi"]
    r, g, b  = feats["rgb_r"], feats["rgb_g"], feats["rgb_b"]
    nir, swir = feats["nir"], feats["swir"]

    # ── layer 1: false-colour background (SWIR/NIR/R → visual contrast) ──
    bg_r = _clip_normalize(swir, 2, 98)   # SWIR → Red channel (bare soil bright)
    bg_g = _clip_normalize(nir,  2, 98)   # NIR  → Green channel (vegetation bright)
    bg_b = _clip_normalize(r,    2, 98)   # Red  → Blue channel

    bg = np.stack([bg_r, bg_g, bg_b], axis=-1)   # (H,W,3) float [0,1]
    # Darken background so edge overlay pops
    bg = np.power(bg, 1.4) * 0.55

    # ── layer 2: edge composite → colormap ──────────────────────────────
    edge_norm = _clip_normalize(edge, 0, 97)
    # Gamma to boost faint edges
    edge_boost = np.power(edge_norm, 0.65)

    rgba_edge = AGRO_CMAP(edge_boost)                 # (H,W,4) float
    rgb_edge  = rgba_edge[..., :3]                    # (H,W,3)

    # Alpha blending: stronger edges → more visible
    alpha_edge = np.clip(edge_boost[..., None] * 1.2, 0, 1)

    # ── layer 3: NDVI vibrancy boost for crop areas ──────────────────────
    ndvi_norm = np.clip((ndvi - 0.1) / 0.7, 0, 1)
    # Slight green tint on high-NDVI pixels to show vegetation context
    veg_tint = np.stack([
        np.zeros_like(ndvi_norm),
        ndvi_norm * 0.15,
        ndvi_norm * 0.05,
    ], axis=-1)

    # ── composite ────────────────────────────────────────────────────────
    composite = bg * (1 - alpha_edge) + rgb_edge * alpha_edge + veg_tint
    composite = np.clip(composite, 0, 1)

    # Mild vignette (edges of image slightly darker)
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    vignette = 1.0 - 0.30 * (xx**2 + yy**2)
    composite *= np.clip(vignette, 0, 1)[..., None]

    out = (composite * 255).clip(0, 255).astype(np.uint8)
    return out


def build_wallpaper(
    feature_img: np.ndarray,
    city: dict,
    out_path: Path,
) -> None:
    """
    Compose 1440×3088 wallpaper:
      - Top header (688 px): city name + feature channel label + TerraINFO brand
      - Middle (2400 px): rendered feature map
    """
    H_HEADER = WP_H - FETCH_H          # 688 px
    assert H_HEADER > 0

    # ── header canvas ────────────────────────────────────────────────────
    header = Image.new("RGB", (WP_W, H_HEADER), color=(3, 5, 18))
    draw   = ImageDraw.Draw(header)

    # Subtle horizontal gradient
    for y in range(H_HEADER):
        alpha = int(20 * (y / H_HEADER))
        draw.line([(0, y), (WP_W, y)], fill=(3 + alpha, 8 + alpha, 30 + alpha * 2))

    # Thin accent line at bottom of header
    draw.rectangle([(0, H_HEADER - 3), (WP_W, H_HEADER)], fill=(30, 180, 200))

    # Fonts
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    ]
    reg_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    font_b = font_r = None
    for fp in font_candidates:
        if Path(fp).exists():
            try:
                font_b = ImageFont.truetype(fp, size=110)
                font_b_sm = ImageFont.truetype(fp, size=38)
                font_b_xs = ImageFont.truetype(fp, size=28)
                break
            except Exception:
                pass
    for fp in reg_candidates:
        if Path(fp).exists():
            try:
                font_r = ImageFont.truetype(fp, size=36)
                break
            except Exception:
                pass
    if font_b is None:
        font_b = font_b_sm = font_b_xs = font_r = ImageFont.load_default()

    # City name (big)
    cx = WP_W // 2
    draw.text((cx + 3, 93), city["name_ru"], font=font_b, fill=(0, 0, 0, 100), anchor="mm")
    draw.text((cx, 90), city["name_ru"], font=font_b, fill=(255, 255, 255), anchor="mm")

    # Region
    draw.text((cx, 170), city["region"].upper(), font=font_b_sm,
              fill=(120, 200, 220), anchor="mm")

    # Feature channel label
    draw.text((cx, 230), "edge_composite", font=font_b_sm,
              fill=(80, 230, 160), anchor="mm")

    # Technical description
    draw.text((cx, 285), "Пространственно-временной признак | TerraINFO v3", font=font_b_xs,
              fill=(140, 160, 180), anchor="mm")

    # Scale note
    draw.text((cx, 330), city["zoom_note"], font=font_b_xs,
              fill=(100, 140, 160), anchor="mm")

    # Colormap legend bar
    BAR_W = 600
    BAR_H = 18
    bar_x = (WP_W - BAR_W) // 2
    bar_y = H_HEADER - 100
    for px in range(BAR_W):
        val = px / BAR_W
        rgba = AGRO_CMAP(val)
        r_v = int(rgba[0] * 255)
        g_v = int(rgba[1] * 255)
        b_v = int(rgba[2] * 255)
        draw.line([(bar_x + px, bar_y), (bar_x + px, bar_y + BAR_H)], fill=(r_v, g_v, b_v))

    draw.text((bar_x - 8,       bar_y + BAR_H // 2), "0",    font=font_b_xs, fill=(180,180,180), anchor="rm")
    draw.text((bar_x + BAR_W + 8, bar_y + BAR_H // 2), "1.0", font=font_b_xs, fill=(180,180,180), anchor="lm")
    draw.text((bar_x + BAR_W // 2, bar_y - 14), "Яркость границ поля", font=font_b_xs,
              fill=(160, 160, 180), anchor="mm")

    # TerraINFO watermark
    draw.text((WP_W - 24, 36), "TerraINFO", font=font_b_xs, fill=(60, 120, 140), anchor="rt")

    # ── assemble wallpaper ───────────────────────────────────────────────
    feat_pil = Image.fromarray(feature_img, mode="RGB")

    wallpaper = Image.new("RGB", (WP_W, WP_H), color=(3, 5, 18))
    wallpaper.paste(header, (0, 0))
    wallpaper.paste(feat_pil, (0, H_HEADER))

    # Thin scanline at seam
    seam_draw = ImageDraw.Draw(wallpaper)
    seam_draw.rectangle([(0, H_HEADER), (WP_W, H_HEADER + 1)], fill=(30, 180, 200))

    wallpaper.save(str(out_path), format="PNG", compress_level=1)
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"  → saved {out_path.name}  ({mb:.1f} MB)")


# ── Main pipeline ────────────────────────────────────────────────────────────

async def process_city(city_key: str, city: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  {city['name_ru']} ({city['lat']}, {city['lon']})")
    print(f"{'='*65}")

    (settings, client,
     compute_all_indices,
     build_valid_mask_from_scl,
     select_dates_by_coverage,
     build_multiyear_composite) = _import_project()

    bbox = bbox_from_center_portrait(city["lat"], city["lon"])
    print(f"  bbox: lon=[{bbox[0]:.4f},{bbox[2]:.4f}]  lat=[{bbox[1]:.4f},{bbox[3]:.4f}]")
    print(f"  fetch: {FETCH_W}×{FETCH_H} px @ 10m/px\n")

    print("[1/3] Fetching multi-temporal Sentinel-2 scenes …")
    band_lists, scl_list = await fetch_all_scenes(client, bbox)

    if len(scl_list) < 2:
        raise RuntimeError(f"Only {len(scl_list)} valid scene(s) — need at least 2. "
                           "Check SentinelHub quota or try again.")

    print(f"\n  {len(scl_list)} valid scenes collected.")

    print("\n[2/3] Computing temporal composite & edge_composite via app pipeline …")
    feats = compute_features(
        band_lists, scl_list,
        compute_all_indices,
        build_valid_mask_from_scl,
        select_dates_by_coverage,
        build_multiyear_composite,
        settings,
    )
    print(f"  edge_composite: min={feats['edge_composite'].min():.3f}  "
          f"max={feats['edge_composite'].max():.3f}  "
          f"mean={feats['edge_composite'].mean():.3f}")

    print("\n[3/3] Rendering wallpaper …")
    feature_img = render_feature_map(feats)

    out_path = PROJECT_ROOT / f"wallpaper_feature_{city_key}.png"
    build_wallpaper(feature_img, city, out_path)


async def main() -> None:
    for city_key, city in CITIES.items():
        await process_city(city_key, city)

    print("\nDone! Wallpapers saved to project root.")


if __name__ == "__main__":
    asyncio.run(main())
