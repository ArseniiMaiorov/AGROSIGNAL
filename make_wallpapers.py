#!/usr/bin/env python3
"""
Download satellite imagery for Perm and Solikamsk and create
wallpaper-quality PNG files for Samsung Galaxy S25 Ultra (1440 × 3088 px).
"""
import io
import math
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ── Target dimensions (S25 Ultra portrait) ──────────────────────────────────
WALLPAPER_W = 1440
WALLPAPER_H = 3088
TILE_SIZE   = 256

CITIES = {
    "perm": {
        "name_ru": "Пермь",
        "name_en": "Perm",
        "lat": 58.0097,
        "lon": 56.2430,
        "zoom": 13,
    },
    "solikamsk": {
        "name_ru": "Соликамск",
        "name_en": "Solikamsk",
        "lat": 59.6404,
        "lon": 56.7742,
        "zoom": 14,
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 AGROSIGNAL/1.0 "
        "(SUAI diploma project; satellite wallpaper generator)"
    )
}


# ── Tile math ────────────────────────────────────────────────────────────────

def deg2tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Top-left corner of a tile → (lat, lon)."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_r)
    return lat, lon


def fetch_tile(x: int, y: int, z: int, retries: int = 4) -> Image.Image:
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 1.5 ** attempt
            print(f"  retry {attempt+1} for {x},{y},{z} ({exc}) — wait {wait:.1f}s")
            time.sleep(wait)


# ── Download & stitch ────────────────────────────────────────────────────────

def download_mosaic(lat: float, lon: float, zoom: int) -> tuple[Image.Image, int, int, int, int]:
    """
    Download a tile grid large enough for WALLPAPER_W × WALLPAPER_H,
    centred on (lat, lon). Returns (image, start_tile_x, start_tile_y, cx_px, cy_px).
    cx_px/cy_px are the pixel coords of the centre within the mosaic.
    """
    tiles_w = math.ceil(WALLPAPER_W / TILE_SIZE) + 2  # a little padding
    tiles_h = math.ceil(WALLPAPER_H / TILE_SIZE) + 2

    cx, cy = deg2tile(lat, lon, zoom)
    start_x = cx - tiles_w // 2
    start_y = cy - tiles_h // 2

    mosaic = Image.new("RGB", (tiles_w * TILE_SIZE, tiles_h * TILE_SIZE))

    total = tiles_w * tiles_h
    done  = 0
    for col in range(tiles_w):
        for row in range(tiles_h):
            tx, ty = start_x + col, start_y + row
            print(f"  tile {done+1}/{total}  ({tx},{ty},{zoom})")
            tile = fetch_tile(tx, ty, zoom)
            mosaic.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
            done += 1
            time.sleep(0.05)        # polite rate-limit

    # Pixel position of the lat/lon centre within the mosaic
    cx_px = (cx - start_x) * TILE_SIZE + TILE_SIZE // 2
    cy_px = (cy - start_y) * TILE_SIZE + TILE_SIZE // 2

    return mosaic, cx_px, cy_px


def crop_centred(img: Image.Image, cx: int, cy: int, w: int, h: int) -> Image.Image:
    left = max(0, cx - w // 2)
    top  = max(0, cy - h // 2)
    right  = left + w
    bottom = top  + h
    # clamp
    if right > img.width:
        right = img.width
        left  = right - w
    if bottom > img.height:
        bottom = img.height
        top    = bottom - h
    return img.crop((left, top, right, bottom))


# ── Overlay ──────────────────────────────────────────────────────────────────

def add_overlay(img: Image.Image, city: dict) -> Image.Image:
    img = img.copy()
    W, H = img.size

    draw = ImageDraw.Draw(img)

    # ── bottom gradient vignette ─────────────────────────────────────────
    grad_h = int(H * 0.28)
    gradient = Image.new("RGBA", (W, grad_h))
    for y in range(grad_h):
        alpha = int(255 * (y / grad_h) ** 1.5)
        ImageDraw.Draw(gradient).line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
    base = img.convert("RGBA")
    base.paste(gradient, (0, H - grad_h), gradient)
    img = base.convert("RGB")
    draw = ImageDraw.Draw(img)

    # ── try to load a system font, fall back to default ──────────────────
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    font_large = font_small = None
    for fp in font_candidates:
        if Path(fp).exists():
            try:
                font_large = ImageFont.truetype(fp, size=120)
                font_small = ImageFont.truetype(fp, size=48)
                break
            except Exception:
                continue
    if font_large is None:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # ── city name (Russian) ──────────────────────────────────────────────
    label_ru = city["name_ru"]
    label_en = city["name_en"].upper()
    label_sub = "ПЕРМСКИЙ КРАЙ · РОССИЯ"

    # shadow
    shadow_offset = 4
    for dx, dy in [(-shadow_offset, -shadow_offset), (shadow_offset, shadow_offset),
                   (-shadow_offset, shadow_offset), (shadow_offset, -shadow_offset)]:
        draw.text((W // 2 + dx, H - 290 + dy), label_ru, font=font_large,
                  fill=(0, 0, 0, 180), anchor="mm")
    # main text
    draw.text((W // 2, H - 290), label_ru, font=font_large,
              fill=(255, 255, 255), anchor="mm")

    # subtitle
    draw.text((W // 2, H - 170), label_sub, font=font_small,
              fill=(200, 220, 255, 200), anchor="mm")

    # AGROSIGNAL watermark (small, top-right)
    wm_font_candidates = [fp for fp in font_candidates if Path(fp).exists()]
    wm_font = None
    if wm_font_candidates:
        try:
            wm_font = ImageFont.truetype(wm_font_candidates[0], size=32)
        except Exception:
            pass
    if wm_font is None:
        wm_font = ImageFont.load_default()

    draw.text((W - 24, 36), "AGROSIGNAL", font=wm_font,
              fill=(255, 255, 255, 120), anchor="rt")

    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def make_wallpaper(city_key: str, out_dir: Path) -> None:
    city = CITIES[city_key]
    print(f"\n{'='*60}")
    print(f"  City: {city['name_ru']}  ({city['lat']:.4f}, {city['lon']:.4f})  zoom={city['zoom']}")
    print(f"{'='*60}")

    print(f"\n[1/3] Downloading {WALLPAPER_W}×{WALLPAPER_H} satellite mosaic …")
    mosaic, cx_px, cy_px = download_mosaic(city["lat"], city["lon"], city["zoom"])

    print(f"\n[2/3] Cropping to wallpaper dimensions …")
    wallpaper = crop_centred(mosaic, cx_px, cy_px, WALLPAPER_W, WALLPAPER_H)

    print(f"\n[3/3] Adding overlay …")
    wallpaper = add_overlay(wallpaper, city)

    out_path = out_dir / f"wallpaper_{city_key}.png"
    wallpaper.save(str(out_path), format="PNG", optimize=False, compress_level=1)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    for city_key in ("perm", "solikamsk"):
        make_wallpaper(city_key, out_dir)

    print("\nDone! Both wallpapers saved to project root.")
