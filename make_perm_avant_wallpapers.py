#!/usr/bin/env python3
"""
Generate two premium smartphone wallpapers for Perm in a neo-brutalist /
cyber-elegance style while preserving the exact satellite topology.

Outputs:
  - wallpaper_perm_avant_perm_krai.png
  - wallpaper_perm_avant_59.png

Format: 1440 x 3200 (9:20 portrait)
"""
from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


WALLPAPER_W = 1440
WALLPAPER_H = 3200
TILE_SIZE = 256

CITY = {
    "name_ru": "Пермь",
    "name_en": "Perm",
    "region": "Пермский край",
    "code": "59",
    "lat": 58.0097,
    "lon": 56.2430,
    "zoom": 13,
}

APP_NAME = "TерраINFO"
BRAND_RU = "ТерраINFO"
OUTPUT_FULL = "wallpaper_perm_avant_perm_krai.png"
OUTPUT_CODE = "wallpaper_perm_avant_59.png"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 AGROSIGNAL/1.0 "
        "(SUAI diploma project; premium Perm wallpaper generator)"
    )
}

OBSIDIAN = np.array([10, 13, 19], dtype=np.float32) / 255.0
TAIGA_NEON = np.array([96, 255, 184], dtype=np.float32) / 255.0
RIVER_CYAN = np.array([86, 214, 255], dtype=np.float32) / 255.0
AMBER_METAL = np.array([255, 179, 82], dtype=np.float32) / 255.0
CHROME = np.array([224, 236, 246], dtype=np.float32) / 255.0
GLASS_FILL = (20, 24, 34, 150)
GLASS_OUTLINE = (180, 255, 233, 84)


@dataclass(frozen=True)
class Variant:
    filename: str
    region_line: str
    accent_label: str
    code_mode: bool


VARIANTS = (
    Variant(
        filename=OUTPUT_FULL,
        region_line="ПЕРМСКИЙ КРАЙ",
        accent_label="KAMA / TAIGA / PERMIAN",
        code_mode=False,
    ),
    Variant(
        filename=OUTPUT_CODE,
        region_line="59",
        accent_label="OKTMO 59 / KAMA / INDUSTRY",
        code_mode=True,
    ),
)


def deg2tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(x: int, y: int, z: int, session: requests.Session, retries: int = 4) -> Image.Image:
    url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    for attempt in range(retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    raise RuntimeError("unreachable")


def download_satellite_base(lat: float, lon: float, zoom: int) -> Image.Image:
    tiles_w = math.ceil(WALLPAPER_W / TILE_SIZE) + 2
    tiles_h = math.ceil(WALLPAPER_H / TILE_SIZE) + 2

    cx, cy = deg2tile(lat, lon, zoom)
    start_x = cx - tiles_w // 2
    start_y = cy - tiles_h // 2

    mosaic = Image.new("RGB", (tiles_w * TILE_SIZE, tiles_h * TILE_SIZE))
    session = requests.Session()

    total = tiles_w * tiles_h
    done = 0
    for col in range(tiles_w):
        for row in range(tiles_h):
            tx = start_x + col
            ty = start_y + row
            print(f"  tile {done + 1}/{total}: ({tx},{ty},{zoom})")
            tile = fetch_tile(tx, ty, zoom, session)
            mosaic.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
            done += 1
            time.sleep(0.05)

    cx_px = (cx - start_x) * TILE_SIZE + TILE_SIZE // 2
    cy_px = (cy - start_y) * TILE_SIZE + TILE_SIZE // 2

    left = max(0, cx_px - WALLPAPER_W // 2)
    top = max(0, cy_px - WALLPAPER_H // 2)
    right = left + WALLPAPER_W
    bottom = top + WALLPAPER_H
    if right > mosaic.width:
        right = mosaic.width
        left = right - WALLPAPER_W
    if bottom > mosaic.height:
        bottom = mosaic.height
        top = bottom - WALLPAPER_H
    return mosaic.crop((left, top, right, bottom))


def normalize(arr: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    lo = np.percentile(arr, p_lo)
    hi = np.percentile(arr, p_hi)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_masks(base: Image.Image) -> dict[str, np.ndarray]:
    arr = np.asarray(base).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    vegetation = normalize(g - (0.72 * r + 0.28 * b) + chroma * 0.20, 10, 98)
    water = normalize((b - g) * 0.9 + (b - r) * 1.4 + (0.42 - luma) * 0.9, 8, 99)
    mineral = normalize((r - g) * 1.2 + (0.62 - luma) * 0.8 + chroma * 0.15, 8, 98)

    grad_y, grad_x = np.gradient(luma)
    edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge = np.power(normalize(edge, 72, 99.8), 0.72)

    detail = np.asarray(base.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    detail = np.power(normalize(detail, 70, 99.5), 0.85)

    return {
        "luma": luma.astype(np.float32),
        "vegetation": vegetation,
        "water": water,
        "mineral": mineral,
        "edge": edge,
        "detail": detail,
    }


def make_base_skin(masks: dict[str, np.ndarray]) -> Image.Image:
    h, w = masks["luma"].shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    luma = masks["luma"]
    vegetation = masks["vegetation"]
    water = masks["water"]
    mineral = masks["mineral"]
    edge = masks["edge"]

    sheen = (np.sin((xx * 2.6 - yy * 1.9) * math.pi * 1.55) + 1.0) * 0.5
    glow_falloff = np.clip(1.0 - ((xx - 0.54) ** 2 * 1.4 + (yy - 0.48) ** 2 * 1.0), 0.25, 1.0)

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb += OBSIDIAN
    rgb += luma[..., None] * np.array([0.06, 0.08, 0.12], dtype=np.float32)
    rgb += vegetation[..., None] * (TAIGA_NEON * 0.22 + CHROME * 0.04)
    rgb += water[..., None] * (RIVER_CYAN * 0.20)
    rgb += mineral[..., None] * (AMBER_METAL * 0.10 + CHROME * 0.05)
    rgb += (edge * sheen * 0.10)[..., None] * CHROME
    rgb *= glow_falloff[..., None]

    vignette = 1.0 - 0.38 * ((xx - 0.52) ** 2 * 1.2 + (yy - 0.50) ** 2 * 1.6)
    rgb *= np.clip(vignette, 0.40, 1.0)[..., None]

    rng = np.random.default_rng(59)
    grain = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    rgb += grain * 0.010

    # Subtle scanline texture to push the "digital instrument" feeling.
    rgb *= 0.985 + 0.015 * ((np.arange(h)[:, None, None] % 4) / 3.0)

    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def mask_to_layer(mask: np.ndarray, color: tuple[int, int, int], blur_radius: int, strength: float) -> Image.Image:
    mask_u8 = np.clip(mask * 255.0 * strength, 0, 255).astype(np.uint8)
    alpha = Image.fromarray(mask_u8).filter(ImageFilter.GaussianBlur(blur_radius))
    layer = Image.new("RGBA", (alpha.width, alpha.height), color + (0,))
    layer.putalpha(alpha)
    return layer


def composite_style(base: Image.Image, masks: dict[str, np.ndarray]) -> Image.Image:
    styled = make_base_skin(masks).convert("RGBA")

    water_glow = mask_to_layer(np.power(masks["water"], 1.0), (76, 219, 255), 24, 0.85)
    veg_glow = mask_to_layer(np.power(masks["vegetation"], 1.2), (89, 255, 183), 18, 0.72)
    mineral_glow = mask_to_layer(np.power(masks["mineral"], 1.35), (255, 178, 70), 14, 0.55)

    line_mix = np.clip(masks["edge"] * 0.70 + masks["detail"] * 0.75, 0.0, 1.0)
    line_glow = mask_to_layer(np.power(line_mix, 0.72), (210, 240, 255), 8, 1.0)
    crisp_alpha = np.clip(np.power(line_mix, 0.86) * 255.0 * 0.58, 0, 255).astype(np.uint8)
    crisp = Image.new("RGBA", styled.size, (160, 255, 233, 0))
    crisp.putalpha(Image.fromarray(crisp_alpha))

    styled = Image.alpha_composite(styled, water_glow)
    styled = Image.alpha_composite(styled, veg_glow)
    styled = Image.alpha_composite(styled, mineral_glow)
    styled = Image.alpha_composite(styled, line_glow)
    styled = Image.alpha_composite(styled, crisp)

    # A restrained constructivist corner flare tied to the true edge field.
    edge_mask = Image.fromarray(np.clip(masks["edge"] * 255.0 * 0.55, 0, 255).astype(np.uint8))
    construct = Image.new("RGBA", styled.size, (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(construct)
    cdraw.rectangle((0, 0, 210, 22), fill=(255, 176, 72, 28))
    cdraw.rectangle((0, 0, 22, 350), fill=(255, 176, 72, 24))
    cdraw.rectangle((styled.width - 22, styled.height - 300, styled.width, styled.height), fill=(94, 255, 192, 20))
    construct.putalpha(ImageChops.multiply(construct.getchannel("A"), edge_mask.filter(ImageFilter.GaussianBlur(48))))
    styled = Image.alpha_composite(styled, construct)

    return styled.convert("RGB")


def choose_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_BOLD = [
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-Bold.ttf",
    "/usr/share/fonts/truetype/clear-sans/ClearSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_REG = [
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf",
    "/usr/share/fonts/truetype/clear-sans/ClearSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def add_glass_panel(img: Image.Image, box: tuple[int, int, int, int], radius: int = 42) -> Image.Image:
    base = img.convert("RGBA")
    x1, y1, x2, y2 = box
    panel_size = (x2 - x1, y2 - y1)
    panel_mask = rounded_mask(panel_size, radius)

    blurred = base.crop(box).filter(ImageFilter.GaussianBlur(24))
    tint = Image.new("RGBA", panel_size, GLASS_FILL)
    panel = Image.alpha_composite(blurred, tint)

    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle((1, 1, panel_size[0] - 2, panel_size[1] - 2), radius=radius, outline=GLASS_OUTLINE, width=2)
    panel_draw.line((36, 28, panel_size[0] - 48, 28), fill=(255, 255, 255, 46), width=2)
    panel_draw.line((28, panel_size[1] - 24, panel_size[0] - 60, panel_size[1] - 24), fill=(88, 255, 198, 32), width=1)

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(panel, (x1, y1), panel_mask)
    return Image.alpha_composite(base, layer).convert("RGB")


def draw_rotated_label(
    img: Image.Image,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    angle: int = 90,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    bbox = font.getbbox(text, anchor="lt")
    width = bbox[2] - bbox[0] + 20
    height = bbox[3] - bbox[1] + 20
    txt = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(txt).text(
        (10, 10),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    rotated = txt.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    layer.alpha_composite(rotated, xy)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def add_layout(img: Image.Image, variant: Variant) -> Image.Image:
    canvas = img.convert("RGBA")
    width, height = canvas.size

    font_city = choose_font(FONT_BOLD, 162)
    font_region = choose_font(FONT_BOLD, 92 if not variant.code_mode else 168)
    font_app = choose_font(FONT_BOLD, 60)
    font_meta = choose_font(FONT_REG, 30)
    font_brand = choose_font(FONT_REG, 32)
    font_code_bg = choose_font(FONT_BOLD, 580)

    # Large translucent subject code as a compositional mass.
    big_text = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    big_draw = ImageDraw.Draw(big_text)
    big_draw.text(
        (width - 48, height - 760),
        CITY["code"],
        font=font_code_bg,
        fill=(215, 233, 243, 22),
        anchor="rd",
        stroke_width=4,
        stroke_fill=(92, 255, 198, 18),
    )
    canvas = Image.alpha_composite(canvas, big_text)

    panel_box = (422, 2210, 1370, 2910)
    canvas = add_glass_panel(canvas.convert("RGB"), panel_box).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    x1, y1, x2, y2 = panel_box
    draw.text((x1 + 54, y1 + 88), CITY["name_ru"].upper(), font=font_city, fill=(244, 248, 252, 255))
    draw.text((x1 + 56, y1 + 270), variant.region_line, font=font_region, fill=(118, 255, 205, 246))
    draw.text((x1 + 58, y1 + 448), APP_NAME, font=font_app, fill=(255, 255, 255, 238))
    draw.text((x1 + 58, y1 + 520), variant.accent_label, font=font_meta, fill=(196, 208, 222, 220))
    draw.text((x1 + 58, y2 - 62), "SATELLITE DATA / EXACT TOPOLOGY / OLED 9:20", font=font_meta, fill=(150, 166, 182, 210))

    draw.line((x1 + 58, y1 + 418, x1 + 266, y1 + 418), fill=(255, 181, 87, 220), width=5)
    draw.line((x1 + 290, y1 + 418, x1 + 790, y1 + 418), fill=(96, 255, 184, 165), width=2)

    # Thin side brand label.
    canvas = draw_rotated_label(
        canvas.convert("RGB"),
        f"{BRAND_RU}™",
        (width - 88, 420),
        font_brand,
        fill=(232, 238, 244, 188),
        angle=90,
    ).convert("RGBA")

    # Vertical city marker with restrained glitch offsets.
    city_shadow = draw_rotated_label(
        canvas.convert("RGB"),
        CITY["name_ru"].upper(),
        (54, 340),
        choose_font(FONT_BOLD, 154),
        fill=(72, 223, 255, 44),
        angle=90,
    ).convert("RGBA")
    city_shadow = draw_rotated_label(
        city_shadow.convert("RGB"),
        CITY["name_ru"].upper(),
        (66, 352),
        choose_font(FONT_BOLD, 154),
        fill=(255, 186, 72, 34),
        angle=90,
    ).convert("RGBA")
    canvas = Image.alpha_composite(canvas, city_shadow)
    canvas = draw_rotated_label(
        canvas.convert("RGB"),
        CITY["name_ru"].upper(),
        (60, 346),
        choose_font(FONT_BOLD, 154),
        fill=(246, 248, 252, 176),
        angle=90,
    ).convert("RGBA")

    # Small coordinate line for technical confidence.
    draw = ImageDraw.Draw(canvas)
    draw.text((74, height - 78), "58.0097 N / 56.2430 E", font=font_meta, fill=(172, 182, 194, 186))

    return canvas.convert("RGB")


def main() -> None:
    print("Downloading exact satellite base for Perm...")
    satellite = download_satellite_base(CITY["lat"], CITY["lon"], CITY["zoom"])

    print("Building topology-preserving masks...")
    masks = build_masks(satellite)

    print("Applying premium visual skin...")
    styled = composite_style(satellite, masks)

    project_root = Path(__file__).resolve().parent
    for variant in VARIANTS:
        print(f"Rendering {variant.filename} ...")
        out = add_layout(styled, variant)
        out_path = project_root / variant.filename
        out.save(out_path, format="PNG", compress_level=1)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  saved -> {out_path.name} ({size_mb:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
