#!/usr/bin/env python3
"""
Premium data-art wallpapers for Пермь (Region 59, Пермский край).

Two versions:
  wallpaper_perm_permskiy_kray.png  — full signature "ПЕРМСКИЙ КРАЙ"
  wallpaper_perm_59.png             — typographic "59"

Visual DNA  : Neo-Brutalism × Cyber-Elegance × Permian Copper
Cultural code: Пермский период (Permian geological era, named after this city) →
               amber/copper chromatic identity. Кама (Kama River) → electric-cyan veins.
               Звериный стиль (Permian Animal Style) → angular geometric accents.
Data source  : TerraINFO pipeline — real Sentinel-2, 10 m/px, Пермский край AOI
"""
from __future__ import annotations

import os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
TILE_NPZ = ROOT / "backend/debug/runs/real_tiles/permkrai_02.npz"

# ── Canvas ───────────────────────────────────────────────────────────────────
WP_W, WP_H = 1440, 3088          # S25 Ultra portrait

# ── Cultural palette — "Пермская медь / Permian Copper" ─────────────────────
# Geological amber/copper gradient for edge composite
_PERMIAN_STOPS = [
    (0.000, (0.004, 0.008, 0.016)),   # obsidian void
    (0.040, (0.060, 0.015, 0.002)),   # deep ember
    (0.130, (0.220, 0.060, 0.008)),   # dark copper
    (0.280, (0.490, 0.170, 0.025)),   # copper shadow
    (0.460, (0.720, 0.310, 0.055)),   # bright copper
    (0.640, (0.880, 0.500, 0.100)),   # warm amber
    (0.800, (0.970, 0.720, 0.210)),   # Permian gold
    (0.920, (1.000, 0.920, 0.620)),   # pale gold
    (1.000, (1.000, 0.980, 0.940)),   # white-gold peak
]
PERMIAN_CMAP = LinearSegmentedColormap.from_list(
    "permian_copper",
    [(t, c) for t, c in _PERMIAN_STOPS],
    N=1024,
)

# Kama River — electric cyan
KAMA_BLUE_LO  = np.array([0.000, 0.200, 0.600])   # deep Kama
KAMA_BLUE_HI  = np.array([0.000, 0.900, 1.000])   # surface shimmer
KAMA_ACCENT   = np.array([0.400, 1.000, 0.980])   # highlight

# Vegetation — OLED-dark emerald
VEG_DARK  = np.array([0.000, 0.060, 0.030])
VEG_LIGHT = np.array([0.050, 0.200, 0.090])

# ── Font discovery ────────────────────────────────────────────────────────────
_FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
]
_FONT_REG_CANDIDATES = [
    "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]
_FONT_MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/croscore/Cousine-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

def _find_font(size: int, candidates: list) -> ImageFont.FreeTypeFont:
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Data helpers ──────────────────────────────────────────────────────────────

def _pct_clip(arr: np.ndarray, lo: float = 1.5, hi: float = 98.5) -> np.ndarray:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    vlo, vhi = np.percentile(valid, lo), np.percentile(valid, hi)
    if vhi - vlo < 1e-7:
        return np.zeros_like(arr)
    return np.clip((arr - vlo) / (vhi - vlo), 0.0, 1.0).astype(np.float32)


def _gamma(arr: np.ndarray, g: float) -> np.ndarray:
    return np.power(np.clip(arr, 0, 1), g).astype(np.float32)


def _upscale_bicubic(arr_2d: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bicubic upscale 2-D float32 array → (h, w)."""
    pil = Image.fromarray((arr_2d * 255).clip(0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((w, h), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _upscale_rgb(arr_3d: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bicubic upscale (H,W,3) float32 array → (h, w, 3)."""
    pil = Image.fromarray((arr_3d * 255).clip(0, 255).astype(np.uint8), mode="RGB")
    pil = pil.resize((w, h), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


# ── Core render ───────────────────────────────────────────────────────────────

def render_feature_layer(
    edge: np.ndarray,
    maxndvi: np.ndarray,
    ndwi: np.ndarray,
    mndwi: np.ndarray,
    r_med: np.ndarray,
    g_med: np.ndarray,
    b_med: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Render the multi-layer premium data visualization at (out_h, out_w, 3).

    Layers (bottom → top):
      L0  Obsidian base
      L1  False-color SWIR/NIR/G — geometric terrain skeleton (very dark)
      L2  Vegetation glow (NDVI) — OLED emerald
      L3  Edge composite — Permian Copper cmap
      L4  Kama River (NDWI water mask) — electric cyan
      L5  Vignette
      L6  Subtle glitch scanlines
    """
    # upscale all inputs to output resolution
    def _up(a): return _upscale_bicubic(a, out_w, out_h)
    ec   = _up(_pct_clip(np.nan_to_num(edge,   nan=0.0), 0, 97))
    ndvi = _up(_pct_clip(np.nan_to_num(maxndvi, nan=0.0), 5, 95))
    wtr  = _up(np.clip((np.nan_to_num(ndwi, nan=-1.0) + np.nan_to_num(mndwi, nan=-1.0)) / 2.0, -1, 1))
    r_u  = _up(_pct_clip(np.nan_to_num(r_med,  nan=0), 2, 98))
    g_u  = _up(_pct_clip(np.nan_to_num(g_med,  nan=0), 2, 98))
    nir_u= _up(_pct_clip(np.nan_to_num(nir,    nan=0), 2, 98))
    sw_u = _up(_pct_clip(np.nan_to_num(swir,   nan=0), 2, 98))

    # ── L0 ── obsidian background ────────────────────────────────────────
    base = np.full((out_h, out_w, 3), fill_value=[0.004, 0.008, 0.016], dtype=np.float32)

    # ── L1 ── false-colour terrain skeleton (SWIR·NIR·G) ─────────────────
    # SWIR → highlights bare soil / built-up  (urban boundaries sharp)
    # NIR  → vegetation structure
    # G    → reflectance texture
    fg = np.stack([sw_u, nir_u, g_u], axis=-1)   # (H,W,3)
    fg = _gamma(fg, 1.8) * 0.18                   # very dark, just texture
    comp = base + fg

    # ── L2 ── vegetation emerald glow ────────────────────────────────────
    veg = np.clip((ndvi - 0.25) / 0.55, 0, 1)   # crop/forest positive
    veg_glow = np.power(veg, 1.6)[..., None]
    veg_col  = VEG_DARK + (VEG_LIGHT - VEG_DARK) * veg_glow
    # alpha proportional to NDVI strength
    va = veg_glow * 0.55
    comp = comp * (1 - va) + veg_col * va

    # ── L3 ── edge composite — Permian Copper ────────────────────────────
    # Gamma to boost faint boundaries (the "heartbeat" of the map)
    ec_boost = _gamma(ec, 0.55)
    rgba_ec  = PERMIAN_CMAP(ec_boost)              # (H,W,4) float
    rgb_ec   = rgba_ec[..., :3].astype(np.float32)

    # Non-linear alpha: only truly significant edges fully opaque
    ea = np.clip(ec_boost * 1.3, 0, 1)[..., None]
    # Min alpha so even faint traces are visible (moody glow)
    ea = np.maximum(ea, ec_boost[..., None] * 0.35)

    comp = comp * (1 - ea) + rgb_ec * ea

    # ── L4 ── Кама / Kama River electric cyan highlight ──────────────────
    # Use combined NDWI+MNDWI; strong positive → open water
    water_raw = np.clip((wtr - 0.04) / 0.40, 0, 1)   # 0..1 water confidence
    # Smooth the water mask so rivers look like flowing neon
    water_pil = Image.fromarray((water_raw * 255).clip(0,255).astype(np.uint8), mode="L")
    water_pil = water_pil.filter(ImageFilter.GaussianBlur(radius=2))
    water_sm  = np.asarray(water_pil, dtype=np.float32) / 255.0

    water_col = (KAMA_BLUE_LO + (KAMA_BLUE_HI - KAMA_BLUE_LO) * water_sm[..., None])
    wa         = (water_sm * 1.15).clip(0, 1)[..., None]
    # Add a fringe glow (halo) around water bodies
    water_halo_pil = water_pil.filter(ImageFilter.GaussianBlur(radius=7))
    water_halo = np.asarray(water_halo_pil, dtype=np.float32) / 255.0 * 0.35
    halo_col   = KAMA_ACCENT
    comp = comp + halo_col * water_halo[..., None]  # additive glow
    comp = comp * (1 - wa) + water_col * wa
    comp = np.clip(comp, 0, 1)

    # ── L5 ── vignette (OLED edge darkening) ─────────────────────────────
    yy = np.linspace(-1, 1, out_h)[:, None]
    xx = np.linspace(-1, 1, out_w)[None, :]
    vig = 1.0 - 0.42 * (xx**2 * 0.5 + yy**2 * 0.7)
    comp *= np.clip(vig, 0.0, 1.0)[..., None]

    # ── L6 ── neo-brutalist glitch scanlines ─────────────────────────────
    # Thin horizontal copper-tinted scans at irregular intervals
    rng = np.random.default_rng(59)   # seed = region code!
    n_scans = 18
    scan_ys = rng.integers(0, out_h, size=n_scans)
    scan_intensity = rng.uniform(0.04, 0.12, size=n_scans)
    copper_accent = np.array([0.88, 0.45, 0.10], dtype=np.float32)
    for y, intensity in zip(scan_ys, scan_intensity):
        if 0 <= y < out_h:
            comp[y, :] = np.clip(comp[y, :] + copper_accent * intensity, 0, 1)
            # tiny horizontal shift on adjacent rows
            shift = rng.integers(-2, 3)
            if shift and 0 < y < out_h - 1:
                comp[y, :] = np.roll(comp[y, :], shift, axis=0)

    return np.clip(comp * 255, 0, 255).astype(np.uint8)


# ── Glassmorphism panel ───────────────────────────────────────────────────────

def glass_panel(
    wallpaper_pil: Image.Image,
    x: int, y: int, w: int, h: int,
    blur_r: int = 18,
    dark_alpha: int = 170,
    border_color: tuple = (50, 200, 220),
    border_w: int = 1,
) -> Image.Image:
    """Paste a frosted-glass rectangle onto wallpaper_pil (in-place crop + blur)."""
    region = wallpaper_pil.crop((x, y, x + w, y + h))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_r))
    # darken
    dark = Image.new("RGBA", (w, h), (0, 5, 15, dark_alpha))
    blurred = blurred.convert("RGBA")
    blurred = Image.alpha_composite(blurred, dark)
    wallpaper_pil.paste(blurred.convert("RGB"), (x, y))
    # thin border
    draw = ImageDraw.Draw(wallpaper_pil)
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=border_color, width=border_w)
    return wallpaper_pil


# ── Decorative elements ────────────────────────────────────────────────────────

def draw_permian_accent(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, color: tuple) -> None:
    """
    Abstract Permian Animal Style marker — a geometric diamond/cross
    inspired by the ancient Finno-Ugric Komi-Perm ornamental tradition.
    """
    s = size
    # Outer diamond
    pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
    draw.polygon(pts, outline=color, fill=None)
    # Inner cross
    draw.line([(cx, cy - s // 2), (cx, cy + s // 2)], fill=color, width=1)
    draw.line([(cx - s // 2, cy), (cx + s // 2, cy)], fill=color, width=1)
    # Corner dots
    for dx, dy in [(0, -s), (s, 0), (0, s), (-s, 0)]:
        draw.ellipse(
            [(cx + dx - 2, cy + dy - 2), (cx + dx + 2, cy + dy + 2)],
            fill=color
        )


def draw_grid_lines(draw: ImageDraw.ImageDraw, w: int, h: int, spacing: int = 90,
                    color: tuple = (20, 30, 40), line_w: int = 1) -> None:
    """Subtle neo-brutalist coordinate grid."""
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=color, width=line_w)
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=color, width=line_w)


def draw_data_label(draw: ImageDraw.ImageDraw, font_mono, x: int, y: int, color: tuple) -> None:
    """Technical data provenance block — monospaced, small."""
    lines = [
        "SRC  // SENTINEL-2A·B · TerraINFO®",
        "AOI  // 58°N 56°E · PERMKRAI",
        "IDX  // edge_composite · v3_16ch",
        "RES  // 10 m/px · 8 SCENES",
    ]
    for i, ln in enumerate(lines):
        draw.text((x, y + i * 24), ln, font=font_mono, fill=color)


# ── VERSION 1 — "ПЕРМСКИЙ КРАЙ" ──────────────────────────────────────────────

def build_v1(feature_layer: np.ndarray, out_path: Path) -> None:
    """
    Layout:
      ┌────────────────────────────────────┐
      │  HEADER (glass overlay, top 740px) │  ← overlays top of map
      │  TерраINFO™    [data label right]  │
      │                                    │
      │  П Е Р М Ь                         │  ← massive, full-width
      │  ── accent line ──                 │
      │  ПЕРМСКИЙ КРАЙ                     │
      │  Пермский период · Кама · Урал     │  ← cultural subtitle
      └────────────────────────────────────┘
      [FEATURE MAP — full bleed continues below]
      ┌────────────────────────────────────┐
      │  FOOTER glass strip (88px)         │
      │  TerraINFO®     edge_composite     │
      └────────────────────────────────────┘
    """
    HEADER_H = 700
    FOOTER_H =  88

    # ── base canvas = feature layer scaled to wallpaper ──────────────────
    wp = Image.fromarray(feature_layer, mode="RGB").resize(
        (WP_W, WP_H), Image.BICUBIC
    )

    # ── overlay grid (subtle depth) ───────────────────────────────────────
    draw = ImageDraw.Draw(wp)
    draw_grid_lines(draw, WP_W, WP_H, spacing=90, color=(12, 18, 28), line_w=1)

    # ── header glass panel ────────────────────────────────────────────────
    wp = glass_panel(wp, 0, 0, WP_W, HEADER_H, blur_r=22, dark_alpha=185,
                     border_color=(40, 160, 180), border_w=0)

    # ── footer glass panel ────────────────────────────────────────────────
    wp = glass_panel(wp, 0, WP_H - FOOTER_H, WP_W, FOOTER_H,
                     blur_r=14, dark_alpha=200, border_color=(40, 160, 180), border_w=0)

    draw = ImageDraw.Draw(wp)

    # ── fonts ─────────────────────────────────────────────────────────────
    f_city   = _find_font(220, _FONT_BOLD_CANDIDATES)   # ПЕРМЬ
    f_region = _find_font(72,  _FONT_BOLD_CANDIDATES)   # ПЕРМСКИЙ КРАЙ
    f_sub    = _find_font(36,  _FONT_REG_CANDIDATES)    # subtitle
    f_brand  = _find_font(34,  _FONT_BOLD_CANDIDATES)   # TерраINFO™
    f_mono   = _find_font(22,  _FONT_MONO_CANDIDATES)   # data label
    f_footer = _find_font(28,  _FONT_REG_CANDIDATES)
    f_footer_b = _find_font(28, _FONT_BOLD_CANDIDATES)

    COPPER_DIM  = (120,  60,  10)
    COPPER_MID  = (220, 120,  40)
    COPPER_BRIGHT = (245, 190, 80)
    CYAN        = ( 40, 200, 215)
    CYAN_DIM    = ( 20, 110, 130)
    WHITE       = (255, 255, 255)
    OFFWHITE    = (235, 235, 230)
    GHOST       = ( 90,  90,  90)

    # ── TерраINFO™ brand (top-left) ──────────────────────────────────────
    draw.text((52, 52), "TерраINFO™", font=f_brand, fill=CYAN)

    # ── corner Permian accent marks ───────────────────────────────────────
    draw_permian_accent(draw, WP_W - 72, 72,   20, COPPER_DIM)
    draw_permian_accent(draw, WP_W - 72, WP_H - 72, 20, COPPER_DIM)
    draw_permian_accent(draw, 72,        WP_H - 72, 20, COPPER_DIM)

    # ── tiny coordinate badge (top-right) ────────────────────────────────
    coord_txt = "58°N · 56°E"
    draw.text((WP_W - 52, 52), coord_txt, font=f_mono, fill=COPPER_DIM, anchor="rt")

    # ── vertical region code "59" along left edge (ghost) ─────────────────
    f_ghost = _find_font(84, _FONT_BOLD_CANDIDATES)
    ghost_img = Image.new("RGBA", (84, 200), (0, 0, 0, 0))
    ghost_draw = ImageDraw.Draw(ghost_img)
    ghost_draw.text((42, 100), "59", font=f_ghost, fill=(50, 50, 50, 120), anchor="mm")
    ghost_rot = ghost_img.rotate(90, expand=True)
    wp.paste(ghost_rot, (0, HEADER_H // 2 - 100), ghost_rot)

    # ── ПЕРМЬ — main city title ───────────────────────────────────────────
    city_y = 200
    # Shadow for depth
    for dx, dy in [(-3, 4), (3, 4)]:
        draw.text((WP_W // 2 + dx, city_y + dy), "ПЕРМЬ",
                  font=f_city, fill=(20, 8, 2), anchor="mm")
    # Main text
    draw.text((WP_W // 2, city_y), "ПЕРМЬ", font=f_city, fill=WHITE, anchor="mm")
    # Copper highlight — very thin bottom-offset
    draw.text((WP_W // 2, city_y + 2), "ПЕРМЬ", font=f_city,
              fill=(*COPPER_BRIGHT, 60), anchor="mm")

    # ── decorative accent line under city name ────────────────────────────
    line_y = city_y + 128
    # Full width copper line
    draw.rectangle([(80, line_y), (WP_W - 80, line_y + 2)],
                   fill=COPPER_MID)
    # Cyan center accent
    cw = 280
    draw.rectangle([(WP_W // 2 - cw // 2, line_y - 1),
                    (WP_W // 2 + cw // 2, line_y + 3)],
                   fill=CYAN)
    # Small diamond at center
    draw_permian_accent(draw, WP_W // 2, line_y + 1, 8, CYAN)

    # ── ПЕРМСКИЙ КРАЙ ─────────────────────────────────────────────────────
    reg_y = line_y + 80
    draw.text((WP_W // 2, reg_y), "ПЕРМСКИЙ КРАЙ",
              font=f_region, fill=COPPER_BRIGHT, anchor="mm")

    # Subtle letter-spacing simulation (character by character not needed — PIL handles it)
    # "Sub-region" context row
    sub_y = reg_y + 90
    draw.text((WP_W // 2, sub_y),
              "Пермский период  ·  р. Кама  ·  Урал  ·  59",
              font=f_sub, fill=COPPER_DIM, anchor="mm")

    # ── data label block (top-right of header) ────────────────────────────
    draw_data_label(draw, f_mono, WP_W - 490, 52, COPPER_DIM)

    # ── thin horizontal separator (neo-brutalism) ─────────────────────────
    sep_y = HEADER_H - 3
    draw.rectangle([(0, sep_y), (WP_W, sep_y + 2)], fill=CYAN)
    draw.rectangle([(0, sep_y + 2), (WP_W, sep_y + 3)], fill=COPPER_MID)

    # ── FOOTER ────────────────────────────────────────────────────────────
    footer_y = WP_H - FOOTER_H
    draw.rectangle([(0, footer_y), (WP_W, footer_y + 1)], fill=CYAN)

    # Left: brand
    draw.text((52, footer_y + FOOTER_H // 2), "TerraINFO®",
              font=f_footer_b, fill=COPPER_DIM, anchor="lm")

    # Center: channel label
    draw.text((WP_W // 2, footer_y + FOOTER_H // 2), "edge_composite",
              font=f_footer, fill=CYAN_DIM, anchor="mm")

    # Right: version
    draw.text((WP_W - 52, footer_y + FOOTER_H // 2), "v3 · 2025",
              font=f_footer, fill=GHOST, anchor="rm")

    # ── save ──────────────────────────────────────────────────────────────
    wp.save(str(out_path), format="PNG", compress_level=1)
    print(f"  → {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


# ── VERSION 2 — "59" ──────────────────────────────────────────────────────────

def build_v2(feature_layer: np.ndarray, out_path: Path) -> None:
    """
    Full-bleed typographic wallpaper.
    The giant "59" IS the design — the data lives inside it.

    Layout: feature map everywhere, massive semi-transparent "59" in bottom half,
    "ПЕРМЬ" and TерраINFO™ floating cleanly.
    """
    FOOTER_H = 72

    # ── base canvas ───────────────────────────────────────────────────────
    wp = Image.fromarray(feature_layer, mode="RGB").resize(
        (WP_W, WP_H), Image.BICUBIC
    )

    # ── subtle grid ───────────────────────────────────────────────────────
    draw = ImageDraw.Draw(wp)
    draw_grid_lines(draw, WP_W, WP_H, spacing=72, color=(10, 16, 24), line_w=1)

    # ── fonts ─────────────────────────────────────────────────────────────
    f_massive = _find_font(900, _FONT_BOLD_CANDIDATES)   # "59"
    f_city_lg = _find_font(150, _FONT_BOLD_CANDIDATES)   # ПЕРМЬ
    f_brand   = _find_font(36,  _FONT_BOLD_CANDIDATES)
    f_mono    = _find_font(22,  _FONT_MONO_CANDIDATES)
    f_footer  = _find_font(26,  _FONT_REG_CANDIDATES)
    f_footer_b= _find_font(26,  _FONT_BOLD_CANDIDATES)

    COPPER_DIM   = (100,  50,   8)
    COPPER_MID   = (200, 100,  30)
    COPPER_BRIGHT= (245, 185,  70)
    CYAN         = ( 40, 200, 215)
    WHITE        = (255, 255, 255)
    GHOST        = ( 55,  55,  55)

    # ── MASSIVE "59" — rendered on RGBA canvas then alpha-pasted ──────────
    # Strategy: render "59" twice:
    #   (a) hollow "59" — outline only (Copper, fully opaque)
    #   (b) fill "59"   — solid (very dark, low alpha) → map shows through

    # We need the bounding box of "59" at this size
    tmp = Image.new("RGBA", (WP_W * 2, WP_H), (0, 0, 0, 0))
    tmp_d = ImageDraw.Draw(tmp)
    # Draw once to measure
    tmp_d.text((WP_W, WP_H // 2 + 200), "59", font=f_massive,
               fill=(255, 255, 255, 255), anchor="mm")
    bbox = tmp.getbbox()
    tmp.close()

    # Now build the "59" layer at wallpaper size
    num_layer = Image.new("RGBA", (WP_W, WP_H), (0, 0, 0, 0))
    nd = ImageDraw.Draw(num_layer)

    # Position: center-x, lower-third
    nx, ny = WP_W // 2, int(WP_H * 0.70)

    # Fill — semi-transparent dark copper (map bleeds through)
    nd.text((nx, ny), "59", font=f_massive,
            fill=(60, 18, 2, 65), anchor="mm")

    # Bright outline — stroke effect (draw at 4 offsets)
    outline_col = (*COPPER_MID, 220)
    for odx, ody in [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1)]:
        nd.text((nx + odx, ny + ody), "59", font=f_massive,
                fill=(10, 3, 0, 0), anchor="mm")   # transparent, just offsets outline

    # Actual outline: draw the text at offsets in copper
    stroke_px = 3
    for sdx in range(-stroke_px, stroke_px + 1):
        for sdy in range(-stroke_px, stroke_px + 1):
            if abs(sdx) == stroke_px or abs(sdy) == stroke_px:
                nd.text((nx + sdx, ny + sdy), "59", font=f_massive,
                        fill=(*COPPER_MID, 200), anchor="mm")

    wp.paste(num_layer.convert("RGB"), (0, 0), num_layer)

    # ── top zone — TерраINFO™ + minimal header ────────────────────────────
    # Small glass strip at top
    wp = glass_panel(wp, 0, 0, WP_W, 130, blur_r=16, dark_alpha=160,
                     border_color=(0, 0, 0, 0), border_w=0)
    draw = ImageDraw.Draw(wp)

    # Cyan top border line
    draw.rectangle([(0, 0), (WP_W, 2)], fill=CYAN)

    draw.text((52, 65), "TерраINFO™", font=f_brand, fill=CYAN, anchor="lm")

    # Vertical "ПЕРМСКИЙ ПЕРИОД" (cultural easter-egg) on right edge
    vert_font = _find_font(22, _FONT_MONO_CANDIDATES)
    vert_text = "ПЕРМСКИЙ ПЕРИОД · ПЕРМСКИЙ ЗВЕРИНЫЙ СТИЛЬ"
    vtmp = Image.new("RGBA", (22, 700), (0, 0, 0, 0))
    vd   = ImageDraw.Draw(vtmp)
    vd.text((11, 350), vert_text, font=vert_font, fill=(*COPPER_DIM, 140), anchor="mm")
    vrot = vtmp.rotate(-90, expand=True)
    wp.paste(vrot, (WP_W - vrot.width - 8, 150), vrot)

    # ── "ПЕРМЬ" — clear, sharp, above the "59" ────────────────────────────
    perm_y = int(WP_H * 0.38)
    # Glass strip under ПЕРМЬ
    wp = glass_panel(wp, 0, perm_y - 100, WP_W, 220,
                     blur_r=20, dark_alpha=150, border_color=(0,0,0,0), border_w=0)
    draw = ImageDraw.Draw(wp)

    # Left accent bar
    draw.rectangle([(52, perm_y - 30), (56, perm_y + 90)], fill=CYAN)

    draw.text((80, perm_y), "ПЕРМЬ", font=f_city_lg, fill=WHITE, anchor="lm")

    # Coord to the right of city name
    draw.text((80 + 600, perm_y + 5), "58°01'N · 56°14'E",
              font=_find_font(30, _FONT_MONO_CANDIDATES), fill=COPPER_DIM, anchor="lm")

    # Horizontal rule under city name
    draw.rectangle([(80, perm_y + 82), (WP_W - 80, perm_y + 84)],
                   fill=COPPER_MID)

    # ── corner Permian accent marks ───────────────────────────────────────
    draw_permian_accent(draw, 52,        WP_H - 120, 16, COPPER_DIM)
    draw_permian_accent(draw, WP_W - 52, 130,        16, COPPER_DIM)

    # ── data label ────────────────────────────────────────────────────────
    draw_data_label(draw, vert_font, 52, WP_H - 210, GHOST)

    # ── footer strip ──────────────────────────────────────────────────────
    wp = glass_panel(wp, 0, WP_H - FOOTER_H, WP_W, FOOTER_H,
                     blur_r=14, dark_alpha=200, border_color=(0,0,0,0), border_w=0)
    draw = ImageDraw.Draw(wp)
    draw.rectangle([(0, WP_H - FOOTER_H), (WP_W, WP_H - FOOTER_H + 1)], fill=COPPER_MID)

    draw.text((52, WP_H - FOOTER_H // 2), "TerraINFO®",
              font=f_footer_b, fill=COPPER_DIM, anchor="lm")
    draw.text((WP_W // 2, WP_H - FOOTER_H // 2), "edge_composite · Пермский край",
              font=f_footer, fill=GHOST, anchor="mm")
    draw.text((WP_W - 52, WP_H - FOOTER_H // 2), "2025",
              font=f_footer, fill=GHOST, anchor="rm")

    # ── save ──────────────────────────────────────────────────────────────
    wp.save(str(out_path), format="PNG", compress_level=1)
    print(f"  → {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading TerraINFO feature tile — permkrai_02.npz …")
    d = np.load(str(TILE_NPZ), allow_pickle=False)

    edge    = d["edgecomposite"].astype(np.float32)
    maxndvi = d["maxndvi"].astype(np.float32)
    ndwi    = d["ndwi_mean"].astype(np.float32)
    mndwi   = d["mndwi_max"].astype(np.float32)
    r_med   = d["red_median"].astype(np.float32)
    g_med   = d["green_median"].astype(np.float32)
    b_med   = d["blue_median"].astype(np.float32)
    nir     = d["nir_median"].astype(np.float32)
    swir    = d["swir_median"].astype(np.float32)

    print(f"  edge: min={edge.min():.3f} max={np.nanmax(edge):.3f} "
          f"nan%={100*np.isnan(edge).mean():.1f}%")
    print(f"  shape: {edge.shape}  bbox: {d['bbox']}")

    print("\nRendering premium feature layer …")
    fl = render_feature_layer(
        edge, maxndvi, ndwi, mndwi, r_med, g_med, b_med, nir, swir,
        out_w=WP_W, out_h=WP_H,
    )
    print(f"  feature layer: {fl.shape}")

    print("\n[1/2] Building VERSION 1 — ПЕРМСКИЙ КРАЙ …")
    build_v1(fl, ROOT / "wallpaper_perm_permskiy_kray.png")

    print("\n[2/2] Building VERSION 2 — 59 …")
    build_v2(fl, ROOT / "wallpaper_perm_59.png")

    print("\nAll done. Files saved to project root.")


if __name__ == "__main__":
    main()
