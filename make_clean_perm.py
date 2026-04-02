#!/usr/bin/env python3
"""
Data-visualization wallpapers  —  6 outputs (1440×3088, S25 Ultra):
  wallpaper_perm_59_edge.png
  wallpaper_perm_59_ndvi.png
  wallpaper_perm_59_ndvi_var.png
  wallpaper_spb_78_edge.png
  wallpaper_spb_78_ndvi.png
  wallpaper_spb_78_ndvi_var.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

ROOT       = Path(__file__).resolve().parent
WP_W, WP_H = 1440, 3088
TILES      = ROOT / "backend/debug/runs/real_tiles"
TILE_PERM  = TILES / "permkrai_01.npz"
TILE_SPB   = TILES / "lenoblast_spb.npz"  # center 59.93°N 30.31°E — real SPB, 50km radius, 12-scene composite
TILE_MRM   = TILES / "murmansk_51.npz"    # center 68.96°N 33.08°E — Murmansk, 30km radius, 9 summer scenes
ZOOM       = 1.38

# ── Colormaps ─────────────────────────────────────────────────────────────────
# True matplotlib-viridis match: 0 = deep indigo/purple (not black)
_VIRIDIS = [
    (0.00, (0.267, 0.004, 0.329)),   # indigo (matplotlib viridis exact)
    (0.13, (0.278, 0.175, 0.484)),   # violet
    (0.25, (0.230, 0.322, 0.546)),   # blue-purple
    (0.38, (0.172, 0.448, 0.558)),   # slate-blue
    (0.50, (0.128, 0.567, 0.551)),   # teal
    (0.63, (0.153, 0.678, 0.506)),   # green-teal
    (0.75, (0.369, 0.789, 0.383)),   # green
    (0.88, (0.678, 0.863, 0.190)),   # yellow-green
    (1.00, (0.993, 0.906, 0.144)),   # bright yellow
]
EDGE_CMAP = LinearSegmentedColormap.from_list(
    "viridis_true", [(t, c) for t, c in _VIRIDIS], N=2048)

# MaxNDVI — vegetation colormap
_NDVI_C = [
    (0.00, (0.267, 0.004, 0.329)),   # indigo (bare soil / water)
    (0.18, (0.180, 0.260, 0.540)),   # slate blue
    (0.35, (0.050, 0.480, 0.360)),   # dark teal
    (0.52, (0.040, 0.560, 0.160)),   # dark green
    (0.68, (0.200, 0.700, 0.030)),   # medium green
    (0.84, (0.570, 0.850, 0.020)),   # lime
    (1.00, (0.910, 0.990, 0.100)),   # bright yellow-green
]
NDVI_CMAP = LinearSegmentedColormap.from_list(
    "ndvi_green", [(t, c) for t, c in _NDVI_C], N=2048)

# NDVI variability (max − mean) — warm seasonal amplitude colormap
_NDVI_VAR = [
    (0.00, (0.080, 0.020, 0.240)),   # deep purple (stable: forest)
    (0.25, (0.350, 0.040, 0.460)),   # purple-magenta
    (0.50, (0.700, 0.140, 0.200)),   # deep rose
    (0.70, (0.920, 0.420, 0.050)),   # amber
    (0.88, (0.990, 0.750, 0.050)),   # gold
    (1.00, (0.995, 0.960, 0.700)),   # pale cream (max variability: crops)
]
NDVI_VAR_CMAP = LinearSegmentedColormap.from_list(
    "ndvi_var", [(t, c) for t, c in _NDVI_VAR], N=2048)

# ── Fonts ─────────────────────────────────────────────────────────────────────
_DISPLAY = [
    "/usr/share/fonts/truetype/lato/Lato-Black.ttf",
    "/usr/share/fonts/truetype/open-sans/OpenSans-ExtraBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_BOLD = [
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_MONO = [
    "/usr/share/fonts/truetype/croscore/Cousine-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_MONO_BOLD = [
    "/usr/share/fonts/truetype/croscore/Cousine-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]

def _font(size: int, candidates: list) -> ImageFont.FreeTypeFont:
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Tile loading ──────────────────────────────────────────────────────────────

def load_tile(npz_path: Path, out_w: int, out_h: int,
              zoom: float = ZOOM, cx_frac: float = 0.5,
              cy_frac: float = 0.5) -> dict[str, np.ndarray]:
    d     = np.load(str(npz_path), allow_pickle=False)
    keys  = ["edgecomposite", "maxndvi", "meanndvi", "ndvistd", "ndwi_mean",
             "mndwi_max", "red_median", "green_median", "nir_median", "swir_median"]
    H0, W0 = d["edgecomposite"].shape
    scale   = max(out_w / W0, out_h / H0) * zoom
    sw, sh  = int(round(W0 * scale)), int(round(H0 * scale))
    cx      = int((sw - out_w) * cx_frac)
    cy      = int((sh - out_h) * cy_frac)
    result: dict[str, np.ndarray] = {}
    for k in keys:
        arr = np.nan_to_num(d[k].astype(np.float32), nan=0.0)
        pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L")
        pil = pil.resize((sw, sh), Image.LANCZOS).crop((cx, cy, cx+out_w, cy+out_h))
        result[k] = np.asarray(pil, dtype=np.float32) / 255.0
    return result


def pct_clip(arr: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    v   = arr[np.isfinite(arr)].ravel()
    vlo = np.percentile(v, lo)
    vhi = np.percentile(v, hi)
    if vhi <= vlo + 1e-7:
        return np.zeros_like(arr)
    return np.clip((arr - vlo) / (vhi - vlo), 0.0, 1.0).astype(np.float32)


# ── Render functions ──────────────────────────────────────────────────────────

def _water_overlay(comp: np.ndarray, ndwi: np.ndarray, H: int, W: int) -> np.ndarray:
    water = np.clip((ndwi - 0.35) / 0.35, 0, 1)
    def blur(r):
        return np.asarray(
            Image.fromarray((water*255).clip(0,255).astype(np.uint8), "L")
                .filter(ImageFilter.GaussianBlur(radius=r)),
            dtype=np.float32) / 255.0
    comp += np.stack([blur(12)*0.0, blur(12)*0.18, blur(12)*0.25], axis=-1)
    cyan  = np.array([0.05, 0.75, 0.95], dtype=np.float32)
    wa    = blur(3)[..., None] * 0.82
    return np.clip(comp*(1-wa) + cyan*wa, 0, 1)


def _vignette(comp: np.ndarray, H: int, W: int) -> np.ndarray:
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    return comp * np.clip(1.0 - 0.36*(xx**2*0.4 + yy**2), 0, 1)[..., None]


def render_edge(t: dict, cmap) -> np.ndarray:
    H, W  = WP_H, WP_W
    ndvi  = pct_clip(t["maxndvi"], 5, 95)
    base  = np.stack([np.power(ndvi, 2.2)*0.08*c for c in (0.3, 0.6, 0.4)], axis=-1)
    ec    = pct_clip(t["edgecomposite"], 0, 99)
    ec_g  = np.power(ec, 0.76)
    rgba  = cmap(ec_g)
    alpha = np.clip(ec_g[..., None] * 0.92, 0, 1)
    comp  = base*(1-alpha) + rgba[..., :3].astype(np.float32)*alpha
    comp  = _water_overlay(comp, t["ndwi_mean"], H, W)
    return (_vignette(comp, H, W) * 255).clip(0, 255).astype(np.uint8)


def render_ndvi(t: dict, cmap) -> np.ndarray:
    H, W  = WP_H, WP_W
    ndvi  = pct_clip(t["maxndvi"], 2, 98)
    rgba  = cmap(ndvi)
    comp  = rgba[..., :3].astype(np.float32)
    comp  = _water_overlay(comp, t["ndwi_mean"], H, W)
    return (_vignette(comp, H, W) * 255).clip(0, 255).astype(np.uint8)


def render_ndvi_var(t: dict, cmap) -> np.ndarray:
    """NDVI STD — true per-pixel standard deviation across all valid scenes."""
    H, W  = WP_H, WP_W
    var   = pct_clip(t["ndvistd"], 2, 98)
    rgba  = cmap(var)
    comp  = rgba[..., :3].astype(np.float32)
    comp  = _water_overlay(comp, t["ndwi_mean"], H, W)
    return (_vignette(comp, H, W) * 255).clip(0, 255).astype(np.uint8)


# ── UI helpers ────────────────────────────────────────────────────────────────

def draw_colorbar(draw: ImageDraw.ImageDraw, cmap,
                  x: int, y: int, w: int, h: int,
                  display_fn=None) -> None:
    """draw_fn(v) -> (r,g,b) overrides cmap if given (for blended layers)."""
    for px in range(w):
        v = px / w
        if display_fn is not None:
            r, g, b = display_fn(v)
        else:
            r, g, b, _ = [int(c*255) for c in cmap(v)]
        draw.line([(x+px, y), (x+px, y+h)], fill=(r, g, b))


def _edge_bar_color(v: float) -> tuple[int, int, int]:
    """Simulate actual render_edge blending so colorbar is accurate."""
    ec_g  = v ** 0.76
    alpha = min(ec_g * 0.92, 1.0)
    # representative dark-green base (ndvi ≈ 0.7 → ndvi^2.2*0.08 * [0.3,0.6,0.4])
    base  = (0.012, 0.024, 0.016)
    rgba  = EDGE_CMAP(ec_g)
    rgb   = tuple(int((base[i]*(1-alpha) + rgba[i]*alpha) * 255) for i in range(3))
    return rgb


def _sh(draw: ImageDraw.ImageDraw, xy, text, font, fill, anchor="lt",
        sc=(0,0,0), sa=215, off=(2,2)) -> None:
    draw.text((xy[0]+off[0], xy[1]+off[1]), text, font=font,
              fill=sc+(sa,), anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _grad(wp: Image.Image, y0: int, h: int, inv: bool) -> None:
    s = Image.new("RGBA", (WP_W, h), (0, 0, 0, 0))
    for row in range(h):
        t = row / h
        a = int(192 * ((1-t) if inv else t) ** 0.54)
        ImageDraw.Draw(s).line([(0, row), (WP_W, row)], fill=(3, 5, 12, a))
    wp.paste(s.convert("RGB"), (0, y0), s.split()[3])


def draw_brand(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    f_b = _font(44, _BOLD)          # ТерраINFO
    f_t = _font(30, _BOLD)          # ™ bigger, white
    draw.text((x+2, y+2), "ТерраINFO", font=f_b, fill=(0,0,0,200), anchor="lt")
    draw.text((x,   y),   "ТерраINFO", font=f_b, fill=(255,255,255), anchor="lt")
    tw = int(draw.textlength("ТерраINFO", font=f_b))
    draw.text((x+tw+2, y-6), "™", font=f_t, fill=(220,220,220), anchor="lt")


def draw_coords(draw: ImageDraw.ImageDraw, H: int,
                phi: str, lam: str) -> None:
    f  = _font(42, _MONO_BOLD)    # bolder font, no backdrop
    cy = H // 2
    draw.rectangle([(42, cy-62), (46, cy+62)], fill=(0, 230, 245))
    _sh(draw, (54, cy-26), phi, f, (255, 255, 255), anchor="lm", sa=245, off=(2,2))
    _sh(draw, (54, cy+26), lam, f, (255, 255, 255), anchor="lm", sa=245, off=(2,2))


def composite_ghost_digits(wp: Image.Image, d1: str, d2: str,
                            f_big: ImageFont.FreeTypeFont,
                            X: int, Y1: int, Y2: int,
                            W: int, H: int,
                            d1_dx: int = 0) -> None:
    """
    d2 rendered first (bottom), d1 on top (brighter).
    d1_dx: horizontal shift for top digit (use to align leg endpoints).
    L-channel mask → data boundary lines stay visible through counter holes.
    max 40% opacity so edge_composite always readable.
    """
    sl = Image.new("L", (W, H), 0)
    dd = ImageDraw.Draw(sl)
    dd.text((X,        Y2), d2, font=f_big, fill=212, anchor="lt")   # bottom digit
    dd.text((X+d1_dx,  Y1), d1, font=f_big, fill=232, anchor="lt")   # top (brighter)
    glow = sl.filter(ImageFilter.GaussianBlur(radius=12))
    combined = Image.blend(sl, glow, 0.38)
    mask = combined.point(lambda v: int(v * 0.40))
    tint = Image.new("RGB", (W, H), (218, 238, 255))
    wp.paste(tint, (0, 0), mask)


# ── Win98 dark CMD window ─────────────────────────────────────────────────────

def draw_cmd_window(wp: Image.Image, x1: int, y1: int,
                    win_w: int, win_h: int,
                    active_ch: str = "edge") -> None:
    BEVEL   = 2
    TITLE_H = 28
    PAD     = 12

    ov = Image.new("RGBA", (win_w+8, win_h+8), (0,0,0,0))
    d  = ImageDraw.Draw(ov)
    d.rectangle([(5,5),(win_w+7,win_h+7)], fill=(0,0,0,148))
    d.rectangle([(0,0),(win_w,win_h)], fill=(50,54,70,255))
    HL = (122, 130, 162, 255)
    SH = (10,  12,  20,  255)
    for i in range(BEVEL):
        d.line([(i,i),(win_w-i,i)],         fill=HL)
        d.line([(i,i),(i,win_h-i)],         fill=HL)
        d.line([(i,win_h-i),(win_w-i,win_h-i)], fill=SH)
        d.line([(win_w-i,i),(win_w-i,win_h-i)], fill=SH)
    for row in range(TITLE_H-BEVEL):
        t = row / max(TITLE_H-BEVEL-1,1)
        d.line([(BEVEL, BEVEL+row),(win_w-BEVEL, BEVEL+row)],
               fill=(int(2+t*5), int(6+t*12), int(72+t*58), 255))
    d.line([(BEVEL, TITLE_H),(win_w-BEVEL, TITLE_H)], fill=(6,10,30,255))
    d.rectangle([(BEVEL, TITLE_H+1),(win_w-BEVEL, win_h-BEVEL)], fill=(5,7,16,245))
    d.line([(BEVEL,TITLE_H+1),(win_w-BEVEL,TITLE_H+1)], fill=(8,10,22,255))
    d.line([(BEVEL,TITLE_H+1),(BEVEL,win_h-BEVEL)],     fill=(8,10,22,255))
    d.line([(BEVEL,win_h-BEVEL),(win_w-BEVEL,win_h-BEVEL)],fill=(58,68,92,255))
    d.line([(win_w-BEVEL,TITLE_H+1),(win_w-BEVEL,win_h-BEVEL)],fill=(58,68,92,255))
    wp.paste(ov.convert("RGB"), (x1-4, y1-4), ov)

    draw  = ImageDraw.Draw(wp)
    f_ttl = _font(13, _MONO_BOLD)
    f_btn = _font(12, _MONO)
    f_cmd = _font(18, _MONO)

    ty_t = y1 + BEVEL + (TITLE_H-BEVEL)//2
    draw.text((x1+BEVEL+8, ty_t), "TERRAINFO — Sentinel-2 Analysis",
              font=f_ttl, fill=(218,228,255), anchor="lm")

    BW   = 22
    BH   = TITLE_H - BEVEL - 4
    by_b = y1 + BEVEL + 2
    bx_r = x1 + win_w - BEVEL - 2
    for lbl, face, txt in [
        ("×",(138,34,34),(255,195,195)),
        ("□",(50,56,80),(198,210,242)),
        ("─",(50,56,80),(198,210,242)),
    ]:
        bx_r -= BW+1
        draw.rectangle([(bx_r,by_b),(bx_r+BW-1,by_b+BH)], fill=face)
        draw.line([(bx_r,by_b),(bx_r+BW-1,by_b)],          fill=(158,166,200))
        draw.line([(bx_r,by_b),(bx_r,by_b+BH)],            fill=(158,166,200))
        draw.line([(bx_r,by_b+BH),(bx_r+BW-1,by_b+BH)],   fill=(12,14,24))
        draw.line([(bx_r+BW-1,by_b),(bx_r+BW-1,by_b+BH)], fill=(12,14,24))
        draw.text((bx_r+BW//2, by_b+BH//2+1), lbl, font=f_btn, fill=txt, anchor="mm")

    GRN = (0,  212, 88)
    WHT = (208,222,248)
    DIM = (100,120,148)
    ACC = (0,  208,178)
    VAL = (152,190,220)
    SEP = (30, 42, 60)

    tx = x1 + BEVEL + PAD
    ty = y1 + TITLE_H + PAD + 4
    LH = 21

    # Active channel markers
    def star(ch): return "[*]" if active_ch == ch else "[ ]"
    def col(ch):  return ACC   if active_ch == ch else DIM

    def prompt(cmd):
        nonlocal ty
        draw.text((tx, ty), "C:\\>", font=f_cmd, fill=GRN)
        if cmd:
            draw.text((tx+int(draw.textlength("C:\\>",font=f_cmd))+5, ty),
                      cmd, font=f_cmd, fill=WHT)
        ty += LH

    def row(k, v, kc=DIM, vc=VAL):
        nonlocal ty
        draw.text((tx, ty), k, font=f_cmd, fill=kc)
        draw.text((tx+int(draw.textlength(k,font=f_cmd)), ty), v, font=f_cmd, fill=vc)
        ty += LH

    def sep():
        nonlocal ty
        draw.line([(tx, ty+4),(x1+win_w-BEVEL-PAD, ty+4)], fill=SEP, width=1)
        ty += LH//2+5

    prompt("query --metadata")
    sep()
    row("Satellite  ", "Sentinel-2 L2A")
    row("Access     ", "SentinelHub API  (OAuth2)")
    row("Level      ", "L2A  Surface Reflectance")
    row("GSD        ", "10 м/пкс  (SWIR: 20→10 м)")
    row("Period     ", "2021 – 2023  (multiyear composite)")
    row("Projection ", "EPSG:4326  WGS84")
    sep()

    prompt("list --channels --verbose")
    sep()
    row(f"{star('edge')} ","edge_composite",              col('edge'), col('edge'))
    row("    ","spectral gradient magnitude",             vc=DIM)
    row(f"{star('ndvi')} ","NDVI  max · mean  ",          col('ndvi'), col('ndvi'))
    row("    ","(B08-B04)/(B08+B04)",                     vc=DIM)
    row(f"{star('var')}  ","NDVI  variability (max−mean)",col('var'),  col('var'))
    row("    ","seasonal amplitude  ← crops bright",      vc=DIM)
    row("[ ] ","NDWI   mean  (B03-B08)/(B03+B08)",       vc=DIM)
    row("[ ] ","B04 B03 B08 B11   10/20m  median",       vc=DIM)
    sep()

    draw.text((tx, ty), "C:\\>", font=f_cmd, fill=GRN)
    draw.text((tx+int(draw.textlength("C:\\>",font=f_cmd))+5, ty),
              "█", font=f_cmd, fill=GRN)


def draw_colorbar_section(draw, cmap, bar_label: str,
                           f_mono, W: int, H: int,
                           display_fn=None) -> None:
    BAR_W, BAR_H = 540, 11
    bx = (W - BAR_W) // 2
    by = H - 130
    draw_colorbar(draw, cmap, bx, by, BAR_W, BAR_H, display_fn=display_fn)
    draw.rectangle([(bx-1,by-1),(bx+BAR_W,by+BAR_H)], outline=(58,78,98), width=1)
    _sh(draw, (bx,       by+BAR_H+8), "0.0", f_mono, (185,215,238), anchor="lt")
    _sh(draw, (bx+BAR_W, by+BAR_H+8), "1.0", f_mono, (185,215,238), anchor="rt")
    _sh(draw, (W//2, by-18), bar_label, f_mono, (158,192,218), anchor="mm")


# ── Master build function ─────────────────────────────────────────────────────

def build_wallpaper(
    layer:       np.ndarray,
    cmap,
    out_path:    Path,
    city_lines:  list[str],          # e.g. ["ПЕРМЬ"] or ["САНКТ-","ПЕТЕРБУРГ"]
    region_code: str,                 # "59" or "78"
    subtitles:   list[str],          # 1-2 lines below city
    phi:         str,                 # "φ  57°26′ N"
    lam:         str,                 # "λ  56°57′ E"
    bar_label:   str,                 # colorbar caption
    active_ch:   str = "edge",        # "edge" | "ndvi" | "var"
    digit_dx:    int = 0,            # horizontal shift for top digit (d1)
) -> None:

    wp   = Image.fromarray(layer, "RGB")
    W, H = WP_W, WP_H

    _grad(wp, 0,       620, inv=True)
    _grad(wp, H - 230, 230, inv=False)

    # ── Ghost region digits ────────────────────────────────────────────────
    # Rule: center of first quarter of top digit = H//2  (+ tiny upward nudge)
    f_big   = _font(950, _DISPLAY)
    bb      = f_big.getbbox(region_code[0])
    digit_h = bb[3] - bb[1]          # actual rendered height of digit glyph
    X  = W - 325
    Y1 = H // 2 - digit_h // 8 - 35  # -35 px nudge upward per user request
    Y2 = Y1 + 635
    composite_ghost_digits(wp, region_code[0], region_code[1],
                           f_big, X, Y1, Y2, W, H, d1_dx=digit_dx)

    draw = ImageDraw.Draw(wp)

    # ── Fonts ──────────────────────────────────────────────────────────────
    n_city   = len(city_lines)
    city_sz  = 205 if n_city == 1 else 135   # 135px keeps 2-line cities clear of CMD
    sub_sz   = max(52, city_sz // 3)

    f_city   = _font(city_sz, _DISPLAY)
    f_sub    = _font(sub_sz,  _BOLD)
    f_sub2   = _font(int(sub_sz*0.88), _BOLD)
    f_mono   = _font(24, _MONO)

    # ── City name ─────────────────────────────────────────────────────────
    y_cursor = 205   # anchor center-y of first city line
    for i, line in enumerate(city_lines):
        if i > 0:
            y_cursor += city_sz + 18
        _sh(draw, (52, y_cursor), line, f_city, (248, 254, 255), anchor="lm")

    # ── Subtitles ─────────────────────────────────────────────────────────
    y_cursor += city_sz // 2 + 28
    for i, sub in enumerate(subtitles):
        font = f_sub if i == 0 else f_sub2
        _sh(draw, (52, y_cursor), sub, font, (242, 248, 255),
            anchor="lm", sc=(0,0,0), sa=215)
        y_cursor += (sub_sz if i == 0 else int(sub_sz*0.88)) + 14

    # ── CMD window — top-right, below city block if text would overlap ────
    CMD_W, CMD_H = 512, 496
    cmd_x_left   = W - CMD_W - 24
    max_city_w   = max(int(draw.textlength(line, font=f_city)) for line in city_lines)
    if 52 + max_city_w + 24 > cmd_x_left:
        # City name too wide → place CMD below subtitle block
        cmd_y = y_cursor + 20
    else:
        cmd_y = 205 - city_sz // 2   # align with city name top
    draw_cmd_window(wp, cmd_x_left, cmd_y, CMD_W, CMD_H, active_ch=active_ch)

    # ── Coordinates — left-center ──────────────────────────────────────────
    draw_coords(draw, H, phi, lam)

    # ── Colorbar — for edge_composite show actual blended display colors ───
    bar_fn = _edge_bar_color if active_ch == "edge" else None
    draw_colorbar_section(draw, cmap, bar_label, f_mono, W, H, display_fn=bar_fn)

    # ── ТерраINFO™ — bottom-left ───────────────────────────────────────────
    draw_brand(draw, 48, H - 58)

    wp.save(str(out_path), format="PNG", compress_level=1)
    print(f"  → {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configs = [
        # ── Perm, region 59 ───────────────────────────────────────────────
        dict(
            tile    = TILE_PERM,
            name    = "Пермь",
            render  = render_edge,
            cmap    = EDGE_CMAP,
            slug    = "perm_59_edge",
            city    = ["ПЕРМЬ"],
            code    = "59",
            subs    = ["СЧАСТЬЕ НЕ ЗА ГОРАМИ"],
            phi     = "φ  57°26′ N",
            lam     = "λ  56°57′ E",
            bar     = "Sentinel-2  ·  edge_composite  ·  нормализованная интенсивность",
            active  = "edge",
        ),
        dict(
            tile    = TILE_PERM,
            name    = "Пермь MaxNDVI",
            render  = render_ndvi,
            cmap    = NDVI_CMAP,
            slug    = "perm_59_ndvi",
            city    = ["ПЕРМЬ"],
            code    = "59",
            subs    = ["СЧАСТЬЕ НЕ ЗА ГОРАМИ"],
            phi     = "φ  57°26′ N",
            lam     = "λ  56°57′ E",
            bar     = "Sentinel-2  ·  NDVI max  ·  индекс растительности",
            active  = "ndvi",
        ),
        dict(
            tile    = TILE_PERM,
            name    = "Пермь NDVI VAR",
            render  = render_ndvi_var,
            cmap    = NDVI_VAR_CMAP,
            slug    = "perm_59_ndvi_var",
            city    = ["ПЕРМЬ"],
            code    = "59",
            subs    = ["СЧАСТЬЕ НЕ ЗА ГОРАМИ"],
            phi     = "φ  57°26′ N",
            lam     = "λ  56°57′ E",
            bar     = "Sentinel-2  ·  NDVI STD  ·  межсезонная изменчивость",
            active  = "var",
        ),
        # ── St. Petersburg, region 78 ──────────────────────────────────────
        dict(
            tile      = TILE_SPB,
            name      = "СПб edge",
            render    = render_edge,
            cmap      = EDGE_CMAP,
            slug      = "spb_78_edge",
            city      = ["САНКТ-", "ПЕТЕРБУРГ"],
            code      = "78",
            subs      = ["ДОБРОЕ УТРО!", "ПОСЛЕДНИЙ ГЕРОЙ"],
            phi       = "φ  59°56′ N",
            lam       = "λ  30°18′ E",
            bar       = "Sentinel-2  ·  edge_composite  ·  нормализованная интенсивность",
            active    = "edge",
            tile_zoom = 1.0,     # show full 50km radius; no extra zoom-in
            cx_frac   = 0.40,    # shift crop left → map slides right, more left area visible
            cy_frac   = 0.60,    # shift crop down → shows more southern/central city area
            digit_dx  = 25,      # small nudge to smooth "7"→"8" seam
        ),
        dict(
            tile      = TILE_SPB,
            name      = "СПб MaxNDVI",
            render    = render_ndvi,
            cmap      = NDVI_CMAP,
            slug      = "spb_78_ndvi",
            city      = ["САНКТ-", "ПЕТЕРБУРГ"],
            code      = "78",
            subs      = ["ДОБРОЕ УТРО!", "ПОСЛЕДНИЙ ГЕРОЙ"],
            phi       = "φ  59°56′ N",
            lam       = "λ  30°18′ E",
            bar       = "Sentinel-2  ·  NDVI max  ·  индекс растительности",
            active    = "ndvi",
            tile_zoom = 1.0,
            cx_frac   = 0.40,
            cy_frac   = 0.60,
            digit_dx  = 25,
        ),
        dict(
            tile      = TILE_SPB,
            name      = "СПб NDVI VAR",
            render    = render_ndvi_var,
            cmap      = NDVI_VAR_CMAP,
            slug      = "spb_78_ndvi_var",
            city      = ["САНКТ-", "ПЕТЕРБУРГ"],
            code      = "78",
            subs      = ["ДОБРОЕ УТРО!", "ПОСЛЕДНИЙ ГЕРОЙ"],
            phi       = "φ  59°56′ N",
            lam       = "λ  30°18′ E",
            bar       = "Sentinel-2  ·  NDVI STD  ·  межсезонная изменчивость",
            active    = "var",
            tile_zoom = 1.0,
            cx_frac   = 0.40,
            cy_frac   = 0.60,
            digit_dx  = 25,
        ),
        # ── Murmansk, region 51 ────────────────────────────────────────────
        dict(
            tile    = TILE_MRM,
            name    = "Мурманск edge",
            render  = render_edge,
            cmap    = EDGE_CMAP,
            slug    = "murmansk_51_edge",
            city    = ["МУРМАНСК"],
            code    = "51",
            subs    = ["НА СЕВЕРЕ — ЖИТЬ!"],
            phi     = "φ  68°58′ N",
            lam     = "λ  33°05′ E",
            bar     = "Sentinel-2  ·  edge_composite  ·  нормализованная интенсивность",
            active  = "edge",
        ),
        dict(
            tile    = TILE_MRM,
            name    = "Мурманск MaxNDVI",
            render  = render_ndvi,
            cmap    = NDVI_CMAP,
            slug    = "murmansk_51_ndvi",
            city    = ["МУРМАНСК"],
            code    = "51",
            subs    = ["НА СЕВЕРЕ — ЖИТЬ!"],
            phi     = "φ  68°58′ N",
            lam     = "λ  33°05′ E",
            bar     = "Sentinel-2  ·  NDVI max  ·  индекс растительности",
            active  = "ndvi",
        ),
        dict(
            tile    = TILE_MRM,
            name    = "Мурманск NDVI VAR",
            render  = render_ndvi_var,
            cmap    = NDVI_VAR_CMAP,
            slug    = "murmansk_51_ndvi_var",
            city    = ["МУРМАНСК"],
            code    = "51",
            subs    = ["НА СЕВЕРЕ — ЖИТЬ!"],
            phi     = "φ  68°58′ N",
            lam     = "λ  33°05′ E",
            bar     = "Sentinel-2  ·  NDVI STD  ·  межсезонная изменчивость",
            active  = "var",
        ),
    ]

    loaded: dict[tuple, dict] = {}   # cache tiles keyed by (path, zoom, cx_frac)

    for i, cfg in enumerate(configs, 1):
        tile_zoom = cfg.get("tile_zoom", ZOOM)
        tile_cx   = cfg.get("cx_frac",   0.5)
        tile_cy   = cfg.get("cy_frac",   0.5)
        tile_key  = (str(cfg["tile"]), tile_zoom, tile_cx, tile_cy)
        if tile_key not in loaded:
            print(f"\nLoading {cfg['tile'].name} (zoom={tile_zoom}, cx={tile_cx}, cy={tile_cy}) …")
            t  = load_tile(cfg["tile"], WP_W, WP_H, zoom=tile_zoom,
                           cx_frac=tile_cx, cy_frac=tile_cy)
            ec = t["edgecomposite"]
            print(f"  ec mean={ec.mean():.3f}  ndvi mean={t['maxndvi'].mean():.3f}")
            loaded[tile_key] = t
        t = loaded[tile_key]

        print(f"\n[{i}/{len(configs)}] {cfg['name']} …")
        layer = cfg["render"](t, cfg["cmap"])
        build_wallpaper(
            layer     = layer,
            cmap      = cfg["cmap"],
            out_path  = ROOT / f"wallpaper_{cfg['slug']}.png",
            city_lines= cfg["city"],
            region_code = cfg["code"],
            subtitles = cfg["subs"],
            phi       = cfg["phi"],
            lam       = cfg["lam"],
            bar_label = cfg["bar"],
            active_ch = cfg["active"],
            digit_dx  = cfg.get("digit_dx", 0),
        )

    print(f"\nDone — {len(configs)} wallpapers generated.")


if __name__ == "__main__":
    main()
