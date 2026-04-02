#!/usr/bin/env python3
"""
Генератор обоев Sentinel-2.

Запуск:  python make_wallpaper.py
Скрипт спрашивает название города, номер региона, подпись,
координаты центра и радиус, затем сам скачивает спутниковые данные
через SentinelHub API и строит 3 обоя (1440×3088):
  wallpapers/{slug}_edge.png
  wallpapers/{slug}_ndvi.png
  wallpapers/{slug}_ndvi_var.png
"""
from __future__ import annotations

import io
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import sobel, gaussian_filter
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

try:
    import requests
    import tifffile
except ImportError:
    print("Установите зависимости:  pip install requests tifffile scipy pillow matplotlib numpy")
    sys.exit(1)

ROOT      = Path(__file__).resolve().parent
TILES_DIR = ROOT / "backend/debug/runs/real_tiles"
OUT_DIR   = ROOT / "wallpapers"
WP_W, WP_H = 1440, 3088
SH_URL    = "https://services.sentinel-hub.com"

# ── Colormaps ─────────────────────────────────────────────────────────────────
EDGE_CMAP = LinearSegmentedColormap.from_list("viridis_true", [
    (0.00, (0.267, 0.004, 0.329)),
    (0.13, (0.278, 0.175, 0.484)),
    (0.25, (0.230, 0.322, 0.546)),
    (0.38, (0.172, 0.448, 0.558)),
    (0.50, (0.128, 0.567, 0.551)),
    (0.63, (0.153, 0.678, 0.506)),
    (0.75, (0.369, 0.789, 0.383)),
    (0.88, (0.678, 0.863, 0.190)),
    (1.00, (0.993, 0.906, 0.144)),
], N=2048)

NDVI_CMAP = LinearSegmentedColormap.from_list("ndvi_green", [
    (0.00, (0.267, 0.004, 0.329)),
    (0.18, (0.180, 0.260, 0.540)),
    (0.35, (0.050, 0.480, 0.360)),
    (0.52, (0.040, 0.560, 0.160)),
    (0.68, (0.200, 0.700, 0.030)),
    (0.84, (0.570, 0.850, 0.020)),
    (1.00, (0.910, 0.990, 0.100)),
], N=2048)

NDVI_VAR_CMAP = LinearSegmentedColormap.from_list("ndvi_var", [
    (0.00, (0.080, 0.020, 0.240)),
    (0.25, (0.350, 0.040, 0.460)),
    (0.50, (0.700, 0.140, 0.200)),
    (0.70, (0.920, 0.420, 0.050)),
    (0.88, (0.990, 0.750, 0.050)),
    (1.00, (0.995, 0.960, 0.700)),
], N=2048)

# ── Fonts ─────────────────────────────────────────────────────────────────────
_DISPLAY  = ["/usr/share/fonts/truetype/lato/Lato-Black.ttf",
             "/usr/share/fonts/truetype/open-sans/OpenSans-ExtraBold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_BOLD     = ["/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
             "/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf",
             "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_MONO     = ["/usr/share/fonts/truetype/croscore/Cousine-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
_MONO_BOLD= ["/usr/share/fonts/truetype/croscore/Cousine-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]


def _font(size: int, candidates: list) -> ImageFont.FreeTypeFont:
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dms(deg: float, is_lat: bool) -> str:
    """57.426 → 'φ  57°26′ N'"""
    prefix = "φ" if is_lat else "λ"
    hemi   = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "W")
    d = abs(deg)
    degs = int(d)
    mins = round((d - degs) * 60)
    if mins == 60:
        degs += 1; mins = 0
    return f"{prefix}  {degs:d}°{mins:02d}′ {hemi}"


def _slug(city: str, code: str) -> str:
    s = city.replace("/", "_").lower()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    return s.strip("_") + f"_{code}"


def _load_env() -> dict[str, str]:
    """Parse ROOT/.env into a dict (best-effort, no dependency on python-dotenv)."""
    env_path = ROOT / ".env"
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ── SentinelHub download ──────────────────────────────────────────────────────

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

# Vegetation, non-veg, water, unclassified — not cloud/shadow
SCL_VALID = {4, 5, 6, 7}

# Clear-sky months 2021–2023 (Apr–Sep)
_BASE_MONTHS = [
    ("2021-05-01","2021-05-31"), ("2021-07-01","2021-07-31"),
    ("2021-08-01","2021-08-31"), ("2022-05-01","2022-05-31"),
    ("2022-06-01","2022-06-30"), ("2022-07-01","2022-07-31"),
    ("2022-08-01","2022-08-31"), ("2022-09-01","2022-09-30"),
    ("2023-05-01","2023-05-31"), ("2023-06-01","2023-06-30"),
    ("2023-07-01","2023-07-31"), ("2023-08-01","2023-08-31"),
]


def _sh_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        f"{SH_URL}/auth/realms/main/protocol/openid-connect/token",
        data={"grant_type": "client_credentials",
              "client_id": client_id, "client_secret": client_secret},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _fetch_scene(token: str, bbox: list[float],
                 date_from: str, date_to: str,
                 w_px: int, h_px: int) -> np.ndarray | None:
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                "timeRange": {"from": f"{date_from}T00:00:00Z",
                              "to":   f"{date_to}T23:59:59Z"},
                "mosaickingOrder": "leastCC",
                "maxCloudCoverage": 80,
            }}],
        },
        "output": {
            "width": w_px, "height": h_px,
            "responses": [{"identifier": "default",
                           "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json",
               "Accept": "image/tiff"}
    for attempt in range(4):
        try:
            r = requests.post(f"{SH_URL}/api/v1/process",
                              headers=headers, json=payload, timeout=240)
            break
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            wait = 15 * (attempt + 1)
            print(f"  сетевая ошибка ({exc.__class__.__name__}), retry {attempt+1}/3 через {wait} с …")
            time.sleep(wait)
    else:
        print("  !! все попытки провалились, пропуск сцены")
        return None
    if r.status_code == 429:
        print("  rate-limit, жду 35 с …")
        time.sleep(35)
        return _fetch_scene(token, bbox, date_from, date_to, w_px, h_px)
    if r.status_code != 200:
        print(f"  !! HTTP {r.status_code}: {r.text[:200]}")
        return None
    arr = tifffile.imread(io.BytesIO(r.content))
    if arr.ndim != 3 or arr.shape[2] < 6:
        return None
    return arr.astype(np.float32)


def download_tile(lat: float, lon: float, radius_km: float,
                  client_id: str, client_secret: str,
                  out_path: Path, px: int = 2048) -> None:
    """Download multi-scene Sentinel-2 composite and save as .npz."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
    print(f"  BBox: {[round(v,4) for v in bbox]}")
    print(f"  Разрешение: {px}×{px} px")

    token = _sh_token(client_id, client_secret)
    print("  Токен получен. Скачиваю сцены …")

    b4_lst, b8_lst, b3_lst, b11_lst, b2_lst = [], [], [], [], []

    for i, (df, dt) in enumerate(_BASE_MONTHS, 1):
        print(f"  [{i}/{len(_BASE_MONTHS)}] {df[:7]} …", end=" ", flush=True)
        arr = _fetch_scene(token, bbox, df, dt, px, px)
        if arr is None:
            print("нет данных")
            continue
        b02, b03, b04, b08, b11, scl = [arr[..., k] for k in range(6)]
        valid = np.isin(scl.astype(np.uint8), list(SCL_VALID))
        vf = valid.mean()
        print(f"valid={vf:.2f}")
        if vf < 0.10:
            print("    пропуск (слишком облачно)")
            continue
        b4_lst .append(np.where(valid, b04, np.nan).astype(np.float32))
        b8_lst .append(np.where(valid, b08, np.nan).astype(np.float32))
        b3_lst .append(np.where(valid, b03, np.nan).astype(np.float32))
        b11_lst.append(np.where(valid, b11, np.nan).astype(np.float32))
        b2_lst .append(np.where(valid, b02, np.nan).astype(np.float32))
        time.sleep(0.5)

    if not b4_lst:
        print("Нет валидных сцен. Прерываю.")
        sys.exit(1)

    print(f"\n  Составляю композит из {len(b4_lst)} сцен …")
    s4  = np.stack(b4_lst,  axis=0)
    s8  = np.stack(b8_lst,  axis=0)
    s3  = np.stack(b3_lst,  axis=0)
    s11 = np.stack(b11_lst, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_stk = (s8 - s4) / (s8 + s4 + 1e-9)
        ndwi_stk = (s3 - s8) / (s3 + s8 + 1e-9)

    # Replace all-NaN pixels with 0 so stats and argmax don't crash
    all_nan = np.isnan(ndvi_stk).all(axis=0)
    ndvi_stk[:, all_nan] = 0.0
    ndwi_stk[:, all_nan] = 0.0

    maxndvi   = np.nanmax(ndvi_stk,  axis=0)
    meanndvi  = np.nanmean(ndvi_stk, axis=0)
    ndvistd   = np.nanstd(ndvi_stk,  axis=0)
    ndwi_mean = np.nanmean(ndwi_stk, axis=0)

    best_idx = np.nanargmax(ndvi_stk, axis=0)
    fi = best_idx.ravel()
    ri = np.arange(fi.size)
    def _pick(stk): return stk.reshape(len(b4_lst), -1)[fi, ri].reshape(px, px)
    red_median   = _pick(s4)
    green_median = _pick(s3)
    nir_median   = _pick(s8)
    swir_median  = _pick(s11)

    ndvi_best = np.nan_to_num(maxndvi, nan=0.0)
    smoothed  = gaussian_filter(ndvi_best, sigma=1.0)
    sx = sobel(smoothed, axis=1).astype(np.float32)
    sy = sobel(smoothed, axis=0).astype(np.float32)
    edgecomposite = np.hypot(sx, sy).astype(np.float32)
    p99 = np.percentile(edgecomposite, 99)
    if p99 > 0:
        edgecomposite = np.clip(edgecomposite / p99, 0, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        edgecomposite = edgecomposite.astype(np.float32),
        maxndvi       = maxndvi      .astype(np.float32),
        meanndvi      = meanndvi     .astype(np.float32),
        ndvistd       = ndvistd      .astype(np.float32),
        ndwi_mean     = ndwi_mean    .astype(np.float32),
        mndwi_max     = ndwi_mean    .astype(np.float32),
        red_median    = red_median   .astype(np.float32),
        green_median  = green_median .astype(np.float32),
        nir_median    = nir_median   .astype(np.float32),
        swir_median   = swir_median  .astype(np.float32),
        bbox          = np.array(bbox, dtype=np.float64),
    )
    print(f"  Тайл сохранён → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


# ── Tile rendering ────────────────────────────────────────────────────────────

def load_tile(npz_path: Path, out_w: int, out_h: int,
              zoom: float = 1.38, cx_frac: float = 0.5,
              cy_frac: float = 0.5) -> dict[str, np.ndarray]:
    d     = np.load(str(npz_path), allow_pickle=False)
    keys  = ["edgecomposite","maxndvi","meanndvi","ndvistd","ndwi_mean",
             "mndwi_max","red_median","green_median","nir_median","swir_median"]
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
    v = arr[np.isfinite(arr)].ravel()
    vlo, vhi = np.percentile(v, lo), np.percentile(v, hi)
    if vhi <= vlo + 1e-7:
        return np.zeros_like(arr)
    return np.clip((arr - vlo) / (vhi - vlo), 0.0, 1.0).astype(np.float32)


def _water_overlay(comp, ndwi, H, W):
    water = np.clip((ndwi - 0.35) / 0.35, 0, 1)
    def blur(r):
        return np.asarray(
            Image.fromarray((water*255).clip(0,255).astype(np.uint8),"L")
                .filter(ImageFilter.GaussianBlur(radius=r)),
            dtype=np.float32) / 255.0
    comp += np.stack([blur(12)*0.0, blur(12)*0.18, blur(12)*0.25], axis=-1)
    cyan = np.array([0.05, 0.75, 0.95], dtype=np.float32)
    wa   = blur(3)[..., None] * 0.82
    return np.clip(comp*(1-wa) + cyan*wa, 0, 1)


def _vignette(comp, H, W):
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    return comp * np.clip(1.0 - 0.36*(xx**2*0.4 + yy**2), 0, 1)[..., None]


def render_edge(t, cmap):
    H, W = WP_H, WP_W
    ndvi = pct_clip(t["maxndvi"], 5, 95)
    base = np.stack([np.power(ndvi, 2.2)*0.08*c for c in (0.3, 0.6, 0.4)], axis=-1)
    ec   = pct_clip(t["edgecomposite"], 0, 99)
    ec_g = np.power(ec, 0.76)
    rgba = cmap(ec_g)
    alpha = np.clip(ec_g[..., None] * 0.92, 0, 1)
    comp  = base*(1-alpha) + rgba[..., :3].astype(np.float32)*alpha
    comp  = _water_overlay(comp, t["ndwi_mean"], H, W)
    return (_vignette(comp, H, W) * 255).clip(0, 255).astype(np.uint8)


def render_ndvi(t, cmap):
    H, W = WP_H, WP_W
    ndvi = pct_clip(t["maxndvi"], 2, 98)
    rgba = cmap(ndvi)
    comp = rgba[..., :3].astype(np.float32)
    comp = _water_overlay(comp, t["ndwi_mean"], H, W)
    return (_vignette(comp, H, W) * 255).clip(0, 255).astype(np.uint8)


def render_ndvi_var(t, cmap):
    H, W = WP_H, WP_W
    var  = pct_clip(t["ndvistd"], 2, 98)
    rgba = cmap(var)
    comp = rgba[..., :3].astype(np.float32)
    comp = _water_overlay(comp, t["ndwi_mean"], H, W)
    return (_vignette(comp, H, W) * 255).clip(0, 255).astype(np.uint8)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _sh(draw, xy, text, font, fill, anchor="lt", sc=(0,0,0), sa=215, off=(2,2)):
    draw.text((xy[0]+off[0], xy[1]+off[1]), text, font=font, fill=sc+(sa,), anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _grad(wp, y0, h, inv):
    s = Image.new("RGBA", (WP_W, h), (0,0,0,0))
    for row in range(h):
        t = row / h
        a = int(192 * ((1-t) if inv else t) ** 0.54)
        ImageDraw.Draw(s).line([(0,row),(WP_W,row)], fill=(3,5,12,a))
    wp.paste(s.convert("RGB"), (0, y0), s.split()[3])


def composite_ghost_digits(wp, d1, d2, f_big, X, Y1, Y2, W, H, d1_dx=0):
    sl = Image.new("L", (W, H), 0)
    dd = ImageDraw.Draw(sl)
    dd.text((X,       Y2), d2, font=f_big, fill=212, anchor="lt")
    dd.text((X+d1_dx, Y1), d1, font=f_big, fill=232, anchor="lt")
    glow     = sl.filter(ImageFilter.GaussianBlur(radius=12))
    combined = Image.blend(sl, glow, 0.38)
    mask     = combined.point(lambda v: int(v * 0.40))
    wp.paste(Image.new("RGB", (W,H), (218,238,255)), (0,0), mask)


def draw_cmd_window(wp, x1, y1, win_w, win_h, active_ch="edge"):
    BEVEL, TITLE_H, PAD = 2, 28, 12
    ov = Image.new("RGBA", (win_w+8, win_h+8), (0,0,0,0))
    d  = ImageDraw.Draw(ov)
    d.rectangle([(5,5),(win_w+7,win_h+7)], fill=(0,0,0,148))
    d.rectangle([(0,0),(win_w,win_h)],     fill=(50,54,70,255))
    HL, SH = (122,130,162,255), (10,12,20,255)
    for i in range(BEVEL):
        d.line([(i,i),(win_w-i,i)],             fill=HL)
        d.line([(i,i),(i,win_h-i)],             fill=HL)
        d.line([(i,win_h-i),(win_w-i,win_h-i)],fill=SH)
        d.line([(win_w-i,i),(win_w-i,win_h-i)],fill=SH)
    for row in range(TITLE_H-BEVEL):
        t = row / max(TITLE_H-BEVEL-1,1)
        d.line([(BEVEL,BEVEL+row),(win_w-BEVEL,BEVEL+row)],
               fill=(int(2+t*5),int(6+t*12),int(72+t*58),255))
    d.line([(BEVEL,TITLE_H),(win_w-BEVEL,TITLE_H)], fill=(6,10,30,255))
    d.rectangle([(BEVEL,TITLE_H+1),(win_w-BEVEL,win_h-BEVEL)], fill=(5,7,16,245))
    d.line([(BEVEL,TITLE_H+1),(win_w-BEVEL,TITLE_H+1)], fill=(8,10,22,255))
    d.line([(BEVEL,TITLE_H+1),(BEVEL,win_h-BEVEL)],     fill=(8,10,22,255))
    d.line([(BEVEL,win_h-BEVEL),(win_w-BEVEL,win_h-BEVEL)], fill=(58,68,92,255))
    d.line([(win_w-BEVEL,TITLE_H+1),(win_w-BEVEL,win_h-BEVEL)], fill=(58,68,92,255))
    wp.paste(ov.convert("RGB"), (x1-4, y1-4), ov)

    draw  = ImageDraw.Draw(wp)
    f_ttl = _font(13, _MONO_BOLD)
    f_btn = _font(12, _MONO)
    f_cmd = _font(18, _MONO)

    ty_t = y1 + BEVEL + (TITLE_H-BEVEL)//2
    draw.text((x1+BEVEL+8, ty_t), "TERRAINFO — Sentinel-2 Analysis",
              font=f_ttl, fill=(218,228,255), anchor="lm")

    BW, BH = 22, TITLE_H-BEVEL-4
    by_b   = y1+BEVEL+2
    bx_r   = x1+win_w-BEVEL-2
    for lbl, face, txt in [("×",(138,34,34),(255,195,195)),
                            ("□",(50,56,80),(198,210,242)),
                            ("─",(50,56,80),(198,210,242))]:
        bx_r -= BW+1
        draw.rectangle([(bx_r,by_b),(bx_r+BW-1,by_b+BH)], fill=face)
        draw.line([(bx_r,by_b),(bx_r+BW-1,by_b)],          fill=(158,166,200))
        draw.line([(bx_r,by_b),(bx_r,by_b+BH)],            fill=(158,166,200))
        draw.line([(bx_r,by_b+BH),(bx_r+BW-1,by_b+BH)],   fill=(12,14,24))
        draw.line([(bx_r+BW-1,by_b),(bx_r+BW-1,by_b+BH)], fill=(12,14,24))
        draw.text((bx_r+BW//2,by_b+BH//2+1), lbl, font=f_btn, fill=txt, anchor="mm")

    GRN,WHT,DIM,ACC,VAL,SEP = (0,212,88),(208,222,248),(100,120,148),(0,208,178),(152,190,220),(30,42,60)
    tx, ty, LH = x1+BEVEL+PAD, y1+TITLE_H+PAD+4, 21

    def star(ch): return "[*]" if active_ch==ch else "[ ]"
    def col(ch):  return ACC   if active_ch==ch else DIM

    def prompt(cmd):
        nonlocal ty
        draw.text((tx,ty), "C:\\>", font=f_cmd, fill=GRN)
        if cmd:
            draw.text((tx+int(draw.textlength("C:\\>",font=f_cmd))+5,ty), cmd, font=f_cmd, fill=WHT)
        ty += LH

    def row(k, v, kc=DIM, vc=VAL):
        nonlocal ty
        draw.text((tx,ty), k, font=f_cmd, fill=kc)
        draw.text((tx+int(draw.textlength(k,font=f_cmd)),ty), v, font=f_cmd, fill=vc)
        ty += LH

    def sep():
        nonlocal ty
        draw.line([(tx,ty+4),(x1+win_w-BEVEL-PAD,ty+4)], fill=SEP, width=1)
        ty += LH//2+5

    prompt("query --metadata"); sep()
    row("Satellite  ","Sentinel-2 L2A")
    row("Access     ","SentinelHub API  (OAuth2)")
    row("Level      ","L2A  Surface Reflectance")
    row("GSD        ","10 м/пкс  (SWIR: 20→10 м)")
    row("Period     ","2021 – 2023  (multiyear composite)")
    row("Projection ","EPSG:4326  WGS84")
    sep()
    prompt("list --channels --verbose"); sep()
    row(f"{star('edge')} ","edge_composite",              col('edge'), col('edge'))
    row("    ","spectral gradient magnitude",             vc=DIM)
    row(f"{star('ndvi')} ","NDVI  max · mean  ",          col('ndvi'), col('ndvi'))
    row("    ","(B08-B04)/(B08+B04)",                     vc=DIM)
    row(f"{star('var')}  ","NDVI  variability (max−mean)",col('var'),  col('var'))
    row("    ","seasonal amplitude  ← crops bright",      vc=DIM)
    row("[ ] ","NDWI   mean  (B03-B08)/(B03+B08)",       vc=DIM)
    row("[ ] ","B04 B03 B08 B11   10/20m  median",       vc=DIM)
    sep()
    draw.text((tx,ty), "C:\\>", font=f_cmd, fill=GRN)
    draw.text((tx+int(draw.textlength("C:\\>",font=f_cmd))+5,ty), "█", font=f_cmd, fill=GRN)


def draw_colorbar_section(draw, cmap, bar_label, f_mono, W, H, display_fn=None):
    BAR_W, BAR_H = 540, 11
    bx, by = (W-BAR_W)//2, H-130
    for px in range(BAR_W):
        v = px / BAR_W
        if display_fn:
            r,g,b = display_fn(v)
        else:
            r,g,b,_ = [int(c*255) for c in cmap(v)]
        draw.line([(bx+px,by),(bx+px,by+BAR_H)], fill=(r,g,b))
    draw.rectangle([(bx-1,by-1),(bx+BAR_W,by+BAR_H)], outline=(58,78,98), width=1)
    f = f_mono
    _sh(draw,(bx,by+BAR_H+8),"0.0",f,(185,215,238),anchor="lt")
    _sh(draw,(bx+BAR_W,by+BAR_H+8),"1.0",f,(185,215,238),anchor="rt")
    _sh(draw,(W//2,by-18),bar_label,f,(158,192,218),anchor="mm")


def _edge_bar_color(v):
    ec_g  = v**0.76
    alpha = min(ec_g*0.92, 1.0)
    base  = (0.012, 0.024, 0.016)
    rgba  = EDGE_CMAP(ec_g)
    return tuple(int((base[i]*(1-alpha)+rgba[i]*alpha)*255) for i in range(3))


def build_wallpaper(layer, cmap, out_path, city_lines, region_code,
                    subtitles, phi, lam, bar_label, active_ch="edge", digit_dx=0,
                    cmd_below=False):
    wp   = Image.fromarray(layer, "RGB")
    W, H = WP_W, WP_H
    _grad(wp, 0, 620, inv=True)
    _grad(wp, H-230, 230, inv=False)

    f_big   = _font(950, _DISPLAY)
    bb      = f_big.getbbox(region_code[0])
    digit_h = bb[3]-bb[1]
    X  = W-325
    Y1 = H//2 - digit_h//8 - 35
    Y2 = Y1+635
    composite_ghost_digits(wp, region_code[0], region_code[1],
                           f_big, X, Y1, Y2, W, H, d1_dx=digit_dx)

    draw   = ImageDraw.Draw(wp)
    n_city = len(city_lines)
    city_sz = 205 if n_city == 1 else 135

    # Auto-shrink single-line city name if it doesn't fit in wallpaper width
    if n_city == 1:
        max_allowed = W - 52 - 30
        while city_sz > 80:
            f_test = _font(city_sz, _DISPLAY)
            if int(draw.textlength(city_lines[0], font=f_test)) <= max_allowed:
                break
            city_sz -= 5

    sub_sz  = max(52, city_sz//3)
    f_city  = _font(city_sz, _DISPLAY)
    f_sub   = _font(sub_sz,  _BOLD)
    f_sub2  = _font(int(sub_sz*0.88), _BOLD)
    f_mono  = _font(24, _MONO)

    y_cursor = 205
    for i, line in enumerate(city_lines):
        if i > 0:
            y_cursor += city_sz+18
        _sh(draw, (52, y_cursor), line, f_city, (248,254,255), anchor="lm")

    y_cursor += city_sz//2+28
    for i, sub in enumerate(subtitles):
        font = f_sub if i==0 else f_sub2
        _sh(draw, (52, y_cursor), sub, font, (242,248,255), anchor="lm", sc=(0,0,0), sa=215)
        y_cursor += (sub_sz if i==0 else int(sub_sz*0.88))+14

    CMD_W, CMD_H = 512, 496
    cmd_x_left   = W-CMD_W-24
    # Check city lines AND subtitles — CMD drops if any text would overlap it
    all_widths = [int(draw.textlength(l, font=f_city)) for l in city_lines]
    if subtitles:
        all_widths.append(int(draw.textlength(subtitles[0], font=f_sub)))
        for s in subtitles[1:]:
            all_widths.append(int(draw.textlength(s, font=f_sub2)))
    max_text_w = max(all_widths)
    if cmd_below or (52+max_text_w+24 > cmd_x_left):
        cmd_y = y_cursor + 20
    else:
        cmd_y = 205 - city_sz // 2
    draw_cmd_window(wp, cmd_x_left, cmd_y, CMD_W, CMD_H, active_ch=active_ch)

    cy = H//2
    f_coord = _font(42, _MONO_BOLD)
    draw.rectangle([(42,cy-62),(46,cy+62)], fill=(0,230,245))
    _sh(draw, (54,cy-26), phi, f_coord, (255,255,255), anchor="lm", sa=245, off=(2,2))
    _sh(draw, (54,cy+26), lam, f_coord, (255,255,255), anchor="lm", sa=245, off=(2,2))

    bar_fn = _edge_bar_color if active_ch=="edge" else None
    draw_colorbar_section(draw, cmap, bar_label, f_mono, W, H, display_fn=bar_fn)

    f_b = _font(44, _BOLD); f_t = _font(30, _BOLD)
    draw.text((50,H-56), "ТерраINFO", font=f_b, fill=(0,0,0,200), anchor="lt")
    draw.text((48,H-58), "ТерраINFO", font=f_b, fill=(255,255,255), anchor="lt")
    tw = int(draw.textlength("ТерраINFO", font=f_b))
    draw.text((48+tw+2,H-64), "™", font=f_t, fill=(220,220,220), anchor="lt")

    wp.save(str(out_path), format="PNG", compress_level=1)
    print(f"  → {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


# ── Interactive prompts ───────────────────────────────────────────────────────

def _ask(prompt: str, default=None) -> str:
    hint = f"  [{default}]" if default is not None else ""
    while True:
        val = input(f"{prompt}{hint}: ").strip()
        if val:
            return val
        if default is not None:
            return str(default)
        print("  (обязательное поле)")


def _ask_float(prompt: str, default=None) -> float:
    while True:
        raw = _ask(prompt, default)
        try:
            return float(raw)
        except ValueError:
            print("  Нужно число, например 57.426")


def _ask_subtitles() -> list[str]:
    items = []
    print("Подпись под названием города (Enter после каждой строки; пустой Enter — завершить):")
    while True:
        val = input("  > ").strip()
        if not val:
            break
        items.append(val)
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  TERRAINFO  —  генератор обоев Sentinel-2  (1440×3088)")
    print("=" * 60)

    # ── Credentials from .env ──────────────────────────────────────────────
    env = _load_env()
    client_id     = (env.get("SH_CLIENT_ID_SECOND_RESERVE")
                  or env.get("SH_CLIENT_ID_RESERVE")
                  or env.get("SH_CLIENT_ID", ""))
    client_secret = (env.get("SH_CLIENT_SECRET_SECOND_RESERVE")
                  or env.get("SH_CLIENT_SECRET_RESERVE")
                  or env.get("SH_CLIENT_SECRET", ""))
    if not client_id or not client_secret:
        print("\nКлюч SentinelHub не найден в .env — введите вручную:")
        client_id     = _ask("  SH_CLIENT_ID")
        client_secret = _ask("  SH_CLIENT_SECRET")
    else:
        print(f"\nКредентиалы загружены из .env  (id: {client_id[:8]}…)")

    # ── City ───────────────────────────────────────────────────────────────
    print()
    print('Название города (для двух строк используйте "/", например: САНКТ-/ПЕТЕРБУРГ):')
    city_raw   = _ask("Город")
    city_lines = [p.strip() for p in city_raw.split("/") if p.strip()]

    # ── Region code ────────────────────────────────────────────────────────
    while True:
        code = _ask("Номер региона (2 цифры)")
        if len(code) == 2 and code.isdigit():
            break
        print("  Нужно ровно 2 цифры, например 59")

    # ── Subtitles ──────────────────────────────────────────────────────────
    print()
    subtitles = _ask_subtitles()

    # ── Coordinates ────────────────────────────────────────────────────────
    print()
    lat = _ask_float("Широта центра (например 57.426)")
    lon = _ask_float("Долгота центра (например 56.955)")
    phi = _to_dms(lat, is_lat=True)
    lam = _to_dms(lon, is_lat=False)
    print(f"  → {phi}   {lam}")

    # ── Radius ─────────────────────────────────────────────────────────────
    radius_km = _ask_float("Радиус охвата, км", default=50)

    # ── Advanced ───────────────────────────────────────────────────────────
    print("\nДополнительно (Enter — оставить по умолчанию):")
    zoom     = _ask_float("  Zoom кадрирования тайла", default=1.38)
    cx       = _ask_float("  Смещение кадра по X (0.0–1.0)", default=0.5)
    cy_frac  = _ask_float("  Смещение кадра по Y (0.0–1.0)", default=0.5)
    digit_dx = int(_ask("  Сдвиг верхней цифры региона, px", default=0))

    # ── Paths ──────────────────────────────────────────────────────────────
    slug     = _slug(city_raw, code)
    tile_path = TILES_DIR / f"{slug}.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download or reuse tile ─────────────────────────────────────────────
    if tile_path.exists():
        print(f"\nТайл уже существует: {tile_path.name}")
        reuse = _ask("Использовать существующий? (y/n)", default="y").lower()
        if reuse != "y":
            tile_path.unlink()

    if not tile_path.exists():
        print(f"\nСкачиваю спутниковые данные для {city_raw} …")
        download_tile(lat, lon, radius_km, client_id, client_secret, tile_path)

    # ── Load & render ──────────────────────────────────────────────────────
    print(f"\nЗагружаю тайл (zoom={zoom}, cx={cx}, cy={cy_frac}) …")
    t  = load_tile(tile_path, WP_W, WP_H, zoom=zoom, cx_frac=cx, cy_frac=cy_frac)
    ec = t["edgecomposite"]
    print(f"  ec mean={ec.mean():.3f}  ndvi mean={t['maxndvi'].mean():.3f}")

    configs = [
        dict(render=render_edge,     cmap=EDGE_CMAP,     suffix="edge",
             active="edge", bar="Sentinel-2  ·  edge_composite  ·  нормализованная интенсивность"),
        dict(render=render_ndvi,     cmap=NDVI_CMAP,     suffix="ndvi",
             active="ndvi", bar="Sentinel-2  ·  NDVI max  ·  индекс растительности"),
        dict(render=render_ndvi_var, cmap=NDVI_VAR_CMAP, suffix="ndvi_var",
             active="var",  bar="Sentinel-2  ·  NDVI STD  ·  межсезонная изменчивость"),
    ]

    print()
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/3] {city_raw} {cfg['suffix']} …")
        layer    = cfg["render"](t, cfg["cmap"])
        out_path = OUT_DIR / f"wallpaper_{slug}_{cfg['suffix']}.png"
        build_wallpaper(
            layer=layer, cmap=cfg["cmap"], out_path=out_path,
            city_lines=city_lines, region_code=code, subtitles=subtitles,
            phi=phi, lam=lam, bar_label=cfg["bar"],
            active_ch=cfg["active"], digit_dx=digit_dx,
        )

    print(f"\nГотово — 3 обоя сохранены в {OUT_DIR}/")


if __name__ == "__main__":
    main()
