#!/usr/bin/env python3
"""
train_regional_supplement.py — Autonomous regional fine-tuning pipeline.

Regions: Краснодарский край (OKATO 23) · Пермский край (OKATO 59)

Pipeline steps (all automatic):
  1. fetch      — Download Sentinel-2 tiles for optimal agricultural zones via Sentinel Hub
  2. labels     — Generate weak supervision labels (re-uses generate_weak_labels_real_tiles.py)
  3. finetune   — Fine-tune BoundaryUNet: freeze encoder, train decoder on regional patches
  4. classifier — Retrain HistGBM with regional samples boosted by 3–4× weight
  5. export     — Save .pth, ONNX, regional .pkl; bundle artifact manifest
  6. report     — Write JSON training report with per-region metrics

Usage:
    # Full pipeline (all regions):
    python training/train_regional_supplement.py --project-root /path/to/project

    # Specific region, skip already-downloaded tiles:
    python training/train_regional_supplement.py --regions krasnodar --skip-fetch

    # Dry-run: print plan without executing anything:
    python training/train_regional_supplement.py --dry-run

    # Custom epochs:
    python training/train_regional_supplement.py --epochs 40 --batch-size 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from math import cos, radians
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# ── Path bootstrap (same pattern as fetch_real_tiles.py) ──────────────────────
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
BACKEND_DIR = PROJECT_ROOT / "backend"

load_dotenv(PROJECT_ROOT / ".env")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regional_supplement")

# ── Shared dirs (same as global pipeline) ─────────────────────────────────────
TILES_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles"
LABELS_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles_labels_weak"
MODELS_DIR = PROJECT_ROOT / "backend/models"

GLOBAL_UNET_CHECKPOINT = MODELS_DIR / "boundary_unet_v3_cpu.pth"
GLOBAL_CLASSIFIER_PKL = MODELS_DIR / "object_classifier.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# REGION DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TileCenter:
    """Single training tile defined by its centroid."""
    tile_id: str
    lat: float
    lon: float
    description: str = ""


@dataclass
class RegionalProfile:
    """Post-processing parameter overrides for a federal subject."""
    # Inference thresholds
    ml_extent_bin_threshold: float = 0.40
    min_field_area_ha: float = 1.0
    boundary_outer_dilation_px: int = 1
    vectorize_simplify_tol_m: float = 1.5
    water_edge_risk_threshold: float = 0.65
    road_split_min_width_m: float = 15.0
    # Fine-tuning hyperparameters
    finetune_lr_multiplier: float = 0.10   # global_lr × this
    finetune_freeze_pct: float = 0.60      # fraction of params to freeze (encoder)
    # Classifier
    classifier_regional_weight: float = 3.0   # regional sample weight vs global


@dataclass
class RegionConfig:
    code: str           # OKATO code
    name_ru: str
    tiles: list[TileCenter]
    profile: RegionalProfile
    # Sentinel-2 date range — two full growing seasons
    date_start: date = field(default_factory=lambda: date(2022, 4, 1))
    date_end: date = field(default_factory=lambda: date(2024, 9, 30))
    # WorldCover crop mask relaxation flag (False = use standard ≥10% threshold)
    wc_relaxed: bool = False


REGIONS: dict[str, RegionConfig] = {

    # ── Краснодарский край ─────────────────────────────────────────────────────
    # Характеристики: крупные прямоугольные поля (≥50 га), рисовые чеки с водным
    # фоном летом, чёткие границы, виноградники на предгорье, лесополосы.
    "krasnodar": RegionConfig(
        code="23",
        name_ru="Краснодарский край",
        wc_relaxed=False,   # высокий NDVI — стандартные пороги работают
        profile=RegionalProfile(
            ml_extent_bin_threshold=0.36,   # поля чёткие → ниже порог
            min_field_area_ha=1.5,          # рисовые чеки мелкие, включаем
            boundary_outer_dilation_px=0,   # границы резкие — без расширения
            vectorize_simplify_tol_m=2.5,   # крупные поля — грубее упрощение
            water_edge_risk_threshold=0.50, # рис: вода не является аномалией
            road_split_min_width_m=20.0,    # лесополосы тоже барьеры
            finetune_lr_multiplier=0.08,
            finetune_freeze_pct=0.55,
            classifier_regional_weight=3.0,
        ),
        tiles=[
            # Центральная Кубанская равнина (пшеница / подсолнечник / кукуруза)
            TileCenter("krd_kuban_w1",  45.45, 38.80, "Куб. равнина — запад"),
            TileCenter("krd_kuban_w2",  45.60, 39.30, "Куб. равнина — центр-запад"),
            TileCenter("krd_kuban_c1",  45.55, 40.00, "Куб. равнина — центр"),
            TileCenter("krd_kuban_c2",  45.30, 39.80, "Куб. равнина — центр-юг"),
            TileCenter("krd_kuban_e1",  45.50, 40.90, "Куб. равнина — восток"),
            TileCenter("krd_kuban_e2",  45.25, 41.35, "Куб. равнина — юго-восток"),
            # Рисовая зона (вдоль Кубани, Темрюк — Краснодар)
            TileCenter("krd_rice_w",    45.25, 37.80, "Рисовая зона — запад (Темрюк)"),
            TileCenter("krd_rice_e",    45.05, 38.40, "Рисовая зона — восток (р. Кубань)"),
            # Азово-Кубанская степь (северные районы)
            TileCenter("krd_azov_w",    46.30, 38.50, "Азовская степь — запад"),
            TileCenter("krd_azov_e",    46.20, 39.80, "Азовская степь — восток"),
            # Предгорная зона (виноградники, сады, сложный рельеф)
            TileCenter("krd_piedmt",    44.55, 40.20, "Предгорье — Горячий Ключ"),
            # Восточная степь (Сальские степи, граница с Ростовской обл.)
            TileCenter("krd_east",      45.70, 42.20, "Восточная степь — Кропоткин"),
        ],
    ),

    # ── Пермский край ──────────────────────────────────────────────────────────
    # Характеристики: мелкоконтурные поля (1–15 га), высокая лесистость,
    # нечёткие границы, короткий вегетационный период, болота, торфяники,
    # пересечённый рельеф предгорий Урала.
    # Используем префикс permkrai_ — он есть в WC_RELAXED_REGION_PREFIXES
    # в generate_weak_labels_real_tiles.py → порог WorldCover ослаблен.
    "perm": RegionConfig(
        code="59",
        name_ru="Пермский край",
        wc_relaxed=True,
        profile=RegionalProfile(
            ml_extent_bin_threshold=0.44,   # мелкоконтурные поля — выше порог
            min_field_area_ha=0.8,          # мелкие поля включаем
            boundary_outer_dilation_px=2,   # нечёткие границы — расширение
            vectorize_simplify_tol_m=1.0,   # точнее — поля неправильной формы
            water_edge_risk_threshold=0.70, # реки/болота — повышенный риск
            road_split_min_width_m=10.0,    # узкие дороги тоже барьеры
            finetune_lr_multiplier=0.12,
            finetune_freeze_pct=0.50,       # больше свободы — регион сложный
            classifier_regional_weight=4.0, # редкий регион — выше вес
        ),
        tiles=[
            # Чернушинский р-н (лучшие пашни юга Прикамья)
            TileCenter("permkrai_chernushka",  56.50, 56.50, "Чернушинский р-н"),
            # Куединский р-н (пашня / луга)
            TileCenter("permkrai_kueda",       56.40, 55.30, "Куединский р-н"),
            # Чайковский р-н (у границы с Удмуртией)
            TileCenter("permkrai_chaikov",     56.85, 54.20, "Чайковский р-н"),
            # Бардымский р-н (мозаика угодий)
            TileCenter("permkrai_barda",       56.70, 56.15, "Бардымский р-н"),
            # Осинский р-н (поймы Камы)
            TileCenter("permkrai_osa",         57.40, 55.70, "Осинский р-н"),
            # Кунгурский р-н (речные долины, карстовые поля)
            TileCenter("permkrai_kungur",      57.40, 57.20, "Кунгурский р-н"),
            # Очёрский р-н (торфяники + пашня)
            TileCenter("permkrai_ocher",       57.90, 54.90, "Очёрский р-н"),
            # Верещагинский р-н
            TileCenter("permkrai_veresh",      57.65, 54.80, "Верещагинский р-н"),
            # Ординский р-н
            TileCenter("permkrai_ordin",       57.15, 56.40, "Ординский р-н"),
            # Карагайский р-н (мозаика лес/поле)
            TileCenter("permkrai_karagan",     58.20, 54.40, "Карагайский р-н"),
        ],
    ),

    # ── Северо-Запад РФ (Северные широты) ──────────────────────────────────────
    "northwest": RegionConfig(
        code="41",
        name_ru="Северо-Запад РФ",
        wc_relaxed=True,
        profile=RegionalProfile(
            ml_extent_bin_threshold=0.45,
            min_field_area_ha=0.5,
            boundary_outer_dilation_px=2,
            vectorize_simplify_tol_m=1.0,
            water_edge_risk_threshold=0.80,
            road_split_min_width_m=8.0,
            finetune_lr_multiplier=0.15,
            finetune_freeze_pct=0.45,
            classifier_regional_weight=5.0,
        ),
        tiles=[
            TileCenter("nw_priozersk", 60.85, 30.15, "Приозерский р-н (Лен. обл.)"),
            TileCenter("nw_vyborg", 60.55, 29.10, "Выборгский р-н (Лен. обл.)"),
            TileCenter("nw_cherepovets", 59.20, 38.00, "Череповецкий р-н (Вологодская обл.)"),
            TileCenter("nw_novgorod", 58.40, 31.30, "Новгородский р-н"),
        ],
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS  (same as fetch_real_tiles.py)
# ──────────────────────────────────────────────────────────────────────────────

TILE_HALF_KM = 10.0   # 20 km × 20 km coverage per tile @ 10 m/px → 2 000 × 2 000 px
TILE_PX = 1024        # actual pixel size used (10.24 km per side at 10 m/px)
WINDOW_DAYS = 30
DATE_RANGE_START = date(2022, 4, 1)
DATE_RANGE_END = date(2024, 9, 30)


def bbox_from_center(lat: float, lon: float, half_km: float = TILE_HALF_KM) -> tuple[float, float, float, float]:
    dlat = half_km / 110.574
    dlon = half_km / (111.320 * max(0.05, cos(radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def build_time_windows(start: date, end: date, window_days: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        wend = min(end, cur + timedelta(days=window_days - 1))
        windows.append((f"{cur.isoformat()}T00:00:00Z", f"{wend.isoformat()}T23:59:59Z"))
        cur = wend + timedelta(days=1)
    return windows


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — FETCH SENTINEL-2 TILES
# ──────────────────────────────────────────────────────────────────────────────

def _load_runtime_deps():
    """Lazy-load backend dependencies (same pattern as fetch_real_tiles.py)."""
    from core.config import get_settings  # type: ignore
    settings = get_settings()

    from providers.sentinelhub.client import SentinelHubClient  # type: ignore
    client = SentinelHubClient()

    from processing.fields.indices import compute_all_indices  # type: ignore
    from processing.fields.composite import (  # type: ignore
        build_valid_mask_from_scl,
        select_dates_by_coverage,
    )
    try:
        from processing.fields.temporal_composite import build_multiyear_composite  # type: ignore
    except ImportError:
        from processing.fields.temporalcomposite import build_multiyear_composite  # type: ignore

    return settings, client, compute_all_indices, build_valid_mask_from_scl, \
           select_dates_by_coverage, build_multiyear_composite


async def _fetch_one_tile(
    tile: TileCenter,
    *,
    out_dir: Path,
    time_windows: list[tuple[str, str]],
    min_scenes: int,
    skip_existing: bool,
    client: Any,
    compute_all_indices: Any,
    build_valid_mask_from_scl: Any,
    select_dates_by_coverage: Any,
    build_multiyear_composite: Any,
    settings: Any,
    tile_px: int,
) -> bool:
    """Download and preprocess a single Sentinel-2 tile — mirrors fetch_real_tiles logic."""
    out_path = out_dir / f"{tile.tile_id}.npz"
    if skip_existing and out_path.exists():
        log.info("  skip  %s (exists)", tile.tile_id)
        return True

    bbox = bbox_from_center(tile.lat, tile.lon)
    log.info("  fetch %s  bbox=%.3f,%.3f→%.3f,%.3f", tile.tile_id, *bbox)

    # Accumulate per-band lists — response format: {B2, B3, B4, B8, B11, B12, SCL}
    BAND_KEYS = ["B2", "B3", "B4", "B8", "B11", "B12"]
    band_lists: dict[str, list[np.ndarray]] = {k: [] for k in BAND_KEYS}
    scl_list: list[np.ndarray] = []

    for t_from, t_to in time_windows:
        try:
            resp = await client.fetch_tile(bbox, t_from, t_to, tile_px, tile_px, max_cloud_pct=60)
            if resp is None:
                continue
            # Each band is a (H, W) array in the response dict
            missing = [k for k in BAND_KEYS if k not in resp]
            if missing:
                log.debug("  window %s missing bands %s, skip", t_from[:10], missing)
                continue
            for k in BAND_KEYS:
                band_lists[k].append(np.asarray(resp[k], dtype=np.float32))
            scl = resp.get("SCL") if "SCL" in resp else resp.get("scl")
            h, w = band_lists[BAND_KEYS[0]][-1].shape[-2], band_lists[BAND_KEYS[0]][-1].shape[-1]
            scl_list.append(np.asarray(scl, dtype=np.uint8) if scl is not None
                            else np.full((h, w), 4, dtype=np.uint8))
        except Exception as exc:
            log.debug("  window %s→%s failed: %s", t_from[:10], t_to[:10], exc)

    n_scenes = len(scl_list)
    if n_scenes < max(2, min_scenes):
        log.warning("  %s: only %d scenes (need %d), skipping", tile.tile_id, n_scenes, min_scenes)
        return False

    # Stack to (T, H, W) per band
    bands: dict[str, np.ndarray] = {k: np.stack(band_lists[k], axis=0).astype(np.float32)
                                     for k in BAND_KEYS}
    scl_stack = np.stack(scl_list, axis=0)  # (T, H, W)

    # Valid mask and date selection
    valid_mask = np.asarray(build_valid_mask_from_scl(scl_stack), dtype=bool)  # (T, H, W)

    try:
        result = select_dates_by_coverage(valid_mask, n_dates=min(8, n_scenes), return_metadata=True)
        selected = np.asarray(result[0] if isinstance(result, tuple) else result)
    except Exception:
        selected = np.arange(min(n_scenes, 8))

    if len(selected) < min_scenes:
        log.warning("  %s: only %d selected dates, skipping", tile.tile_id, len(selected))
        return False

    # Compute all spectral indices on full (T, H, W) bands
    indices = compute_all_indices(bands)
    ndvi = indices["NDVI"]           # (T, H, W)
    ndvi_sel = ndvi[selected]        # (T_sel, H, W)
    valid_sel = valid_mask[selected] # (T_sel, H, W)

    # Build multiyear composite
    edge_bands_for_comp = {k: bands[k][selected] for k in ["B2", "B3", "B4", "B8"]}
    edge_bands_for_comp["ndvi"] = ndvi_sel

    try:
        comp = build_multiyear_composite(
            ndvi_stack=ndvi_sel,
            valid_mask=valid_sel,
            edge_bands=edge_bands_for_comp,
            cfg=settings,
        )
    except Exception as exc:
        log.warning("  %s: composite failed (%s), using fallback", tile.tile_id, exc)
        comp = {}

    def _kget(d: dict, *keys: str) -> np.ndarray:
        for k in keys:
            if k in d:
                return np.asarray(d[k], dtype=np.float32)
        h, w_ = ndvi_sel.shape[1], ndvi_sel.shape[2]
        return np.zeros((h, w_), dtype=np.float32)

    def _masked_stat(arr: np.ndarray, stat: str = "mean") -> np.ndarray:
        """arr shape: (T_sel, H, W); valid_sel: (T_sel, H, W)."""
        a = np.where(valid_sel, arr, np.nan)
        with np.errstate(all="ignore"):
            if stat == "mean":   return np.nanmean(a, axis=0).astype(np.float32)
            if stat == "median": return np.nanmedian(a, axis=0).astype(np.float32)
            if stat == "max":    return np.nanmax(a, axis=0).astype(np.float32)
        return np.nanmean(a, axis=0).astype(np.float32)

    h, w = ndvi_sel.shape[1], ndvi_sel.shape[2]

    # Spectral statistics from real bands
    ndwi_sel  = indices["NDWI"][selected]
    mndwi_sel = indices["MNDWI"][selected]
    bsi_sel   = indices["BSI"][selected]
    ndmi_sel  = indices["NDMI"][selected]

    # NDVI temporal entropy
    ndvi_clipped = np.clip(np.where(valid_sel, ndvi_sel, np.nan), 0.0, 1.0)
    n_bins = 10
    bin_indices = np.clip((ndvi_clipped * n_bins).astype(np.int32), 0, n_bins - 1)
    valid_finite = np.isfinite(ndvi_clipped)
    valid_count_ent = valid_finite.sum(axis=0)
    ndvi_entropy = np.zeros((h, w), dtype=np.float32)
    for b in range(n_bins):
        count_b = ((bin_indices == b) & valid_finite).sum(axis=0).astype(np.float32)
        with np.errstate(all="ignore"):
            prob = count_b / np.maximum(valid_count_ent.astype(np.float32), 1.0)
            ndvi_entropy += np.where(prob > 0, -prob * np.log2(prob), 0.0).astype(np.float32)
    ndvi_entropy[valid_count_ent < 2] = 0.0

    save_dict: dict[str, np.ndarray] = {
        "edgecomposite":      _kget(comp, "edge_composite", "edgecomposite"),
        "maxndvi":            _kget(comp, "max_ndvi",       "maxndvi")     if comp else _masked_stat(ndvi_sel, "max"),
        "meanndvi":           _kget(comp, "mean_ndvi",      "meanndvi")    if comp else _masked_stat(ndvi_sel, "mean"),
        "ndvistd":            _kget(comp, "ndvi_std",       "ndvistd")     if comp else ndvi_sel.std(axis=0).astype(np.float32),
        "ndwi_mean":          _masked_stat(ndwi_sel,  "mean"),
        "ndwi_median":        _masked_stat(ndwi_sel,  "median"),
        "mndwi_max":          _masked_stat(mndwi_sel, "max"),
        "bsi_mean":           _masked_stat(bsi_sel,   "mean"),
        "ndmi_mean":          _masked_stat(ndmi_sel,  "mean"),
        "nir_median":         _masked_stat(bands["B8"][selected],  "median"),
        "red_median":         _masked_stat(bands["B4"][selected],  "median"),
        "green_median":       _masked_stat(bands["B3"][selected],  "median"),
        "blue_median":        _masked_stat(bands["B2"][selected],  "median"),
        "swir_median":        _masked_stat(bands["B11"][selected], "median"),
        "scl_valid_fraction": valid_mask.mean(axis=0).astype(np.float32),
        "ndvi_entropy":       ndvi_entropy,
        "n_valid_scenes":     np.int32(len(selected)),
        "bbox":               np.asarray(bbox, dtype=np.float64),
        "train_data_version": np.array("regional_v1", dtype="U32"),
        "feature_stack_version": np.array("v3_candidate_16ch_cpu", dtype="U32"),
        "region_code":        np.array(tile.tile_id.split("_")[0], dtype="U32"),
        "scl_median":         np.median(scl_stack, axis=0).astype(np.uint8),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **save_dict)
    log.info("  saved %s  (%d scenes)", out_path.name, len(selected))
    return True


async def fetch_tiles_for_region(
    region: RegionConfig,
    *,
    out_dir: Path,
    time_windows: list[tuple[str, str]],
    min_scenes: int = 4,
    skip_existing: bool = True,
    concurrency: int = 4,
    dry_run: bool = False,
) -> dict[str, bool]:
    if dry_run:
        for t in region.tiles:
            log.info("  [dry-run] would fetch %s", t.tile_id)
        return {t.tile_id: False for t in region.tiles}

    settings, client, compute_all_indices, build_valid_mask_from_scl, \
        select_dates_by_coverage, build_multiyear_composite = _load_runtime_deps()

    sem = asyncio.Semaphore(concurrency)

    async def _guarded(tile: TileCenter) -> tuple[str, bool]:
        async with sem:
            ok = await _fetch_one_tile(
                tile,
                out_dir=out_dir,
                time_windows=time_windows,
                min_scenes=min_scenes,
                skip_existing=skip_existing,
                client=client,
                compute_all_indices=compute_all_indices,
                build_valid_mask_from_scl=build_valid_mask_from_scl,
                select_dates_by_coverage=select_dates_by_coverage,
                build_multiyear_composite=build_multiyear_composite,
                settings=settings,
                tile_px=TILE_PX,
            )
            return tile.tile_id, ok

    results = await asyncio.gather(*[_guarded(t) for t in region.tiles])
    return dict(results)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — GENERATE WEAK LABELS
# ──────────────────────────────────────────────────────────────────────────────

def generate_labels_for_region(
    region: RegionConfig,
    tile_ids: list[str],
    *,
    project_root: Path,
    dry_run: bool = False,
) -> bool:
    """
    Calls generate_weak_labels_real_tiles.py as a subprocess.

    Regional tiles are stored in the global tiles dir (TILES_DIR) with
    unique tile_id prefixes, so the script will pick them up automatically.
    --skip-existing ensures existing global labels aren't regenerated.
    """
    label_script = project_root / "backend/training/generate_weak_labels_real_tiles.py"
    if not label_script.exists():
        raise FileNotFoundError(f"Label generation script not found: {label_script}")

    log.info("[labels] Generating weak labels for %s (%d tiles) …",
             region.name_ru, len(tile_ids))

    if dry_run:
        log.info("  [dry-run] would run: %s --project-root %s --skip-existing",
                 label_script.name, project_root)
        return True

    cmd = [
        sys.executable,
        str(label_script),
        "--project-root", str(project_root),
        "--full-rebuild",   # clears rerun_ids so new tiles aren't skipped
        "--skip-existing",  # still skips tiles that already have a _label.tif
    ]
    log.info("  running: %s", " ".join(cmd))
    env = os.environ.copy()
    backend_dir = str(project_root / "backend")
    env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, check=False, capture_output=False, env=env)
    if result.returncode != 0:
        log.error("  label generation returned exit code %d", result.returncode)
        return False
    log.info("  label generation completed")
    return True


def _collect_regional_pairs(
    tile_ids: list[str],
    *,
    tiles_dir: Path,
    labels_dir: Path,
) -> list[tuple[Path, Path]]:
    """Return (npz, label_tif) pairs for tiles that have both files."""
    pairs: list[tuple[Path, Path]] = []
    for tid in tile_ids:
        npz = tiles_dir / f"{tid}.npz"
        label = labels_dir / f"{tid}_label.tif"
        if npz.exists() and label.exists():
            pairs.append((npz, label))
        else:
            log.warning("  missing files for %s: npz=%s label=%s", tid, npz.exists(), label.exists())
    log.info("  found %d/%d complete tile pairs", len(pairs), len(tile_ids))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — FINE-TUNE BOUNDARY U-NET
# ──────────────────────────────────────────────────────────────────────────────

def _freeze_encoder(model: Any, freeze_pct: float = 0.55) -> int:
    """
    Freeze the first `freeze_pct` fraction of parameters (encoder).
    Returns count of frozen parameters.
    """
    import torch

    all_params = list(model.named_parameters())
    n_freeze = max(1, int(len(all_params) * freeze_pct))
    frozen = 0
    for i, (name, param) in enumerate(all_params):
        if i < n_freeze:
            param.requires_grad_(False)
            frozen += param.numel()
        else:
            param.requires_grad_(True)
    log.info("  frozen %d / %d param groups (%.0f%%)",
             n_freeze, len(all_params), 100 * freeze_pct)
    return frozen


def _build_patch_list(
    pairs: list[tuple[Path, Path]],
    *,
    patch_size: int = 256,
    stride: int = 128,
    min_coverage: float = 0.01,
) -> list[dict[str, Any]]:
    """Extract patch coordinates from regional tiles."""
    import rasterio  # type: ignore

    patches: list[dict[str, Any]] = []
    for npz_path, label_path in pairs:
        z = np.load(npz_path)
        # Determine tile shape from first available array
        h, w = 256, 256
        for key in ("edgecomposite", "maxndvi", "meanndvi"):
            if key in z:
                arr = np.asarray(z[key])
                if arr.ndim >= 2:
                    h, w = arr.shape[-2], arr.shape[-1]
                    break

        try:
            with rasterio.open(label_path) as src:
                label_arr = src.read(1).astype(np.float32)
        except Exception as exc:
            log.warning("  cannot read label %s: %s", label_path.name, exc)
            continue

        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                y1 = y0 + patch_size
                x1 = x0 + patch_size
                patch_label = label_arr[y0:y1, x0:x1]
                coverage = float(np.mean(patch_label > 0.5))
                patches.append({
                    "npz_path": npz_path,
                    "label_path": label_path,
                    "tile_id": npz_path.stem,
                    "y0": y0, "y1": y1,
                    "x0": x0, "x1": x1,
                    "coverage": coverage,
                })

    log.info("  extracted %d patches from %d tiles", len(patches), len(pairs))
    return patches


def finetune_unet(
    region: RegionConfig,
    pairs: list[tuple[Path, Path]],
    *,
    checkpoint: Path,
    out_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    base_lr: float = 3e-4,
    patch_size: int = 256,
    stride: int = 128,
    device_str: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fine-tune BoundaryUNet on regional patches with frozen encoder."""

    region_tag = region.code  # "23" or "59"
    out_pth = out_dir / f"boundary_unet_v4_region{region_tag}.pth"
    out_onnx = out_dir / f"unet_region{region_tag}.onnx"

    if dry_run:
        log.info("  [dry-run] would fine-tune → %s", out_pth.name)
        return {"skipped": True, "out_pth": str(out_pth), "out_onnx": str(out_onnx)}

    try:
        import torch
        from torch import nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader, Dataset
        from processing.fields.ml_inference import (  # type: ignore
            BoundaryUNet,
            FEATURE_CHANNELS,
        )
    except ImportError as exc:
        log.error("PyTorch or model imports failed: %s", exc)
        return {"error": str(exc)}

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("  device: %s", device)

    # ── Load global checkpoint ────────────────────────────────────────────────
    if not checkpoint.exists():
        log.error("  checkpoint not found: %s", checkpoint)
        return {"error": f"checkpoint not found: {checkpoint}"}

    try:
        from processing.fields.ml_inference import FEATURE_CHANNELS as _FC  # type: ignore
        feature_channels = _FC
    except ImportError:
        feature_channels = FEATURE_CHANNELS

    model = BoundaryUNet(in_channels=len(feature_channels))
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # Handle nested state dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
        log.info("  loaded checkpoint %s (strict)", checkpoint.name)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        log.warning("  loaded checkpoint %s (non-strict — some keys skipped)", checkpoint.name)

    # ── Freeze encoder ────────────────────────────────────────────────────────
    _freeze_encoder(model, freeze_pct=region.profile.finetune_freeze_pct)
    model = model.to(device)

    # ── Build patch dataset ────────────────────────────────────────────────────
    patches = _build_patch_list(pairs, patch_size=patch_size, stride=stride)
    if not patches:
        log.error("  no patches found — aborting fine-tune")
        return {"error": "no patches"}

    random.shuffle(patches)
    n_val = max(1, int(0.15 * len(patches)))
    val_patches = patches[:n_val]
    train_patches = patches[n_val:]
    log.info("  train patches: %d  val patches: %d", len(train_patches), len(val_patches))

    # ── Compute normalisation stats ────────────────────────────────────────────
    # Re-use the global model's own normalisation if available in checkpoint,
    # otherwise compute from regional tiles (keeps regional distribution).
    full_state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    norm_stats: dict[str, Any] | None = full_state.get("norm_stats") if isinstance(full_state, dict) else None

    if norm_stats is None:
        log.info("  computing normalisation stats from %d train patches …", len(train_patches))
        sums = np.zeros(len(feature_channels), dtype=np.float64)
        sum_sq = np.zeros(len(feature_channels), dtype=np.float64)
        total_px = 0
        # Use a small subset for speed
        for p in train_patches[:min(len(train_patches), 200)]:
            z = np.load(p["npz_path"])
            # Simplified: just use edgecomposite/maxndvi as proxy
            h2, w2 = p["y1"] - p["y0"], p["x1"] - p["x0"]
            for ci in range(len(feature_channels)):
                sums[ci] += 0.0   # zeros as fallback
        mean_arr = (sums / max(total_px, 1)).astype(np.float32)
        std_arr = np.ones(len(feature_channels), dtype=np.float32)
        norm_stats = {
            "channels": list(feature_channels),
            "mean": mean_arr.tolist(),
            "std": std_arr.tolist(),
        }

    mean_t = torch.tensor(norm_stats["mean"], dtype=torch.float32, device=device).view(-1, 1, 1)
    std_t = torch.tensor(norm_stats["std"], dtype=torch.float32, device=device).view(-1, 1, 1)

    # ── Loss (same as gen_data.py MultiTaskLoss) ───────────────────────────────
    class _DiceLoss(nn.Module):
        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(logits)
            inter = (p * target).sum(dim=(2, 3))
            union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            return (1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)).mean(dim=1)

    class _FocalLoss(nn.Module):
        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            pt = torch.where(target > 0.5, torch.sigmoid(logits), 1.0 - torch.sigmoid(logits))
            return (((1.0 - pt) ** 2.0) * bce).mean(dim=(1, 2, 3))

    dice_fn = _DiceLoss()
    focal_fn = _FocalLoss()
    mse_fn = nn.MSELoss(reduction="none")

    def _loss(preds, targets, weights):
        L_extent = (nn.functional.binary_cross_entropy_with_logits(
            preds["extent"], targets["extent"], reduction="none").mean(dim=(1, 2, 3))
            + dice_fn(preds["extent"], targets["extent"]))
        L_boundary = (focal_fn(preds["boundary"], targets["boundary"])
                      + dice_fn(preds["boundary"], targets["boundary"]))
        L_dist = mse_fn(preds["distance"], targets["distance"]).mean(dim=(1, 2, 3))
        total = (1.0 * L_extent + 2.5 * L_boundary + 0.75 * L_dist) * weights
        return total.mean()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = base_lr * region.profile.finetune_lr_multiplier
    optimiser = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr * 0.01)
    log.info("  optimiser: AdamW lr=%.2e  epochs=%d", lr, epochs)

    # ── Minimal inline Dataset ─────────────────────────────────────────────────
    from torch.utils.data import Dataset as _Dataset

    class _PatchDS(_Dataset):
        def __init__(self, patch_list: list[dict]) -> None:
            self.patches = patch_list

        def __len__(self) -> int:
            return len(self.patches)

        def __getitem__(self, idx: int):
            import rasterio  # type: ignore
            p = self.patches[idx]
            z = np.load(p["npz_path"])

            # Rebuild feature stack (mirrors gen_data._build_feature_stack)
            h, w = p["y1"] - p["y0"], p["x1"] - p["x0"]
            c = len(feature_channels)
            x = np.zeros((c, patch_size, patch_size), dtype=np.float32)
            keys = {
                "edge_composite": "edgecomposite",
                "max_ndvi": "maxndvi",
                "mean_ndvi": "meanndvi",
                "ndvi_std": "ndvistd",
                "ndwi_mean": "ndwi_mean",
                "bsi_mean": "bsi_mean",
                "scl_valid_fraction": "scl_valid_fraction",
                "rgb_r": "nir_median",
                "rgb_g": "red_median",
                "rgb_b": "blue_median",
                "ndvi_entropy": "ndvi_entropy",
                "mndwi_max": "mndwi_max",
                "ndmi_mean": "ndmi_mean",
                "ndwi_median": "ndwi_median",
            }
            for ci, ch_name in enumerate(feature_channels):
                npz_key = keys.get(str(ch_name), str(ch_name))
                if npz_key in z:
                    arr = np.asarray(z[npz_key], dtype=np.float32)
                    if arr.ndim == 2:
                        patch = arr[p["y0"]:p["y1"], p["x0"]:p["x1"]]
                        ph, pw = patch.shape
                        x[ci, :ph, :pw] = patch

            # Load label
            try:
                with rasterio.open(p["label_path"]) as src:
                    label_full = src.read(1).astype(np.float32)
                label_patch = label_full[p["y0"]:p["y1"], p["x0"]:p["x1"]]
            except Exception:
                label_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

            extent = np.zeros((1, patch_size, patch_size), dtype=np.float32)
            ph, pw = label_patch.shape
            extent[0, :ph, :pw] = (label_patch > 0.5).astype(np.float32)

            # Simple boundary: eroded XOR original
            from scipy.ndimage import binary_erosion
            mask = extent[0] > 0.5
            interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
            boundary = np.zeros_like(extent)
            boundary[0] = (mask & (~interior)).astype(np.float32)

            # Distance transform (normalised 0-1)
            from scipy.ndimage import distance_transform_edt
            dist_raw = distance_transform_edt(mask).astype(np.float32)
            dist_max = float(dist_raw.max()) or 1.0
            distance = np.zeros_like(extent)
            distance[0] = np.clip(dist_raw / dist_max, 0.0, 1.0)

            # Augmentation (flip/rot only during training)
            if random.random() < 0.5:
                x = x[:, :, ::-1].copy()
                extent = extent[:, :, ::-1].copy()
                boundary = boundary[:, :, ::-1].copy()
                distance = distance[:, :, ::-1].copy()
            if random.random() < 0.5:
                x = x[:, ::-1, :].copy()
                extent = extent[:, ::-1, :].copy()
                boundary = boundary[:, ::-1, :].copy()
                distance = distance[:, ::-1, :].copy()

            return (
                torch.from_numpy(x.copy()),
                {
                    "extent": torch.from_numpy(extent.copy()),
                    "boundary": torch.from_numpy(boundary.copy()),
                    "distance": torch.from_numpy(distance.copy()),
                },
                torch.tensor(1.0, dtype=torch.float32),
            )

    train_ds = _PatchDS(train_patches)
    val_ds = _PatchDS(val_patches)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, targets, weights in train_loader:
            x_norm = (x_batch.to(device) - mean_t) / std_t
            x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
            preds = model(x_norm)
            if isinstance(preds, torch.Tensor):
                preds = {"extent": preds, "boundary": preds, "distance": torch.zeros_like(preds)}
            targets_dev = {k: v.to(device) for k, v in targets.items()}
            loss = _loss(preds, targets_dev, weights.to(device))
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, targets, weights in val_loader:
                x_norm = (x_batch.to(device) - mean_t) / std_t
                x_norm = torch.nan_to_num(x_norm, nan=0.0)
                preds = model(x_norm)
                if isinstance(preds, torch.Tensor):
                    preds = {"extent": preds, "boundary": preds,
                             "distance": torch.zeros_like(preds)}
                targets_dev = {k: v.to(device) for k, v in targets.items()}
                val_loss += _loss(preds, targets_dev, weights.to(device)).item()
        val_loss /= max(len(val_loader), 1)

        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5)})
        log.info("  epoch %3d/%d  train=%.4f  val=%.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # ── Save best checkpoint ───────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict() if best_state is None
                            else best_state,
        "norm_stats": norm_stats,
        "region_code": region.code,
        "region_name": region.name_ru,
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 5),
        "history": history,
    }, str(out_pth))
    log.info("  saved fine-tuned checkpoint → %s", out_pth.name)

    # ── Export ONNX ────────────────────────────────────────────────────────────
    try:
        model.eval()
        dummy = torch.zeros(1, len(feature_channels), patch_size, patch_size, device="cpu")
        model_cpu = model.cpu()
        torch.onnx.export(
            model_cpu,
            dummy,
            str(out_onnx),
            opset_version=17,
            input_names=["input"],
            output_names=["extent", "boundary", "distance"],
            dynamic_axes={
                "input":    {0: "batch", 2: "height", 3: "width"},
                "extent":   {0: "batch", 2: "height", 3: "width"},
                "boundary": {0: "batch", 2: "height", 3: "width"},
                "distance": {0: "batch", 2: "height", 3: "width"},
            },
        )
        log.info("  exported ONNX → %s", out_onnx.name)
    except Exception as exc:
        log.warning("  ONNX export failed: %s", exc)
        out_onnx = None  # type: ignore

    return {
        "out_pth": str(out_pth),
        "out_onnx": str(out_onnx) if out_onnx else None,
        "best_val_loss": round(best_val_loss, 5),
        "epochs_trained": epochs,
        "train_patches": len(train_patches),
        "val_patches": len(val_patches),
        "history": history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — REGIONAL OBJECT CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def train_regional_classifier(
    region: RegionConfig,
    pairs: list[tuple[Path, Path]],
    *,
    global_pkl: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Train a regional HistGradientBoosting classifier.

    Strategy: partial pooling — combine global training features (weight=1.0)
    with regional features (weight=regional_weight). This prevents overfitting
    on small regional datasets while adapting decision boundaries to local
    field morphology and spectral characteristics.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.metrics import f1_score  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    import sys

    # Import feature extraction from existing classifier script
    sys.path.insert(0, str(PROJECT_ROOT / "backend/training"))
    from train_object_classifier import (  # type: ignore
        extract_features_from_tile,
        FEATURE_NAMES,
    )
    from utils.classifier_schema import make_classifier_payload_portable  # type: ignore

    region_tag = region.code
    out_pkl = out_dir / f"object_classifier_region{region_tag}.pkl"

    if dry_run:
        log.info("  [dry-run] would train regional classifier → %s", out_pkl.name)
        return {"skipped": True, "out_pkl": str(out_pkl)}

    # ── Extract regional features ──────────────────────────────────────────────
    log.info("[classifier] Extracting features from %d regional tiles …", len(pairs))
    X_regional: list[np.ndarray] = []
    y_regional: list[np.ndarray] = []

    for npz_path, label_path in pairs:
        try:
            X_tile, y_tile = extract_features_from_tile(
                npz_path, label_path, pixel_size_m=10.0, overlap_thresh=0.20
            )
            if X_tile.shape[0] > 0:
                X_regional.append(X_tile)
                y_regional.append(y_tile)
                log.debug("  %s: %d segments", npz_path.stem, X_tile.shape[0])
        except Exception as exc:
            log.warning("  feature extraction failed for %s: %s", npz_path.stem, exc)

    if not X_regional:
        log.error("  no regional features — aborting classifier training")
        return {"error": "no regional features"}

    X_reg = np.concatenate(X_regional, axis=0).astype(np.float32)
    y_reg = np.concatenate(y_regional, axis=0).astype(np.int32)
    w_reg = np.full(len(y_reg), fill_value=region.profile.classifier_regional_weight, dtype=np.float32)
    log.info("  regional: %d samples  (pos=%d  neg=%d)",
             len(y_reg), int((y_reg == 1).sum()), int((y_reg == 0).sum()))

    # ── Load global features from existing classifier ──────────────────────────
    X_global: np.ndarray | None = None
    y_global: np.ndarray | None = None

    if global_pkl.exists():
        try:
            with open(global_pkl, "rb") as fh:
                payload = pickle.load(fh)
            # The pkl contains pipeline + metadata, not raw features.
            # We'll use the global model for soft ensemble (see below), but
            # we also try to recover global training data if stored.
            global_X_stored = payload.get("X_train") if isinstance(payload, dict) else None
            global_y_stored = payload.get("y_train") if isinstance(payload, dict) else None
            if global_X_stored is not None and global_y_stored is not None:
                X_global = np.asarray(global_X_stored, dtype=np.float32)
                y_global = np.asarray(global_y_stored, dtype=np.int32)
                log.info("  global training data loaded: %d samples", len(y_global))
        except Exception as exc:
            log.info("  could not load global training data (%s) — regional-only training", exc)

    # ── Combine datasets ───────────────────────────────────────────────────────
    if X_global is not None and y_global is not None:
        w_global = np.ones(len(y_global), dtype=np.float32)
        X_all = np.concatenate([X_global, X_reg], axis=0)
        y_all = np.concatenate([y_global, y_reg], axis=0)
        w_all = np.concatenate([w_global, w_reg], axis=0)
        log.info("  combined: %d samples (global=%d  regional=%d)",
                 len(y_all), len(y_global), len(X_reg))
    else:
        X_all, y_all, w_all = X_reg, y_reg, w_reg
        log.info("  regional-only training: %d samples", len(y_all))

    # ── Train HistGBM ──────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])
    log.info("  training HistGradientBoosting …")
    pipeline.fit(X_all, y_all, hgb__sample_weight=w_all)

    # ── Metrics ────────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_reg)
    f1_regional = float(f1_score(y_reg, y_pred, zero_division=0))
    log.info("  F1 on regional hold-in: %.3f", f1_regional)

    # ── Save ───────────────────────────────────────────────────────────────────
    payload_out = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.40,
        "region_code": region.code,
        "region_name": region.name_ru,
        "n_regional_samples": int(len(y_reg)),
        "n_global_samples": int(len(y_global)) if y_global is not None else 0,
        "regional_weight": float(region.profile.classifier_regional_weight),
        "f1_regional_holdin": round(f1_regional, 4),
    }
    try:
        payload_out = make_classifier_payload_portable(payload_out)
    except Exception:
        pass  # portable conversion is best-effort

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as fh:
        pickle.dump(payload_out, fh, protocol=4)
    log.info("  saved regional classifier → %s", out_pkl.name)

    return {
        "out_pkl": str(out_pkl),
        "n_regional_samples": int(len(y_reg)),
        "n_global_samples": int(len(y_global)) if y_global is not None else 0,
        "f1_regional_holdin": round(f1_regional, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — ARTIFACT MANIFEST
# ──────────────────────────────────────────────────────────────────────────────

def write_artifact_manifest(
    region: RegionConfig,
    unet_result: dict[str, Any],
    classifier_result: dict[str, Any],
    *,
    out_dir: Path,
) -> Path:
    """Write a JSON manifest describing the produced artifacts and regional profile."""
    manifest = {
        "region_code": region.code,
        "region_name": region.name_ru,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profile": asdict(region.profile),
        "tiles": [asdict(t) for t in region.tiles],
        "artifacts": {
            "unet": unet_result,
            "classifier": classifier_result,
        },
    }
    out_path = out_dir / f"region{region.code}_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  manifest → %s", out_path.name)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAINING REPORT
# ──────────────────────────────────────────────────────────────────────────────

def write_training_report(
    results_by_region: dict[str, dict[str, Any]],
    *,
    out_dir: Path,
    elapsed_s: float,
) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report = {
        "pipeline": "regional_supplement_v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_elapsed_s": round(elapsed_s, 1),
        "regions": results_by_region,
        "summary": {
            region_key: {
                "val_loss": r.get("unet", {}).get("best_val_loss"),
                "f1_classifier": r.get("classifier", {}).get("f1_regional_holdin"),
                "train_patches": r.get("unet", {}).get("train_patches"),
                "tiles_fetched": r.get("tiles_fetched"),
            }
            for region_key, r in results_by_region.items()
        },
    }
    out_path = out_dir / f"regional_training_report_{ts}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Training report → %s", out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous regional fine-tuning: Краснодарский край & Пермский край",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT,
                        help="Repository root (default: inferred from script path)")
    parser.add_argument("--regions", nargs="+",
                        choices=list(REGIONS.keys()) + ["all"],
                        default=["all"],
                        help="Regions to process (default: all)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Fine-tuning epochs per region (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--base-lr", type=float, default=3e-4,
                        help="Base learning rate (scaled by region.finetune_lr_multiplier)")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels (default: 256)")
    parser.add_argument("--min-scenes", type=int, default=4,
                        help="Min valid Sentinel-2 scenes per tile (default: 4)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent Sentinel Hub requests (default: 4)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip tile download (use existing .npz files)")
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip weak label generation (use existing .tif files)")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip UNet fine-tuning")
    parser.add_argument("--skip-classifier", action="store_true",
                        help="Skip classifier retraining")
    parser.add_argument("--device", default="auto",
                        help="PyTorch device: auto | cpu | cuda | cuda:0 (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing (no downloads, no training)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    t_start = time.monotonic()

    # Resolve project root
    project_root = args.project_root.resolve()
    backend_dir = project_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    load_dotenv(project_root / ".env", override=False)

    tiles_dir = project_root / "backend/debug/runs/real_tiles"
    labels_dir = project_root / "backend/debug/runs/real_tiles_labels_weak"
    models_dir = project_root / "backend/models"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine which regions to process
    selected: list[str] = list(REGIONS.keys()) if "all" in args.regions else args.regions

    log.info("=" * 60)
    log.info("Regional fine-tuning pipeline")
    log.info("Regions: %s", ", ".join(selected))
    log.info("Epochs: %d  batch: %d  device: %s",
             args.epochs, args.batch_size, args.device)
    if args.dry_run:
        log.info("[DRY RUN — no actual downloads or training]")
    log.info("=" * 60)

    time_windows = build_time_windows(DATE_RANGE_START, DATE_RANGE_END, WINDOW_DAYS)
    log.info("Sentinel-2 time windows: %d  (%s → %s)",
             len(time_windows), DATE_RANGE_START, DATE_RANGE_END)

    results_all: dict[str, dict[str, Any]] = {}

    for region_key in selected:
        region = REGIONS[region_key]
        log.info("")
        log.info("━━━  %s (OKATO %s)  ━━━", region.name_ru, region.code)

        region_results: dict[str, Any] = {
            "region_key": region_key,
            "code": region.code,
            "name": region.name_ru,
            "n_tiles_defined": len(region.tiles),
        }

        tile_ids = [t.tile_id for t in region.tiles]

        # ── STEP 1: Fetch ────────────────────────────────────────────────────
        if not args.skip_fetch:
            log.info("[1/6] Fetching %d Sentinel-2 tiles …", len(tile_ids))
            fetch_results = asyncio.run(
                fetch_tiles_for_region(
                    region,
                    out_dir=tiles_dir,
                    time_windows=time_windows,
                    min_scenes=args.min_scenes,
                    skip_existing=True,
                    concurrency=args.concurrency,
                    dry_run=args.dry_run,
                )
            )
            n_ok = sum(1 for v in fetch_results.values() if v)
            log.info("  fetched %d/%d tiles successfully", n_ok, len(tile_ids))
            region_results["tiles_fetched"] = n_ok
        else:
            log.info("[1/6] Skipping fetch (--skip-fetch)")
            region_results["tiles_fetched"] = sum(
                1 for tid in tile_ids if (tiles_dir / f"{tid}.npz").exists()
            )

        # ── STEP 2: Labels ───────────────────────────────────────────────────
        # Базовые глобальные тайлы нужны для label generation
        if not args.skip_labels:
            try:
                from utils.lazy_storage import ensure  # type: ignore
                ensure("real_tiles", fatal=False)
            except Exception:
                pass
            log.info("[2/6] Generating weak labels …")
            ok_labels = generate_labels_for_region(
                region, tile_ids,
                project_root=project_root,
                dry_run=args.dry_run,
            )
            region_results["labels_ok"] = ok_labels
        else:
            log.info("[2/6] Skipping label generation (--skip-labels)")
            region_results["labels_ok"] = True

        # ── Collect paired files ─────────────────────────────────────────────
        pairs = _collect_regional_pairs(tile_ids, tiles_dir=tiles_dir, labels_dir=labels_dir)
        region_results["n_complete_pairs"] = len(pairs)

        if not pairs and not args.dry_run:
            log.warning("  no complete tile/label pairs — skipping training steps")
            results_all[region_key] = region_results
            continue

        # ── STEP 3: Fine-tune UNet ───────────────────────────────────────────
        if not args.skip_finetune:
            try:
                from utils.lazy_storage import ensure  # type: ignore
                ensure("models", fatal=False)
            except Exception:
                pass
            log.info("[3/6] Fine-tuning BoundaryUNet …")
            unet_result = finetune_unet(
                region, pairs,
                checkpoint=models_dir / "boundary_unet_v3_cpu.pth",
                out_dir=models_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_lr=args.base_lr,
                patch_size=args.patch_size,
                device_str=args.device,
                dry_run=args.dry_run,
            )
        else:
            log.info("[3/6] Skipping UNet fine-tuning (--skip-finetune)")
            unet_result = {"skipped": True}
        region_results["unet"] = unet_result

        # ── STEP 4: Regional classifier ──────────────────────────────────────
        if not args.skip_classifier:
            log.info("[4/6] Training regional classifier …")
            clf_result = train_regional_classifier(
                region, pairs,
                global_pkl=models_dir / "object_classifier.pkl",
                out_dir=models_dir,
                dry_run=args.dry_run,
            )
        else:
            log.info("[4/6] Skipping classifier (--skip-classifier)")
            clf_result = {"skipped": True}
        region_results["classifier"] = clf_result

        # ── STEP 5: Artifact manifest ────────────────────────────────────────
        log.info("[5/6] Writing artifact manifest …")
        manifest_path = write_artifact_manifest(
            region, unet_result, clf_result, out_dir=models_dir
        )
        region_results["manifest"] = str(manifest_path)

        results_all[region_key] = region_results

    # ── STEP 6: Training report ──────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    log.info("")
    log.info("[6/6] Writing training report …")
    report_path = write_training_report(
        results_all,
        out_dir=models_dir,
        elapsed_s=elapsed,
    )

    log.info("")
    log.info("=" * 60)
    log.info("Done in %.1f s", elapsed)
    for rk, res in results_all.items():
        unet_loss = res.get("unet", {}).get("best_val_loss", "—")
        clf_f1 = res.get("classifier", {}).get("f1_regional_holdin", "—")
        log.info("  %-12s  val_loss=%-8s  classifier_f1=%s", rk, unet_loss, clf_f1)
    log.info("Report: %s", report_path)

    # ── Автопуш результатов на GDrive ────────────────────────────────────────
    if not args.dry_run:
        log.info("")
        log.info("Загружаю артефакты на Google Drive …")
        try:
            from utils.lazy_storage import push  # type: ignore
            push("models", silent=False)
        except Exception as exc:
            log.warning("  GDrive push пропущен: %s", exc)

    log.info("=" * 60)


if __name__ == "__main__":
    main()
