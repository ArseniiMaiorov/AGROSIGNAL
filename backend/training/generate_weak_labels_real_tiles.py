#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import asyncio
import inspect
import argparse
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from utils.nan_safe import nanmax_safe, nanmean_safe, nanmedian_safe

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
BACKEND_DIR = PROJECT_ROOT / "backend"
load_dotenv(PROJECT_ROOT / ".env")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATABASE_URL",      "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

W, H = 256, 256

RANGE_START = date(2023, 4, 15)
RANGE_END   = date(2025, 4, 14)
WINDOW_DAYS = 30

IN_NPZ_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles"
OUT_LABEL_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles_labels_weak"
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG_DIR = OUT_LABEL_DIR / "preview_png"
OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_LABEL_DIR / "weak_labels_summary.csv"
ROAD_RETRY_TILES = {
    "lenoblast_03",
    "moscow_01",
    "rostov_01",
    "saratov_01",
    "volgograd_01",
    "volgograd_02",
}
WC_RELAXED_REGION_PREFIXES = (
    "samara_",
    "orenburg_",
    "bashkortostan_",
    "chelyabinsk_",
    "tatarstan_",
    "saratov_",
    "penza_",
    "volgograd_",
    "permkrai_",
    "kurgan_",
    "tyumen_",
    "omsk_",
    "novosibirsk_",
    "altai_",
    "kirov_",
)


# ── utils ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak labels from real Sentinel-2 tiles")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root (defaults to repository root inferred from script path)",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Rebuild summary from scratch and rerun all tiles (no selective fallback rerun)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles that already have a _label.tif file (resume after crash)",
    )
    return parser.parse_args()


def _apply_project_root(project_root: Path) -> None:
    global PROJECT_ROOT, BACKEND_DIR, IN_NPZ_DIR, OUT_LABEL_DIR, OUT_PNG_DIR, SUMMARY_CSV
    PROJECT_ROOT = project_root.resolve()
    BACKEND_DIR = PROJECT_ROOT / "backend"
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    IN_NPZ_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles"
    OUT_LABEL_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles_labels_weak"
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PNG_DIR = OUT_LABEL_DIR / "preview_png"
    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV = OUT_LABEL_DIR / "weak_labels_summary.csv"

def import_callable(module_path: str, *names: str):
    mod = __import__(module_path, fromlist=["*"])
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise ImportError(f"None of {names} found in {module_path}")


def _resolve_fn(mod, *candidates: str):
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"None of {candidates} found in {mod.__name__}")


def _get_settings():
    try:
        from core.config import getsettings
        return getsettings()
    except Exception:
        from core.config import get_settings
        return get_settings()


def _get_sentinel_client():
    from providers.sentinelhub.client import SentinelHubClient
    return SentinelHubClient()


def make_weak_settings(settings):
    """
    Relaxed copy of settings для weak-label генерации:
    - порог минимальной площади поля 0.05 ha вместо ~1 ha
    - больше компонент после merge
    """
    # Support both legacy compact names and current underscore field names.
    override_candidates: dict[str, tuple[str, ...]] = {
        "POST_MIN_FIELD_AREA_HA": ("POST_MIN_FIELD_AREA_HA", "POSTMINFIELDAREAHA"),
        "POST_MERGE_MAX_COMPONENTS": ("POST_MERGE_MAX_COMPONENTS", "POSTMERGEMAXCOMPONENTS"),
        "POST_GROW_MAX_ITERS": ("POST_GROW_MAX_ITERS", "POSTGROWMAXITERS"),
        "POST_MERGE_BUFFER_PX": ("POST_MERGE_BUFFER_PX", "POSTMERGEBUFFERPX"),
        # Weak-label mode: avoid fully zeroing candidates in dense road networks.
        "POST_ROAD_HARD_EXCLUSION": ("POST_ROAD_HARD_EXCLUSION", "POSTROADHARDEXCLUSION"),
        "POST_ROAD_BUFFER_PX": ("POST_ROAD_BUFFER_PX", "POSTROADBUFFERPX"),
        "ROAD_OSM_TAGS": ("ROAD_OSM_TAGS", "ROADOSMTAGS"),
        "ROAD_OSM_BUFFER_DEFAULT_M": ("ROAD_OSM_BUFFER_DEFAULT_M", "ROADOSMBUFFERDEFAULTM"),
        "ROAD_OSM_BUFFER_MAP": ("ROAD_OSM_BUFFER_MAP", "ROADOSMBUFFERMAP"),
    }
    override_values: dict[str, Any] = {
        "POST_MIN_FIELD_AREA_HA": 0.05,
        "POST_MERGE_MAX_COMPONENTS": 512,
        "POST_GROW_MAX_ITERS": 3,
        "POST_MERGE_BUFFER_PX": 2,
        "POST_ROAD_HARD_EXCLUSION": False,
        "POST_ROAD_BUFFER_PX": 1,
        # Keep only major roads as hard barriers in weak-label generation.
        "ROAD_OSM_TAGS": ("motorway", "trunk", "primary", "secondary"),
        "ROAD_OSM_BUFFER_DEFAULT_M": 6,
        "ROAD_OSM_BUFFER_MAP": {
            "motorway": 20,
            "trunk": 16,
            "primary": 12,
            "secondary": 10,
            "tertiary": 8,
            "unclassified": 6,
            "residential": 6,
            "service": 5,
            "track": 7,
            "path": 4,
        },
    }
    valid = (
        set(settings.model_fields.keys())
        if hasattr(settings, "model_fields")
        else set(settings.__fields__.keys())
        if hasattr(settings, "__fields__")
        else set()
    )
    filtered: dict[str, Any] = {}
    for logical_name, candidates in override_candidates.items():
        value = override_values[logical_name]
        for candidate in candidates:
            if candidate in valid:
                filtered[candidate] = value
                break

    try:
        return settings.model_copy(update=filtered)   # Pydantic v2
    except AttributeError:
        return settings.copy(update=filtered)          # Pydantic v1


def settings_with_overrides(settings, override_values: dict[str, Any]):
    """Return a settings copy with safe key/alias matching for overrides."""
    valid = (
        set(settings.model_fields.keys())
        if hasattr(settings, "model_fields")
        else set(settings.__fields__.keys())
        if hasattr(settings, "__fields__")
        else set()
    )
    filtered: dict[str, Any] = {}
    for key, value in override_values.items():
        candidates = (key, key.replace("_", ""))
        for candidate in candidates:
            if candidate in valid:
                filtered[candidate] = value
                break
    try:
        return settings.model_copy(update=filtered)   # Pydantic v2
    except AttributeError:
        return settings.copy(update=filtered)          # Pydantic v1


def make_weak_road_retry_settings(settings):
    """Second-pass weak settings when road barrier suppresses all candidates."""
    return settings_with_overrides(
        settings,
        {
            "POST_ROAD_HARD_EXCLUSION": False,
            # Force near-empty spectral road candidate in weak-label retry.
            "POST_ROAD_MAX_NDVI": -1.0,
            "POST_ROAD_NIR_MAX": 0.0,
            "POST_ROAD_NDBI_MIN": 1.0,
            "POST_ROAD_BUFFER_PX": 0,
            # Keep OSM road fetch narrow; zero tags can break some providers.
            "ROAD_OSM_TAGS": ("motorway", "trunk", "primary"),
            "ROAD_OSM_BUFFER_DEFAULT_M": 2,
            "ROAD_OSM_BUFFER_MAP": {
                "motorway": 8,
                "trunk": 6,
                "primary": 4,
                "secondary": 3,
                "tertiary": 2,
                "unclassified": 2,
                "residential": 2,
                "service": 1,
                "track": 1,
                "path": 1,
            },
        },
    )


def _is_wc_relaxed_region(tile_id: str, lat: float, lon: float) -> bool:
    if tile_id.startswith(WC_RELAXED_REGION_PREFIXES):
        return True
    # Fallback geo-heuristic for Volga/Ural/Siberia-like AOIs.
    return bool((lon >= 44.0 and lat >= 50.0) or lon >= 60.0)


def get_adaptive_worldcover_gate_pct(tile_id: str, lat: float, lon: float) -> tuple[float, str]:
    """Adaptive WC overlap threshold (percent) by region + latitude."""
    if lat > 60.0:
        base = 4.0
    elif lat > 57.0:
        base = 5.0
    else:
        base = 10.0

    profile = "default"
    if _is_wc_relaxed_region(tile_id, lat, lon):
        profile = "volga_ural_siberia_relaxed"
        if lat > 57.0:
            base = min(base, 2.0)
        elif lat >= 52.0:
            base = min(base, 3.0)
        else:
            base = min(base, 4.0)
    return float(base), profile


def compute_mask_temporal_stats(
    ndvi_stack: np.ndarray,
    valid_stack: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | bool | int]:
    """Temporal coherence summary for a mask based on mean NDVI profile."""
    ndvi = np.asarray(ndvi_stack, dtype=np.float32)
    valid = np.asarray(valid_stack, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if ndvi.ndim != 3 or valid.ndim != 3 or m.ndim != 2 or not m.any():
        return {
            "valid_dates": 0,
            "ndvi_amplitude": 0.0,
            "has_growth_peak": False,
            "ndvi_entropy": 4.0,
        }

    t = ndvi.shape[0]
    profile = np.full(t, np.nan, dtype=np.float32)
    for ti in range(t):
        px = ndvi[ti][m & valid[ti]]
        if px.size:
            profile[ti] = float(np.nanmean(px))

    profile = profile[np.isfinite(profile)]
    if profile.size < 3:
        return {
            "valid_dates": int(profile.size),
            "ndvi_amplitude": 0.0,
            "has_growth_peak": False,
            "ndvi_entropy": 4.0,
        }

    amplitude = float(np.max(profile) - np.min(profile))
    peak_idx = int(np.argmax(profile))
    rise = np.diff(profile[: peak_idx + 1]) if peak_idx > 0 else np.asarray([], dtype=np.float32)
    drop = np.diff(profile[peak_idx:]) if peak_idx < (profile.size - 1) else np.asarray([], dtype=np.float32)
    has_growth_peak = bool(
        rise.size > 0
        and drop.size > 0
        and np.any(rise > 0.02)
        and np.any(drop < -0.02)
        and amplitude >= 0.16
    )

    scaled = np.clip(
        (profile - np.min(profile)) / max(amplitude, 1e-6),
        0.0,
        1.0,
    )
    hist, _ = np.histogram(scaled, bins=np.linspace(0.0, 1.0, 7))
    probs = hist.astype(np.float32) / float(max(1, hist.sum()))
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 4.0

    return {
        "valid_dates": int(profile.size),
        "ndvi_amplitude": amplitude,
        "has_growth_peak": has_growth_peak,
        "ndvi_entropy": entropy,
    }


def temporal_is_strong(stats: dict[str, float | bool | int], lat: float) -> bool:
    amp = float(stats.get("ndvi_amplitude", 0.0))
    entropy = float(stats.get("ndvi_entropy", 4.0))
    has_peak = bool(stats.get("has_growth_peak", False))
    amp_min = 0.18 if lat > 57.0 else 0.22
    entropy_max = 2.4 if lat > 57.0 else 2.2
    return bool(has_peak or (amp >= amp_min and entropy <= entropy_max))


def targeted_quality_rescue(
    *,
    candidate_mask: np.ndarray,
    hard_exclusion: np.ndarray,
    max_ndvi: np.ndarray,
    lat: float,
    max_cover_frac: float = 0.55,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Soft rescue pass for quality-gate failed tiles with shape/area guards."""
    from scipy.ndimage import binary_erosion, label as nd_label

    candidate = np.asarray(candidate_mask, dtype=bool) & ~np.asarray(hard_exclusion, dtype=bool)
    if not candidate.any():
        return np.zeros_like(candidate, dtype=bool), {
            "accepted": False,
            "components_kept": 0,
            "coverage_frac": 0.0,
        }

    labels, n_labels = nd_label(candidate)
    min_area_px = 24 if lat > 57.0 else 36
    max_area_px = max(min_area_px + 1, int(max_cover_frac * candidate.size))
    min_ndvi_max = 0.20 if lat > 57.0 else 0.28
    max_shape_index = 3.8

    out = np.zeros_like(candidate, dtype=bool)
    components_kept = 0

    for comp_id in range(1, int(n_labels) + 1):
        comp = labels == comp_id
        area = int(np.count_nonzero(comp))
        if area < min_area_px or area > max_area_px:
            continue

        ndvi_max_val = float(np.nanmax(max_ndvi[comp])) if comp.any() else 0.0
        if ndvi_max_val < min_ndvi_max:
            continue

        boundary = comp & ~binary_erosion(comp, structure=np.ones((3, 3), dtype=bool))
        perimeter = max(1, int(np.count_nonzero(boundary)))
        shape_index = float(perimeter / (2.0 * np.sqrt(np.pi * max(area, 1.0))))
        if shape_index > max_shape_index:
            continue

        out |= comp
        components_kept += 1

    return out, {
        "accepted": bool(out.any()),
        "components_kept": int(components_kept),
        "coverage_frac": float(out.mean()),
    }


def build_time_windows(start: date, end: date, window_days: int) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        wend = min(end, cur + timedelta(days=window_days - 1))
        windows.append((f"{cur.isoformat()}T00:00:00Z", f"{wend.isoformat()}T23:59:59Z"))
        cur = wend + timedelta(days=1)
    return windows


def center_latlon_from_bbox(bbox: tuple) -> tuple[float, float]:
    minx, miny, maxx, maxy = bbox
    return (0.5 * (miny + maxy), 0.5 * (minx + maxx))


def approx_pixel_area_m2(bbox: tuple) -> float:
    import math
    minx, miny, maxx, maxy = bbox
    lat = 0.5 * (miny + maxy)
    w_m = (maxx - minx) * 111_320.0 * max(0.05, math.cos(math.radians(lat)))
    h_m = (maxy - miny) * 110_574.0
    return float((w_m / W) * (h_m / H))


def infer_tile_dims_from_npz(z: np.lib.npyio.NpzFile) -> tuple[int, int]:
    """Infer (width, height) from arrays stored in a tile NPZ."""
    for key in ("B2", "B3", "B4", "B8", "edgecomposite", "maxndvi", "meanndvi", "ndvistd"):
        if key not in z:
            continue
        arr = np.asarray(z[key])
        if arr.ndim >= 2:
            h, w = int(arr.shape[-2]), int(arr.shape[-1])
            if h > 0 and w > 0:
                return w, h
    return int(W), int(H)


def median_composite(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, stack, np.nan)
    med = nanmedian_safe(masked, axis=0, fill_value=np.nan)
    return np.where(np.isfinite(med), med, 0.0).astype(np.float32)


def mean_composite(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, stack, np.nan)
    m = nanmean_safe(masked, axis=0, fill_value=np.nan)
    return np.where(np.isfinite(m), m, 0.0).astype(np.float32)


def call_select_dates(fn, valid_mask: np.ndarray, n_dates: int):
    params = set(inspect.signature(fn).parameters.keys())
    kw: dict[str, Any] = {}
    kw["min_valid_pct" if "min_valid_pct" in params else "minvalidpct"] = 0.30
    if "n_dates" in params:
        kw["n_dates"] = n_dates
    elif "ndates" in params:
        kw["ndates"] = n_dates
    if "min_good_dates" in params:
        kw["min_good_dates"] = 2
    elif "mingooddates" in params:
        kw["mingooddates"] = 2
    if "return_metadata" in params:
        kw["return_metadata"] = True
    elif "returnmetadata" in params:
        kw["returnmetadata"] = True
    out = fn(valid_mask, **kw)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def build_pheno_thresholds(PhenoThresholds, settings):
    DEFAULTS: dict[str, float | int] = {
        "ndwi_water":      0.10,  "mndwi_water":     0.05,
        "bsi_built":       0.15,  "std_built":       0.02,
        "ndvi_forest_min": 0.45,  "delta_forest":    0.15,
        "ndvi_grass_mean": 0.35,  "delta_grass":     0.20,
        "ndvi_crop_max":   0.62,  "ndvi_crop_min":   0.25,
        "delta_crop":      0.30,  "msi_crop":        1.10,
        "n_valid_min":     4,
    }
    SETTINGS_ALIASES: dict[str, list[str]] = {
        "ndwi_water":      ["PHENO_NDWI_WATER",     "PHENONDWIWATER"],
        "mndwi_water":     ["PHENO_MNDWI_WATER",    "PHENOMNDWIWATER"],
        "bsi_built":       ["PHENO_BSI_BUILT",       "PHENOBSIBUILT"],
        "std_built":       ["PHENO_STD_BUILT",       "PHENOSTDBUILT"],
        "ndvi_forest_min": ["PHENO_NDVI_FOREST_MIN", "PHENONDVIFORESTMIN"],
        "delta_forest":    ["PHENO_DELTA_FOREST",    "PHENODELTAFOREST"],
        "ndvi_grass_mean": ["PHENO_NDVI_GRASS_MEAN", "PHENONDVIGRASSMEAN"],
        "delta_grass":     ["PHENO_DELTA_GRASS",     "PHENODELTAGRASS"],
        "ndvi_crop_max":   ["PHENO_NDVI_CROP_MAX",   "PHENONDVICROPMAX"],
        "ndvi_crop_min":   ["PHENO_NDVI_CROP_MIN",   "PHENONDVICROPMIN"],
        "delta_crop":      ["PHENO_DELTA_CROP",      "PHENODELTACROP"],
        "msi_crop":        ["PHENO_MSI_CROP",        "PHENOMSICROP"],
        "n_valid_min":     ["PHENO_NVALID_MIN",      "PHENONVALIDMIN"],
    }
    sig_params = set(inspect.signature(PhenoThresholds).parameters.keys()) - {"self"}
    kw: dict[str, Any] = {}
    for canonical, default in DEFAULTS.items():
        variants = [
            canonical,
            canonical.replace("_", ""),
            "".join(w.capitalize() for w in canonical.split("_")),
            canonical[0] + "".join(w.capitalize() for w in canonical.split("_"))[1:],
        ]
        real_param = next((v for v in variants if v in sig_params), None)
        if real_param is None:
            continue
        value = default
        for alias in SETTINGS_ALIASES.get(canonical, []):
            if hasattr(settings, alias):
                value = getattr(settings, alias)
                break
        kw[real_param] = type(default)(value)
    return PhenoThresholds(**kw)


def _fill_kwargs(fn, value_map: dict[str, Any]) -> dict[str, Any]:
    fn_params = inspect.signature(fn).parameters
    result: dict[str, Any] = {}
    for k, v in value_map.items():
        if k in fn_params and k not in result:
            result[k] = v
    for name, param in fn_params.items():
        if name in result:
            continue
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            raise TypeError(
                f"Required param '{name}' missing for {getattr(fn, '__name__', fn)}"
            )
    return result


def get_regional_overrides(lat: float) -> dict[str, float]:
    """Latitude-adaptive phenological threshold overrides."""
    if lat > 57.0:  # Northern Russia (LO, Pskov, Tver, Perm)
        return {
            "ndvi_crop_max": 0.50,
            "delta_crop": 0.20,
            "ndvi_crop_min": 0.15,
            "ndvi_forest_min": 0.40,
            "ndvi_grass_mean": 0.25,
            "delta_grass": 0.12,
        }
    elif lat > 53.0:  # Central Russia (Moscow, Tula, etc.)
        return {
            "ndvi_crop_max": 0.55,
            "delta_crop": 0.25,
        }
    elif lat < 48.0:  # Southern Russia (Krasnodar, Stavropol, Rostov)
        return {
            "ndvi_crop_max": 0.70,
            "delta_crop": 0.35,
        }
    return {}


def lite_postprocess(mask: np.ndarray, hard_exclusion: np.ndarray) -> np.ndarray:
    """Fallback если run_postprocess всё ещё недоступен."""
    m = mask.astype(bool) & (~hard_exclusion.astype(bool))
    try:
        from skimage.morphology import binary_closing, disk
        m = binary_closing(m, disk(2))
    except Exception:
        pass
    return m.astype(np.uint8)


def build_osm_pseudo_mask(
    *,
    osm_farmland_mask: np.ndarray,
    max_ndvi: np.ndarray,
    hard_exclusion: np.ndarray,
    worldcover_crop_mask: np.ndarray | None,
    min_cover: float = 0.05,
    max_cover: float = 0.95,
    min_ndvi_max: float = 0.30,
    worldcover_ndvi_min: float = 0.35,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Build OSM pseudo-GT mask with NDVI/WorldCover consistency checks."""
    mask = np.asarray(osm_farmland_mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool), {
            "osm_used": False,
            "osm_cover": 0.0,
            "osm_ndvi_max": 0.0,
            "worldcover_overlap": 0.0,
        }

    if worldcover_crop_mask is not None:
        wc_mask = np.asarray(worldcover_crop_mask, dtype=bool)
        worldcover_overlap = float((mask & wc_mask).sum()) / float(max(1, mask.sum()))
        # Keep polygon pixels supported by WorldCover OR high NDVI.
        mask = mask & (wc_mask | (max_ndvi >= float(worldcover_ndvi_min)))
    else:
        worldcover_overlap = 0.0

    mask &= ~np.asarray(hard_exclusion, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool), {
            "osm_used": False,
            "osm_cover": 0.0,
            "osm_ndvi_max": 0.0,
            "worldcover_overlap": worldcover_overlap,
        }

    cover = float(mask.mean())
    ndvi_max_inside = float(np.nanmax(max_ndvi[mask])) if mask.any() else 0.0
    use_mask = bool(min_cover <= cover <= max_cover and ndvi_max_inside >= min_ndvi_max)
    if not use_mask:
        mask = np.zeros_like(mask, dtype=bool)
        cover = 0.0

    return mask, {
        "osm_used": use_mask,
        "osm_cover": cover,
        "osm_ndvi_max": ndvi_max_inside,
        "worldcover_overlap": worldcover_overlap,
    }


# ── selective rerun ───────────────────────────────────────────────────────────

def load_rerun_ids(*, full_rebuild: bool) -> set[str]:
    import pandas as pd
    if full_rebuild:
        print("📋 Full rebuild mode: rerunning all tiles and replacing summary.")
        return set()
    if not SUMMARY_CSV.exists():
        print("📋 No existing summary — full run.")
        return set()
    df = pd.read_csv(SUMMARY_CSV)
    if "used_fallback" not in df.columns:
        print("📋 Old summary format — full run.")
        return set()
    ids = set(df.loc[df["used_fallback"].astype(bool), "tile_id"].tolist())
    if ids:
        print(f"📋 Existing summary found. Rerunning fallback tiles: {sorted(ids)}")
    else:
        print("📋 Existing summary: no fallback tiles — nothing to rerun.")
    return ids


def merge_summary(new_rows: list[dict], *, full_rebuild: bool) -> None:
    import pandas as pd
    if full_rebuild:
        merged = pd.DataFrame(new_rows).sort_values("tile_id").reset_index(drop=True)
    elif SUMMARY_CSV.exists():
        old = pd.read_csv(SUMMARY_CSV)
        new = pd.DataFrame(new_rows)
        merged = pd.concat(
            [old[~old["tile_id"].isin(new["tile_id"])], new],
            ignore_index=True,
        )
        merged = merged.sort_values("tile_id").reset_index(drop=True)
    else:
        merged = pd.DataFrame(new_rows).sort_values("tile_id").reset_index(drop=True)
    merged.to_csv(SUMMARY_CSV, index=False)


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    global W, H
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import from_bounds

    args = _parse_args()
    _apply_project_root(args.project_root)
    run_id = uuid.uuid4().hex[:12]
    rerun_ids = load_rerun_ids(full_rebuild=bool(args.full_rebuild))

    settings      = _get_settings()
    weak_settings = make_weak_settings(settings)
    weak_settings_road_retry = make_weak_road_retry_settings(weak_settings)
    weak_label_min_coverage_pct = float(getattr(settings, "WEAK_LABEL_MIN_COVERAGE_PCT", 0.5))
    weak_label_max_fallback_ratio = float(getattr(settings, "WEAK_LABEL_MAX_FALLBACK_RATIO", 0.35))
    weak_label_osm_override_enabled = bool(
        getattr(settings, "WEAK_LABEL_OSM_OVERRIDE_ENABLED", True)
    )
    weak_label_temporal_override_enabled = bool(
        getattr(settings, "WEAK_LABEL_TEMPORAL_OVERRIDE_ENABLED", True)
    )
    client        = _get_sentinel_client()
    from processing.priors.worldcover import CROPLAND_CLASS, WorldCoverPrior
    from providers.osm import build_osm_farmland_mask, fetch_osm_farmland_geometries

    wc_prior = WorldCoverPrior(
        year=int(getattr(settings, "WORLDCOVER_YEAR", 2021)),
        cache_dir=Path(str(getattr(settings, "PRIORS_CACHE_DIR", "/tmp/autodetect_priors_cache"))),
    )

    build_valid_mask_from_scl = import_callable(
        "processing.fields.composite",
        "build_valid_mask_from_scl", "buildvalidmaskfromscl",
    )
    select_dates_by_coverage = import_callable(
        "processing.fields.composite",
        "select_dates_by_coverage", "selectdatesbycoverage",
    )
    compute_all_indices = import_callable(
        "processing.fields.indices",
        "compute_all_indices", "computeallindices",
    )
    compute_phenometrics = import_callable(
        "processing.fields.composite",
        "compute_phenometrics", "computephenometrics",
    )
    try:
        run_postprocess = import_callable(
            "processing.fields.postprocess",
            "run_postprocess", "runpostprocess",
        )
    except Exception:
        run_postprocess = None

    pheno_mod = __import__("processing.fields.phenoclassify", fromlist=["*"])
    classify_landcover = _resolve_fn(
        pheno_mod,
        "classify_land_cover", "classify_landcover",
        "classifylandcover", "classifyLandCover",
    )
    PhenoThresholds = _resolve_fn(pheno_mod, "PhenoThresholds")
    WATER   = getattr(pheno_mod, "WATER")
    CROP    = getattr(pheno_mod, "CROP")
    FOREST  = getattr(pheno_mod, "FOREST")
    BUILTUP = getattr(pheno_mod, "BUILTUP", None)

    time_windows = build_time_windows(RANGE_START, RANGE_END, WINDOW_DAYS)

    async def fetch_tile(bbox, tf, tt, max_cloud_pct=60):
        if hasattr(client, "fetchtile"):
            return await client.fetchtile(bbox, tf, tt, W, H, max_cloud_pct)
        return await client.fetch_tile(bbox, tf, tt, W, H, max_cloud_pct=max_cloud_pct)

    async def fetch_time_series(
        bbox: tuple[float, float, float, float],
        *,
        max_cloud_pct: int = 60,
    ) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
        band_lists: dict[str, list[np.ndarray]] = {
            k: [] for k in ["B2", "B3", "B4", "B8", "B11", "B12"]
        }
        scl_list: list[np.ndarray] = []
        concurrency = max(1, int(getattr(settings, "SENTINEL_CONCURRENT_REQUESTS", 4)))
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch_window(tf: str, tt: str):
            async with semaphore:
                try:
                    return tf, tt, await fetch_tile(bbox, tf, tt, max_cloud_pct=max_cloud_pct)
                except Exception as e:
                    print(f"   ⚠️  {tf[:10]} skip: {str(e)[:120]}")
                    return None

        results = await asyncio.gather(*[_fetch_window(tf, tt) for tf, tt in time_windows])
        for item in results:
            if item is None:
                continue
            _, _, result = item
            try:
                for k in band_lists:
                    band_lists[k].append(np.asarray(result[k], dtype=np.float32))
                scl = result.get("SCL")
                if scl is None:
                    scl = np.full((H, W), 4, dtype=np.uint8)
                scl_list.append(np.asarray(scl, dtype=np.uint8))
            except Exception as e:
                print(f"   ⚠️  malformed scene skipped: {str(e)[:120]}")

        return band_lists, scl_list

    npz_paths = sorted(IN_NPZ_DIR.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"No tiles found in {IN_NPZ_DIR}")

    new_rows: list[dict] = []

    for npz_path in npz_paths:
        tile_id = npz_path.stem

        if rerun_ids and tile_id not in rerun_ids:
            print(f"   ⏭  {tile_id}  skip (already ok)")
            continue

        if args.skip_existing:
            existing_tif = OUT_LABEL_DIR / f"{tile_id}_label.tif"
            if existing_tif.exists() and existing_tif.stat().st_size > 0:
                print(f"   ⏭  {tile_id}  skip (label already exists)")
                continue

        z    = np.load(npz_path)
        tile_w, tile_h = infer_tile_dims_from_npz(z)
        W, H = int(tile_w), int(tile_h)
        bbox = tuple(map(float, z["bbox"].tolist()))
        lat, lon = center_latlon_from_bbox(bbox)

        print(f"\n🧩 {tile_id}  center=({lat:.5f},{lon:.5f})  tile={W}x{H}")

        band_lists, scl_list = await fetch_time_series(bbox)

        if len(scl_list) < 2:
            print("   ❌ too few scenes, skip")
            continue

        bands      = {k: np.stack(v, axis=0).astype(np.float32) for k, v in band_lists.items()}
        scl        = np.stack(scl_list, axis=0).astype(np.uint8)
        valid_mask = np.asarray(build_valid_mask_from_scl(scl), dtype=bool)
        indices    = compute_all_indices(bands)

        selected, sel_meta = call_select_dates(
            select_dates_by_coverage, valid_mask, n_dates=min(8, valid_mask.shape[0])
        )
        selected = np.asarray(selected) if selected is not None else np.asarray([], dtype=int)
        if selected.size == 0:
            print(f"   ❌ no selected dates, meta={sel_meta}")
            continue

        validsel = valid_mask[selected]
        ndvi     = indices["NDVI"][selected]
        ndwi     = indices["NDWI"][selected]
        mndwi    = indices.get("MNDWI", indices["NDWI"])[selected]
        bsi      = indices["BSI"][selected]
        msi      = indices["MSI"][selected]

        scl_water_mask = np.any(scl[selected] == 6, axis=0)
        pheno     = compute_phenometrics(ndvi, validsel)
        ndwimed   = median_composite(ndwi,  validsel)
        bsimed    = median_composite(bsi,   validsel)
        msimed    = median_composite(msi,   validsel)
        nirmed    = median_composite(bands["B8"][selected],  validsel)
        swirmed   = median_composite(bands["B11"][selected], validsel)
        mndwimax = nanmax_safe(np.where(validsel, mndwi, np.nan), axis=0, fill_value=np.nan)
        mndwimax   = np.where(np.isfinite(mndwimax), mndwimax, 0.0).astype(np.float32)
        ndwimean   = mean_composite(ndwi, validsel)
        validcount = validsel.sum(axis=0).astype(np.int32)

        thresholds = build_pheno_thresholds(PhenoThresholds, settings)

        # Apply latitude-adaptive overrides
        regional = get_regional_overrides(lat)
        if regional:
            for attr_name, new_val in regional.items():
                # Try exact name, camelCase, and no-underscore variants
                variants = [
                    attr_name,
                    attr_name.replace("_", ""),
                    "".join(w.capitalize() for w in attr_name.split("_")),
                ]
                variants[2] = variants[2][0].lower() + variants[2][1:]  # camelCase
                for v in variants:
                    if hasattr(thresholds, v):
                        setattr(thresholds, v, new_val)
                        break
            print(f"   🌡️  regional overrides (lat={lat:.1f}): {regional}")

        classes = np.asarray(classify_landcover(**_fill_kwargs(classify_landcover, {
            "pheno":          pheno,    "ndwimed":        ndwimed,
            "ndwi_med":       ndwimed,  "bsimed":         bsimed,
            "bsi_med":        bsimed,   "msimed":         msimed,
            "msi_med":        msimed,   "validcount":     validcount,
            "valid_count":    validcount, "thresholds":   thresholds,
            "mndwimax":       mndwimax, "mndwi_max":      mndwimax,
            "ndwimax":        mndwimax, "ndwi_max":       mndwimax,
            "sclwatermask":   scl_water_mask,
            "scl_water_mask": scl_water_mask,
            "ndwimean":       ndwimean, "ndwi_mean":      ndwimean,
        })))

        watermask     = (classes == WATER)
        forestmask    = (classes == FOREST)
        builtmask     = (classes == BUILTUP) if BUILTUP is not None \
                         else np.zeros((H, W), dtype=bool)
        candidatemask = (classes == CROP)
        hard_exclusion = watermask | forestmask | builtmask

        edgecomposite = z["edgecomposite"].astype(np.float32)
        maxndvi       = z["maxndvi"].astype(np.float32)
        ndvistd       = z["ndvistd"].astype(np.float32)
        transform     = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], W, H)
        tile_crs      = "EPSG:4326"

        worldcover_crop_mask: np.ndarray | None = None
        try:
            worldcover_grid = wc_prior.load_worldcover_grid(
                bbox,
                transform,
                (H, W),
                tile_crs,
            )
            worldcover_crop_mask = (worldcover_grid == CROPLAND_CLASS)
        except Exception as e:
            print(f"   ⚠️  worldcover load skipped: {str(e)[:120]}")
            worldcover_crop_mask = None

        # Relaxed OSM parameters for northern regions
        osm_min_area = 0.3 if lat > 57.0 else 1.0
        osm_min_ndvi = 0.20 if lat > 57.0 else 0.30
        osm_landuse_tags = ("farmland", "farm", "meadow", "orchard") if lat > 57.0 else (
            "farmland",
            "farm",
            "meadow",
        )

        osm_farmland_geoms = []
        last_osm_err = ""
        for attempt in range(3):
            try:
                osm_farmland_geoms = await fetch_osm_farmland_geometries(
                    bbox,
                    min_area_ha=osm_min_area,
                    max_area_ha=500.0,
                    min_compactness=0.1,
                    landuse_tags=osm_landuse_tags,
                )
                break
            except Exception as e:
                last_osm_err = str(e)[:160]
                if attempt < 2:
                    delay_s = float(1 << attempt)
                    print(
                        f"   ⚠️  OSM fetch retry {attempt + 1}/3 in {delay_s:.0f}s: "
                        f"{last_osm_err}"
                    )
                    await asyncio.sleep(delay_s)
                else:
                    osm_fetch_failed = True
                    print(f"   ⚠️  OSM farmland fetch skipped after retries: {last_osm_err}")
        osm_farmland_mask = build_osm_farmland_mask(
            osm_farmland_geoms,
            transform,
            (H, W),
            tile_crs,
        )
        osm_pseudo_mask, osm_meta = build_osm_pseudo_mask(
            osm_farmland_mask=osm_farmland_mask,
            max_ndvi=maxndvi,
            hard_exclusion=hard_exclusion,
            worldcover_crop_mask=worldcover_crop_mask,
            min_cover=0.02 if lat > 57.0 else 0.05,
            max_cover=0.95,
            min_ndvi_max=osm_min_ndvi,
            worldcover_ndvi_min=0.25 if lat > 57.0 else 0.35,
        )

        # ── debug: сколько кандидатов до postprocess ──────────────
        _cand_pct = float(candidatemask.mean()) * 100.0
        print(f"   📐 candidatemask before postprocess: "
              f"{candidatemask.sum()} px ({_cand_pct:.2f}%)")

        # ── run_postprocess с relaxed weak_settings ────────────────
        final_mask_u8: np.ndarray
        used_fallback = False
        postprocess_zeroed_candidates = False
        quality_gate_override_reason = ""
        weak_label_source = "postprocess"
        fallback_reason = "none"
        osm_fetch_failed = False
        postprocess_removed_by_road_ratio = 0.0

        if run_postprocess is not None:
            def _run_postprocess_once(cfg_obj):
                out = run_postprocess(**_fill_kwargs(run_postprocess, {
                    "candidatemask":      candidatemask.astype(bool),
                    "candidate_mask":     candidatemask.astype(bool),
                    "watermask":          watermask.astype(bool),
                    "water_mask":         watermask.astype(bool),
                    "classes":            classes,
                    "ndvi":               maxndvi,
                    "ndwi":               ndwimed,
                    "cfg":                cfg_obj,
                    "nir":                nirmed,
                    "swir":               swirmed,
                    "edgecomposite":      edgecomposite,
                    "edge_composite":     edgecomposite,
                    "ndvistd":            ndvistd,
                    "ndvi_std":           ndvistd,
                    "worldcovermask":     None,
                    "worldcover_mask":    None,
                    "bbox":               bbox,
                    "tiletransform":      transform,
                    "tile_transform":     transform,
                    "outshape":           (H, W),
                    "out_shape":          (H, W),
                    "crsepsg":            4326,
                    "crs_epsg":           4326,
                    "returndebugsteps":   True,
                    "return_debug_steps": True,
                }))
                if isinstance(out, tuple):
                    post_mask = np.asarray(out[0], dtype=np.uint8)
                    debug_meta = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
                else:
                    post_mask = np.asarray(out, dtype=np.uint8)
                    debug_meta = {}
                return post_mask, debug_meta

            try:
                post_mask, post_debug = _run_postprocess_once(weak_settings)
                final_mask_u8 = np.asarray(post_mask, dtype=np.uint8)

                if final_mask_u8.sum() == 0 and candidatemask.sum() > 0:
                    postprocess_zeroed_candidates = True
                    fallback_reason = "postprocess_zeroed_candidates"
                    debug_masks = (post_debug or {}).get("masks", {}) if isinstance(post_debug, dict) else {}
                    dbg_candidate = np.asarray(debug_masks.get("step_00_candidate_initial", np.zeros((H, W), dtype=np.uint8))) > 0
                    dbg_road = np.asarray(debug_masks.get("step_01_road_mask", np.zeros((H, W), dtype=np.uint8))) > 0
                    candidate_px = int(np.count_nonzero(dbg_candidate))
                    road_hit_px = int(np.count_nonzero(dbg_candidate & dbg_road))
                    road_hit_ratio = float(road_hit_px / max(1, candidate_px))
                    postprocess_removed_by_road_ratio = round(road_hit_ratio, 6)

                    def _rescue_accept(mask_u8: np.ndarray) -> tuple[bool, str]:
                        mask = np.asarray(mask_u8, dtype=bool)
                        if not mask.any():
                            return False, "empty_mask"
                        coverage_pct = float(mask.mean()) * 100.0
                        if coverage_pct < 0.3 or coverage_pct > 60.0:
                            return False, f"coverage_out_of_range({coverage_pct:.2f}%)"
                        ndvi_max_val = float(np.nanmax(maxndvi[mask]))
                        if ndvi_max_val < 0.25:
                            return False, f"ndvi_max_too_low({ndvi_max_val:.3f})"
                        temporal_stats = compute_mask_temporal_stats(ndvi, validsel, mask)
                        if not temporal_is_strong(temporal_stats, lat):
                            return False, "temporal_not_strong"
                        return True, "accepted"

                    should_retry_no_road = (
                        tile_id in ROAD_RETRY_TILES
                        or road_hit_ratio >= 0.90
                        or (road_hit_ratio >= 0.80 and candidate_px >= 3000)
                    )
                    if should_retry_no_road:
                        retry_mask, _retry_debug = _run_postprocess_once(weak_settings_road_retry)
                        accepted, reason = _rescue_accept(retry_mask)
                        if accepted:
                            final_mask_u8 = np.asarray(retry_mask, dtype=np.uint8)
                            weak_label_source = "postprocess_road_retry"
                            print(
                                "   ↻ road-barrier retry accepted: "
                                f"road_hit={100.0 * road_hit_ratio:.1f}% "
                                f"pixels={int(np.count_nonzero(final_mask_u8))}"
                            )
                        elif int(np.count_nonzero(retry_mask)) > 0:
                            print(
                                "   ⚠️  road-barrier retry rejected by acceptance gate: "
                                f"{reason}"
                            )
                        elif tile_id in ROAD_RETRY_TILES:
                            # Final rescue in weak-label mode: bypass road barrier entirely.
                            direct_mask = (candidatemask.astype(bool) & ~hard_exclusion).astype(np.uint8)
                            accepted, reason = _rescue_accept(direct_mask)
                            if accepted:
                                final_mask_u8 = direct_mask
                                weak_label_source = "postprocess_road_off_rescue"
                                print(
                                    "   ↻ road-barrier direct rescue accepted: "
                                    f"road_hit={100.0 * road_hit_ratio:.1f}% "
                                    f"pixels={int(np.count_nonzero(final_mask_u8))}"
                                )
                            elif int(np.count_nonzero(direct_mask)) > 0:
                                print(
                                    "   ⚠️  road-barrier direct rescue rejected by acceptance gate: "
                                    f"{reason}"
                                )

                # если postprocess вернул ноль, но кандидаты были — fallback
                if final_mask_u8.sum() == 0 and candidatemask.sum() > 0:
                    print(f"   ⚠️  postprocess zeroed all candidates — using lite_postprocess")
                    used_fallback = True
                    fallback_reason = "postprocess_zeroed_candidates"
                    final_mask_u8 = lite_postprocess(candidatemask, hard_exclusion)
                    weak_label_source = "lite_postprocess"

            except ValueError as e:
                if "expected 3" in str(e) and "got 2" in str(e):
                    print(f"   ⚠️  run_postprocess ValueError: {e}")
                    used_fallback = True
                    fallback_reason = "postprocess_value_error"
                    final_mask_u8 = lite_postprocess(candidatemask, hard_exclusion)
                    weak_label_source = "lite_postprocess"
                else:
                    raise
        else:
            used_fallback = True
            fallback_reason = "postprocess_unavailable"
            final_mask_u8 = lite_postprocess(candidatemask, hard_exclusion)
            weak_label_source = "lite_postprocess"

        if osm_pseudo_mask.any():
            final_mask_u8 = np.maximum(final_mask_u8, osm_pseudo_mask.astype(np.uint8))
            print(
                "   🌾 OSM pseudo-GT merged: "
                f"polygons={len(osm_farmland_geoms)} "
                f"cover={100.0 * float(osm_meta['osm_cover']):.2f}% "
                f"ndvi_max={float(osm_meta['osm_ndvi_max']):.3f}"
            )
            # If postprocess was zeroed but OSM pseudo-GT restored a meaningful mask,
            # we treat this tile as successfully recovered (not a hard fallback).
            if used_fallback:
                used_fallback = False
                weak_label_source = "osm_pseudo_merge"
                print("   ↻ fallback recovered by OSM pseudo-GT merge")
        else:
            print(
                "   ℹ️  OSM pseudo-GT skipped: "
                f"polygons={len(osm_farmland_geoms)} "
                f"cover={100.0 * float(osm_meta['osm_cover']):.2f}% "
                f"wc_overlap={100.0 * float(osm_meta['worldcover_overlap']):.2f}%"
            )

        # Targeted rescue for persistent fallback tiles:
        # use the pre-postprocess crop candidate cleaned by hard exclusions.
        if used_fallback and tile_id in ROAD_RETRY_TILES:
            rescue_mask = candidatemask.astype(bool) & ~hard_exclusion
            rescue_cover = float(rescue_mask.mean())
            rescue_ndvi_max = float(np.nanmax(maxndvi[rescue_mask])) if rescue_mask.any() else 0.0
            if worldcover_crop_mask is not None and rescue_mask.any():
                rescue_wc_overlap = float(
                    np.mean(worldcover_crop_mask[rescue_mask].astype(np.float32))
                )
            else:
                rescue_wc_overlap = 1.0

            rescue_min_cover = 0.005 if lat > 57.0 else 0.01
            rescue_max_cover = 0.50
            if lat > 57.0:
                rescue_ndvi_min = 0.20
                rescue_wc_min = 0.0
            elif lat < 53.0:
                rescue_ndvi_min = 0.18
                rescue_wc_min = 0.0
            else:
                rescue_ndvi_min = 0.22
                rescue_wc_min = 0.05

            rescue_ok = (
                rescue_min_cover <= rescue_cover <= rescue_max_cover
                and rescue_ndvi_max >= rescue_ndvi_min
                and rescue_wc_overlap >= rescue_wc_min
            )
            if rescue_ok:
                final_mask_u8 = rescue_mask.astype(np.uint8)
                used_fallback = False
                weak_label_source = "targeted_candidate_rescue"
                print(
                    "   ↻ fallback recovered by targeted candidate rescue: "
                    f"cover={100.0 * rescue_cover:.2f}% "
                    f"ndvi_max={rescue_ndvi_max:.3f} "
                    f"wc_overlap={100.0 * rescue_wc_overlap:.2f}%"
                )

        # If fallback mask itself looks field-like and consistent with priors,
        # count it as a recovered result instead of hard fallback.
        if used_fallback and final_mask_u8.any():
            fallback_mask = final_mask_u8.astype(bool)
            fallback_cover = float(fallback_mask.mean())
            fallback_ndvi_max = float(np.nanmax(maxndvi[fallback_mask])) if fallback_mask.any() else 0.0
            if worldcover_crop_mask is not None and fallback_mask.any():
                fallback_wc_overlap = float(
                    np.mean(worldcover_crop_mask[fallback_mask].astype(np.float32))
                )
            else:
                fallback_wc_overlap = 1.0

            min_cover_qc = 0.01 if lat > 57.0 else 0.02
            ndvi_min_qc = 0.20 if lat > 57.0 else 0.30
            wc_min_qc = 0.20 if _is_wc_relaxed_region(tile_id, lat, lon) else 0.35
            quality_ok = (
                min_cover_qc <= fallback_cover <= 0.85
                and fallback_ndvi_max >= ndvi_min_qc
                and fallback_wc_overlap >= wc_min_qc
            )
            if quality_ok:
                used_fallback = False
                weak_label_source = "fallback_quality_gate_accept"
                print(
                    "   ↻ fallback accepted by quality gate: "
                    f"cover={100.0 * fallback_cover:.2f}% "
                    f"ndvi_max={fallback_ndvi_max:.3f} "
                    f"wc_overlap={100.0 * fallback_wc_overlap:.2f}%"
                )

        # v5 quality gate for weak-label acceptance.
        final_mask = final_mask_u8.astype(bool)
        final_coverage_pct = float(final_mask.mean()) * 100.0
        final_ndvi_max = float(np.nanmax(maxndvi[final_mask])) if final_mask.any() else 0.0
        if worldcover_crop_mask is not None and final_mask.any():
            final_wc_overlap_pct = float(
                np.mean(worldcover_crop_mask[final_mask].astype(np.float32))
            ) * 100.0
        else:
            final_wc_overlap_pct = 100.0 if final_mask.any() else 0.0

        min_coverage_pct = float(weak_label_min_coverage_pct) * (0.6 if lat > 57.0 else 1.0)
        min_coverage_pct = float(np.clip(min_coverage_pct, 0.05, 5.0))
        min_ndvi_gate = 0.25 if lat > 57.0 else 0.35
        min_wc_overlap_pct, wc_gate_profile = get_adaptive_worldcover_gate_pct(tile_id, lat, lon)

        final_temporal_stats = compute_mask_temporal_stats(ndvi, validsel, final_mask)
        final_temporal_strong = temporal_is_strong(final_temporal_stats, lat)
        osm_signal_strong = bool(
            weak_label_osm_override_enabled
            and osm_meta.get("osm_used", False)
            and float(osm_meta.get("osm_ndvi_max", 0.0)) >= (0.22 if lat > 57.0 else 0.30)
            and float(osm_meta.get("osm_cover", 0.0)) >= (0.005 if lat > 57.0 else 0.01)
        )

        coverage_ok = final_coverage_pct > min_coverage_pct
        ndvi_ok = final_ndvi_max >= min_ndvi_gate
        wc_ok = final_wc_overlap_pct >= min_wc_overlap_pct
        quality_gate_passed = bool(coverage_ok and ndvi_ok and wc_ok)

        if not quality_gate_passed and osm_pseudo_mask.any():
            candidate = np.asarray(osm_pseudo_mask, dtype=bool)
            candidate_cov_pct = float(candidate.mean()) * 100.0
            candidate_ndvi_max = float(np.nanmax(maxndvi[candidate])) if candidate.any() else 0.0
            if worldcover_crop_mask is not None and candidate.any():
                candidate_wc_overlap_pct = float(
                    np.mean(worldcover_crop_mask[candidate].astype(np.float32))
                ) * 100.0
            else:
                candidate_wc_overlap_pct = 100.0 if candidate.any() else 0.0
            candidate_temporal_stats = compute_mask_temporal_stats(ndvi, validsel, candidate)
            candidate_temporal_strong = temporal_is_strong(candidate_temporal_stats, lat)

            candidate_cov_ok = candidate_cov_pct > min_coverage_pct
            candidate_ndvi_ok = candidate_ndvi_max >= min_ndvi_gate
            candidate_wc_ok = candidate_wc_overlap_pct >= min_wc_overlap_pct
            candidate_override = bool(
                weak_label_osm_override_enabled
                and weak_label_temporal_override_enabled
                and candidate_cov_ok
                and candidate_ndvi_ok
                and (not candidate_wc_ok)
                and osm_signal_strong
                and candidate_temporal_strong
            )
            candidate_ok = bool(
                candidate_cov_ok and candidate_ndvi_ok and (candidate_wc_ok or candidate_override)
            )
            if candidate_ok:
                final_mask_u8 = candidate.astype(np.uint8)
                final_mask = candidate
                final_coverage_pct = candidate_cov_pct
                final_ndvi_max = candidate_ndvi_max
                final_wc_overlap_pct = candidate_wc_overlap_pct
                final_temporal_stats = candidate_temporal_stats
                final_temporal_strong = candidate_temporal_strong
                quality_gate_passed = True
                used_fallback = False
                weak_label_source = "osm_pseudo_override"
                if candidate_override:
                    quality_gate_override_reason = "osm_ndvi_temporal_wc_override"
                print(
                    "   ↻ weak-label quality gate recovered by OSM pseudo-GT: "
                    f"coverage={final_coverage_pct:.2f}% "
                    f"ndvi_max={final_ndvi_max:.3f} "
                    f"wc_overlap={final_wc_overlap_pct:.2f}%"
                )

        coverage_ok = final_coverage_pct > min_coverage_pct
        ndvi_ok = final_ndvi_max >= min_ndvi_gate
        wc_ok = final_wc_overlap_pct >= min_wc_overlap_pct
        if (not quality_gate_passed) and coverage_ok and ndvi_ok and (not wc_ok):
            if (
                weak_label_osm_override_enabled
                and weak_label_temporal_override_enabled
                and osm_signal_strong
                and final_temporal_strong
            ):
                quality_gate_passed = True
                used_fallback = False
                quality_gate_override_reason = "osm_ndvi_temporal_wc_override"
                weak_label_source = "osm_ndvi_temporal_override"
                print(
                    "   ↻ weak-label accepted despite low WorldCover: "
                    f"wc_overlap={final_wc_overlap_pct:.2f}%<{min_wc_overlap_pct:.2f}% "
                    f"amp={float(final_temporal_stats['ndvi_amplitude']):.3f} "
                    f"entropy={float(final_temporal_stats['ndvi_entropy']):.3f}"
                )

        if not quality_gate_passed:
            targeted_mask, targeted_meta = targeted_quality_rescue(
                candidate_mask=candidatemask,
                hard_exclusion=hard_exclusion,
                max_ndvi=maxndvi,
                lat=lat,
            )
            if bool(targeted_meta.get("accepted", False)):
                final_mask_u8 = targeted_mask.astype(np.uint8)
                final_mask = targeted_mask
                final_coverage_pct = float(final_mask.mean()) * 100.0
                final_ndvi_max = float(np.nanmax(maxndvi[final_mask])) if final_mask.any() else 0.0
                if worldcover_crop_mask is not None and final_mask.any():
                    final_wc_overlap_pct = float(
                        np.mean(worldcover_crop_mask[final_mask].astype(np.float32))
                    ) * 100.0
                else:
                    final_wc_overlap_pct = 100.0 if final_mask.any() else 0.0
                final_temporal_stats = compute_mask_temporal_stats(ndvi, validsel, final_mask)
                quality_gate_passed = True
                used_fallback = False
                weak_label_source = "targeted_quality_rescue"
                quality_gate_override_reason = "targeted_soft_gate"
                print(
                    "   ↻ weak-label quality gate recovered by targeted pass: "
                    f"components={int(targeted_meta.get('components_kept', 0))} "
                    f"coverage={final_coverage_pct:.2f}%"
                )

        if not quality_gate_passed:
            used_fallback = True
            weak_label_source = "quality_gate_failed"
            fallback_reason = "quality_gate_failed"
            print(
                "   ⚠️  weak-label quality gate failed: "
                f"coverage={final_coverage_pct:.2f}%/{min_coverage_pct:.2f}% "
                f"ndvi_max={final_ndvi_max:.3f}/{min_ndvi_gate:.3f} "
                f"wc_overlap={final_wc_overlap_pct:.2f}%/{min_wc_overlap_pct:.2f}% "
                f"profile={wc_gate_profile} "
                f"amp={float(final_temporal_stats['ndvi_amplitude']):.3f} "
                f"entropy={float(final_temporal_stats['ndvi_entropy']):.3f}"
            )

        # ── save TIF ───────────────────────────────────────────────
        label_tif = OUT_LABEL_DIR / f"{tile_id}_label.tif"
        with rasterio.open(
            label_tif, "w", driver="GTiff",
            height=H, width=W, count=1, dtype="uint8",
            crs="EPSG:4326", transform=transform, nodata=0,
            compress="deflate", predictor=1,
            tiled=True, blockxsize=256, blockysize=256,
        ) as ds:
            ds.write(final_mask_u8, 1)

        # ── save PNG (2 панели: NDVI + weak label) ─────────────────
        png_path = OUT_PNG_DIR / f"{tile_id}.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(maxndvi, cmap="RdYlGn", vmin=0, vmax=0.9)
        axes[0].set_title("max NDVI")
        axes[0].axis("off")
        axes[1].imshow(maxndvi, cmap="gray")
        axes[1].imshow(
            np.ma.masked_where(final_mask_u8 == 0, final_mask_u8),
            cmap="Reds", alpha=0.55,
        )
        suffix = " ⚠ fallback" if used_fallback else " ✓ postprocess"
        axes[1].set_title(f"{tile_id}{suffix}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        cov = final_coverage_pct
        weak_label_source = (str(weak_label_source).strip() or "unknown")
        if used_fallback and fallback_reason == "none":
            fallback_reason = "fallback_unknown"
        if not used_fallback:
            fallback_reason = "none"
        new_rows.append({
            "run_id":               run_id,
            "tile_id":              tile_id,
            "label_tif":            str(label_tif),
            "preview_png":          str(png_path),
            "label_coverage_pct":   round(cov, 6),
            "n_selected_dates":     int(selected.size),
            "pixel_area_m2_approx": round(float(approx_pixel_area_m2(bbox)), 4),
            "used_fallback":        bool(used_fallback),
            "osm_polygons":         int(len(osm_farmland_geoms)),
            "osm_used":             bool(osm_meta["osm_used"]),
            "osm_cover_pct":        round(float(osm_meta["osm_cover"]) * 100.0, 6),
            "osm_worldcover_overlap_pct": round(float(osm_meta["worldcover_overlap"]) * 100.0, 6),
            "osm_ndvi_max":         round(float(osm_meta["osm_ndvi_max"]), 6),
            "quality_gate_passed":  bool(quality_gate_passed),
            "quality_gate_failed":  bool(not quality_gate_passed),
            "quality_gate_override": quality_gate_override_reason,
            "weak_label_source":   weak_label_source,
            "fallback_reason":      fallback_reason,
            "osm_fetch_failed":     bool(osm_fetch_failed),
            "wc_gate_profile":      wc_gate_profile,
            "wc_gate_min_overlap_pct": round(float(min_wc_overlap_pct), 6),
            "final_ndvi_max":       round(final_ndvi_max, 6),
            "final_wc_overlap_pct": round(final_wc_overlap_pct, 6),
            "temporal_valid_dates": int(final_temporal_stats["valid_dates"]),
            "temporal_amp":         round(float(final_temporal_stats["ndvi_amplitude"]), 6),
            "temporal_entropy":     round(float(final_temporal_stats["ndvi_entropy"]), 6),
            "temporal_has_peak":    bool(final_temporal_stats["has_growth_peak"]),
            "postprocess_zeroed_candidates": bool(postprocess_zeroed_candidates),
            "postprocess_removed_by_road_ratio": float(postprocess_removed_by_road_ratio),
        })
        status = "⚠ fallback" if used_fallback else "✅"
        print(f"   {status} {label_tif.name}  "
              f"coverage={cov:.2f}%  selected={int(selected.size)}")

    # ── merge + финальный вывод ────────────────────────────────────
    if new_rows:
        do_full_rebuild = bool(args.full_rebuild) and not bool(args.skip_existing)
        merge_summary(new_rows, full_rebuild=do_full_rebuild)
        print(f"\n📄 summary (merged) → {SUMMARY_CSV}")
    else:
        print("\n✅ Nothing to rerun — all tiles already ok.")

    final_df = pd.read_csv(SUMMARY_CSV)
    print(f"\n{'─'*65}")
    print(final_df[["tile_id", "label_coverage_pct",
                     "n_selected_dates", "used_fallback"]].to_string(index=False))

    n_fallback = int(final_df["used_fallback"].sum())
    fallback_ratio = float(n_fallback) / float(max(1, len(final_df)))
    if n_fallback:
        print(
            f"\n⚠️  {n_fallback} tile(s) still on fallback "
            f"({100.0 * fallback_ratio:.2f}%)"
        )
    else:
        print(f"\n🎉 All {len(final_df)} tiles via full postprocess pipeline")

    if fallback_ratio > weak_label_max_fallback_ratio:
        print(
            "⚠️  fallback ratio above configured limit: "
            f"{fallback_ratio:.3f} > {weak_label_max_fallback_ratio:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
