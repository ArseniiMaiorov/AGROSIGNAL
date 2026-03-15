#!/usr/bin/env python3
"""
train_object_classifier.py
CLI для обучения ObjectClassifier на основе weak labels.
used_fallback=True → sample_weight=0.5.
quality_gate_failed=True → sample_weight=0.25.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
import sys

import numpy as np

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DATABASE_URL",      "postgresql+asyncpg://localhost/stub")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg://localhost/stub")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from utils.classifier_schema import (
    make_classifier_payload_portable,
    validate_classifier_file,
    validate_classifier_payload,
)
from utils.training import compute_tile_quality_weight

DEFAULT_TILES_DIR  = PROJECT_ROOT / "backend/debug/runs/real_tiles"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "backend/debug/runs/real_tiles_labels_weak"
DEFAULT_SUMMARY    = DEFAULT_LABELS_DIR / "weak_labels_summary.csv"
DEFAULT_OUTPUT     = PROJECT_ROOT / "backend/models/object_classifier.pkl"

FEATURE_NAMES = [
    "area_m2",
    "perimeter_m",
    "shape_index",
    "compactness",
    "elongation",
    "ndvi_mean",
    "ndvi_max",
    "ndvi_delta",
    "ndwi_mean",
    "msi_mean",
    "bsi_mean",
    "ndvi_variance",
    "worldcover_crop_pct",
    "growth_amplitude",
    "has_growth_peak",
    "ndvi_entropy",
    "neighbor_field_pct",
    "distance_to_road_m",
    "scl_valid_fraction_mean",
]


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if np.isnan(value):
            return default
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


# ── сегментация ───────────────────────────────────────────────────────────────

def segment_image(maxndvi: np.ndarray, edgecomp: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    try:
        from skimage.filters import gaussian
        from skimage.segmentation import watershed

        smoothed = gaussian(maxndvi, sigma=2.0)
        distance = ndi.distance_transform_edt(smoothed > 0.15)
        from skimage.feature import peak_local_max
        coords = peak_local_max(distance, min_distance=6, labels=(smoothed > 0.10))
        mask_m = np.zeros(distance.shape, dtype=bool)
        mask_m[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_m)
        gradient = 1.0 - gaussian(edgecomp, sigma=1.0)
        seg = watershed(gradient, markers, mask=(maxndvi > 0.05), compactness=0.001)
        return seg.astype(np.int32)
    except Exception as exc:
        print(f"   ℹ️  fallback segment_image: {exc}")

    seg_fallback, _ = ndi.label(maxndvi > 0.15)
    return seg_fallback.astype(np.int32)


def compute_shape_index(area: float, perimeter: float) -> float:
    return perimeter / (2.0 * np.sqrt(np.pi * max(area, 1e-6)))


def compute_compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def estimate_axes_lengths(seg_mask: np.ndarray) -> tuple[float, float]:
    """Estimate major/minor axis lengths from second moments (regionprops-free)."""
    ys, xs = np.nonzero(seg_mask)
    if ys.size < 2:
        return 0.0, 0.0

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    x -= x.mean()
    y -= y.mean()

    cov_xx = float(np.mean(x * x))
    cov_yy = float(np.mean(y * y))
    cov_xy = float(np.mean(x * y))
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov)  # ascending order
    eigvals = np.clip(eigvals, 0.0, None)

    major_axis = float(4.0 * np.sqrt(eigvals[1]))
    minor_axis = float(4.0 * np.sqrt(eigvals[0]))
    return major_axis, minor_axis


# ── извлечение признаков ──────────────────────────────────────────────────────

def extract_features_from_tile(
    npz_path: Path,
    label_tif: Path,
    pixel_size_m: float = 10.0,
    # порог перекрытия: снижен до 0.20 чтобы поймать больше crop-сегментов
    overlap_thresh: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    import rasterio
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

    z = np.load(npz_path)
    maxndvi = z["maxndvi"].astype(np.float32)
    meanndvi = z["meanndvi"].astype(np.float32) if "meanndvi" in z else np.clip(maxndvi * 0.8, 0.0, 1.0)
    ndvistd = z["ndvistd"].astype(np.float32)
    edgecomp = z["edgecomposite"].astype(np.float32)
    ndwi = (
        z["ndwi_mean"].astype(np.float32)
        if "ndwi_mean" in z
        else z["ndwi"].astype(np.float32)
        if "ndwi" in z
        else np.zeros_like(maxndvi)
    )
    bsi = z["bsi_mean"].astype(np.float32) if "bsi_mean" in z else np.zeros_like(maxndvi)
    nir = z["nir_median"].astype(np.float32) if "nir_median" in z else np.clip(maxndvi, 0.0, 1.0)
    swir = z["swir_median"].astype(np.float32) if "swir_median" in z else np.zeros_like(maxndvi)
    scl_valid_fraction = (
        z["scl_valid_fraction"].astype(np.float32)
        if "scl_valid_fraction" in z
        else np.ones_like(maxndvi, dtype=np.float32)
    )
    with np.errstate(all="ignore"):
        msi = swir / np.maximum(nir, 1e-3)
    msi = np.nan_to_num(msi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    ndvi_delta_map = np.clip(maxndvi - meanndvi, 0.0, 1.0).astype(np.float32)
    growth_amplitude_map = ndvi_delta_map
    has_growth_peak_map = (
        (growth_amplitude_map >= 0.20) & (ndvistd >= 0.05)
    ).astype(np.float32)
    ndvi_entropy_map = np.clip(
        np.log2(1.0 + np.square(ndvistd) * 32.0),
        0.0,
        4.0,
    ).astype(np.float32)
    worldcover_crop_proxy = (maxndvi >= 0.35).astype(np.float32)

    road_threshold = float(np.nanpercentile(edgecomp, 75)) if edgecomp.size else 0.35
    road_mask = (edgecomp >= road_threshold) & (maxndvi < 0.25)
    if np.any(road_mask):
        distance_to_road_m = distance_transform_edt(~road_mask).astype(np.float32) * float(pixel_size_m)
    else:
        distance_to_road_m = np.full_like(maxndvi, 1000.0, dtype=np.float32)

    with rasterio.open(label_tif) as src:
        label_mask = src.read(1).astype(bool)

    seg_map = segment_image(maxndvi, edgecomp)
    if seg_map.max() == 0:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, np.int32)

    rows_X, rows_y = [], []

    labels = np.unique(seg_map)
    labels = labels[labels > 0]
    for seg_label in labels:
        seg_mask = seg_map == seg_label
        area_px = float(np.count_nonzero(seg_mask))
        if area_px < 4:
            continue

        boundary = seg_mask & (~binary_erosion(seg_mask, structure=np.ones((3, 3), dtype=bool)))
        perim_px = float(np.count_nonzero(boundary))
        area_m2 = area_px * (pixel_size_m ** 2)
        perim_m = perim_px * pixel_size_m
        major_axis, minor_axis = estimate_axes_lengths(seg_mask)
        elongation = major_axis / max(1e-6, minor_axis) if major_axis > 0.0 and minor_axis > 0.0 else 1.0

        intersection   = int((seg_mask & label_mask).sum())
        overlap_ratio  = intersection / max(1, int(seg_mask.sum()))
        y_lbl = 1 if overlap_ratio >= overlap_thresh else 0

        ring = binary_dilation(seg_mask, iterations=1) & (~seg_mask)
        neighbor_labels = np.unique(seg_map[ring])
        neighbor_labels = neighbor_labels[(neighbor_labels > 0) & (neighbor_labels != seg_label)]
        neighbor_field_pct = min(1.0, float(len(neighbor_labels)) / 8.0)

        ndvi_vals = maxndvi[seg_mask]
        ndvi_mean_val = float(np.nanmean(ndvi_vals)) if ndvi_vals.size else 0.0
        ndvi_max_val = float(np.nanmax(ndvi_vals)) if ndvi_vals.size else 0.0

        rows_X.append([
            area_m2,
            perim_m,
            compute_shape_index(area_m2, perim_m),
            compute_compactness(area_m2, perim_m),
            float(elongation),
            ndvi_mean_val,
            ndvi_max_val,
            float(np.nanmean(ndvi_delta_map[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(ndwi[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(msi[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(bsi[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(np.square(ndvistd[seg_mask]))) if seg_mask.any() else 0.0,
            float(np.nanmean(worldcover_crop_proxy[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(growth_amplitude_map[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(has_growth_peak_map[seg_mask])) if seg_mask.any() else 0.0,
            float(np.nanmean(ndvi_entropy_map[seg_mask])) if seg_mask.any() else 0.0,
            float(neighbor_field_pct),
            float(np.nanmean(distance_to_road_m[seg_mask])) if seg_mask.any() else 1000.0,
            float(np.nanmean(scl_valid_fraction[seg_mask])) if seg_mask.any() else 1.0,
        ])
        rows_y.append(y_lbl)

    if not rows_X:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, np.int32)

    return (
        np.nan_to_num(np.array(rows_X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.array(rows_y, dtype=np.int32),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root (defaults to repository root inferred from script path)",
    )
    parser.add_argument("--tiles-dir",      type=Path,  default=None)
    parser.add_argument("--labels-dir",     type=Path,  default=None)
    parser.add_argument("--summary",        type=Path,  default=None)
    parser.add_argument("--output",         type=Path,  default=None)
    parser.add_argument("--min-score",      type=float, default=0.40)
    parser.add_argument("--max-iter",       type=int,   default=300)
    parser.add_argument("--pixel-size-m",   type=float, default=10.0)
    parser.add_argument("--overlap-thresh", type=float, default=0.20,
                        help="min overlap_ratio to label segment as crop")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    default_tiles_dir = project_root / "backend/debug/runs/real_tiles"
    default_labels_dir = project_root / "backend/debug/runs/real_tiles_labels_weak"
    args.tiles_dir = (args.tiles_dir or default_tiles_dir).resolve()
    args.labels_dir = (args.labels_dir or default_labels_dir).resolve()
    args.summary = (args.summary or (default_labels_dir / "weak_labels_summary.csv")).resolve()
    args.output = (args.output or (project_root / "backend/models/object_classifier.pkl")).resolve()

    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight

    summary = pd.read_csv(args.summary)
    print(f"\n📋 Loaded summary: {len(summary)} tiles")
    preview_cols = ["tile_id", "label_coverage_pct", "used_fallback"]
    if "quality_gate_failed" in summary.columns:
        preview_cols.append("quality_gate_failed")
    elif "quality_gate_passed" in summary.columns:
        preview_cols.append("quality_gate_passed")
    print(summary[preview_cols].to_string(index=False))

    all_X, all_y, all_w = [], [], []

    for _, row in summary.iterrows():
        tile_id       = row["tile_id"]
        used_fallback = _to_bool(row.get("used_fallback", False), default=False)
        if "quality_gate_failed" in row:
            quality_gate_failed = _to_bool(row.get("quality_gate_failed"), default=False)
        elif "quality_gate_passed" in row:
            quality_gate_failed = not _to_bool(row.get("quality_gate_passed"), default=True)
        else:
            quality_gate_failed = False
        npz_path      = args.tiles_dir  / f"{tile_id}.npz"
        label_tif     = args.labels_dir / f"{tile_id}_label.tif"

        if not npz_path.exists():
            print(f"   ⚠️  {tile_id}: npz not found, skip"); continue
        if not label_tif.exists():
            print(f"   ⚠️  {tile_id}: tif not found, skip"); continue

        print(
            f"\n🧩 {tile_id}  fallback={used_fallback}  "
            f"quality_gate_failed={quality_gate_failed}"
        )
        X, y = extract_features_from_tile(
            npz_path, label_tif,
            args.pixel_size_m, args.overlap_thresh,
        )
        if X.shape[0] == 0:
            print("   ⚡ no segments"); continue

        tile_weight = compute_tile_quality_weight(
            used_fallback=used_fallback,
            quality_gate_failed=quality_gate_failed,
        )
        w = np.full(len(y), tile_weight, dtype=np.float32)
        all_X.append(X)
        all_y.append(y)
        all_w.append(w)

        print(f"   segments={len(y)}  "
              f"crop={100.0*y.mean():.1f}%  "
              f"weight={tile_weight:.2f}")

    if not all_X:
        raise SystemExit("❌ No features extracted")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    w = np.concatenate(all_w)

    n_crop    = int(y.sum())
    n_noncrop = int((y == 0).sum())
    print(f"\n{'─'*55}")
    print(f"📊 Dataset : {len(y)} segments")
    print(f"   crop    : {n_crop}  ({100.0*n_crop/len(y):.1f}%)")
    print(f"   non-crop: {n_noncrop}  ({100.0*n_noncrop/len(y):.1f}%)")

    if n_crop == 0:
        raise SystemExit(
            "❌ 0 crop segments — снизь --overlap-thresh (сейчас "
            f"{args.overlap_thresh}) или проверь weak label TIF"
        )

    # ── обучение ──────────────────────────────────────────────────────────
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])

    # ── cross-val (sklearn >= 1.4: fit_params убран из cross_val_score) ───
    # делаем manual CV чтобы корректно передать sample_weight
    if len(np.unique(y)) >= 2 and n_crop >= 4:
        n_splits = min(5, n_crop)  # не больше числа crop-образцов
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_f1s = []
            for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                w_tr        = w[train_idx]
                class_w = compute_sample_weight(class_weight="balanced", y=y_tr).astype(np.float32)
                clf.fit(X_tr, y_tr, hgb__sample_weight=(w_tr * class_w))
                y_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                fold_f1s.append(f1)
            arr = np.array(fold_f1s)
            print(f"🔁 CV F1: {arr.mean():.3f} ± {arr.std():.3f}  "
                  f"(folds={n_splits})")
    else:
        print("⚠️  not enough crop samples for CV — training directly")

    # финальное обучение на всём датасете
    class_w_all = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    clf.fit(X, y, hgb__sample_weight=(w * class_w_all))

    print("\n🌲 Model: HistGradientBoostingClassifier (LightGBM-like histogram boosting)")
    print("   Feature importances are not exposed natively; use permutation importance if needed.")

    # ── сохранение ────────────────────────────────────────────────────────
    payload = {
        "pipeline":      clf,
        "feature_names": FEATURE_NAMES,
        "threshold":     args.min_score,
    }
    payload = make_classifier_payload_portable(payload)
    validate_classifier_payload(
        {
            "pipeline": payload["pipeline"],
            "feature_columns": FEATURE_NAMES,
        },
        FEATURE_NAMES,
    )

    saved_via_oc = False
    try:
        _oc_mod = __import__("processing.fields.object_classifier", fromlist=["*"])
        OC = getattr(_oc_mod, "ObjectClassifier", None)
        if OC is not None:
            oc = OC()
            oc._pipeline = clf
            oc._feature_columns = tuple(FEATURE_NAMES)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            oc.save(args.output)
            saved_via_oc = True
            print(f"\n✅ Saved via ObjectClassifier → {args.output}")
    except Exception as e:
        print(f"   ℹ️  ObjectClassifier wrapper: {e}")

    if not saved_via_oc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(payload, f, protocol=5)
        print(f"\n✅ Saved raw pickle → {args.output}")

    validate_classifier_file(args.output, FEATURE_NAMES)
    print(f"🔍 Схема classifier проверена: {len(FEATURE_NAMES)} признаков")

    print(f"\n💡 Добавь в .env:")
    print(f"   USEOBJECTCLASSIFIER=true")
    print(f"   OBJECTCLASSIFIERPATH={args.output}")
    print(f"   OBJECTMINSCORE={args.min_score}")


if __name__ == "__main__":
    main()
