#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from processing.fields.ml_inference import FieldBoundaryInferencer
from core.config import get_settings
from training.gen_data import _build_feature_stack, _build_targets


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / max(union, 1.0)


def _compute_f1(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    tp = float(np.logical_and(mask_a, mask_b).sum())
    fp = float(np.logical_and(mask_a, ~mask_b).sum())
    fn = float(np.logical_and(~mask_a, mask_b).sum())
    return (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)


def _load_label_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
    return arr.astype(bool)


def main() -> None:
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX parity check for BoundaryUNet")
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--torch-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--max-tiles", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("backend/debug/runs/ml_parity.json"))
    args = parser.parse_args()

    torch_inf = FieldBoundaryInferencer(
        str(args.torch_model),
        norm_stats_path=str(args.norm),
        use_onnx=False,
    )
    onnx_inf = FieldBoundaryInferencer(
        str(args.onnx_model),
        norm_stats_path=str(args.norm),
        use_onnx=True,
    )

    rows = []
    for npz_path in sorted(args.tiles_dir.glob("*.npz"))[: max(1, args.max_tiles)]:
        tile_id = npz_path.stem
        label_path = args.labels_dir / f"{tile_id}_label.tif"
        if not label_path.exists():
            continue

        feature_channels = tuple(getattr(torch_inf, "feature_channels", ()))
        if feature_channels:
            x, _edge_valid_mask = _build_feature_stack(npz_path, feature_channels=feature_channels)
        else:
            x, _edge_valid_mask = _build_feature_stack(npz_path)
        gt_extent, gt_boundary, _ = _build_targets(label_path)

        pred_t = torch_inf.predict(x)
        pred_o = onnx_inf.predict(x)

        d_extent = float(np.mean(np.abs(pred_t["extent"] - pred_o["extent"])))
        d_boundary = float(np.mean(np.abs(pred_t["boundary"] - pred_o["boundary"])))
        d_distance = float(np.mean(np.abs(pred_t["distance"] - pred_o["distance"])))

        pt_extent = np.asarray(pred_t["extent"]) > 0.5
        po_extent = np.asarray(pred_o["extent"]) > 0.5
        pt_boundary = np.asarray(pred_t["boundary"]) > 0.5
        po_boundary = np.asarray(pred_o["boundary"]) > 0.5

        row = {
            "tile_id": tile_id,
            "delta_prob_extent": d_extent,
            "delta_prob_boundary": d_boundary,
            "delta_prob_distance": d_distance,
            "iou_torch": _compute_iou(pt_extent, gt_extent > 0.5),
            "iou_onnx": _compute_iou(po_extent, gt_extent > 0.5),
            "f1_torch": _compute_f1(pt_boundary, gt_boundary > 0.5),
            "f1_onnx": _compute_f1(po_boundary, gt_boundary > 0.5),
            "boundary_iou_torch": _compute_iou(pt_boundary, gt_boundary > 0.5),
            "boundary_iou_onnx": _compute_iou(po_boundary, gt_boundary > 0.5),
        }
        rows.append(row)

    if not rows:
        raise SystemExit("No comparable tiles found (check --tiles-dir and --labels-dir)")

    mean_delta_extent = float(np.mean([r["delta_prob_extent"] for r in rows]))
    mean_delta_boundary = float(np.mean([r["delta_prob_boundary"] for r in rows]))
    mean_delta_distance = float(np.mean([r["delta_prob_distance"] for r in rows]))
    iou_gap = float(np.mean([abs(r["iou_torch"] - r["iou_onnx"]) for r in rows]))
    f1_gap = float(np.mean([abs(r["f1_torch"] - r["f1_onnx"]) for r in rows]))
    boundary_iou_gap = float(
        np.mean([abs(r["boundary_iou_torch"] - r["boundary_iou_onnx"]) for r in rows])
    )

    summary = {
        "count": len(rows),
        "extent_threshold_used": float(getattr(cfg, "ML_EXTENT_BIN_THRESHOLD", 0.42)),
        "geometry_refine_profile": str(getattr(cfg, "GEOMETRY_REFINE_PROFILE", "balanced")),
        "date_selection_profile": str(getattr(cfg, "DATE_SELECTION_PROFILE", "adaptive_region")),
        "region_profile_mode": str(getattr(cfg, "REGION_PROFILE_MODE", "auto_only")),
        "region_lat_south_max": float(getattr(cfg, "REGION_LAT_SOUTH_MAX", 48.0)),
        "region_lat_north_min": float(getattr(cfg, "REGION_LAT_NORTH_MIN", 57.0)),
        "ml_feature_profile": str(getattr(torch_inf, "feature_profile", "unknown")),
        "channels": list(getattr(torch_inf, "feature_channels", [])),
        "mean_abs_delta": {
            "extent": mean_delta_extent,
            "boundary": mean_delta_boundary,
            "distance": mean_delta_distance,
        },
        "metric_gap": {
            "iou_abs_diff": iou_gap,
            "f1_abs_diff": f1_gap,
            "boundary_iou_abs_diff": boundary_iou_gap,
        },
        "parity_passed": bool(
            max(mean_delta_extent, mean_delta_boundary, mean_delta_distance) < 0.02
            and iou_gap < 0.01
            and f1_gap < 0.01
            and boundary_iou_gap < 0.01
        ),
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
