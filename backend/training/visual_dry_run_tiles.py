#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from processing.fields.ml_inference import FieldBoundaryInferencer
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


def _load_preview_background(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as z:
        max_ndvi = np.asarray(z["maxndvi"], dtype=np.float32)
        mean_ndvi = np.asarray(z["meanndvi"], dtype=np.float32)
        edge = np.asarray(z["edgecomposite"], dtype=np.float32)
    return np.stack(
        [
            np.clip(max_ndvi, 0.0, 1.0),
            np.clip(mean_ndvi, 0.0, 1.0),
            np.clip(edge, 0.0, 1.0),
        ],
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual dry-run for south/north tiles")
    parser.add_argument(
        "--tile-ids",
        nargs="+",
        default=["krasnodar_01", "lenoblast_01"],
        help="Tile ids from backend/debug/runs/real_tiles",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=PROJECT_ROOT / "backend/debug/runs/real_tiles",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "backend/debug/runs/real_tiles_labels_weak",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "backend/models/boundary_unet_v2.onnx",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=PROJECT_ROOT / "backend/models/boundary_unet_v2.norm.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "backend/debug/runs/visual_dry_run",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inferencer = FieldBoundaryInferencer(
        str(args.model),
        norm_stats_path=str(args.norm),
        use_onnx=args.model.suffix.lower() == ".onnx",
    )
    feature_channels = tuple(getattr(inferencer, "feature_channels", ()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for tile_id in args.tile_ids:
        npz_path = args.tiles_dir / f"{tile_id}.npz"
        label_path = args.labels_dir / f"{tile_id}_label.tif"
        if not npz_path.exists():
            raise FileNotFoundError(f"Tile not found: {npz_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found: {label_path}")

        x, _edge_valid_mask = _build_feature_stack(npz_path, feature_channels=feature_channels)
        gt_extent, gt_boundary, _distance = _build_targets(label_path)
        pred = inferencer.predict(x, tile_size=args.tile_size, overlap=args.overlap)

        pred_extent_prob = np.asarray(pred["extent"], dtype=np.float32)
        pred_boundary_prob = np.asarray(pred["boundary"], dtype=np.float32)
        pred_extent = pred_extent_prob >= float(args.threshold)
        pred_boundary = pred_boundary_prob >= float(args.threshold)
        gt_extent_mask = gt_extent > 0.5
        gt_boundary_mask = gt_boundary > 0.5

        weak_extent_iou = _compute_iou(pred_extent, gt_extent_mask)
        weak_boundary_f1 = _compute_f1(pred_boundary, gt_boundary_mask)
        weak_boundary_iou = _compute_iou(pred_boundary, gt_boundary_mask)
        background = _load_preview_background(npz_path)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"{tile_id} | score={float(pred.get('score', 0.0)):.3f} | "
            f"weak_extent_iou={weak_extent_iou:.3f} | weak_boundary_f1={weak_boundary_f1:.3f}",
            fontsize=12,
        )

        axes[0, 0].imshow(background)
        axes[0, 0].set_title("Composite RGB")
        axes[0, 1].imshow(gt_extent_mask, cmap="Greens")
        axes[0, 1].set_title("Weak Label Extent")
        axes[0, 2].imshow(gt_boundary_mask, cmap="magma")
        axes[0, 2].set_title("Weak Label Boundary")

        axes[1, 0].imshow(pred_extent_prob, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("Pred Extent Prob")
        axes[1, 1].imshow(background)
        axes[1, 1].contour(gt_extent_mask.astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=1.0)
        axes[1, 1].contour(pred_extent.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=1.0)
        axes[1, 1].set_title("Extent Overlay: weak=yellow, pred=red")
        axes[1, 2].imshow(pred_boundary_prob, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 2].set_title("Pred Boundary Prob")

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        png_path = args.out_dir / f"{tile_id}_dry_run.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        rows.append(
            {
                "tile_id": tile_id,
                "score": float(pred.get("score", 0.0)),
                "weak_extent_iou": float(weak_extent_iou),
                "weak_boundary_f1": float(weak_boundary_f1),
                "weak_boundary_iou": float(weak_boundary_iou),
                "png": str(png_path),
            }
        )

    summary = {
        "model": str(args.model),
        "norm": str(args.norm),
        "tile_size": int(args.tile_size),
        "overlap": int(args.overlap),
        "threshold": float(args.threshold),
        "rows": rows,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
