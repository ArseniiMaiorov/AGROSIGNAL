#!/usr/bin/env python3
"""Remove BoundaryUNet training artifacts for a clean retrain."""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PATHS_TO_DELETE = [
    ROOT / "backend" / "debug" / "runs" / "real_tiles",
    ROOT / "backend" / "debug" / "runs" / "real_tiles_labels_weak",
    ROOT / "backend" / "debug" / "runs" / "boundary_dataset_stats.json",
    ROOT / "backend" / "debug" / "runs" / "ml_parity.json",
]

MODEL_GLOBS = ["boundary_unet_v1.*", "boundary_unet_v2.*"]


def clean() -> None:
    for path in PATHS_TO_DELETE:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Removed dir:  {path}")
        elif path.is_file():
            path.unlink()
            print(f"Removed file: {path}")

    models_dir = ROOT / "backend" / "models"
    for pattern in MODEL_GLOBS:
        for model_path in models_dir.glob(pattern):
            model_path.unlink()
            print(f"Removed model: {model_path}")

    print("Clean complete. Ready for retrain from scratch.")


if __name__ == "__main__":
    clean()
