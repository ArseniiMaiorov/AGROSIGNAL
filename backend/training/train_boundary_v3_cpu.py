#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "backend" / "training" / "config" / "boundary_unet_v3_cpu.yaml"


def _as_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _tiles_dir_from_config(config: dict[str, object]) -> Path:
    return _as_project_path(str(config.get("tiles_dir", "backend/debug/runs/real_tiles")))


def _labels_dir_from_config(config: dict[str, object]) -> Path:
    return _as_project_path(str(config.get("labels_dir", "backend/debug/runs/real_tiles_labels_weak")))


def _snapshot_root_from_config(config: dict[str, object]) -> Path:
    return _as_project_path(str(config.get("snapshot_root", "backend/debug/runs/training_snapshot_boundary_v3")))


def _has_live_s1_signal(tiles_dir: Path, *, sample_limit: int = 8) -> bool | None:
    if not tiles_dir.exists():
        return None
    tile_paths = sorted(tiles_dir.glob("*.npz"))[:sample_limit]
    if not tile_paths:
        return None
    for tile_path in tile_paths:
        try:
            with np.load(tile_path, allow_pickle=False) as data:
                if "s1_vv_mean" not in data.files or "s1_vh_mean" not in data.files:
                    continue
                s1_vv = data["s1_vv_mean"]
                s1_vh = data["s1_vh_mean"]
                if (
                    abs(float(s1_vv.mean())) > 1e-6
                    or abs(float(s1_vh.mean())) > 1e-6
                    or float(s1_vv.std()) > 1e-6
                    or float(s1_vh.std()) > 1e-6
                ):
                    return True
        except Exception:
            continue
    return False


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_training_snapshot(config: dict[str, object]) -> tuple[Path, Path, dict[str, object]]:
    tiles_dir = _tiles_dir_from_config(config)
    labels_dir = _labels_dir_from_config(config)
    snapshot_root = _snapshot_root_from_config(config)
    snapshot_tiles_dir = snapshot_root / "tiles"
    snapshot_labels_dir = snapshot_root / "labels"

    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    snapshot_tiles_dir.mkdir(parents=True, exist_ok=True)
    snapshot_labels_dir.mkdir(parents=True, exist_ok=True)

    paired_tiles = 0
    skipped_tiles = 0
    for tile_path in sorted(tiles_dir.glob("*.npz")):
        label_path = labels_dir / f"{tile_path.stem}_label.tif"
        if not label_path.exists():
            skipped_tiles += 1
            continue
        _link_or_copy(tile_path, snapshot_tiles_dir / tile_path.name)
        _link_or_copy(label_path, snapshot_labels_dir / label_path.name)
        paired_tiles += 1

    if paired_tiles == 0:
        raise RuntimeError(
            f"No paired tile/label files found for training snapshot. "
            f"tiles_dir={tiles_dir} labels_dir={labels_dir}"
        )

    summary = {
        "snapshot_root": str(snapshot_root.resolve()),
        "tiles_dir": str(snapshot_tiles_dir.resolve()),
        "labels_dir": str(snapshot_labels_dir.resolve()),
        "paired_tiles": paired_tiles,
        "skipped_tiles": skipped_tiles,
    }
    return snapshot_tiles_dir, snapshot_labels_dir, summary


def _resolve_training_profile(config: dict[str, object]) -> tuple[str, str]:
    requested_profile = str(config.get("feature_profile", "v1_18ch"))
    requested_stack = str(config.get("feature_stack_version", "v3_candidate_20ch_cpu"))
    if requested_profile != "v1_18ch":
        return requested_profile, requested_stack

    s1_signal = _has_live_s1_signal(_tiles_dir_from_config(config))
    if s1_signal is None:
        return requested_profile, requested_stack
    if s1_signal:
        return requested_profile, requested_stack

    fallback_stack = requested_stack
    if requested_stack == "v3_candidate_20ch_cpu":
        fallback_stack = "v3_candidate_16ch_cpu"
    return "v2_16ch", fallback_stack


def _build_gen_data_command(
    config: dict[str, object],
    *,
    python_bin: str,
    feature_profile: str,
    feature_stack_version: str,
    tiles_dir: Path,
    labels_dir: Path,
) -> list[str]:
    command = [
        python_bin,
        str(PROJECT_ROOT / "backend" / "training" / "gen_data.py"),
        "--tiles-dir",
        str(tiles_dir),
        "--labels-dir",
        str(labels_dir),
        "--output-model",
        str(_as_project_path(str(config["output_model"]))),
        "--output-norm",
        str(_as_project_path(str(config["output_norm"]))),
        "--output-onnx",
        str(_as_project_path(str(config["output_onnx"]))),
        "--output-stats",
        str(_as_project_path(str(config["output_stats"]))),
        "--epochs",
        str(config.get("epochs", 24)),
        "--patience",
        str(config.get("patience", 8)),
        "--batch-size",
        str(config.get("batch_size", 2)),
        "--eval-batch-size",
        str(config.get("eval_batch_size", 1)),
        "--accumulation-steps",
        str(config.get("accumulation_steps", 8)),
        "--patch-size",
        str(config.get("patch_size", 256)),
        "--stride",
        str(config.get("stride", 64)),
        "--tile-cache-size",
        str(config.get("tile_cache_size", 2)),
        "--num-workers",
        str(config.get("num_workers", 0)),
        "--lr",
        str(config.get("lr", 1e-4)),
        "--weight-decay",
        str(config.get("weight_decay", 1e-4)),
        "--dropout-bottleneck",
        str(config.get("dropout_bottleneck", 0.30)),
        "--dropout-dec4",
        str(config.get("dropout_dec4", 0.20)),
        "--dropout-dec3",
        str(config.get("dropout_dec3", 0.10)),
        "--progress-interval",
        str(config.get("progress_interval", 50)),
        "--eval-progress-interval",
        str(config.get("eval_progress_interval", 50)),
        "--ml-feature-profile",
        feature_profile,
        "--model-version",
        str(config.get("model_version", "boundary_unet_v3_cpu")),
        "--train-data-version",
        str(config.get("train_data_version", "open_public_ru_v3_cpu")),
        "--feature-stack-version",
        feature_stack_version,
        "--geometry-refine-profile",
        str(config.get("geometry_refine_profile", "balanced")),
        "--ml-extent-bin-threshold",
        str(config.get("extent_threshold", 0.42)),
    ]
    if bool(config.get("cpu", True)):
        command.append("--cpu")
    return command


def _write_candidate_manifest(
    config: dict[str, object],
    *,
    config_path: Path,
    feature_profile: str,
    feature_stack_version: str,
) -> Path:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "model_name": config.get("model_name", "boundary_unet_v3_cpu"),
        "model_version": config.get("model_version", "boundary_unet_v3_cpu"),
        "train_data_version": config.get("train_data_version", "open_public_ru_v3_cpu"),
        "feature_stack_version": feature_stack_version,
        "feature_profile": feature_profile,
        "outputs": {
            "model": str(_as_project_path(str(config["output_model"])).resolve()),
            "norm": str(_as_project_path(str(config["output_norm"])).resolve()),
            "onnx": str(_as_project_path(str(config["output_onnx"])).resolve()),
            "stats": str(_as_project_path(str(config["output_stats"])).resolve()),
        },
        "cpu_safe": bool(config.get("cpu", True)),
        "low_mem": bool(config.get("low_mem", True)),
    }
    out_path = PROJECT_ROOT / "backend" / "debug" / "runs" / "boundary_unet_v3_cpu_candidate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-safe wrapper for BoundaryUNet v3 candidate training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Invalid config payload: {config_path}")

    feature_profile, feature_stack_version = _resolve_training_profile(config)
    if feature_profile != str(config.get("feature_profile", "v1_18ch")):
        print(
            f"[profile] requested={config.get('feature_profile', 'v1_18ch')} "
            f"resolved={feature_profile} feature_stack_version={feature_stack_version}"
        )

    if args.dry_run:
        tiles_dir = _tiles_dir_from_config(config)
        labels_dir = _labels_dir_from_config(config)
    else:
        tiles_dir, labels_dir, snapshot_summary = _prepare_training_snapshot(config)
        print(
            f"[snapshot] paired_tiles={snapshot_summary['paired_tiles']} "
            f"skipped_tiles={snapshot_summary['skipped_tiles']} "
            f"root={snapshot_summary['snapshot_root']}"
        )

    command = _build_gen_data_command(
        config,
        python_bin=args.python_bin,
        feature_profile=feature_profile,
        feature_stack_version=feature_stack_version,
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
    )
    if args.dry_run:
        print(" ".join(command))
        return 0

    env = dict(os.environ)
    env.setdefault("ML_FEATURE_PROFILE", feature_profile)
    env.setdefault("MODEL_VERSION", str(config.get("model_version", "boundary_unet_v3_cpu")))
    env.setdefault("TRAIN_DATA_VERSION", str(config.get("train_data_version", "open_public_ru_v3_cpu")))
    env.setdefault("FEATURE_STACK_VERSION", feature_stack_version)

    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    manifest = _write_candidate_manifest(
        config,
        config_path=config_path,
        feature_profile=feature_profile,
        feature_stack_version=feature_stack_version,
    )
    print(manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
