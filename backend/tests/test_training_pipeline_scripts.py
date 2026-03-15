from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from training.train_boundary_v3_cpu import _prepare_training_snapshot


ROOT = Path(__file__).resolve().parents[2]
PY_BIN = str(ROOT / ".venv" / "bin" / "python")


def test_prepare_open_boundary_corpus_writes_manifest(tmp_path):
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "demo.json").write_text(
        json.dumps(
            {
                "source": "Demo Open Fields",
                "artifact_uri": "https://example.invalid/demo",
                "license": "open",
                "country": ["RU"],
                "year": "2025",
                "label_type": "field_boundaries",
                "prepare_recipe": "demo",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend" / "training" / "prepare_open_boundary_corpus.py"),
            "--registry-dir",
            str(registry_dir),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source_count"] == 1
    assert payload["sources"][0]["source"] == "Demo Open Fields"


def test_summarize_release_qa_by_band_groups_rows(tmp_path):
    results = tmp_path / "results.jsonl"
    rows = [
        {"region": "Krasnodar", "status": "done", "queue_wall_s": 10, "field_count": 50, "stale_running": False, "empty_output": False},
        {"region": "Krasnodar", "status": "failed", "queue_wall_s": 20, "field_count": 0, "stale_running": False, "empty_output": True},
        {"region": "Novosibirsk", "status": "done", "queue_wall_s": 30, "field_count": 15, "stale_running": False, "empty_output": False},
    ]
    results.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend" / "training" / "scripts" / "summarize_release_qa_by_band.py"),
            str(results),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["region_bands"]["south"]["runs"] == 2
    assert payload["region_bands"]["south"]["fail_count"] == 1
    assert payload["region_bands"]["north"]["runs"] == 1


def test_train_boundary_v3_cpu_dry_run(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    config = tmp_path / "boundary.yaml"
    config.write_text(
        "\n".join(
            [
                "model_name: boundary_unet_v3_cpu",
                "model_version: boundary_unet_v3_cpu",
                "train_data_version: open_public_ru_v3_cpu",
                "feature_stack_version: v3_candidate_20ch_cpu",
                "feature_profile: v1_18ch",
                f"tiles_dir: {tiles_dir}",
                "output_model: backend/models/boundary_unet_v3_cpu.pth",
                "output_norm: backend/models/boundary_unet_v3_cpu.norm.json",
                "output_onnx: backend/models/boundary_unet_v3_cpu.onnx",
                "output_stats: backend/debug/runs/boundary_unet_v3_cpu_stats.json",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend" / "training" / "train_boundary_v3_cpu.py"),
            "--config",
            str(config),
            "--dry-run",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert "gen_data.py" in completed.stdout
    assert "--ml-feature-profile v1_18ch" in completed.stdout
    assert "--labels-dir" in completed.stdout
    assert "--progress-interval 50" in completed.stdout
    assert "--eval-progress-interval 50" in completed.stdout


def test_train_boundary_v3_cpu_dry_run_falls_back_without_s1_signal(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    np.savez_compressed(
        tiles_dir / "demo_tile.npz",
        edge_composite=np.ones((4, 4), dtype=np.float32),
        max_ndvi=np.ones((4, 4), dtype=np.float32),
    )
    config = tmp_path / "boundary.yaml"
    config.write_text(
        "\n".join(
            [
                "model_name: boundary_unet_v3_cpu",
                "model_version: boundary_unet_v3_cpu",
                "train_data_version: open_public_ru_v3_cpu",
                "feature_stack_version: v3_candidate_20ch_cpu",
                "feature_profile: v1_18ch",
                f"tiles_dir: {tiles_dir}",
                "output_model: backend/models/boundary_unet_v3_cpu.pth",
                "output_norm: backend/models/boundary_unet_v3_cpu.norm.json",
                "output_onnx: backend/models/boundary_unet_v3_cpu.onnx",
                "output_stats: backend/debug/runs/boundary_unet_v3_cpu_stats.json",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend" / "training" / "train_boundary_v3_cpu.py"),
            "--config",
            str(config),
            "--dry-run",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert "resolved=v2_16ch" in completed.stdout
    assert "--ml-feature-profile v2_16ch" in completed.stdout
    assert "--feature-stack-version v3_candidate_16ch_cpu" in completed.stdout


def test_prepare_training_snapshot_links_only_paired_tiles(tmp_path):
    tiles_dir = tmp_path / "tiles"
    labels_dir = tmp_path / "labels"
    tiles_dir.mkdir()
    labels_dir.mkdir()

    np.savez_compressed(tiles_dir / "demo_01.npz", edge_composite=np.ones((2, 2), dtype=np.float32))
    np.savez_compressed(tiles_dir / "demo_02.npz", edge_composite=np.ones((2, 2), dtype=np.float32))
    (labels_dir / "demo_01_label.tif").write_bytes(b"label")

    config = {
        "tiles_dir": str(tiles_dir),
        "labels_dir": str(labels_dir),
        "snapshot_root": str(tmp_path / "snapshot"),
    }
    snapshot_tiles, snapshot_labels, summary = _prepare_training_snapshot(config)

    assert snapshot_tiles.exists()
    assert snapshot_labels.exists()
    assert (snapshot_tiles / "demo_01.npz").exists()
    assert not (snapshot_tiles / "demo_02.npz").exists()
    assert (snapshot_labels / "demo_01_label.tif").exists()
    assert summary["paired_tiles"] == 1
    assert summary["skipped_tiles"] == 1


def test_train_yield_scripts_support_dry_run():
    corpus = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_yield_corpus.sh")],
        check=True,
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "PY_BIN": PY_BIN},
        capture_output=True,
        text=True,
    )
    assert "yield_corpus_v2.jsonl" in corpus.stdout

    baseline = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_yield_baseline.sh")],
        check=True,
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "PY_BIN": PY_BIN},
        capture_output=True,
        text=True,
    )
    assert "yield_baseline_v2.pkl" in baseline.stdout

    ensemble = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_yield_ensemble.sh")],
        check=True,
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "PY_BIN": PY_BIN},
        capture_output=True,
        text=True,
    )
    assert "yield_ensemble_v2.pkl" in ensemble.stdout


def test_train_orchestrated_supports_dry_run():
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_orchestrated.sh")],
        check=True,
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "PY_BIN": PY_BIN,
            "STAGES": "download,prepare,train-boundary,train-yield-corpus,train-yield-baseline,train-yield-ensemble,benchmark",
        },
        capture_output=True,
        text=True,
    )
    assert "Controlled Sentinel Hub failover is enabled" in completed.stdout
    assert "boundary_unet_v3_cpu" in completed.stdout
    assert "yield_baseline_v2.pkl" in completed.stdout
