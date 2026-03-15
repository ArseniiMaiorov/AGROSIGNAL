"""Tests for v3 U-Net edge fallback utilities."""
from __future__ import annotations

import numpy as np
import pytest

from processing.fields.ml_inference import (
    BoundaryUNet,
    FEATURE_CHANNELS,
    FieldBoundaryInferencer,
    get_feature_channels,
)
from processing.fields.unet_edge import (
    UNetEdgeDetector,
    build_unet_edge_input,
    predict_edge_map,
)


def test_build_unet_edge_input_supports_optional_s1_channels():
    shape = (8, 8)
    edgecomp = np.full(shape, 0.2, dtype=np.float32)
    maxndvi = np.full(shape, 0.6, dtype=np.float32)
    ndvistd = np.full(shape, 0.2, dtype=np.float32)
    scl = np.zeros(shape, dtype=np.float32)
    vv_edge = np.full(shape, 0.3, dtype=np.float32)
    vhvv_ratio = np.full(shape, 0.4, dtype=np.float32)

    stacked = build_unet_edge_input(
        edgecomp,
        maxndvi,
        ndvistd,
        scl,
        vv_edge=vv_edge,
        vhvv_ratio=vhvv_ratio,
    )

    assert stacked.shape == (6, 8, 8)
    assert stacked.dtype == np.float32
    assert np.all(stacked >= 0.0)
    assert np.all(stacked <= 1.0)


def test_predict_edge_map_uses_heuristic_fallback_without_model_weights():
    edgecomp = np.zeros((16, 16), dtype=np.float32)
    edgecomp[:, 8] = 1.0

    maxndvi = np.full((16, 16), 0.3, dtype=np.float32)
    maxndvi[:, 8:] = 0.7
    ndvistd = np.full((16, 16), 0.1, dtype=np.float32)
    ndvistd[:, 8:] = 0.2
    scl = np.zeros((16, 16), dtype=np.float32)

    detector = UNetEdgeDetector(checkpoint_path=None, heuristic_only=True)
    prob = predict_edge_map(
        detector,
        edgecomp,
        maxndvi,
        ndvistd,
        scl,
        threshold=0.4,
    )

    assert prob.shape == edgecomp.shape
    assert prob.dtype == np.float32
    assert float(prob.max()) <= 1.0
    assert float(prob.min()) >= 0.0
    assert float(prob[:, 8].mean()) > float(prob[:, 0].mean())


def test_boundary_unet_forward_shape():
    torch = pytest.importorskip("torch")
    model = BoundaryUNet(in_channels=len(FEATURE_CHANNELS))
    x = torch.randn(2, len(FEATURE_CHANNELS), 64, 64)
    out = model(x)
    assert set(out.keys()) == {"extent", "boundary", "distance"}
    assert out["extent"].shape == (2, 1, 64, 64)
    assert out["boundary"].shape == (2, 1, 64, 64)
    assert out["distance"].shape == (2, 1, 64, 64)


def test_field_boundary_inferencer_torch_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    model = BoundaryUNet(in_channels=len(FEATURE_CHANNELS))

    ckpt_path = tmp_path / "boundary_unet_v1.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_stats": {
                "mean": [0.0] * len(FEATURE_CHANNELS),
                "std": [1.0] * len(FEATURE_CHANNELS),
            },
        },
        ckpt_path,
    )

    inferencer = FieldBoundaryInferencer(str(ckpt_path), device="cpu")
    feature_stack = np.random.rand(len(FEATURE_CHANNELS), 96, 96).astype(np.float32)
    pred = inferencer.predict(feature_stack, tile_size=64, overlap=16)

    assert pred["extent"].shape == (96, 96)
    assert pred["boundary"].shape == (96, 96)
    assert pred["distance"].shape == (96, 96)
    assert 0.0 <= float(pred["score"]) <= 1.0


def test_field_boundary_inferencer_autodetects_v1_profile_from_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    channels_v1 = get_feature_channels("v1_18ch")
    model = BoundaryUNet(in_channels=len(channels_v1))

    ckpt_path = tmp_path / "boundary_unet_v1.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_stats": {
                "mean": [0.0] * len(channels_v1),
                "std": [1.0] * len(channels_v1),
            },
        },
        ckpt_path,
    )

    inferencer = FieldBoundaryInferencer(
        str(ckpt_path),
        device="cpu",
        feature_profile="v2_16ch",
    )
    assert inferencer.feature_profile == "v1_18ch"
    assert len(inferencer.feature_channels) == len(channels_v1)

    feature_stack = np.random.rand(len(channels_v1), 64, 64).astype(np.float32)
    pred = inferencer.predict(feature_stack, tile_size=64, overlap=16)
    assert pred["extent"].shape == (64, 64)


def test_field_boundary_inferencer_supports_tta_and_multiscale(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model = BoundaryUNet(in_channels=len(FEATURE_CHANNELS))

    ckpt_path = tmp_path / "boundary_unet_v2.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_stats": {
                "mean": [0.0] * len(FEATURE_CHANNELS),
                "std": [1.0] * len(FEATURE_CHANNELS),
            },
        },
        ckpt_path,
    )

    inferencer = FieldBoundaryInferencer(str(ckpt_path), device="cpu")
    calls: list[tuple[int, int]] = []
    original = inferencer._predict_spatial_batch

    def _wrapped(arr, *, tile_size, overlap, **kwargs):
        calls.append((arr.shape[-2], arr.shape[-1]))
        return original(arr, tile_size=tile_size, overlap=overlap, **kwargs)

    monkeypatch.setattr(inferencer, "_predict_spatial_batch", _wrapped)
    feature_stack = np.random.rand(len(FEATURE_CHANNELS), 96, 96).astype(np.float32)
    pred = inferencer.predict(
        feature_stack,
        tile_size=64,
        overlap=16,
        tta_mode="rotate4",
        scales=(1.0, 0.75),
    )

    assert pred["extent"].shape == (96, 96)
    assert pred["boundary"].shape == (96, 96)
    assert pred["distance"].shape == (96, 96)
    assert len(calls) == 8
    assert (72, 72) in calls
    assert pred["tta_transform_count"] == 4
    assert pred["uncertainty_source"] == "tta_disagreement"
    assert 0.0 <= float(pred["tta_consensus"]) <= 1.0
    assert 0.0 <= float(pred["boundary_uncertainty"]) <= 1.0
    assert 0.0 <= float(pred["geometry_confidence"]) <= 1.0


def test_field_boundary_inferencer_emits_progress_callback(tmp_path):
    torch = pytest.importorskip("torch")
    model = BoundaryUNet(in_channels=len(FEATURE_CHANNELS))

    ckpt_path = tmp_path / "boundary_unet_v2.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_stats": {
                "mean": [0.0] * len(FEATURE_CHANNELS),
                "std": [1.0] * len(FEATURE_CHANNELS),
            },
        },
        ckpt_path,
    )

    inferencer = FieldBoundaryInferencer(str(ckpt_path), device="cpu")
    feature_stack = np.random.rand(len(FEATURE_CHANNELS), 96, 96).astype(np.float32)
    events: list[tuple[str, int, int]] = []

    inferencer.predict(
        feature_stack,
        tile_size=64,
        overlap=16,
        progress_callback=lambda stage, done, total: events.append((stage, done, total)),
    )

    assert events
    assert any(stage == "ml_patches" for stage, _, _ in events)
    assert any(stage == "ml_blend" for stage, _, _ in events)
    assert any(done == total for _, done, total in events)
