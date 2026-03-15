from __future__ import annotations

import numpy as np
import rasterio

from processing.fields.ml_inference import get_feature_channels
from training.gen_data import (
    PatchDataset,
    TileGroupedBatchSampler,
    _build_feature_stack,
    _collect_samples,
    _compute_norm_stats,
    _resolve_norm_stat_samples,
)


def _write_tile_npz(path, shape=(32, 32)):
    h, w = shape
    edge = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    np.savez(
        path,
        edgecomposite=edge,
        maxndvi=np.clip(0.4 + 0.4 * xx, 0.0, 1.0).astype(np.float32),
        meanndvi=np.clip(0.2 + 0.5 * yy, 0.0, 1.0).astype(np.float32),
        ndvistd=np.clip(0.05 + 0.1 * (xx * yy), 0.0, 1.0).astype(np.float32),
        ndwi_mean=(xx - yy).astype(np.float32),
        bsi_mean=(yy - 0.5 * xx).astype(np.float32),
        scl_valid_fraction=np.clip(0.6 + 0.4 * yy, 0.0, 1.0).astype(np.float32),
        nir_median=np.clip(0.3 + 0.5 * xx, 0.0, 1.0).astype(np.float32),
        red_median=np.clip(0.2 + 0.4 * yy, 0.0, 1.0).astype(np.float32),
        blue_median=np.clip(0.1 + 0.2 * (1.0 - xx), 0.0, 1.0).astype(np.float32),
        ndvi_entropy=np.clip(0.1 + 0.5 * (xx + yy) * 0.5, 0.0, 5.0).astype(np.float32),
        mndwi_max=(0.5 * xx - yy).astype(np.float32),
        ndmi_mean=(yy - xx).astype(np.float32),
        ndwi_median=(0.3 * xx - 0.2 * yy).astype(np.float32),
        green_median=np.clip(0.2 + 0.3 * xx, 0.0, 1.0).astype(np.float32),
        swir_median=np.clip(0.1 + 0.3 * yy, 0.0, 1.0).astype(np.float32),
    )


def _write_label_tif(path, shape=(32, 32)):
    h, w = shape
    label = np.zeros((h, w), dtype=np.uint8)
    label[8:24, 8:24] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(label, 1)


def test_training_uses_lazy_patch_loading(tmp_path):
    tiles_dir = tmp_path / "tiles"
    labels_dir = tmp_path / "labels"
    tiles_dir.mkdir()
    labels_dir.mkdir()

    _write_tile_npz(tiles_dir / "demo_01.npz")
    _write_label_tif(labels_dir / "demo_01_label.tif")

    feature_channels = get_feature_channels("v2_16ch")
    samples, tile_quality = _collect_samples(
        tiles_dir,
        labels_dir,
        patch_size=16,
        stride=16,
        feature_channels=feature_channels,
    )

    assert len(samples) == 4
    assert samples[0].tile_id == "demo_01"
    assert not hasattr(samples[0], "x")
    assert tile_quality == {}

    norm = _compute_norm_stats(
        samples,
        feature_channels,
        patch_size=16,
        tile_cache_size=1,
    )
    ds = PatchDataset(
        samples,
        norm,
        train=False,
        feature_channels=feature_channels,
        patch_size=16,
        tile_cache_size=1,
    )
    x, target, weight, edge_valid = ds[0]
    assert tuple(x.shape) == (16, 16, 16)
    assert tuple(target["extent"].shape) == (1, 16, 16)
    assert float(weight.item()) == 1.0
    assert 0.0 <= float(edge_valid.item()) <= 1.0


def test_tile_grouped_batch_sampler_groups_tile_indices():
    samples = [
        type("S", (), {"tile_id": "a"})(),
        type("S", (), {"tile_id": "a"})(),
        type("S", (), {"tile_id": "b"})(),
        type("S", (), {"tile_id": "b"})(),
    ]
    sampler = TileGroupedBatchSampler(samples, batch_size=2, seed=7)
    batches = list(iter(sampler))
    assert len(batches) == 2
    for batch in batches:
        tile_ids = {samples[idx].tile_id for idx in batch}
        assert len(tile_ids) == 1


def test_build_feature_stack_uses_visible_band_green_fallback(tmp_path):
    npz_path = tmp_path / "tile.npz"
    red = np.full((4, 4), 0.6, dtype=np.float32)
    blue = np.full((4, 4), 0.2, dtype=np.float32)
    np.savez(
        npz_path,
        edgecomposite=np.zeros((4, 4), dtype=np.float32),
        maxndvi=np.zeros((4, 4), dtype=np.float32),
        meanndvi=np.ones((4, 4), dtype=np.float32),
        ndvistd=np.zeros((4, 4), dtype=np.float32),
        red_median=red,
        blue_median=blue,
    )

    stack, _mask = _build_feature_stack(npz_path, feature_channels=("green_median",))

    assert stack.shape == (1, 4, 4)
    assert np.allclose(stack[0], 0.4)


def test_resolve_norm_stat_samples_prefers_raw_train_split():
    raw_samples = [type("S", (), {"tile_id": "raw"})()]
    rebalanced_samples = [type("S", (), {"tile_id": "balanced"})()]

    resolved = _resolve_norm_stat_samples(raw_samples, rebalanced_samples)

    assert resolved is raw_samples
