"""Tests for the SAM workflow wiring."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from processing.fields.sam_field_boundary import (
    build_sam_composite,
    build_sam_input_composite,
    run_sam_sequential,
)


class DummyCfg:
    SAM_MODEL_TYPE = "vit_b"
    SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"
    SAM_POINTS_PER_SIDE = 16


def test_build_sam_composite_alias_matches_input_builder():
    max_ndvi = np.full((8, 8), 0.6, dtype=np.float32)
    edge = np.full((8, 8), 0.2, dtype=np.float32)
    mean_ndvi = np.full((8, 8), 0.4, dtype=np.float32)

    direct = build_sam_input_composite(max_ndvi, edge, mean_ndvi)
    alias = build_sam_composite(max_ndvi, edge, mean_ndvi)

    assert alias.shape == (8, 8, 3)
    assert np.array_equal(alias, direct)


def test_run_sam_sequential_delegates_to_segmentation(monkeypatch, tmp_path):
    composite = np.zeros((8, 8, 3), dtype=np.uint8)
    expected = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:32634",
    )

    def _fake_run(*args, **kwargs):
        return expected

    monkeypatch.setattr(
        "processing.fields.sam_field_boundary.run_sam_segmentation",
        _fake_run,
    )

    result = run_sam_sequential(
        composite_uint8=composite,
        transform=None,
        crs_epsg=32634,
        cfg=DummyCfg(),
        output_dir=Path(tmp_path),
    )

    assert len(result) == 1
    assert result.crs.to_epsg() == 32634
