"""Tests for v3 SAM-primary fallback segmentation."""
from __future__ import annotations

import numpy as np

from processing.fields.sam_primary import (
    SAM2FieldSegmentor,
    build_sam_primary_image,
    masks_to_label_raster,
)


class DummyCfg:
    PHENO_FIELD_MAX_NDVI_MIN = 0.45
    PHENO_FIELD_NDVI_STD_MIN = 0.15
    SAM_POINT_SPACING = 20
    SAM_PRED_IOU_THRESHOLD = 0.85


def test_build_sam_primary_image_returns_uint8_rgb():
    edge_prob = np.full((8, 8), 0.2, dtype=np.float32)
    maxndvi = np.full((8, 8), 0.6, dtype=np.float32)
    ndvistd = np.full((8, 8), 0.2, dtype=np.float32)

    image = build_sam_primary_image(edge_prob, maxndvi, ndvistd)

    assert image.shape == (8, 8, 3)
    assert image.dtype == np.uint8


def test_sam2_segmentor_fallback_uses_candidate_mask_when_model_unavailable():
    edge_prob = np.zeros((12, 12), dtype=np.float32)
    maxndvi = np.full((12, 12), 0.6, dtype=np.float32)
    ndvistd = np.full((12, 12), 0.2, dtype=np.float32)
    candidate_mask = np.zeros((12, 12), dtype=bool)
    candidate_mask[2:8, 3:9] = True

    segmentor = SAM2FieldSegmentor(checkpoint_path="/missing/sam2.pt", device="cpu")
    masks, scores = segmentor.segment_fields(
        edge_prob,
        maxndvi,
        ndvistd,
        transform=None,
        cfg=DummyCfg(),
        candidate_mask=candidate_mask,
    )

    assert len(masks) == 1
    assert scores.shape == (1,)
    assert masks[0].dtype == bool
    assert np.array_equal(masks[0], candidate_mask)


def test_masks_to_label_raster_assigns_dense_labels():
    mask_a = np.zeros((6, 6), dtype=bool)
    mask_b = np.zeros((6, 6), dtype=bool)
    mask_a[1:3, 1:3] = True
    mask_b[3:5, 3:5] = True

    labels = masks_to_label_raster([mask_a, mask_b], shape=(6, 6))

    assert labels.dtype == np.int32
    assert labels[1, 1] == 1
    assert labels[4, 4] == 2
    assert labels.max() == 2
