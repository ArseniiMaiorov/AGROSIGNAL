"""Tests for watershed segmentation."""
import numpy as np
import pytest

from processing.fields.segmentation import (
    _selective_split_refine,
    build_watershed_surface,
    find_markers,
    watershed_segment,
)


class TestWatershedSurface:
    def test_basic_surface(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        edge = np.random.uniform(0, 1, (50, 50)).astype(np.float32)
        surface, distance = build_watershed_surface(mask, edge, lambda_edge=0.5)
        assert surface.shape == (50, 50)
        assert distance.shape == (50, 50)
        assert distance[25, 25] > distance[10, 10]  # center farther from edge

    def test_empty_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        edge = np.zeros((20, 20), dtype=np.float32)
        surface, distance = build_watershed_surface(mask, edge)
        assert (distance == 0).all()


class TestFindMarkers:
    def test_single_blob(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[15:35, 15:35] = True
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)
        markers = find_markers(dist, mask, min_distance=5)
        assert markers.max() >= 1  # at least one marker

    def test_two_blobs(self):
        mask = np.zeros((100, 50), dtype=bool)
        mask[5:25, 10:40] = True
        mask[55:75, 10:40] = True
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)
        markers = find_markers(dist, mask, min_distance=5)
        assert markers.max() >= 2  # at least two markers

    def test_grid_seed_mode(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:56, 8:56] = True
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(mask)
        markers = find_markers(dist, mask, min_distance=4, seed_mode="grid", grid_step=16)
        assert markers.max() >= 4

    def test_custom_seed_mode(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[8:56, 8:56] = True
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(mask)
        markers = find_markers(
            dist,
            mask,
            min_distance=4,
            seed_mode="custom",
            custom_points=[(16, 16), (48, 48)],
        )
        assert markers.max() == 2


class TestWatershedSegment:
    def test_basic_segmentation(self):
        h, w = 100, 100
        candidate = np.zeros((h, w), dtype=bool)
        candidate[10:45, 10:45] = True
        candidate[55:90, 10:45] = True
        edge = np.zeros((h, w), dtype=np.float32)
        edge[45:55, :] = 1.0  # strong edge between two regions
        edge[:, 45:55] = 0.5

        labels = watershed_segment(edge, candidate, lambda_edge=0.5, min_distance=8)
        assert labels.shape == (h, w)
        assert labels[20, 20] > 0  # inside first region
        assert labels[70, 20] > 0  # inside second region
        assert labels[50, 50] == 0  # between regions

    def test_empty_candidate(self):
        edge = np.random.uniform(0, 1, (30, 30)).astype(np.float32)
        candidate = np.zeros((30, 30), dtype=bool)
        labels = watershed_segment(edge, candidate)
        assert (labels == 0).all()

    def test_with_osm_mask(self):
        h, w = 60, 60
        candidate = np.ones((h, w), dtype=bool)
        osm_mask = np.zeros((h, w), dtype=bool)
        osm_mask[25:35, :] = True  # road across the middle
        edge = np.zeros((h, w), dtype=np.float32)

        labels = watershed_segment(edge, candidate, osm_mask=osm_mask, min_distance=8)
        assert (labels[30, 30] == 0)  # masked by OSM

    def test_output_no_negative_labels(self):
        candidate = np.ones((40, 40), dtype=bool)
        edge = np.random.uniform(0, 0.5, (40, 40)).astype(np.float32)
        labels = watershed_segment(edge, candidate, min_distance=10)
        assert labels.min() >= 0

    def test_supports_oriented_watershed_inputs(self):
        candidate = np.zeros((30, 30), dtype=bool)
        candidate[5:25, 5:25] = True
        edge = np.zeros((30, 30), dtype=np.float32)
        edge[14:16, :] = 1.0
        ndvi = np.linspace(0.2, 0.8, 30 * 30, dtype=np.float32).reshape(30, 30)

        class DummyCfg:
            OWT_EDGE_WEIGHT = 0.7
            OWT_NDVI_WEIGHT = 0.3
            OWT_SIGMA_ORIENTATION = 1.5
            OWT_SIGMA_STRENGTH = 0.8

        labels = watershed_segment(
            edge,
            candidate,
            min_distance=6,
            ndvi=ndvi,
            cfg=DummyCfg(),
        )

        assert labels.shape == candidate.shape
        assert labels.max() >= 1

    def test_supports_precomputed_distance(self):
        candidate = np.zeros((40, 40), dtype=bool)
        candidate[6:34, 6:34] = True
        edge = np.zeros((40, 40), dtype=np.float32)
        from scipy.ndimage import distance_transform_edt

        distance = distance_transform_edt(candidate).astype(np.float32)
        labels = watershed_segment(
            edge,
            candidate,
            min_distance=6,
            seed_mode="grid",
            precomputed_distance=distance,
        )
        assert labels.shape == candidate.shape
        assert labels.max() >= 1

    def test_selective_split_skips_weak_internal_cracks(self):
        candidate = np.zeros((48, 48), dtype=bool)
        candidate[6:42, 6:42] = True
        edge = np.zeros((48, 48), dtype=np.float32)
        edge[:, 23:25] = 0.06
        ndvi = np.full((48, 48), 0.62, dtype=np.float32)
        ndvi_std = np.full((48, 48), 0.08, dtype=np.float32)
        boundary_prob = np.zeros((48, 48), dtype=np.float32)
        boundary_prob[:, 23:25] = 0.08

        class DummyCfg:
            SELECTIVE_SPLIT_ENABLED = True
            WATERSHED_ONLY_IF_SPLIT_SCORE = True
            SELECTIVE_SPLIT_SCORE_MIN = 0.62
            SELECTIVE_SPLIT_MIN_SHARED_BOUNDARY_PX = 24
            SELECTIVE_SPLIT_MIN_BOUNDARY_PROB = 0.58
            SELECTIVE_SPLIT_MIN_EDGE_SCORE = 0.22
            SELECTIVE_SPLIT_MIN_FEATURE_DELTA = 0.12
            WATERSHED_ROLLBACK_COMPONENT_RATIO_MAX = 1.8
            WATERSHED_ROLLBACK_MAX_INTERNAL_BOUNDARY_CONF = 0.55

        labels, diagnostics = watershed_segment(
            edge,
            candidate,
            min_distance=6,
            ndvi=ndvi,
            ndvi_std=ndvi_std,
            boundary_prob=boundary_prob,
            cfg=DummyCfg(),
            return_diagnostics=True,
        )

        assert labels.max() == 1
        assert diagnostics["watershed_applied"] is False
        assert diagnostics["watershed_skipped_reason"] in {
            "no_strong_internal_boundary",
            "no_internal_boundaries",
        }

    def test_selective_split_rolls_back_low_confidence_oversegmentation(self):
        candidate = np.zeros((48, 48), dtype=bool)
        candidate[4:44, 4:44] = True
        labels = np.zeros((48, 48), dtype=np.int32)
        label = 1
        for row in range(4, 44, 8):
            for col in range(4, 44, 8):
                labels[row:row + 7, col:col + 7] = label
                label += 1
        for row in range(11, 44, 8):
            labels[row:row + 1, 4:44] = 0
        for col in range(11, 44, 8):
            labels[4:44, col:col + 1] = 0
        edge = np.full((48, 48), 1.0, dtype=np.float32)
        ndvi = np.zeros((48, 48), dtype=np.float32)
        ndvi_std = np.zeros((48, 48), dtype=np.float32)
        label = 1
        for row in range(4, 44, 8):
            for col in range(4, 44, 8):
                region_mask = labels == label
                ndvi[region_mask] = 1.0 if (label % 2 == 0) else 0.0
                ndvi_std[region_mask] = 1.0 if (label % 2 == 0) else 0.0
                label += 1
        boundary_prob = np.full((48, 48), 0.12, dtype=np.float32)

        class DummyCfg:
            SELECTIVE_SPLIT_ENABLED = True
            WATERSHED_ONLY_IF_SPLIT_SCORE = False
            SELECTIVE_SPLIT_SCORE_MIN = 0.62
            SELECTIVE_SPLIT_MIN_SHARED_BOUNDARY_PX = 1
            SELECTIVE_SPLIT_MIN_BOUNDARY_PROB = 0.0
            SELECTIVE_SPLIT_MIN_EDGE_SCORE = 0.0
            SELECTIVE_SPLIT_MIN_FEATURE_DELTA = 0.0
            WATERSHED_ROLLBACK_COMPONENT_RATIO_MAX = 1.2
            WATERSHED_ROLLBACK_MAX_INTERNAL_BOUNDARY_CONF = 0.55

        labels, diagnostics = _selective_split_refine(
            labels,
            candidate,
            edge_score=edge,
            ndvi=ndvi,
            ndvi_std=ndvi_std,
            boundary_prob=boundary_prob,
            cfg=DummyCfg(),
        )

        assert diagnostics["watershed_rollback_reason"] == "oversegmentation_low_boundary_conf"
        assert diagnostics["components_after_watershed"] == 1
        assert labels.max() == 1
