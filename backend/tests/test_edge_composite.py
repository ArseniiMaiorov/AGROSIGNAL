"""Tests for multi-temporal edge composite."""
import numpy as np
import pytest

from processing.fields.edge_composite import (
    build_multitemporal_edge_composite,
    compute_edge_stats,
    compute_canny_edges,
    compute_scharr_edges,
    normalize_band,
)


class TestNormalizeBand:
    def test_basic_normalization(self):
        band = np.array([[0.0, 0.5, 1.0]])
        mask = np.ones_like(band, dtype=bool)
        normed = normalize_band(band, mask)
        np.testing.assert_allclose(normed, [[0.0, 0.5, 1.0]], atol=1e-6)

    def test_with_mask(self):
        band = np.array([[0.2, 0.8, 100.0]])
        mask = np.array([[True, True, False]])
        normed = normalize_band(band, mask)
        assert normed[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert normed[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert normed[0, 2] == 0.0  # masked out

    def test_constant_band(self):
        band = np.full((5, 5), 0.3)
        mask = np.ones_like(band, dtype=bool)
        normed = normalize_band(band, mask)
        assert (normed == 0.0).all()

    def test_all_masked(self):
        band = np.array([[1.0, 2.0]])
        mask = np.zeros_like(band, dtype=bool)
        normed = normalize_band(band, mask)
        assert (normed == 0.0).all()


class TestCannyEdges:
    def test_output_binary(self):
        img = np.random.uniform(0, 1, (50, 50)).astype(np.float32)
        edges = compute_canny_edges(img)
        assert edges.shape == (50, 50)
        unique = set(np.unique(edges))
        assert unique.issubset({0.0, 1.0})

    def test_sharp_edge(self):
        img = np.zeros((50, 50), dtype=np.float32)
        img[:, 25:] = 1.0
        edges = compute_canny_edges(img, sigma=1.0)
        assert edges.sum() > 0  # should detect the edge


class TestScharrEdges:
    def test_output_range(self):
        img = np.random.uniform(0, 1, (50, 50)).astype(np.float32)
        grad = compute_scharr_edges(img)
        assert grad.shape == (50, 50)
        assert grad.min() >= 0.0
        assert grad.max() <= 1.0 + 1e-6

    def test_uniform_image(self):
        img = np.full((30, 30), 0.5, dtype=np.float32)
        grad = compute_scharr_edges(img)
        assert grad.max() < 1e-6  # no edges in uniform image


class TestEdgeComposite:
    def test_basic_composite(self):
        t, h, w = 3, 50, 50
        bands = {
            "B2": np.random.uniform(0, 1, (t, h, w)).astype(np.float32),
            "B3": np.random.uniform(0, 1, (t, h, w)).astype(np.float32),
            "B4": np.random.uniform(0, 1, (t, h, w)).astype(np.float32),
            "B8": np.random.uniform(0, 1, (t, h, w)).astype(np.float32),
        }
        valid = np.ones((t, h, w), dtype=bool)
        composite = build_multitemporal_edge_composite(bands, valid, alpha=0.7)
        assert composite.shape == (h, w)
        assert composite.dtype == np.float32
        assert composite.min() >= 0.0
        assert composite.max() <= 1.0 + 1e-6

    def test_all_masked(self):
        t, h, w = 2, 20, 20
        bands = {"B2": np.random.uniform(0, 1, (t, h, w)).astype(np.float32)}
        valid = np.zeros((t, h, w), dtype=bool)
        composite = build_multitemporal_edge_composite(bands, valid)
        assert (composite == 0.0).all()

    def test_single_date(self):
        t, h, w = 1, 30, 30
        bands = {
            "B4": np.random.uniform(0, 1, (t, h, w)).astype(np.float32),
            "B8": np.random.uniform(0, 1, (t, h, w)).astype(np.float32),
        }
        valid = np.ones((t, h, w), dtype=bool)
        composite = build_multitemporal_edge_composite(bands, valid)
        assert composite.shape == (h, w)

    def test_composite_preserves_soft_gradient(self):
        rng = np.random.default_rng(42)
        t, h, w = 3, 64, 64
        bands = {
            "B2": rng.uniform(0.0, 1.0, (t, h, w)).astype(np.float32),
            "B3": rng.uniform(0.0, 1.0, (t, h, w)).astype(np.float32),
            "B4": rng.uniform(0.0, 1.0, (t, h, w)).astype(np.float32),
            "B8": rng.uniform(0.0, 1.0, (t, h, w)).astype(np.float32),
        }
        valid = np.ones((t, h, w), dtype=bool)
        composite_lo = build_multitemporal_edge_composite(
            bands,
            valid,
            binary_threshold=0.01,
            closing_radius=1,
            soft_clip_percentile=95.0,
        )
        composite_hi = build_multitemporal_edge_composite(
            bands,
            valid,
            binary_threshold=0.90,
            closing_radius=6,
            soft_clip_percentile=95.0,
        )
        np.testing.assert_allclose(composite_lo, composite_hi, atol=1e-6)

    def test_edge_stats(self):
        edge = np.array([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
        stats = compute_edge_stats(edge, bins=2)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(1.0)
        assert stats["histogram"] == [1, 3]

    def test_edge_stats_ignores_nan_values(self):
        edge = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
        stats = compute_edge_stats(edge, bins=2)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(0.0)
        assert stats["mean"] == pytest.approx(0.0)
        assert stats["histogram"] == [0, 0]

    def test_skips_low_coverage_dates(self):
        t, h, w = 2, 40, 40
        bands = {
            "B2": np.zeros((t, h, w), dtype=np.float32),
            "ndvi": np.zeros((t, h, w), dtype=np.float32),
        }
        bands["B2"][0, :, 20:] = 1.0
        bands["B2"][1, :5, :5] = 1.0
        bands["ndvi"][0, 20:, :] = 1.0
        bands["ndvi"][1, :5, :5] = 1.0

        valid = np.zeros((t, h, w), dtype=bool)
        valid[0] = True
        valid[1, :5, :5] = True

        composite_all = build_multitemporal_edge_composite(
            bands,
            valid,
            coverage_threshold=0.0,
        )
        composite_filtered = build_multitemporal_edge_composite(
            bands,
            valid,
            coverage_threshold=0.30,
        )
        assert composite_filtered.shape == composite_all.shape
        assert np.count_nonzero(composite_filtered) <= np.count_nonzero(composite_all)
