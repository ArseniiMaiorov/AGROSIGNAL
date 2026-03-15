"""Tests for composite and phenometric computations."""
import numpy as np
import pytest

from processing.fields.composite import (
    build_median_composite,
    build_valid_mask_from_scl,
    compute_phenometrics,
    select_dates_adaptive,
    select_dates_by_coverage,
)
from core.config import Settings


class TestValidMask:
    def test_valid_classes(self):
        scl = np.array([[[4, 5, 6, 3, 8, 0, 1, 10, 11, 2]]])
        mask = build_valid_mask_from_scl(scl)
        expected = np.array([[[True, True, True, False, False, False, False, False, False, False]]])
        np.testing.assert_array_equal(mask, expected)

    def test_shape_preserved(self):
        scl = np.random.randint(0, 12, (7, 100, 100), dtype=np.uint8)
        mask = build_valid_mask_from_scl(scl)
        assert mask.shape == scl.shape
        assert mask.dtype == bool

    def test_all_valid(self):
        scl = np.full((3, 5, 5), 4, dtype=np.uint8)
        mask = build_valid_mask_from_scl(scl)
        assert mask.all()

    def test_all_invalid(self):
        scl = np.full((3, 5, 5), 8, dtype=np.uint8)
        mask = build_valid_mask_from_scl(scl)
        assert not mask.any()


class TestSelectDates:
    def test_selects_good_dates(self):
        valid = np.zeros((10, 50, 50), dtype=bool)
        valid[0] = True
        valid[2] = True
        valid[5] = True
        valid[7] = True
        valid[9] = True
        selected = select_dates_by_coverage(valid, min_valid_pct=0.5, n_dates=3)
        assert len(selected) <= 5
        assert all(s in [0, 2, 5, 7, 9] for s in selected)

    def test_returns_all_if_few(self):
        valid = np.ones((3, 10, 10), dtype=bool)
        selected = select_dates_by_coverage(valid, n_dates=7)
        assert len(selected) == 3

    def test_handles_low_coverage(self):
        valid = np.zeros((5, 100, 100), dtype=bool)
        valid[0, :10, :10] = True  # 1% coverage
        selected = select_dates_by_coverage(valid, min_valid_pct=0.5, n_dates=3)
        assert len(selected) > 0

    def test_returns_metadata_for_low_quality_input(self):
        valid = np.zeros((4, 20, 20), dtype=bool)
        valid[0] = True
        selected, meta = select_dates_by_coverage(
            valid,
            min_valid_pct=0.5,
            n_dates=3,
            min_good_dates=3,
            return_metadata=True,
        )
        assert len(selected) > 0
        assert meta["low_quality_input"] is True
        assert meta["good_date_count"] == 1

    def test_select_dates_adaptive_returns_region_scores(self):
        valid = np.zeros((5, 10, 10), dtype=bool)
        valid[0] = True
        valid[2] = True
        valid[4] = True
        ndvi = np.zeros((5, 10, 10), dtype=np.float32)
        ndvi[0] = 0.2
        ndvi[2] = 0.6
        ndvi[4] = 0.4
        ndwi = np.zeros_like(ndvi)
        mndwi = np.zeros_like(ndvi)
        selected, meta = select_dates_adaptive(
            valid,
            {"NDVI": ndvi, "NDWI": ndwi, "MNDWI": mndwi},
            [("2025-05-01T00:00:00Z", "2025-05-07T23:59:59Z")] * 5,
            58.7,
            3,
            2,
            Settings(),
            return_metadata=True,
        )
        assert len(selected) >= 2
        assert meta["region_band"] == "north"
        assert len(meta["score_total"]) == 5
        assert len(meta["score_components"]) == 5
        assert meta["selected_date_confidence_mean"] >= 0.0


class TestPhenometrics:
    def test_basic_phenometrics(self):
        ndvi = np.array([
            [[0.2, 0.3], [0.1, 0.4]],
            [[0.6, 0.7], [0.5, 0.8]],
            [[0.4, 0.5], [0.3, 0.6]],
        ])
        valid = np.ones_like(ndvi, dtype=bool)
        pheno = compute_phenometrics(ndvi, valid)

        assert set(pheno.keys()) == {"ndvi_min", "ndvi_max", "ndvi_mean", "ndvi_std", "ndvi_delta"}
        np.testing.assert_allclose(pheno["ndvi_min"], [[0.2, 0.3], [0.1, 0.4]], atol=1e-6)
        np.testing.assert_allclose(pheno["ndvi_max"], [[0.6, 0.7], [0.5, 0.8]], atol=1e-6)
        np.testing.assert_allclose(pheno["ndvi_delta"], [[0.4, 0.4], [0.4, 0.4]], atol=1e-6)

    def test_with_masked_values(self):
        ndvi = np.array([
            [[0.2, 0.3]],
            [[0.6, 0.7]],
        ])
        valid = np.array([
            [[True, False]],
            [[True, True]],
        ])
        pheno = compute_phenometrics(ndvi, valid)
        assert pheno["ndvi_min"][0, 0] == pytest.approx(0.2)
        assert pheno["ndvi_max"][0, 1] == pytest.approx(0.7)

    def test_empty_mask_returns_correct_shape(self):
        ndvi = np.zeros((3, 4, 5), dtype=np.float32)
        valid = np.zeros_like(ndvi, dtype=bool)
        pheno = compute_phenometrics(ndvi, valid)
        assert pheno["ndvi_mean"].shape == (4, 5)


class TestMedianComposite:
    def test_basic_median(self):
        stack = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
        valid = np.ones_like(stack, dtype=bool)
        median = build_median_composite(stack, valid)
        np.testing.assert_allclose(median, [[3.0, 4.0]], atol=1e-6)

    def test_with_masked(self):
        stack = np.array([[[1.0]], [[100.0]], [[3.0]]])
        valid = np.array([[[True]], [[False]], [[True]]])
        median = build_median_composite(stack, valid)
        np.testing.assert_allclose(median, [[2.0]], atol=1e-6)

    def test_empty_mask_returns_correct_shape(self):
        stack = np.zeros((2, 3, 4), dtype=np.float32)
        valid = np.zeros_like(stack, dtype=bool)
        median = build_median_composite(stack, valid)
        assert median.shape == (3, 4)
