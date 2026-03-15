"""Tests for spectral index calculations."""
import numpy as np
import pytest

from processing.fields.indices import (
    compute_all_indices,
    compute_bsi,
    compute_mndwi,
    compute_msi,
    compute_ndmi,
    compute_ndvi,
    compute_ndwi,
)


class TestNDVI:
    def test_basic_ndvi(self):
        b4 = np.array([[0.1, 0.2], [0.05, 0.15]])
        b8 = np.array([[0.5, 0.3], [0.6, 0.4]])
        ndvi = compute_ndvi(b4, b8)
        expected = (b8 - b4) / (b8 + b4)
        np.testing.assert_allclose(ndvi, expected, atol=1e-6)

    def test_ndvi_range(self):
        b4 = np.random.uniform(0.01, 0.3, (100, 100)).astype(np.float32)
        b8 = np.random.uniform(0.01, 0.6, (100, 100)).astype(np.float32)
        ndvi = compute_ndvi(b4, b8)
        valid = np.isfinite(ndvi)
        assert ndvi[valid].min() >= -1.0
        assert ndvi[valid].max() <= 1.0

    def test_ndvi_zero_division(self):
        b4 = np.array([[0.0, 0.1]])
        b8 = np.array([[0.0, 0.2]])
        ndvi = compute_ndvi(b4, b8)
        assert np.isnan(ndvi[0, 0])
        assert np.isfinite(ndvi[0, 1])

    def test_ndvi_3d(self):
        b4 = np.random.uniform(0.01, 0.2, (5, 50, 50)).astype(np.float32)
        b8 = np.random.uniform(0.1, 0.5, (5, 50, 50)).astype(np.float32)
        ndvi = compute_ndvi(b4, b8)
        assert ndvi.shape == (5, 50, 50)


class TestNDWI:
    def test_basic_ndwi(self):
        b3 = np.array([[0.3, 0.1]])
        b8 = np.array([[0.1, 0.5]])
        ndwi = compute_ndwi(b3, b8)
        expected = (b3 - b8) / (b3 + b8)
        np.testing.assert_allclose(ndwi, expected, atol=1e-6)


class TestNDMI:
    def test_basic_ndmi(self):
        b8 = np.array([[0.4, 0.2]])
        b11 = np.array([[0.1, 0.3]])
        ndmi = compute_ndmi(b8, b11)
        expected = (b8 - b11) / (b8 + b11)
        np.testing.assert_allclose(ndmi, expected, atol=1e-6)


class TestMNDWI:
    def test_basic_mndwi(self):
        b3 = np.array([[0.4, 0.2]])
        b11 = np.array([[0.1, 0.3]])
        mndwi = compute_mndwi(b3, b11)
        expected = (b3 - b11) / (b3 + b11)
        np.testing.assert_allclose(mndwi, expected, atol=1e-6)


class TestBSI:
    def test_basic_bsi(self):
        b2 = np.array([[0.1]])
        b4 = np.array([[0.15]])
        b8 = np.array([[0.4]])
        b11 = np.array([[0.2]])
        bsi = compute_bsi(b2, b4, b8, b11)
        num = (0.2 + 0.15) - (0.4 + 0.1)
        den = (0.2 + 0.15) + (0.4 + 0.1)
        np.testing.assert_allclose(bsi[0, 0], num / den, atol=1e-6)


class TestMSI:
    def test_basic_msi(self):
        b8 = np.array([[0.5, 0.0]])
        b11 = np.array([[0.2, 0.1]])
        msi = compute_msi(b8, b11)
        assert abs(msi[0, 0] - 0.4) < 1e-6
        assert np.isnan(msi[0, 1]) or np.isinf(msi[0, 1]) or msi[0, 1] > 100


class TestComputeAll:
    def test_all_indices_keys(self):
        bands = {
            "B2": np.ones((10, 10)) * 0.05,
            "B3": np.ones((10, 10)) * 0.06,
            "B4": np.ones((10, 10)) * 0.08,
            "B8": np.ones((10, 10)) * 0.35,
            "B11": np.ones((10, 10)) * 0.15,
        }
        result = compute_all_indices(bands)
        assert set(result.keys()) == {"NDVI", "NDWI", "MNDWI", "NDMI", "BSI", "MSI"}
        for v in result.values():
            assert v.shape == (10, 10)
