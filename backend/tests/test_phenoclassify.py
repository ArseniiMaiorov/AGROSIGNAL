"""Tests for phenological classification."""
import numpy as np
import pytest

from processing.fields.phenoclassify import (
    BUILTUP,
    CROP,
    FOREST,
    GRASS,
    OTHER,
    WATER,
    PhenoThresholds,
    classify_land_cover,
    compute_hydro_masks,
)


def _make_pheno(h=10, w=10, **overrides):
    defaults = {
        "ndvi_min": np.full((h, w), 0.1),
        "ndvi_max": np.full((h, w), 0.5),
        "ndvi_mean": np.full((h, w), 0.3),
        "ndvi_std": np.full((h, w), 0.1),
        "ndvi_delta": np.full((h, w), 0.4),
    }
    defaults.update(overrides)
    return defaults


class TestClassifyLandCover:
    def test_water_detection(self):
        pheno = _make_pheno()
        ndwi_max = np.full((10, 10), 0.5)  # above threshold
        bsi_med = np.zeros((10, 10))
        msi_med = np.zeros((10, 10))
        valid_count = np.full((10, 10), 5)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count)
        assert (classes == WATER).all()

    def test_builtup_detection(self):
        pheno = _make_pheno(ndvi_std=np.full((10, 10), 0.01))
        ndwi_max = np.full((10, 10), 0.0)
        bsi_med = np.full((10, 10), 0.3)  # above threshold
        msi_med = np.zeros((10, 10))
        valid_count = np.full((10, 10), 5)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count)
        assert (classes == BUILTUP).all()

    def test_forest_detection(self):
        pheno = _make_pheno(
            ndvi_min=np.full((10, 10), 0.5),
            ndvi_delta=np.full((10, 10), 0.1),
        )
        ndwi_max = np.full((10, 10), 0.0)
        bsi_med = np.full((10, 10), 0.0)
        msi_med = np.zeros((10, 10))
        valid_count = np.full((10, 10), 5)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count)
        assert (classes == FOREST).all()

    def test_crop_detection(self):
        pheno = _make_pheno(
            ndvi_min=np.full((10, 10), 0.1),
            ndvi_max=np.full((10, 10), 0.7),
            ndvi_mean=np.full((10, 10), 0.4),
            ndvi_std=np.full((10, 10), 0.15),
            ndvi_delta=np.full((10, 10), 0.6),
        )
        ndwi_max = np.full((10, 10), 0.0)
        bsi_med = np.full((10, 10), 0.0)
        msi_med = np.full((10, 10), 0.5)
        valid_count = np.full((10, 10), 5)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count)
        assert (classes == CROP).all()

    def test_custom_thresholds(self):
        thresholds = PhenoThresholds(ndwi_water=0.1)
        pheno = _make_pheno()
        ndwi_max = np.full((10, 10), 0.15)
        bsi_med = np.zeros((10, 10))
        msi_med = np.zeros((10, 10))
        valid_count = np.full((10, 10), 5)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count, thresholds)
        assert (classes == WATER).all()

    def test_hard_scl_water_mask_takes_priority(self):
        pheno = _make_pheno()
        ndwi_max = np.zeros((10, 10))
        mndwi_max = np.zeros((10, 10))
        bsi_med = np.zeros((10, 10))
        msi_med = np.zeros((10, 10))
        valid_count = np.full((10, 10), 5)
        scl_water_mask = np.ones((10, 10), dtype=bool)
        classes = classify_land_cover(
            pheno,
            ndwi_max,
            bsi_med,
            msi_med,
            valid_count,
            mndwi_max=mndwi_max,
            scl_water_mask=scl_water_mask,
        )
        assert (classes == WATER).all()

    def test_low_valid_observation_count_becomes_other(self):
        pheno = _make_pheno(
            ndvi_min=np.full((10, 10), 0.1),
            ndvi_max=np.full((10, 10), 0.8),
            ndvi_delta=np.full((10, 10), 0.7),
        )
        ndwi_max = np.zeros((10, 10))
        bsi_med = np.zeros((10, 10))
        msi_med = np.zeros((10, 10))
        valid_count = np.full((10, 10), 2)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count)
        assert (classes == OTHER).all()

    def test_output_shape(self):
        pheno = _make_pheno(h=50, w=80)
        ndwi_max = np.zeros((50, 80))
        bsi_med = np.zeros((50, 80))
        msi_med = np.zeros((50, 80))
        valid_count = np.full((50, 80), 5)
        classes = classify_land_cover(pheno, ndwi_max, bsi_med, msi_med, valid_count)
        assert classes.shape == (50, 80)
        assert classes.dtype == np.uint8

    def test_compute_hydro_masks_splits_open_water_and_seasonal_wet(self):
        thresholds = PhenoThresholds()
        ndwi_max = np.zeros((8, 8), dtype=np.float32)
        ndwi_max[2:4, 2:4] = 0.2
        ndwi_mean = np.zeros((8, 8), dtype=np.float32)
        ndwi_mean[4:6, 4:6] = 0.08
        mndwi_max = np.zeros((8, 8), dtype=np.float32)
        valid_count = np.full((8, 8), 5, dtype=np.int32)

        masks = compute_hydro_masks(
            ndwi_max,
            valid_count,
            thresholds,
            mndwi_max=mndwi_max,
            ndwi_mean=ndwi_mean,
            hydro_profile="water_aware",
            open_water_ndwi=0.14,
            seasonal_wet_ndwi=0.06,
            riparian_buffer_px=1,
        )

        assert masks["open_water_mask"][2:4, 2:4].all()
        assert masks["seasonal_wet_mask"][4:6, 4:6].all()
        assert not masks["open_water_mask"][4:6, 4:6].any()
        assert masks["riparian_soft_mask"].any()
