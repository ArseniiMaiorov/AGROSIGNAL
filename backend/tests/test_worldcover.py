"""Tests for ESA WorldCover prior helpers."""
import numpy as np

from processing.priors.worldcover import (
    CROPLAND_CLASS,
    SHRUBLAND_CLASS,
    TREE_CLASS,
    WATER_CLASS,
    WETLAND_CLASS,
    _mask_for_classes,
    load_worldcover_prior,
)


class TestWorldCoverHelpers:
    def test_mask_for_exclusion_classes(self):
        data = np.array([
            [CROPLAND_CLASS, WATER_CLASS],
            [WETLAND_CLASS, SHRUBLAND_CLASS],
        ], dtype=np.uint8)
        mask = _mask_for_classes(data, (WATER_CLASS, WETLAND_CLASS, TREE_CLASS, SHRUBLAND_CLASS))
        expected = np.array([
            [False, True],
            [True, True],
        ])
        np.testing.assert_array_equal(mask, expected)

    def test_mask_for_cropland_class(self):
        data = np.array([[0, CROPLAND_CLASS, WATER_CLASS]], dtype=np.uint8)
        mask = _mask_for_classes(data, (CROPLAND_CLASS,))
        np.testing.assert_array_equal(mask, np.array([[False, True, False]]))

    def test_load_worldcover_prior_uses_forest_and_priority_classes_only(self):
        wc_mask = np.array(
            [
                [TREE_CLASS, WETLAND_CLASS, CROPLAND_CLASS],
                [TREE_CLASS, WETLAND_CLASS, TREE_CLASS],
            ],
            dtype=np.uint8,
        )
        pheno_masks = {
            "is_forest": np.array(
                [
                    [True, False, True],
                    [False, True, True],
                ],
                dtype=bool,
            )
        }

        weak_prior = load_worldcover_prior(wc_mask, pheno_masks, cfg=None)

        expected = np.array(
            [
                [True, False, False],
                [False, True, True],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(weak_prior, expected)
