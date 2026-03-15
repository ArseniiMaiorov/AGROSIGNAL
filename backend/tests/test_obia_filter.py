"""Tests for OBIA filtering."""
import numpy as np
import pytest

from processing.fields.obia_filter import compute_segment_properties, filter_segments


def _make_labels_and_pheno():
    labels = np.zeros((100, 100), dtype=np.int32)
    # Large square field (30x30 = 900 px => 90000 m2 at 10m)
    labels[10:40, 10:40] = 1
    # Small square (5x5 = 25 px => 2500 m2 at 10m) — below 3000 threshold
    labels[60:65, 60:65] = 2
    # Long narrow strip (3x50 = 150 px => 15000 m2 but bad shape)
    labels[80:83, 10:60] = 3
    # Thin border ring (396 px => 39600 m2) with a very large perimeter — coastline artifact
    labels[0, :] = 4
    labels[-1, :] = 4
    labels[:, 0] = 4
    labels[:, -1] = 4

    ndvi_delta = np.full((100, 100), 0.3)
    ndvi_delta[80:83, 10:60] = 0.05  # low dynamic — should be filtered
    ndwi = np.zeros((100, 100), dtype=float)
    ndwi[60:65, 60:65] = 0.4
    pheno = {"ndvi_delta": ndvi_delta, "ndwi": ndwi}
    return labels, pheno


class TestComputeProperties:
    def test_properties_count(self):
        labels, pheno = _make_labels_and_pheno()
        props = compute_segment_properties(labels, pheno, pixel_size_m=10.0)
        assert len(props) == 4
        assert all("mean_ndwi" in p for p in props)

    def test_area_calculation(self):
        labels, pheno = _make_labels_and_pheno()
        props = compute_segment_properties(labels, pheno, pixel_size_m=10.0)
        areas = {p["label"]: p["area_m2"] for p in props}
        assert areas[1] == pytest.approx(90000, rel=0.01)
        assert areas[2] == pytest.approx(2500, rel=0.01)

    def test_shape_index(self):
        labels, pheno = _make_labels_and_pheno()
        props = compute_segment_properties(labels, pheno, pixel_size_m=10.0)
        si = {p["label"]: p["shape_index"] for p in props}
        # Square should have low SI (~1.1-1.3), narrow strip should have high SI
        assert si[1] < 2.0
        assert si[3] > si[1]

    def test_properties_emit_progress_callback(self):
        labels, pheno = _make_labels_and_pheno()
        events: list[tuple[str, int, int]] = []

        props = compute_segment_properties(
            labels,
            pheno,
            pixel_size_m=10.0,
            progress_callback=lambda stage, done, total: events.append((stage, done, total)),
        )

        assert len(props) == 4
        assert events
        assert events[0][0] == "props"


class TestFilterSegments:
    def test_removes_small_segments(self):
        labels, pheno = _make_labels_and_pheno()
        filtered = filter_segments(labels, pheno, min_area_m2=3000, max_shape_index=999,
                                   min_ndvi_delta=0.0, pixel_size_m=10.0)
        assert (filtered == 2).sum() == 0  # small segment removed
        assert (filtered == 1).sum() > 0   # large remains

    def test_removes_low_ndvi_delta(self):
        labels, pheno = _make_labels_and_pheno()
        filtered = filter_segments(labels, pheno, min_area_m2=0, max_shape_index=999,
                                   min_ndvi_delta=0.15, pixel_size_m=10.0)
        assert (filtered == 3).sum() == 0  # low delta removed
        assert (filtered == 1).sum() > 0   # high delta remains

    def test_preserves_valid_segments(self):
        labels, pheno = _make_labels_and_pheno()
        filtered = filter_segments(labels, pheno, min_area_m2=0, max_shape_index=999,
                                   min_ndvi_delta=0.0, pixel_size_m=10.0)
        assert (filtered == 1).sum() == (labels == 1).sum()

    def test_output_shape(self):
        labels, pheno = _make_labels_and_pheno()
        filtered = filter_segments(labels, pheno)
        assert filtered.shape == labels.shape

    def test_removes_high_ndwi_segments(self):
        labels, pheno = _make_labels_and_pheno()
        filtered = filter_segments(
            labels,
            pheno,
            min_area_m2=0,
            max_shape_index=999,
            min_ndvi_delta=0.0,
            max_mean_ndwi=0.2,
            pixel_size_m=10.0,
        )
        assert (filtered == 2).sum() == 0

    def test_removes_coastline_artifact_segments(self):
        labels, pheno = _make_labels_and_pheno()
        filtered = filter_segments(
            labels,
            pheno,
            min_area_m2=0,
            max_shape_index=999,
            min_ndvi_delta=0.0,
            pixel_size_m=10.0,
        )
        assert (filtered == 4).sum() == 0

    def test_removes_segments_with_high_internal_water_fraction(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[2:18, 2:18] = 1
        pheno = {
            "ndvi_delta": np.full((20, 20), 0.3, dtype=float),
            "ndwi": np.zeros((20, 20), dtype=float),
        }
        lc_fractions = {
            "water_frac": np.zeros((20, 20), dtype=float),
            "tree_frac": np.zeros((20, 20), dtype=float),
            "cropland_frac": np.ones((20, 20), dtype=float),
            "shrubland_frac": np.zeros((20, 20), dtype=float),
            "wetland_frac": np.zeros((20, 20), dtype=float),
        }
        lc_fractions["water_frac"][2:18, 2:18] = 0.25

        filtered = filter_segments(
            labels,
            pheno,
            min_area_m2=0,
            max_shape_index=999,
            min_ndvi_delta=0.0,
            pixel_size_m=10.0,
            lc_fractions=lc_fractions,
            max_internal_water_frac=0.10,
        )

        assert not np.any(filtered == 1)

    def test_relax_labels_can_keep_high_tree_fraction_segment(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[2:18, 2:18] = 1
        pheno = {
            "ndvi_delta": np.full((20, 20), 0.3, dtype=float),
            "ndwi": np.zeros((20, 20), dtype=float),
        }
        lc_fractions = {
            "water_frac": np.zeros((20, 20), dtype=float),
            "tree_frac": np.zeros((20, 20), dtype=float),
            "cropland_frac": np.ones((20, 20), dtype=float),
            "shrubland_frac": np.zeros((20, 20), dtype=float),
            "wetland_frac": np.zeros((20, 20), dtype=float),
        }
        lc_fractions["tree_frac"][2:18, 2:18] = 0.24

        filtered_strict = filter_segments(
            labels,
            pheno,
            min_area_m2=0,
            max_shape_index=999,
            min_ndvi_delta=0.0,
            pixel_size_m=10.0,
            lc_fractions=lc_fractions,
            max_internal_tree_frac=0.20,
        )
        filtered_relaxed = filter_segments(
            labels,
            pheno,
            min_area_m2=0,
            max_shape_index=999,
            min_ndvi_delta=0.0,
            pixel_size_m=10.0,
            lc_fractions=lc_fractions,
            max_internal_tree_frac=0.20,
            relax_labels={1},
            relax_tree_multiplier=1.5,
        )

        assert not np.any(filtered_strict == 1)
        assert np.any(filtered_relaxed == 1)
