from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from training.run_holdout_ab import _boundary_metrics, _validate_holdout_items


def test_validate_holdout_items_accepts_valid_config(tmp_path):
    gt = tmp_path / "gt.geojson"
    gt.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    items = [
        {
            "id": "holdout_ok",
            "request": {"aoi": {"type": "point_radius", "lat": 55.0, "lon": 37.0, "radius_km": 2}},
            "ground_truth_geojson": str(gt),
        }
    ]
    _validate_holdout_items(tmp_path / "holdout.json", items)


def test_validate_holdout_items_accepts_aoi_only_manifest(tmp_path):
    gt = tmp_path / "gt.geojson"
    gt.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    items = [
        {
            "id": "holdout_aoi_only",
            "aoi": {"type": "point_radius", "lat": 55.0, "lon": 37.0, "radius_km": 2},
            "ground_truth_geojson": str(gt),
        }
    ]
    _validate_holdout_items(tmp_path / "holdout.json", items)


def test_validate_holdout_items_rejects_placeholder_path(tmp_path):
    items = [
        {
            "id": "bad_placeholder",
            "request": {"aoi": {"type": "point_radius", "lat": 55.0, "lon": 37.0, "radius_km": 2}},
            "ground_truth_geojson": "/absolute/path/to/holdout_01_gt.geojson",
        }
    ]
    with pytest.raises(ValueError, match="placeholder"):
        _validate_holdout_items(tmp_path / "holdout.json", items)


def test_validate_holdout_items_rejects_missing_gt(tmp_path):
    missing_gt = tmp_path / "missing.geojson"
    items = [
        {
            "id": "bad_missing",
            "request": {"aoi": {"type": "point_radius", "lat": 55.0, "lon": 37.0, "radius_km": 2}},
            "ground_truth_geojson": str(missing_gt),
        }
    ]
    with pytest.raises(ValueError, match="Missing ground truth files"):
        _validate_holdout_items(tmp_path / "holdout.json", items)


def test_boundary_metrics_for_identical_polygons():
    pred = gpd.GeoDataFrame(geometry=[box(29.0, 58.0, 29.01, 58.01)], crs="EPSG:4326")
    gt = gpd.GeoDataFrame(geometry=[box(29.0, 58.0, 29.01, 58.01)], crs="EPSG:4326")

    metrics = _boundary_metrics(pred, gt)

    assert metrics["boundary_iou"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["hausdorff_95_px"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["centroid_shift_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["area_ratio_pred_gt"] == pytest.approx(1.0, rel=1e-3)
