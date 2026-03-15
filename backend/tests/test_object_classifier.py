"""Tests for object-classifier compatibility helpers."""
import pickle

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_bounds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from shapely.geometry import MultiPolygon, box

from processing.fields.object_classifier import (
    LEGACY_FEATURE_COLUMNS,
    ObjectClassifier,
    compute_object_features,
)
from utils.classifier_schema import make_classifier_payload_portable, validate_classifier_file


class DummyPipeline:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1], dtype=np.int32)
        self.last_X = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.last_X = X
        return np.column_stack(
            [
                np.full(len(X), 0.2, dtype=np.float64),
                np.full(len(X), 0.8, dtype=np.float64),
            ]
        )


def test_loads_legacy_payload_and_projects_geometry(tmp_path):
    model_path = tmp_path / "legacy_object_classifier.pkl"
    payload = {
        "pipeline": DummyPipeline(),
        "feature_names": LEGACY_FEATURE_COLUMNS,
        "threshold": 0.4,
    }
    with model_path.open("wb") as f:
        pickle.dump(payload, f)

    clf = ObjectClassifier.load(model_path)
    gdf = gpd.GeoDataFrame(
        {
            "ndvi_mean": [0.42],
            "ndvi_std": [0.11],
            "ndvi_max": [0.67],
            "edge_mean": [0.19],
            "edge_max": [0.45],
            "ndwi_mean": [0.08],
        },
        geometry=[MultiPolygon([box(29.0, 58.0, 29.01, 58.01)])],
        crs="EPSG:4326",
    )

    scores = clf.predict_proba(gdf)
    loaded_pipeline = clf._pipeline

    assert scores.tolist() == pytest.approx([0.8])
    assert loaded_pipeline.last_X.shape == (1, len(LEGACY_FEATURE_COLUMNS))
    assert loaded_pipeline.last_X[0, 0] > 1000.0
    ndvi_idx = LEGACY_FEATURE_COLUMNS.index("ndvi_mean")
    assert loaded_pipeline.last_X[0, ndvi_idx] == pytest.approx(0.42)


def test_compute_object_features_supports_legacy_raster_columns():
    gdf = gpd.GeoDataFrame(
        geometry=[box(0.0, 0.0, 20.0, 20.0)],
        crs="EPSG:3857",
    )
    raster = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float32,
    )
    transform = from_bounds(0.0, 0.0, 20.0, 20.0, 2, 2)

    enriched = compute_object_features(
        gdf,
        raster_data={
            "ndvi_std": raster,
            "ndvistd_mean": raster,
            "edge_mean": raster,
            "edge_max": raster,
        },
        transform=transform,
    )

    assert enriched.at[0, "ndvi_std"] == pytest.approx(0.25)
    assert enriched.at[0, "ndvistd_mean"] == pytest.approx(0.25)
    assert enriched.at[0, "edge_mean"] == pytest.approx(0.25)
    assert enriched.at[0, "edge_max"] == pytest.approx(0.4)


def test_make_classifier_payload_portable_strips_hist_gradient_rng(tmp_path):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("hgb", HistGradientBoostingClassifier(random_state=42, max_iter=8)),
        ]
    )
    pipeline.fit(X, y)

    hgb = pipeline.named_steps["hgb"]
    assert getattr(hgb, "_feature_subsample_rng", None) is not None

    payload = {
        "pipeline": pipeline,
        "feature_columns": ["f1", "f2"],
    }
    make_classifier_payload_portable(payload)

    assert getattr(hgb, "_feature_subsample_rng", None) is None

    model_path = tmp_path / "portable_object_classifier.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)

    meta = validate_classifier_file(model_path, ["f1", "f2"])
    assert meta["feature_count"] == 2
