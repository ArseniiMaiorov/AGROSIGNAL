"""ML-based object classifier for filtering detected field polygons."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np

from core.logging import get_logger
from utils.classifier_schema import make_classifier_payload_portable, validate_classifier_payload
from utils.geometry import compactness, elongation, legacy_shape_index, shape_index
from utils.pickle_compat import load_pickle_compat

logger = get_logger(__name__)
ProgressCallback = Callable[[str, int, int], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int,
) -> None:
    if progress_callback is None:
        return
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    progress_callback(str(stage), safe_completed, safe_total)


@dataclass
class ObjectClassifierConfig:
    model_type: str = "hist_gradient_boosting"
    max_iter: int = 300
    max_depth: int | None = 8
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    random_state: int = 42


FEATURE_COLUMNS = [
    "area_m2",
    "perimeter_m",
    "shape_index",
    "compactness",
    "elongation",
    "ndvi_mean",
    "ndvi_max",
    "ndvi_delta",
    "ndwi_mean",
    "msi_mean",
    "bsi_mean",
    "ndvi_variance",
    "worldcover_crop_pct",
    "growth_amplitude",
    "has_growth_peak",
    "ndvi_entropy",
    "neighbor_field_pct",
    "distance_to_road_m",
    "scl_valid_fraction_mean",
]

LEGACY_FEATURE_COLUMNS = [
    "area_px",
    "perimeter_px",
    "shape_index",
    "extent",
    "solidity",
    "ndvi_mean",
    "ndvi_std",
    "ndvi_max",
    "ndvistd_mean",
    "edge_mean",
    "edge_max",
    "ndwi_mean",
]

_EXTRA_MEAN_RASTER_COLUMNS = {
    "ndvi_std",
    "ndvistd_mean",
    "edge_mean",
    "growth_amplitude",
    "has_growth_peak",
    "ndvi_entropy",
    "distance_to_road_m",
    "scl_valid_fraction_mean",
}
_EXTRA_MAX_RASTER_COLUMNS = {
    "edge_max",
}
_SUPPORTED_RASTER_COLUMNS = (
    set(FEATURE_COLUMNS)
    | _EXTRA_MEAN_RASTER_COLUMNS
    | _EXTRA_MAX_RASTER_COLUMNS
)

_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "ndvi_mean": ("ndvi_mean", "ndvimean", "mean_ndvi"),
    "ndvi_std": ("ndvi_std", "ndvistd"),
    "ndvi_max": ("ndvi_max", "ndvimax", "max_ndvi"),
    "ndvistd_mean": ("ndvistd_mean", "ndvistdmean", "ndvi_std", "ndvistd"),
    "edge_mean": ("edge_mean", "edgemean"),
    "edge_max": ("edge_max", "edgemax"),
    "ndwi_mean": ("ndwi_mean", "ndwimean"),
    "ndvi_delta": ("ndvi_delta", "ndvidelta"),
    "msi_mean": ("msi_mean", "msimean"),
    "bsi_mean": ("bsi_mean", "bsimean"),
    "ndvi_variance": ("ndvi_variance",),
    "worldcover_crop_pct": ("worldcover_crop_pct", "worldcover_croppct"),
    "growth_amplitude": ("growth_amplitude", "growthamplitude", "ndvi_delta"),
    "has_growth_peak": ("has_growth_peak", "hasgrowthpeak"),
    "ndvi_entropy": ("ndvi_entropy", "ndvientropy", "ndvi_temporal_entropy"),
    "neighbor_field_pct": ("neighbor_field_pct", "neighborfieldpct"),
    "distance_to_road_m": ("distance_to_road_m", "distancetoroadm"),
    "scl_valid_fraction_mean": (
        "scl_valid_fraction_mean",
        "sclvalidfractionmean",
        "scl_valid_fraction",
    ),
}
def _to_projected_for_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is not None and getattr(gdf.crs, "is_geographic", False):
        try:
            return gdf.to_crs("EPSG:3857")
        except Exception as exc:
            logger.warning("object_classifier_projection_failed", error=str(exc), exc_info=True)
    return gdf


def _column_values(df: gpd.GeoDataFrame, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in df.columns:
            return np.nan_to_num(df[name].to_numpy(dtype=np.float64, copy=False), nan=0.0)
    return None


def compute_object_features(
    gdf: gpd.GeoDataFrame,
    raster_data: dict[str, np.ndarray] | None = None,
    worldcover_mask: np.ndarray | None = None,
    transform=None,
    progress_callback: ProgressCallback | None = None,
) -> gpd.GeoDataFrame:
    """Compute feature columns for each polygon in the GeoDataFrame.

    If raster_data / worldcover_mask are provided, zonal stats are computed.
    Otherwise only geometry-based features are filled; spectral features default to 0.
    Legacy raster columns used by the existing tile-trained classifier
    (`ndvi_std`, `ndvistd_mean`, `edge_mean`, `edge_max`) are also supported.

    Args:
        gdf: GeoDataFrame with polygon geometries in a projected CRS.
        raster_data: dict with keys like 'ndvi_mean', 'ndvi_max', 'ndvi_delta',
                     'ndwi_mean', 'msi_mean', 'bsi_mean', 'ndvi_variance' — each (H, W).
        worldcover_mask: boolean (H, W) cropland mask.
        transform: rasterio Affine transform mapping pixel coords to CRS of gdf.

    Returns:
        GeoDataFrame with FEATURE_COLUMNS added.
    """
    result = gdf.copy()

    areas = result.geometry.area
    perimeters = result.geometry.length
    result["area_m2"] = areas
    result["perimeter_m"] = perimeters
    result["shape_index"] = [shape_index(a, p) for a, p in zip(areas, perimeters)]
    result["compactness"] = [compactness(a, p) for a, p in zip(areas, perimeters)]
    result["elongation"] = [elongation(g) for g in result.geometry]

    # Spectral / contextual defaults
    for col in [
        "ndvi_mean",
        "ndvi_max",
        "ndvi_delta",
        "ndwi_mean",
        "msi_mean",
        "bsi_mean",
        "ndvi_variance",
        "worldcover_crop_pct",
        "growth_amplitude",
        "has_growth_peak",
        "ndvi_entropy",
        "neighbor_field_pct",
        "distance_to_road_m",
        "scl_valid_fraction_mean",
    ]:
        if col not in result.columns:
            result[col] = 0.0
    for col in sorted(_EXTRA_MEAN_RASTER_COLUMNS | _EXTRA_MAX_RASTER_COLUMNS):
        if col not in result.columns:
            result[col] = 0.0

    # Zonal stats from rasters (if available)
    if raster_data is not None and transform is not None:
        try:
            from rasterio.features import geometry_mask

            h, w = next(iter(raster_data.values())).shape
            total_rows = max(len(result.index), 1)
            _emit_progress(progress_callback, "object_zonal", 0, total_rows)
            for row_pos, idx in enumerate(result.index, start=1):
                geom = result.at[idx, "geometry"]
                try:
                    mask = ~geometry_mask([geom], out_shape=(h, w), transform=transform)
                except Exception:
                    _emit_progress(progress_callback, "object_zonal", row_pos, total_rows)
                    continue
                if not mask.any():
                    _emit_progress(progress_callback, "object_zonal", row_pos, total_rows)
                    continue
                for key, raster in raster_data.items():
                    col = key
                    if col in _SUPPORTED_RASTER_COLUMNS:
                        vals = raster[mask]
                        if col in _EXTRA_MAX_RASTER_COLUMNS:
                            result.at[idx, col] = float(np.nanmax(vals))
                        else:
                            result.at[idx, col] = float(np.nanmean(vals))

                if worldcover_mask is not None:
                    wc_vals = worldcover_mask[mask]
                    result.at[idx, "worldcover_crop_pct"] = float(wc_vals.mean())
                if row_pos % 8 == 0 or row_pos == total_rows:
                    _emit_progress(progress_callback, "object_zonal", row_pos, total_rows)
        except Exception as exc:
            logger.warning("zonal_stats_failed", error=str(exc), exc_info=True)

    # Пространственный контекст: плотность соседних кандидатов.
    try:
        if len(result) > 1:
            work = _to_projected_for_geometry(result)
            buffered = work[["geometry"]].copy()
            buffered["geometry"] = buffered.geometry.buffer(20.0)
            left = buffered.reset_index().rename(columns={"index": "src_idx"})
            right = work[["geometry"]].reset_index().rename(columns={"index": "dst_idx"})
            joined = gpd.sjoin(left, right, how="left", predicate="intersects")
            joined = joined[joined["src_idx"] != joined["dst_idx"]]
            neighbor_vals = np.zeros(len(work), dtype=np.float32)
            if not joined.empty:
                counts = joined.groupby("src_idx")["dst_idx"].nunique()
                indices = counts.index.to_numpy(dtype=np.int64, copy=False)
                values = np.minimum(1.0, counts.to_numpy(dtype=np.float32, copy=False) / 8.0)
                neighbor_vals[indices] = values
            result["neighbor_field_pct"] = neighbor_vals
    except Exception as exc:
        logger.warning("neighbor_context_failed", error=str(exc), exc_info=True)

    return result


class ObjectClassifier:
    """Histogram-gradient boosting classifier for field-candidate polygons."""

    def __init__(self, config: ObjectClassifierConfig | None = None) -> None:
        self.config = config or ObjectClassifierConfig()
        self._pipeline: Any | None = None
        self._feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)

    def fit(
        self,
        gdf: gpd.GeoDataFrame,
        target_col: str = "is_field",
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Train on a labelled GeoDataFrame (must contain FEATURE_COLUMNS + target_col)."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.utils.class_weight import compute_sample_weight

        X = gdf[FEATURE_COLUMNS].values.astype(np.float64)
        y = gdf[target_col].values.astype(int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        balanced_weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            balanced_weights = balanced_weights * sample_weight

        clf = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])
        self._feature_columns = tuple(FEATURE_COLUMNS)
        self._pipeline.fit(X, y, clf__sample_weight=balanced_weights)
        logger.info(
            "object_classifier_trained",
            n_samples=len(y),
            positive=int(y.sum()),
            negative=int((1 - y).sum()),
        )
        importances = self.feature_importances()
        if importances is not None:
            ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:5]
            logger.info("object_classifier_feature_importance_top5", top5=ranked)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "pipeline": self._pipeline,
            "feature_columns": list(self._feature_columns),
        }
        payload = make_classifier_payload_portable(payload)
        validate_classifier_payload(payload, self._feature_columns)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("object_classifier_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> ObjectClassifier:
        data = load_pickle_compat(path)
        return cls._from_serialized(data)

    @classmethod
    def _from_serialized(cls, data: Any) -> ObjectClassifier:
        if isinstance(data, cls):
            return data

        if isinstance(data, dict) and "config" in data and "pipeline" in data:
            obj = cls(config=data["config"])
            obj._pipeline = data["pipeline"]
            obj._feature_columns = tuple(
                data.get("feature_columns")
                or data.get("feature_names")
                or FEATURE_COLUMNS
            )
            validate_classifier_payload(data, obj._feature_columns)
            return obj

        if isinstance(data, dict) and "pipeline" in data:
            obj = cls()
            obj._pipeline = data["pipeline"]
            obj._feature_columns = tuple(
                data.get("feature_columns")
                or data.get("feature_names")
                or FEATURE_COLUMNS
            )
            validate_classifier_payload(
                {
                    "pipeline": obj._pipeline,
                    "feature_columns": list(obj._feature_columns),
                },
                obj._feature_columns,
            )
            return obj

        pipeline = getattr(data, "pipeline", None)
        if pipeline is not None:
            obj = cls()
            obj._pipeline = pipeline
            obj._feature_columns = tuple(
                getattr(data, "feature_columns", None)
                or getattr(data, "feature_names", None)
                or FEATURE_COLUMNS
            )
            validate_classifier_payload(
                {
                    "pipeline": obj._pipeline,
                    "feature_columns": list(obj._feature_columns),
                },
                obj._feature_columns,
            )
            return obj

        raise ValueError("Unsupported object classifier payload")

    def _uses_legacy_feature_space(self) -> bool:
        return any(
            name in self._feature_columns
            for name in ("area_px", "perimeter_px", "extent", "solidity")
        )

    def _feature_values(
        self,
        gdf: gpd.GeoDataFrame,
        name: str,
        geometry_cache: dict[str, Any] | None = None,
    ) -> np.ndarray:
        existing = _column_values(gdf, _FEATURE_ALIASES.get(name, (name,)))
        if existing is not None:
            return existing

        geometry_cache = geometry_cache or {}
        geom_gdf = geometry_cache["gdf"]
        areas_m2 = geometry_cache["areas_m2"]
        perimeters_m = geometry_cache["perimeters_m"]
        legacy = geometry_cache["legacy"]
        area_px = geometry_cache["area_px"]
        perimeter_px = geometry_cache["perimeter_px"]

        if name == "area_m2":
            return np.nan_to_num(areas_m2, nan=0.0)
        if name == "perimeter_m":
            return np.nan_to_num(perimeters_m, nan=0.0)
        if name == "area_px":
            return np.nan_to_num(area_px, nan=0.0)
        if name == "perimeter_px":
            return np.nan_to_num(perimeter_px, nan=0.0)
        if name == "shape_index":
            if legacy:
                return np.array(
                    [legacy_shape_index(a, p) for a, p in zip(area_px, perimeter_px)],
                    dtype=np.float64,
                )
            return np.array([shape_index(a, p) for a, p in zip(areas_m2, perimeters_m)], dtype=np.float64)
        if name == "compactness":
            return np.array([compactness(a, p) for a, p in zip(areas_m2, perimeters_m)], dtype=np.float64)
        if name == "elongation":
            return np.array([elongation(geom) for geom in geom_gdf.geometry], dtype=np.float64)
        if name == "extent":
            values = []
            for geom in geom_gdf.geometry:
                if geom is None or geom.is_empty:
                    values.append(0.0)
                    continue
                minx, miny, maxx, maxy = geom.bounds
                bbox_area = max((maxx - minx) * (maxy - miny), 1e-9)
                values.append(float(geom.area) / bbox_area)
            return np.array(values, dtype=np.float64)
        if name == "solidity":
            values = []
            for geom in geom_gdf.geometry:
                if geom is None or geom.is_empty:
                    values.append(0.0)
                    continue
                try:
                    hull_area = max(float(geom.convex_hull.area), 1e-9)
                except Exception:
                    hull_area = 1e-9
                values.append(float(geom.area) / hull_area)
            return np.array(values, dtype=np.float64)
        if name == "ndvi_variance":
            std_values = _column_values(gdf, ("ndvi_std", "ndvistd", "ndvistd_mean", "ndvistdmean"))
            if std_values is not None:
                return np.square(std_values)
        if name == "growth_amplitude":
            delta_values = _column_values(gdf, ("ndvi_delta", "ndvidelta"))
            if delta_values is not None:
                return np.maximum(delta_values, 0.0)
        if name == "has_growth_peak":
            peak_values = _column_values(gdf, ("has_growth_peak", "hasgrowthpeak"))
            if peak_values is not None:
                return np.clip(peak_values, 0.0, 1.0)
            delta_values = _column_values(gdf, ("ndvi_delta", "ndvidelta"))
            if delta_values is not None:
                return (delta_values >= 0.20).astype(np.float64)
        if name == "ndvi_entropy":
            entropy_values = _column_values(gdf, ("ndvi_entropy", "ndvientropy", "ndvi_temporal_entropy"))
            if entropy_values is not None:
                return np.maximum(entropy_values, 0.0)
            var_values = self._feature_values(gdf, "ndvi_variance", geometry_cache)
            return np.clip(np.log2(1.0 + np.maximum(var_values, 0.0) * 32.0), 0.0, 4.0)
        if name == "distance_to_road_m":
            road_values = _column_values(gdf, ("distance_to_road_m", "distancetoroadm"))
            if road_values is not None:
                return np.maximum(road_values, 0.0)
            return np.full(len(gdf), 1000.0, dtype=np.float64)
        if name == "scl_valid_fraction_mean":
            scl_values = _column_values(gdf, ("scl_valid_fraction_mean", "scl_valid_fraction"))
            if scl_values is not None:
                return np.clip(scl_values, 0.0, 1.0)
            return np.ones(len(gdf), dtype=np.float64)

        return np.zeros(len(gdf), dtype=np.float64)

    def _build_feature_matrix(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        feature_names = self._feature_columns or tuple(FEATURE_COLUMNS)
        geom_gdf = _to_projected_for_geometry(gdf)
        areas_m2 = geom_gdf.geometry.area.to_numpy(dtype=np.float64, copy=False)
        perimeters_m = geom_gdf.geometry.length.to_numpy(dtype=np.float64, copy=False)
        geometry_cache = {
            "gdf": geom_gdf,
            "areas_m2": areas_m2,
            "perimeters_m": perimeters_m,
            "legacy": self._uses_legacy_feature_space(),
            "area_px": areas_m2 / 100.0,
            "perimeter_px": perimeters_m / 10.0,
        }
        columns = [self._feature_values(gdf, name, geometry_cache) for name in feature_names]
        if not columns:
            return np.empty((0, 0), dtype=np.float64)
        return np.column_stack(columns).astype(np.float64, copy=False)

    def predict_proba(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Return P(is_field) for each row.

        Missing feature columns are derived from geometry / known aliases when possible,
        so both current and legacy model payloads can run against the merged polygons.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not fitted or loaded")

        X = np.nan_to_num(self._build_feature_matrix(gdf), nan=0.0)
        proba = self._pipeline.predict_proba(X)
        # Column index for positive class (1)
        pos_idx = list(self._pipeline.classes_).index(1)
        return proba[:, pos_idx]

    def feature_importances(self) -> dict[str, float] | None:
        """Return feature importances when the underlying estimator exposes them."""
        if self._pipeline is None:
            return None
        clf = self._pipeline.named_steps.get("clf")
        if clf is None or not hasattr(clf, "feature_importances_"):
            return None
        values = getattr(clf, "feature_importances_", None)
        if values is None:
            return None
        return {
            feature: float(value)
            for feature, value in zip(self._feature_columns, values)
        }
