"""Tests for configuration."""
import pytest

from core.config import Settings, get_adaptive_season_window, get_px_area_m2, get_settings


def _settings_defaults() -> Settings:
    return Settings(_env_file=None)


class TestSettings:
    def test_defaults(self):
        s = _settings_defaults()
        assert s.DEFAULT_CENTER_LAT == pytest.approx(58.689077)
        assert s.DEFAULT_CENTER_LON == pytest.approx(29.892103)
        assert s.DEFAULT_RADIUS_KM == 15.0
        assert s.MAX_RADIUS_KM == 40.0
        assert s.DEFAULT_RESOLUTION_M == 10
        assert s.MAX_CLOUD_PCT == 40
        assert s.MIN_FIELD_AREA_HA == 0.5
        assert s.S2_TEMPORAL_SLICES == 7
        assert s.S2_SEASON_START == "04-15"
        assert s.S2_SEASON_END == "10-15"
        assert s.S2_MIN_GOOD_DATES == 4
        assert s.ALLOW_SYNTHETIC_DATA is False

    def test_sentinel_defaults(self):
        s = _settings_defaults()
        assert "sentinel-hub.com" in s.SH_BASE_URL
        assert s.SH_MAX_RETRIES == 4
        assert s.SH_RETRY_BASE_DELAY_S == 2.0
        assert s.SH_RETRY_MAX_DELAY_S == 30.0
        assert s.SH_FAILOVER_ENABLED is True
        assert s.SENTINEL_CONCURRENT_REQUESTS == 4
        assert s.TILE_MEMORY_CLEANUP_ENABLED is True
        assert s.TILE_MEMORY_GC_EVERY_TILE is True
        assert s.TILE_MAX_RUNTIME_S == 180
        assert s.MAX_CANDIDATES_PER_TILE_FAST == 256
        assert s.MAX_CANDIDATES_PER_TILE_STANDARD == 768
        assert s.MAX_CANDIDATES_PER_TILE_QUALITY == 1200
        assert s.FAST_TILE_MEMORY_SOFT_LIMIT_MB == 1600
        assert s.STANDARD_TILE_MEMORY_SOFT_LIMIT_MB == 2400
        assert s.QUALITY_TILE_MEMORY_SOFT_LIMIT_MB == 3000
        assert s.WORKER_HEARTBEAT_STALE_S == 240
        assert s.SENTINEL_FETCH_KEEPALIVE_S == pytest.approx(10.0)
        assert s.TILE_SIZE_PX == 1024
        assert s.DATE_SELECTION_PROFILE == "adaptive_region"
        assert s.DATE_SELECTION_MIN_VALID_PCT == pytest.approx(0.50)
        assert s.DATE_SELECTION_TARGET_DATES == 7
        assert s.DATE_SELECTION_MIN_GOOD_DATES == 4
        assert s.DATE_SELECTION_WEIGHT_COVERAGE == pytest.approx(0.40)
        assert s.DATE_SELECTION_WEIGHT_PHENO == pytest.approx(0.30)
        assert s.DATE_SELECTION_WEIGHT_UNIQUENESS == pytest.approx(0.20)
        assert s.DATE_SELECTION_WEIGHT_WATER_PENALTY == pytest.approx(0.10)
        assert s.REGION_PROFILE_MODE == "auto_only"
        assert s.REGION_LAT_SOUTH_MAX == pytest.approx(48.0)
        assert s.REGION_LAT_NORTH_MIN == pytest.approx(57.0)
        assert s.MODE == "production"

    def test_pheno_thresholds(self):
        s = _settings_defaults()
        assert s.PHENO_NDWI_WATER == 0.10
        assert s.PHENO_MNDWI_WATER == 0.05
        assert s.PHENO_NDVI_CROP_MIN == 0.25
        assert s.PHENO_DELTA_CROP == 0.3
        assert s.PHENO_N_VALID_MIN == 4

    def test_segmentation_defaults(self):
        s = _settings_defaults()
        assert s.EDGE_COVERAGE_THRESHOLD == 0.30
        assert s.EDGE_CANNY_SIGMA == 1.2
        assert s.WATERSHED_LAMBDA == 1.0
        assert s.WATERSHED_LAMBDA_CANDIDATES == (0.2, 0.5, 1.0, 2.0)
        assert s.WATERSHED_MIN_DISTANCE == 14
        assert s.OBIA_MAX_SHAPE_INDEX == 2.0
        assert s.OBIA_MAX_MEAN_NDWI == 0.2
        assert s.OWT_EDGE_WEIGHT == 0.7
        assert s.OWT_NDVI_WEIGHT == 0.3
        assert s.MERGE_BOUNDARY_THRESH == 0.25

    def test_worldcover_defaults(self):
        s = _settings_defaults()
        assert s.USE_WEAK_WORLDCOVER_BARRIER is True
        assert s.WC_EXCLUDE_CLASSES == (10, 20, 80, 90)
        assert s.WC_TREE_HARD_EXCLUSION is True
        assert s.WC_MIN_CROPLAND_FRAC == 0.05
        assert s.WC_MAX_NONCROP_FRAC == 0.80
        assert s.TILE_MIN_OBS_COUNT == 4
        assert s.AUTO_DETECT_VERSION == 4
        assert s.FRAMEWORK_SAM_ENABLED is False
        assert s.FRAMEWORK_SAM_FIELD_DET is True
        assert s.FRAMEWORK_USE_WEAK_WORLDCOVER is True
        assert s.FRAMEWORK_SKIP_WATERSHED_FOR_LARGEST is True
        assert s.FEATURE_UNET_EDGE is True
        assert s.FEATURE_SAM2_PRIMARY is False
        assert s.FEATURE_S1_FUSION is False
        assert s.FEATURE_SNIC_REFINE is False
        assert s.FEATURE_ML_PRIMARY is True
        assert s.ML_MODEL_PATH.endswith("boundary_unet_v2.onnx")
        assert s.ML_MODEL_NORM_STATS_PATH.endswith("boundary_unet_v2.norm.json")
        assert s.ML_INFERENCE_DEVICE == "auto"
        assert s.ML_FALLBACK_ON_LOW_SCORE is True
        assert s.ML_SCORE_THRESHOLD == pytest.approx(0.35)
        assert s.ML_TILE_SIZE == 512
        assert s.ML_OVERLAP == 128
        assert s.ML_USE_ONNX is True
        assert s.ML_FEATURE_PROFILE == "v2_16ch"
        assert s.ML_TTA_STANDARD_MODE == "flip2"
        assert s.ML_TTA_QUALITY_MODE == "rotate4"
        assert s.ML_MULTI_SCALE_STANDARD is False
        assert s.ML_MULTI_SCALE_QUALITY is True
        assert s.ML_MULTI_SCALE_AUX_SCALES == (0.75,)
        assert s.MODEL_VERSION == "boundary_unet_v2"
        assert s.TRAIN_DATA_VERSION == "real_tiles_v5"
        assert s.FEATURE_STACK_VERSION == "v5_16ch"
        assert s.ONNX_OPSET_VERSION == 18
        assert s.WEATHER_HTTP_TIMEOUT_S == pytest.approx(20.0)
        assert s.WEATHER_HTTP_CONNECT_TIMEOUT_S == pytest.approx(8.0)
        assert s.WEATHER_HTTP_RETRIES == 3
        assert s.WEATHER_HTTP_RETRY_BACKOFF_S == pytest.approx(1.5)
        assert s.WEAK_LABEL_MIN_COVERAGE_PCT == pytest.approx(0.5)
        assert s.WEAK_LABEL_MAX_FALLBACK_RATIO == pytest.approx(0.35)
        assert s.WEAK_LABEL_OSM_OVERRIDE_ENABLED is True
        assert s.WEAK_LABEL_TEMPORAL_OVERRIDE_ENABLED is True
        assert s.ML_EXTENT_BIN_THRESHOLD == pytest.approx(0.42)
        assert s.ML_EXTENT_CALIBRATION_ENABLED is True
        assert s.GEOMETRY_REFINE_PROFILE == "balanced"
        assert s.BOUNDARY_QUALITY_PROFILE == "quality_first"

    def test_object_classifier_defaults(self):
        s = _settings_defaults()
        assert s.USE_OBJECT_CLASSIFIER is True
        assert s.OBJECT_MIN_SCORE == 0.7

    def test_postprocess_defaults(self):
        s = _settings_defaults()
        assert s.POST_ROAD_MAX_NDVI == 0.22
        assert s.POST_ROAD_NIR_MAX == 0.15
        assert s.POST_ROAD_NDBI_MIN == -0.10
        assert s.POST_FOREST_NDVI_MIN == 0.65
        assert s.POST_MORPH_CLOSE_RADIUS == 2
        assert s.POST_MERGE_BUFFER_PX == 4
        assert s.POST_MERGE_NDVI_DIFF_MAX == 0.12
        assert s.POST_MERGE_OVERLAP_MIN == 0.30
        assert s.POST_MERGE_BARRIER_RATIO == 0.08
        assert s.POST_MERGE_MAX_COMPONENTS == 2000
        assert s.POST_GROW_NDVI_RELAX == 0.11
        assert s.POST_GROW_BOUNDARY_STOP_THRESHOLD == pytest.approx(0.35)
        assert s.POST_GROW_MAX_ITERS == 10
        assert s.POST_GAP_EDGE_THRESHOLD == 0.30
        assert s.SELECTIVE_SPLIT_ENABLED is True
        assert s.WATERSHED_ONLY_IF_SPLIT_SCORE is True
        assert s.SELECTIVE_SPLIT_SCORE_MIN == pytest.approx(0.70)
        assert s.SELECTIVE_SPLIT_MIN_SHARED_BOUNDARY_PX == 32
        assert s.SELECTIVE_SPLIT_MIN_BOUNDARY_PROB == pytest.approx(0.64)
        assert s.SELECTIVE_SPLIT_MIN_EDGE_SCORE == pytest.approx(0.28)
        assert s.SELECTIVE_SPLIT_MIN_FEATURE_DELTA == pytest.approx(0.15)
        assert s.WATERSHED_ROLLBACK_COMPONENT_RATIO_MAX == pytest.approx(1.6)
        assert s.WATERSHED_ROLLBACK_MAX_INTERNAL_BOUNDARY_CONF == pytest.approx(0.58)
        assert s.POST_CONVEX_MIN_HA == 1.5
        assert s.POST_CONVEX_RATIO_MAX == 1.6
        assert s.POST_MIN_FIELD_AREA_HA == 0.7
        assert s.NORTH_POST_MIN_FIELD_AREA_HA == pytest.approx(0.45)
        assert s.POST_PX_AREA_M2 == 100
        assert s.WATERSHED_COMPACTNESS == pytest.approx(0.001)
        assert s.WATERSHED_GRADIENT_EDGE_W == 0.5
        assert s.DEBUG_COMPARE_VERSIONS is False
        assert s.SNAKE_REFINE_ENABLED is True
        assert s.SNAKE_MAX_PX_DIST == 15.0
        assert s.ROAD_OSM_BUFFER_DEFAULT_M == 12
        assert s.TEMPORAL_CLOUD_SCL_CLASSES == (0, 1, 2, 3, 8, 9, 10, 11)
        assert s.POST_MERGE_SMOOTH is True
        assert s.POST_SIMPLIFY_TOLERANCE == 4.0
        assert s.POST_BUFFER_SMOOTH_M == 2.0
        assert s.TEMPORAL_YEARS_BACK == 1
        assert s.TEMPORAL_BEST_N_SCENES == 8
        assert s.TEMPORAL_CLOUD_MAX_PCT == 20
        assert s.TEMPORAL_SCL_INVALID == (0, 1, 2, 3, 8, 9, 10, 11)
        assert s.TEMPORAL_NDVI_VALID_MIN == 0.05
        assert s.PHENO_FIELD_NDVI_STD_MIN == 0.15
        assert s.PHENO_GRASS_NDVI_STD_MAX == 0.12
        assert s.PHENO_FOREST_NDVI_STD_MAX == 0.05
        assert s.PHENO_FIELD_MAX_NDVI_MIN == 0.45
        assert s.PHENO_FIELD_MAX_NDVI_MAX == 0.62
        assert s.PHENO_GRASS_MAX_NDVI_MAX == 0.55
        assert s.PHENO_FOREST_MAX_NDVI_MIN == 0.65
        assert s.UNET_EDGE_MODEL == "/app/models/unet_edge_best.pth"
        assert s.UNET_EDGE_THRESHOLD == 0.5
        assert s.UNET_DEVICE == "cpu"
        assert s.SAM2_CHECKPOINT == "/app/models/sam2_hiera_large.pt"
        assert s.SAM_POINT_SPACING == 20
        assert s.SAM_PRED_IOU_THRESHOLD == 0.85
        assert s.S1_ENABLED is False
        assert s.S1_ACQUISITION_MODE == "IW"
        assert s.S1_POLARIZATION == "DV"
        assert s.S1_LEE_FILTER_ENABLE is True
        assert s.S1_LEE_WINDOW_SIZE == 5
        assert s.SNIC_REFINE_ENABLED is False
        assert s.SNIC_N_SEGMENTS == 1000
        assert s.SNIC_COMPACTNESS == 0.01
        assert s.SNIC_MERGE_NDVI_THRESH == 0.05
        assert s.SAM_ENABLED is False
        assert s.SAM_FIELD_DET is True
        assert s.SAM_MODEL_TYPE == "vit_b"
        assert s.SAM_POINTS_PER_SIDE == 16
        assert s.HYBRID_MERGE_MIN_IOU == 0.20
        assert s.HYBRID_SAM_MIN_CROP_RATIO == 0.30
        assert s.HYBRID_SAM_MAX_FOREST_RATIO == 0.40
        assert s.HYBRID_SAM_MAX_WATER_RATIO == 0.20
        assert s.HYBRID_SNAP_TOLERANCE_M == 15.0
        assert s.OBIA_MAX_HOLE_FRAC == 0.10
        assert s.OBIA_MAX_HOLE_NONCROP_FRAC == 0.35
        assert s.OBIA_MAX_INTERNAL_TREE_FRAC == 0.20
        assert s.OBIA_MAX_INTERNAL_WATER_FRAC == 0.10
        assert s.POST_BOUNDARY_DILATION_PX == 1
        assert s.POST_BOUNDARY_DILATION_MAX_PX == 3
        assert s.POST_LARGE_FIELD_RESCUE_ENABLED is True
        assert s.POST_LARGE_FIELD_RESCUE_MIN_AREA_HA == pytest.approx(2.0)
        assert s.SNAKE_REFINE_MODE == "guarded"
        assert s.SNAKE_MAX_CENTROID_SHIFT_M == pytest.approx(6.0)
        assert s.SNAKE_MIN_AREA_RATIO == pytest.approx(0.90)
        assert s.SNAKE_MAX_AREA_RATIO == pytest.approx(1.12)
        assert s.VECTORIZE_SIMPLIFY_TOL_M == pytest.approx(0.8)
        assert s.BOUNDARY_SMOOTH_SIMPLIFY_TOL_M == pytest.approx(1.2)
        assert s.TOPOLOGY_SIMPLIFY_ENABLED is False
        assert s.TOPOLOGY_SIMPLIFY_TOL_M == pytest.approx(1.0)
        assert s.OBIA_RELAX_IF_ML_CONFIDENT is True
        assert s.OBIA_RELAX_MIN_BOUNDARY_CONF == pytest.approx(0.65)
        assert s.ROAD_BARRIER_SOFT_RETRY_ENABLED is True
        assert s.ROAD_CLASS_PROFILE == "major_minor"
        assert s.ROAD_MAJOR_BUFFER_PX == 2
        assert s.ROAD_MINOR_BUFFER_PX == 1
        assert s.ROAD_SNAP_REJECT_ENABLED is True
        assert s.ROAD_SNAP_REJECT_BUFFER_PX == 2
        assert s.ROAD_SNAP_REJECT_MAX_OVERLAP_RATIO == pytest.approx(0.08)
        assert s.ROAD_SOFT_BARRIER_MINOR_ONLY is True
        assert s.ROAD_PARALLEL_EDGE_PENALTY_ENABLED is True
        assert s.ROAD_DIRECTIONAL_PENALTY_ENABLED is True
        assert s.REGION_PROFILE_DEBUG_LOGGING is True
        assert s.REGION_PROFILE_RECORD_DIAGNOSTICS is True
        assert s.SOUTH_PROFILE_ENABLED is True
        assert s.SOUTH_ML_EXTENT_BIN_THRESHOLD == pytest.approx(0.34)
        assert s.SOUTH_POST_GROW_MAX_ITERS == 6
        assert s.SOUTH_POST_GROW_NDVI_RELAX == pytest.approx(0.17)
        assert s.SOUTH_COMPONENT_BRIDGE_ENABLED is True
        assert s.SOUTH_MERGE_MIN_OVERLAP_RATIO == pytest.approx(0.03)
        assert s.NORTH_PROFILE_ENABLED is True
        assert s.NORTH_ML_EXTENT_BIN_THRESHOLD == pytest.approx(0.40)
        assert s.NORTH_VECTORIZE_SIMPLIFY_TOL_M == pytest.approx(0.25)
        assert s.NORTH_BOUNDARY_SMOOTH_SIMPLIFY_TOL_M == pytest.approx(0.50)
        assert s.NORTH_TOPOLOGY_SIMPLIFY_ENABLED is False
        assert s.NORTH_STAGE_ROLLBACK_MIN_AREA_RATIO == pytest.approx(0.95)
        assert s.NORTH_BOUNDARY_OUTER_DILATION_PX == 1
        assert s.HYDRO_BOUNDARY_PROFILE == "water_aware"
        assert s.HYDRO_OPEN_WATER_NDWI == pytest.approx(0.14)
        assert s.HYDRO_SEASONAL_WET_NDWI == pytest.approx(0.06)
        assert s.HYDRO_RIPARIAN_BUFFER_PX == 2
        assert s.HYDRO_FIELD_NEAR_WATER_RESCUE_ENABLED is True
        assert s.SAM_RUNTIME_POLICY == "safe_optional"
        assert s.SAM_MAX_TILE_PIXELS == 600_000
        assert s.SAM_MAX_COMPONENTS == 64
        assert s.SAM_MAX_CANDIDATE_COVERAGE_PCT == pytest.approx(18.0)
        assert s.SAM_MAX_EST_MEMORY_MB == 2200
        assert s.SAM_TIMEOUT_S == 45
        assert s.SAM_USE_CROP_BOXES is True
        assert s.SAM_CROP_BOX_PADDING_PX == 24
        assert s.BOUNDARY_OUTER_EXPAND_WATER_AWARE is True
        assert s.BOUNDARY_OUTER_EXPAND_MAX_PX == 4
        assert s.BOUNDARY_OUTER_EXPAND_NEAR_WATER_MAX_PX == 2
        assert s.BOUNDARY_VECTOR_SNAP_GRID_M == pytest.approx(1.0)

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_accepts_compact_wctree_alias(self):
        s = Settings.model_validate({"WCTREEHARD_EXCLUSION": False})
        assert s.WC_TREE_HARD_EXCLUSION is False

    def test_accepts_v3_aliases(self):
        s = Settings.model_validate(
            {
                "UNETEDGEMODEL": "/tmp/model.pth",
                "S1ENABLED": True,
                "SNICREFINEENABLED": True,
            }
        )
        assert s.UNET_EDGE_MODEL == "/tmp/model.pth"
        assert s.S1_ENABLED is True
        assert s.SNIC_REFINE_ENABLED is True

    def test_accepts_compact_runtime_env_aliases(self):
        s = Settings.model_validate(
            {
                "DATABASEURL": "postgresql+asyncpg://u:p@localhost:5432/db",
                "DATABASEURLSYNC": "postgresql+psycopg://u:p@localhost:5432/db",
                "REDISURL": "redis://localhost:6379/0",
                "CELERYBROKERURL": "redis://localhost:6379/0",
                "CELERYRESULTBACKEND": "redis://localhost:6379/1",
                "ALLOWSYNTHETICDATA": True,
                "USEOBJECTCLASSIFIER": False,
                "OBJECTCLASSIFIERPATH": "/tmp/object.pkl",
                "OBJECTMINSCORE": 0.4,
                "MODELVERSION": "boundary_unet_v2",
                "TRAINDATAVERSION": "dataset_2026_03",
                "FEATURESTACKVERSION": "v5_16ch",
                "ONNXOPSETVERSION": 19,
                "WEAKLABELMINCOVERAGEPCT": 0.4,
                "WEAKLABELMAXFALLBACKRATIO": 0.2,
                "WEAKLABELOSMOVERRIDEENABLED": False,
                "WEAKLABELTEMPORALOVERRIDEENABLED": False,
            }
        )
        assert s.DATABASE_URL.endswith("@localhost:5432/db")
        assert s.DATABASE_URL_SYNC.endswith("@localhost:5432/db")
        assert s.REDIS_URL == "redis://localhost:6379/0"
        assert s.CELERY_BROKER_URL == "redis://localhost:6379/0"
        assert s.CELERY_RESULT_BACKEND == "redis://localhost:6379/1"
        assert s.ALLOW_SYNTHETIC_DATA is True
        assert s.USE_OBJECT_CLASSIFIER is False
        assert s.OBJECT_CLASSIFIER_PATH == "/tmp/object.pkl"
        assert s.OBJECT_MIN_SCORE == 0.4
        assert s.MODEL_VERSION == "boundary_unet_v2"
        assert s.TRAIN_DATA_VERSION == "dataset_2026_03"
        assert s.FEATURE_STACK_VERSION == "v5_16ch"
        assert s.ONNX_OPSET_VERSION == 19
        assert s.WEAK_LABEL_MIN_COVERAGE_PCT == pytest.approx(0.4)
        assert s.WEAK_LABEL_MAX_FALLBACK_RATIO == pytest.approx(0.2)
        assert s.WEAK_LABEL_OSM_OVERRIDE_ENABLED is False
        assert s.WEAK_LABEL_TEMPORAL_OVERRIDE_ENABLED is False

    def test_road_osm_defaults(self):
        s = _settings_defaults()
        assert s.ROAD_OSM_ENABLED is False
        assert s.ROAD_OSM_REQUEST_TIMEOUT_S == 45
        assert s.ROAD_OSM_OVERPASS_RATE_LIMIT is False
        assert s.ROAD_OSM_BUFFER_DEFAULT_M == 12
        assert "motorway" in s.ROAD_OSM_TAGS
        assert "trunk" in s.ROAD_OSM_TAGS
        assert isinstance(s.ROAD_OSM_BUFFER_MAP, dict)
        assert s.ROAD_OSM_BUFFER_MAP.get("motorway") == 25

    def test_edge_composite_config(self):
        s = _settings_defaults()
        assert s.EDGE_BINARY_THRESHOLD == pytest.approx(0.12)
        assert s.EDGE_CLOSING_RADIUS == 2
        assert s.EDGE_SOFT_CLIP_ENABLED is True
        assert s.EDGE_SOFT_CLIP_PERCENTILE == pytest.approx(95.0)

    def test_rejects_invalid_balanced_profile_values(self):
        with pytest.raises(Exception):
            Settings.model_validate({"GEOMETRYREFINEPROFILE": "bad"})
        with pytest.raises(Exception):
            Settings.model_validate({"BOUNDARYQUALITYPROFILE": "bad"})
        with pytest.raises(Exception):
            Settings.model_validate({"SNAKEREFINEMODE": "bad"})
        with pytest.raises(Exception):
            Settings.model_validate({"SAMRUNTIMEPOLICY": "bad"})
        with pytest.raises(Exception):
            Settings.model_validate({"HYDROBOUNDARYPROFILE": "bad"})
        with pytest.raises(Exception):
            Settings.model_validate({"MODE": "invalid"})
        with pytest.raises(Exception):
            Settings.model_validate({"MLFEATUREPROFILE": "v9_unknown"})

    def test_get_px_area_m2(self):
        assert get_px_area_m2(10) == pytest.approx(100.0)
        assert get_px_area_m2(20) == pytest.approx(400.0)

    def test_adaptive_season_window(self):
        s = _settings_defaults()
        assert get_adaptive_season_window(45.0, s) == ("03-01", "11-01")
        assert get_adaptive_season_window(58.0, s) == ("04-25", "10-25")
        assert get_adaptive_season_window(53.0, s) == ("04-05", "10-20")
