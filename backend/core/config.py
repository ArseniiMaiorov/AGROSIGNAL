from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

BOOL_TRUE_LITERALS: tuple[str, ...] = ("1", "true", "t", "yes", "y", "on")
BOOL_FALSE_LITERALS: tuple[str, ...] = ("0", "false", "f", "no", "n", "off")
BOOL_ALLOWED_LITERALS: tuple[str, ...] = BOOL_TRUE_LITERALS + BOOL_FALSE_LITERALS


def parse_env_bool(value: Any, *, field_name: str) -> bool:
    """Parse a boolean env value with explicit accepted literals."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in BOOL_TRUE_LITERALS:
            return True
        if token in BOOL_FALSE_LITERALS:
            return False
    allowed = ", ".join(BOOL_ALLOWED_LITERALS)
    raise ValueError(
        f"{field_name} must be a boolean value. Allowed literals: {allowed}. Got: {value!r}"
    )


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_prefix="",
        case_sensitive=False,
        # Support both local launches from repo root and from ./backend.
        env_file=(".env", "../.env"),
    )
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://agromap:agromap_dev_password@postgres:5432/agromap",
        validation_alias=AliasChoices("DATABASE_URL", "DATABASEURL"),
    )
    DATABASE_URL_SYNC: str = Field(
        default="postgresql+psycopg://agromap:agromap_dev_password@postgres:5432/agromap",
        validation_alias=AliasChoices("DATABASE_URL_SYNC", "DATABASEURLSYNC"),
    )
    REDIS_URL: str = Field(
        default="redis://redis:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "REDISURL"),
    )
    CELERY_BROKER_URL: str = Field(
        default="redis://redis:6379/0",
        validation_alias=AliasChoices("CELERY_BROKER_URL", "CELERYBROKERURL"),
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://redis:6379/1",
        validation_alias=AliasChoices("CELERY_RESULT_BACKEND", "CELERYRESULTBACKEND"),
    )

    SH_CLIENT_ID: str = ""
    SH_CLIENT_SECRET: str = ""
    SH_CLIENT_ID_reserv: str = ""
    SH_CLIENT_SECRET_reserv: str = ""
    SH_CLIENT_ID_second_reserv: str = ""
    SH_CLIENT_SECRET_second_reserv: str = ""
    SH_BASE_URL: str = "https://services.sentinel-hub.com"
    SH_MAX_RETRIES: int = 4
    SH_RETRY_BASE_DELAY_S: float = 2.0
    SH_RETRY_MAX_DELAY_S: float = 30.0
    SH_FAILOVER_ENABLED: bool = True
    SH_FAILOVER_COOLDOWN_S: int = 1800
    SH_RATE_LIMIT_COOLDOWN_S: int = 60

    ERA5_CDS_URL: str = "https://cds.climate.copernicus.eu/api"
    ERA5_CDS_KEY: str = ""
    ERA5_CACHE_DIR: str = "/tmp/era5_cache"
    ERA5_CACHE_TTL_HOURS: int = 24
    OPENWEATHER_API_KEY: str = ""
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/3.0"
    OPENMETEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    OPENMETEO_ARCHIVE_BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    WEATHER_PROVIDER: str = "openmeteo"
    WEATHER_CACHE_TTL_MINUTES: int = 60
    STATUS_CACHE_TTL_SECONDS: int = 30
    ARCHIVE_DIR: str = "debug/archives"
    ARCHIVE_TTL_DAYS: int = 30
    DEFAULT_CROP_CODE: str = "wheat"
    YIELD_MODEL_VERSION: str = "agronomy_tabular_v2"
    SCENE_CACHE_DIR: str = "cache/sentinel_scenes"
    SCENE_CACHE_TTL_DAYS: int = 30
    SH_RETRY_BUDGET: int = 12
    WEATHER_ALLOW_STALE_ON_FAILURE: bool = True
    AUTH_REQUIRED: bool = True
    AUTH_ACCESS_TTL_MINUTES: int = 30
    AUTH_REFRESH_TTL_DAYS: int = 14
    AUTH_JWT_SECRET: str = "change-me-for-production"
    AUTH_BOOTSTRAP_ENABLED: bool = True
    AUTH_BOOTSTRAP_ORG_NAME: str = "Default Organization"
    AUTH_BOOTSTRAP_ADMIN_EMAIL: str = "admin@local"
    AUTH_BOOTSTRAP_ADMIN_PASSWORD: str = "admin12345"
    DATA_IMPORT_DIR: str = "debug/imports"
    DATA_IMPORT_MAX_ERRORS: int = 100
    MLFLOW_ENABLED: bool = True
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MINIO_ENDPOINT: str = "http://minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "autodetect-artifacts"

    DEFAULT_CENTER_LAT: float = 58.689077
    DEFAULT_CENTER_LON: float = 29.892103
    DEFAULT_RADIUS_KM: float = 15.0
    MAX_RADIUS_KM: float = 40.0
    DEFAULT_RESOLUTION_M: int = 10
    MAX_CLOUD_PCT: int = 40
    MIN_FIELD_AREA_HA: float = 0.5
    S2_TEMPORAL_SLICES: int = 7
    S2_SEASON_START: str = "04-15"
    S2_SEASON_END: str = "10-15"
    S2_MIN_GOOD_DATES: int = 4
    ALLOW_SYNTHETIC_DATA: bool = Field(
        default=False,
        validation_alias=AliasChoices("ALLOW_SYNTHETIC_DATA", "ALLOWSYNTHETICDATA"),
    )
    TILE_SIZE_PX: int = 1024
    TILE_OVERLAP_M: int = 500
    SENTINEL_CONCURRENT_REQUESTS: int = 4
    TILE_MEMORY_CLEANUP_ENABLED: bool = True
    TILE_MEMORY_GC_EVERY_TILE: bool = True
    TILE_MAX_RUNTIME_S: int = 180
    ENABLE_CANDIDATE_RANKER: bool = True
    # Maximum candidate components per tile before spatial cap is applied.
    # Prevents O(n²) memory spike in branch-agreement / suppress stages.
    # Candidates are sorted by pixel area (largest kept) before truncation.
    MAX_CANDIDATES_PER_TILE: int = 1800
    MAX_CANDIDATES_PER_TILE_FAST: int = 256
    MAX_CANDIDATES_PER_TILE_STANDARD: int = 768
    MAX_CANDIDATES_PER_TILE_QUALITY: int = 1200
    POST_MERGE_MAX_COMPONENTS_FAST: int = 512
    FAST_TILE_MEMORY_SOFT_LIMIT_MB: int = 1600
    STANDARD_TILE_MEMORY_SOFT_LIMIT_MB: int = 2400
    QUALITY_TILE_MEMORY_SOFT_LIMIT_MB: int = 3000
    WORKER_HEARTBEAT_STALE_S: int = 240
    SENTINEL_FETCH_KEEPALIVE_S: int = 10
    CACHE_TTL_SECONDS: int = 86400 * 3
    LOG_LEVEL: str = "INFO"
    WEATHER_HTTP_TIMEOUT_S: float = 20.0
    WEATHER_HTTP_CONNECT_TIMEOUT_S: float = 8.0
    WEATHER_HTTP_RETRIES: int = 3
    WEATHER_HTTP_RETRY_BACKOFF_S: float = 1.5
    DATE_SELECTION_PROFILE: str = "adaptive_region"
    DATE_SELECTION_MIN_VALID_PCT: float = 0.50
    DATE_SELECTION_TARGET_DATES: int = 7
    DATE_SELECTION_MIN_GOOD_DATES: int = 4
    DATE_SELECTION_WEIGHT_COVERAGE: float = 0.40
    DATE_SELECTION_WEIGHT_PHENO: float = 0.30
    DATE_SELECTION_WEIGHT_UNIQUENESS: float = 0.20
    DATE_SELECTION_WEIGHT_WATER_PENALTY: float = 0.10
    DATE_SELECTION_NORTH_SHIFT_DAYS: int = 14
    DATE_SELECTION_SOUTH_SHIFT_DAYS: int = -14
    DATE_SELECTION_ALLOW_LOW_CONFIDENCE_FALLBACK: bool = True
    REGION_PROFILE_MODE: str = "auto_only"
    REGION_LAT_SOUTH_MAX: float = 48.0
    REGION_LAT_NORTH_MIN: float = 57.0
    MODE: str = "production"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_DETECT: str = "5/minute"
    GRID_LAYER_CELL_PX: int = 16

    PHENO_NDWI_WATER: float = 0.10
    PHENO_MNDWI_WATER: float = 0.05
    PHENO_BSI_BUILT: float = 0.15
    PHENO_STD_BUILT: float = 0.02
    PHENO_NDVI_FOREST_MIN: float = 0.45
    PHENO_DELTA_FOREST: float = 0.15
    PHENO_NDVI_GRASS_MEAN: float = 0.35
    PHENO_DELTA_GRASS: float = 0.2
    PHENO_NDVI_CROP_MAX: float = 0.62
    PHENO_NDVI_CROP_MIN: float = 0.25
    PHENO_DELTA_CROP: float = 0.3
    PHENO_MSI_CROP: float = 1.1
    PHENO_N_VALID_MIN: int = 4

    EDGE_ALPHA: float = 0.7
    EDGE_COVERAGE_THRESHOLD: float = 0.30
    EDGE_CANNY_SIGMA: float = 1.2
    EDGE_WEAK_THRESHOLD: float = 0.1
    EDGE_BINARY_THRESHOLD: float = 0.12
    EDGE_CLOSING_RADIUS: int = 2
    EDGE_SOFT_CLIP_ENABLED: bool = True
    EDGE_SOFT_CLIP_PERCENTILE: float = 95.0
    BOUNDARY_RECOVERY_BRANCH_SCORE_BOOST: float = 0.08
    RECOVERY_SECOND_PASS_SCORE_BOOST: float = 0.05
    BOUNDARY_RECOVERY_EDGE_SEED_THRESHOLD: float = 0.18
    BOUNDARY_RECOVERY_EDGE_SEED_PERCENTILE: float = 72.0
    BOUNDARY_RECOVERY_DILATION_PX: int = 2
    BOUNDARY_RECOVERY_BOUNDARY_HALO_PX: int = 1
    RECOVERY_BOUNDARY_ANCHOR_ENABLED: bool = False
    RECOVERY_BOUNDARY_ANCHOR_DILATION_PX: int = 2
    RECOVERY_EDGE_GUIDE_HALO_FACTOR: float = 0.85
    RECOVERY_TEMPORAL_COHERENCE_RELAXED: bool = False
    RECOVERY_TEMPORAL_GROWTH_AMPLITUDE_MIN: float = 0.14
    RECOVERY_TEMPORAL_ENTROPY_MAX: float = 3.1
    WATERSHED_LAMBDA: float = 1.0
    WATERSHED_LAMBDA_CANDIDATES: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0)
    WATERSHED_MIN_DISTANCE: int = 14
    SELECTIVE_SPLIT_ENABLED: bool = True
    WATERSHED_ONLY_IF_SPLIT_SCORE: bool = True
    SELECTIVE_SPLIT_SCORE_MIN: float = 0.70
    SELECTIVE_SPLIT_MIN_SHARED_BOUNDARY_PX: int = 32
    SELECTIVE_SPLIT_MIN_BOUNDARY_PROB: float = 0.64
    SELECTIVE_SPLIT_MIN_EDGE_SCORE: float = 0.28
    SELECTIVE_SPLIT_MIN_FEATURE_DELTA: float = 0.15
    WATERSHED_ROLLBACK_COMPONENT_RATIO_MAX: float = 1.6
    WATERSHED_ROLLBACK_MAX_INTERNAL_BOUNDARY_CONF: float = 0.58
    OBIA_MAX_SHAPE_INDEX: float = 2.4      # relaxed: elongated/irregular agricultural fields
    OBIA_MIN_NDVI_DELTA: float = 0.12     # relaxed: uniform crops (same growth stage) were being missed
    OBIA_MAX_MEAN_NDWI: float = 0.2
    SIMPLIFY_TOL_M: float = 3.0

    # Two-step shape filter
    OBIA_SHAPE_INDEX_IDEAL: float = 1.4
    OBIA_SHAPE_INDEX_HARD_MAX: float = 2.5
    OBIA_SHAPE_NDVI_DELTA_OVERRIDE: float = 0.35

    # WorldCover prior
    PRIORS_CACHE_DIR: str = "/tmp/autodetect_priors_cache"
    WORLDCOVER_YEAR: int = 2021
    USE_WORLDCOVER_PRIOR: bool = True
    USE_WEAK_WORLDCOVER_BARRIER: bool = True
    WC_EXCLUDE_CLASSES: tuple[int, ...] = (10, 20, 80, 90)
    WC_TREE_HARD_EXCLUSION: bool = Field(
        default=True,
        validation_alias=AliasChoices("WC_TREE_HARD_EXCLUSION", "WCTREEHARD_EXCLUSION"),
    )
    WC_MIN_CROPLAND_FRAC: float = 0.05
    WC_MAX_NONCROP_FRAC: float = 0.80

    # Tile skip thresholds
    TILE_MIN_OBS_COUNT: int = 4
    TILE_MAX_LOW_OBS_PCT: float = 0.5
    SKIP_WEAK_EDGE_TILES: bool = True
    MIN_VALID_SCENES_FOR_BOUNDARY: int = 3

    # Framework switches
    AUTO_DETECT_VERSION: int = 4
    FRAMEWORK_SAM_ENABLED: bool = False
    FRAMEWORK_SAM_FIELD_DET: bool = True
    FRAMEWORK_USE_WEAK_WORLDCOVER: bool = True
    FRAMEWORK_SKIP_WATERSHED_FOR_LARGEST: bool = True
    FEATURE_UNET_EDGE: bool = True
    FEATURE_SAM2_PRIMARY: bool = False
    FEATURE_S1_FUSION: bool = False
    FEATURE_SNIC_REFINE: bool = False
    FEATURE_ML_PRIMARY: bool = Field(
        default=True,
        validation_alias=AliasChoices("FEATURE_ML_PRIMARY", "FEATUREMLPRIMARY"),
    )
    ML_MODEL_PATH: str = Field(
        default="models/boundary_unet_v3_cpu.onnx",
        validation_alias=AliasChoices("ML_MODEL_PATH", "MLMODELPATH"),
    )
    ML_MODEL_NORM_STATS_PATH: str = Field(
        default="models/boundary_unet_v3_cpu.norm.json",
        validation_alias=AliasChoices("ML_MODEL_NORM_STATS_PATH", "MLMODELNORMSTATSPATH"),
    )
    ML_INFERENCE_DEVICE: str = Field(
        default="auto",
        validation_alias=AliasChoices("ML_INFERENCE_DEVICE", "MLINFERENCEDEVICE"),
    )
    ML_FALLBACK_ON_LOW_SCORE: bool = Field(
        default=True,
        validation_alias=AliasChoices("ML_FALLBACK_ON_LOW_SCORE", "MLFALLBACKONLOWSCORE"),
    )
    ML_SCORE_THRESHOLD: float = Field(
        default=0.35,
        validation_alias=AliasChoices("ML_SCORE_THRESHOLD", "MLSCORETHRESHOLD"),
    )
    ML_EXTENT_BIN_THRESHOLD: float = Field(
        default=0.42,
        validation_alias=AliasChoices("ML_EXTENT_BIN_THRESHOLD", "MLEXTENTBINTHRESHOLD"),
    )
    ML_EXTENT_CALIBRATION_ENABLED: bool = Field(
        default=True,
        validation_alias=AliasChoices("ML_EXTENT_CALIBRATION_ENABLED", "MLEXTENTCALIBRATIONENABLED"),
    )
    ML_TILE_SIZE: int = Field(
        default=512,
        validation_alias=AliasChoices("ML_TILE_SIZE", "MLTILESIZE"),
    )
    ML_OVERLAP: int = Field(
        default=128,
        validation_alias=AliasChoices("ML_OVERLAP", "MLOVERLAP"),
    )
    ML_USE_ONNX: bool = Field(
        default=True,
        validation_alias=AliasChoices("ML_USE_ONNX", "MLUSEONNX"),
    )
    ML_TTA_STANDARD_MODE: str = Field(
        default="flip2",
        validation_alias=AliasChoices("ML_TTA_STANDARD_MODE", "MLTTASTANDARDMODE"),
    )
    ML_TTA_QUALITY_MODE: str = Field(
        default="rotate4",
        validation_alias=AliasChoices("ML_TTA_QUALITY_MODE", "MLTTAQUALITYMODE"),
    )
    ML_MULTI_SCALE_STANDARD: bool = Field(
        default=False,
        validation_alias=AliasChoices("ML_MULTI_SCALE_STANDARD", "MLMULTISCALESTANDARD"),
    )
    ML_MULTI_SCALE_QUALITY: bool = Field(
        default=True,
        validation_alias=AliasChoices("ML_MULTI_SCALE_QUALITY", "MLMULTISCALEQUALITY"),
    )
    ML_MULTI_SCALE_AUX_SCALES: tuple[float, ...] = Field(
        default=(0.75,),
        validation_alias=AliasChoices("ML_MULTI_SCALE_AUX_SCALES", "MLMULTISCALEAUXSCALES"),
    )
    ML_FEATURE_PROFILE: str = Field(
        default="v2_16ch",
        validation_alias=AliasChoices("ML_FEATURE_PROFILE", "MLFEATUREPROFILE"),
    )
    MODEL_VERSION: str = Field(
        default="boundary_unet_v3_cpu",
        validation_alias=AliasChoices("MODEL_VERSION", "MODELVERSION"),
    )
    TRAIN_DATA_VERSION: str = Field(
        default="open_public_ru_v3_cpu",
        validation_alias=AliasChoices("TRAIN_DATA_VERSION", "TRAINDATAVERSION"),
    )
    FEATURE_STACK_VERSION: str = Field(
        default="v3_candidate_16ch_cpu",
        validation_alias=AliasChoices("FEATURE_STACK_VERSION", "FEATURESTACKVERSION"),
    )
    ONNX_OPSET_VERSION: int = Field(
        default=18,
        validation_alias=AliasChoices("ONNX_OPSET_VERSION", "ONNXOPSETVERSION"),
    )
    GEOMETRY_REFINE_PROFILE: str = Field(
        default="balanced",
        validation_alias=AliasChoices("GEOMETRY_REFINE_PROFILE", "GEOMETRYREFINEPROFILE"),
    )
    BOUNDARY_QUALITY_PROFILE: str = Field(
        default="quality_first",
        validation_alias=AliasChoices("BOUNDARY_QUALITY_PROFILE", "BOUNDARYQUALITYPROFILE"),
    )
    # Candidate model promotion: when a new model finishes training,
    # set ML_CANDIDATE_MODEL_PATH to try it alongside the baseline.
    ML_CANDIDATE_MODEL_PATH: str = Field(
        default="",
        validation_alias=AliasChoices("ML_CANDIDATE_MODEL_PATH", "MLCANDIDATEMODELPATH"),
    )
    ML_CANDIDATE_NORM_STATS_PATH: str = Field(
        default="",
        validation_alias=AliasChoices("ML_CANDIDATE_NORM_STATS_PATH", "MLCANDIDATENORMSTATSPATH"),
    )

    # Weak-label reliability gates for training-data generation.
    WEAK_LABEL_MIN_COVERAGE_PCT: float = Field(
        default=0.5,
        validation_alias=AliasChoices("WEAK_LABEL_MIN_COVERAGE_PCT", "WEAKLABELMINCOVERAGEPCT"),
    )
    WEAK_LABEL_MAX_FALLBACK_RATIO: float = Field(
        default=0.35,
        validation_alias=AliasChoices("WEAK_LABEL_MAX_FALLBACK_RATIO", "WEAKLABELMAXFALLBACKRATIO"),
    )
    WEAK_LABEL_OSM_OVERRIDE_ENABLED: bool = Field(
        default=True,
        validation_alias=AliasChoices("WEAK_LABEL_OSM_OVERRIDE_ENABLED", "WEAKLABELOSMOVERRIDEENABLED"),
    )
    WEAK_LABEL_TEMPORAL_OVERRIDE_ENABLED: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "WEAK_LABEL_TEMPORAL_OVERRIDE_ENABLED",
            "WEAKLABELTEMPORALOVERRIDEENABLED",
        ),
    )

    # Object classifier
    OBJECT_CLASSIFIER_PATH: str = Field(
        default="models/object_classifier.pkl",
        validation_alias=AliasChoices("OBJECT_CLASSIFIER_PATH", "OBJECTCLASSIFIERPATH"),
    )
    OBJECT_MIN_SCORE: float = Field(
        default=0.7,
        validation_alias=AliasChoices("OBJECT_MIN_SCORE", "OBJECTMINSCORE"),
    )
    USE_OBJECT_CLASSIFIER: bool = Field(
        default=True,
        validation_alias=AliasChoices("USE_OBJECT_CLASSIFIER", "USEOBJECTCLASSIFIER"),
    )

    # Postprocessing
    POST_ROAD_MAX_NDVI: float = 0.22
    POST_ROAD_NIR_MAX: float = 0.15
    POST_ROAD_NDBI_MIN: float = -0.10
    POST_ROAD_HARD_EXCLUSION: bool = True
    POST_ROAD_HOUGH_THRESHOLD: int = 45
    POST_ROAD_HOUGH_MIN_LEN: int = 5
    POST_ROAD_HOUGH_MAX_GAP: int = 3
    POST_ROAD_BUFFER_PX: int = 3

    POST_FOREST_NDVI_MIN: float = 0.65
    POST_FOREST_MIN_AREA_PX: int = 50

    POST_MORPH_CLOSE_RADIUS: int = 2

    POST_MERGE_BUFFER_PX: int = 4
    POST_MERGE_NDVI_DIFF_MAX: float = 0.12
    POST_MERGE_OVERLAP_MIN: float = 0.30
    POST_MERGE_BARRIER_RATIO: float = 0.08
    POST_MERGE_MAX_COMPONENTS: int = 2000

    POST_GROW_NDVI_RELAX: float = 0.11
    POST_GROW_MAX_ITERS: int = 10
    POST_BOUNDARY_DILATION_PX: int = 1
    POST_BOUNDARY_DILATION_MAX_PX: int = 3

    # Gap close edge threshold: holes with mean edge probability below this
    # value are filled.  0.15 was too conservative — most internal holes have
    # edge values 0.15-0.35.  Raised to 0.30 to fill more valid gaps.
    POST_GAP_EDGE_THRESHOLD: float = 0.30
    POST_GAP_CLOSE_MAX_HA: float = 1.0
    POST_LARGE_FIELD_RESCUE_ENABLED: bool = True
    POST_LARGE_FIELD_RESCUE_MIN_AREA_HA: float = 2.0

    POST_CONVEX_MIN_HA: float = 1.5
    POST_CONVEX_RATIO_MAX: float = 1.6
    POST_PREMERGE_MIN_PX: int = 10

    POST_MIN_FIELD_AREA_HA: float = 0.50   # relaxed from 0.70: catch smaller valid fields
    NORTH_POST_MIN_FIELD_AREA_HA: float = 0.35
    # Runtime value is overwritten from request resolution via get_px_area_m2().
    POST_PX_AREA_M2: int = 100
    WATERSHED_COMPACTNESS: float = 0.001
    WATERSHED_GRADIENT_EDGE_W: float = 0.5
    POST_GROW_BOUNDARY_STOP_THRESHOLD: float = 0.35

    # Tile quality controller thresholds (v4 pipeline)
    QC_MIN_COVERAGE_FRACTION: float = 0.20
    QC_MIN_VALID_SCENES: int = 3
    QC_MIN_EDGE_P90: float = 0.08
    QC_MIN_NDVI_TEMPORAL_STD: float = 0.04
    QC_BOUNDARY_RECOVERY_EDGE_P90: float = 0.12

    # Candidate ranker thresholds (v4 pipeline)
    CANDIDATE_MIN_SCORE: float = 0.18       # relaxed from 0.25: catch more valid candidates
    CANDIDATE_NMS_IOU_THRESHOLD: float = 0.46  # relaxed from 0.40: less aggressive duplicate suppression

    # V4 evalscript: use red-edge bands when available
    USE_REDEDGE_EVALSCRIPT: bool = True

    # Smart infill — max_hole_frac controls how large internal holes can be
    # relative to the component.  0.08 (8%) was too strict — real fields can
    # have 15-25% internal voids from trees, ponds, or sensor gaps.
    POST_INFILL_MAX_HOLE_FRAC: float = 0.25
    POST_INFILL_MIN_ALLOW_FRAC: float = 0.45
    POST_INFILL_APPLY_TO_LARGE: bool = True

    # OBIA hole filter
    OBIA_MAX_HOLE_FRAC: float = 0.16         # relaxed from 0.10: fields with ponds/trees were rejected
    OBIA_MAX_HOLE_NONCROP_FRAC: float = 0.40
    OBIA_MAX_INTERNAL_TREE_FRAC: float = 0.25  # relaxed from 0.20
    OBIA_MAX_INTERNAL_WATER_FRAC: float = 0.14  # relaxed from 0.10
    OBIA_RELAX_IF_ML_CONFIDENT: bool = True
    OBIA_RELAX_MIN_BOUNDARY_CONF: float = 0.60  # lower trigger threshold for relaxation
    OBIA_RELAX_SHAPE_MULTIPLIER: float = 1.55  # relaxed from 1.35
    OBIA_RELAX_HOLE_MULTIPLIER: float = 1.80   # relaxed from 1.50
    OBIA_RELAX_TREE_MULTIPLIER: float = 1.65   # relaxed from 1.50

    # Tile merge gate
    MERGE_TILE_MIN_IOU: float = 0.50
    MERGE_TILE_ONLY_IN_OVERLAP: bool = True

    # Boundary-first refinement (v4)
    OWT_EDGE_WEIGHT: float = 0.7
    OWT_NDVI_WEIGHT: float = 0.3
    OWT_SIGMA_ORIENTATION: float = 1.5
    OWT_SIGMA_STRENGTH: float = 0.8
    BOUNDARY_FILL_THRESH: float = 0.35
    BOUNDARY_FILL_MAX_REGIONS: int = 4000
    MERGE_BOUNDARY_THRESH: float = 0.25
    DEBUG_COMPARE_VERSIONS: bool = False

    # Active contour refinement
    SNAKE_REFINE_ENABLED: bool = True
    SNAKE_REFINE_MODE: str = "guarded"
    SNAKE_ALPHA: float = 0.015
    SNAKE_BETA: float = 10.0
    SNAKE_W_EDGE: float = -1.0
    SNAKE_MAX_PX_DIST: float = 15.0
    SNAKE_MAX_CENTROID_SHIFT_M: float = 6.0
    SNAKE_MIN_AREA_RATIO: float = 0.90
    SNAKE_MAX_AREA_RATIO: float = 1.12

    # Road barrier
    ROAD_OSM_ENABLED: bool = Field(
        default=False,
        validation_alias=AliasChoices("ROAD_OSM_ENABLED", "ROADOSMENABLED"),
    )
    ROAD_OSM_REQUEST_TIMEOUT_S: int = Field(
        default=45,
        validation_alias=AliasChoices("ROAD_OSM_REQUEST_TIMEOUT_S", "ROADOSMREQUESTTIMEOUTS"),
    )
    ROAD_OSM_OVERPASS_RATE_LIMIT: bool = Field(
        default=False,
        validation_alias=AliasChoices("ROAD_OSM_OVERPASS_RATE_LIMIT", "ROADOSMOVERPASSRATELIMIT"),
    )
    ROAD_OSM_TAGS: tuple[str, ...] = (
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "track",
        "path",
        "living_street",
        "road",
    )
    ROAD_OSM_BUFFER_DEFAULT_M: int = 12
    ROAD_OSM_BUFFER_MAP: dict[str, int] = {
        "motorway": 25,
        "trunk": 20,
        "primary": 15,
        "secondary": 12,
        "tertiary": 10,
        "unclassified": 8,
        "residential": 10,
        "service": 8,
        "track": 10,
        "path": 6,
    }
    ROAD_BARRIER_SOFT_RETRY_ENABLED: bool = True
    ROAD_BARRIER_SOFT_RETRY_MIN_REMOVED_RATIO: float = 0.15
    ROAD_CLASS_PROFILE: str = "major_minor"
    ROAD_MAJOR_BUFFER_PX: int = 2
    ROAD_MINOR_BUFFER_PX: int = 1
    ROAD_SNAP_REJECT_ENABLED: bool = True
    ROAD_SNAP_REJECT_BUFFER_PX: int = 2
    ROAD_SNAP_REJECT_MAX_OVERLAP_RATIO: float = 0.08
    ROAD_SOFT_BARRIER_MINOR_ONLY: bool = True
    ROAD_PARALLEL_EDGE_PENALTY_ENABLED: bool = True
    ROAD_DIRECTIONAL_PENALTY_ENABLED: bool = True
    REGION_PROFILE_DEBUG_LOGGING: bool = True
    REGION_PROFILE_RECORD_DIAGNOSTICS: bool = True

    # Regional boundary profiles
    SOUTH_PROFILE_ENABLED: bool = True
    SOUTH_ML_EXTENT_BIN_THRESHOLD: float = 0.34
    SOUTH_POST_GROW_MAX_ITERS: int = 6
    SOUTH_POST_GROW_NDVI_RELAX: float = 0.17
    SOUTH_CLEAN_ROLLBACK_MIN_AREA_RATIO: float = 0.75
    SOUTH_GAP_CLOSE_MAX_HA: float = 1.00
    SOUTH_POST_MIN_FIELD_AREA_HA: float = 0.12
    SOUTH_SKIP_WATERSHED_LARGE_COMPONENTS: bool = True
    SOUTH_WATERSHED_SKIP_MIN_AREA_HA: float = 4.0
    SOUTH_COMPONENT_BRIDGE_ENABLED: bool = True
    SOUTH_COMPONENT_BRIDGE_MAX_COMPONENTS: int = 400
    SOUTH_COMPONENT_BRIDGE_MAX_GAP_PX: int = 3
    SOUTH_COMPONENT_BRIDGE_MAX_NDVI_DIFF: float = 0.08
    SOUTH_COMPONENT_BRIDGE_MAX_BOUNDARY_PROB: float = 0.45
    SOUTH_MERGE_MIN_OVERLAP_RATIO: float = 0.03
    SOUTH_MAX_COMPONENTS_PER_FIELD_TARGET: int = 3

    NORTH_PROFILE_ENABLED: bool = True
    NORTH_ML_EXTENT_BIN_THRESHOLD: float = 0.40
    NORTH_POST_GROW_MAX_ITERS: int = 3
    NORTH_CLEAN_ROLLBACK_MIN_AREA_RATIO: float = 0.92
    NORTH_MERGE_BOUNDARY_THRESH: float = 0.20
    NORTH_GAP_CLOSE_MAX_HA: float = 0.35
    NORTH_VECTORIZE_SIMPLIFY_TOL_M: float = 0.25
    NORTH_BOUNDARY_SMOOTH_SIMPLIFY_TOL_M: float = 0.50
    NORTH_POST_BUFFER_SMOOTH_M: float = 0.0
    NORTH_TOPOLOGY_SIMPLIFY_ENABLED: bool = False
    NORTH_STAGE_ROLLBACK_MIN_AREA_RATIO: float = 0.95
    NORTH_STAGE_ROLLBACK_MAX_CENTROID_SHIFT_M: float = 5.0
    NORTH_BOUNDARY_OUTER_DILATION_PX: int = 1
    NORTH_WATERSHED_WEAKEN_ENABLED: bool = True

    # Temporal stack
    TEMPORAL_CLOUD_SCL_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 8, 9, 10, 11)
    TEMPORAL_YEARS_BACK: int = 1
    TEMPORAL_BEST_N_SCENES: int = 8
    TEMPORAL_CLOUD_MAX_PCT: int = 20
    TEMPORAL_SCL_INVALID: tuple[int, ...] = (0, 1, 2, 3, 8, 9, 10, 11)
    TEMPORAL_NDVI_VALID_MIN: float = 0.05

    # NDVI phenology priors
    PHENO_FIELD_NDVI_STD_MIN: float = 0.15
    PHENO_GRASS_NDVI_STD_MAX: float = 0.12
    PHENO_FOREST_NDVI_STD_MAX: float = 0.05
    PHENO_FIELD_CANDIDATE_NDVI_OFFSET: float = 0.10
    PHENO_FIELD_MAX_NDVI_MIN: float = 0.45
    PHENO_FIELD_MAX_NDVI_MAX: float = 0.62
    PHENO_GRASS_MAX_NDVI_MAX: float = 0.55
    PHENO_FOREST_MAX_NDVI_MIN: float = 0.65
    PHENO_FOREST_DECID_MAX_NDVI_MIN: float = 0.60
    PHENO_FOREST_DECID_NDVI_STD_MIN: float = 0.07
    PHENO_FOREST_DECID_NDVI_STD_MAX: float = 0.18

    # Hybrid SAM pipeline (kept optional to preserve the default runtime path)
    MODEL_DIR: str = "/app/models"
    UNET_EDGE_MODEL: str = Field(
        default="/app/models/unet_edge_best.pth",
        validation_alias=AliasChoices("UNET_EDGE_MODEL", "UNETEDGEMODEL"),
    )
    UNET_EDGE_THRESHOLD: float = Field(
        default=0.5,
        validation_alias=AliasChoices("UNET_EDGE_THRESHOLD", "UNETEDGETHRESHOLD"),
    )
    UNET_DEVICE: str = Field(
        default="cpu",
        validation_alias=AliasChoices("UNET_DEVICE", "UNETDEVICE"),
    )
    SAM2_CHECKPOINT: str = Field(
        default="/app/models/sam2_hiera_large.pt",
        validation_alias=AliasChoices("SAM2_CHECKPOINT", "SAMCHECKPOINT"),
    )
    SAM_POINT_SPACING: int = Field(
        default=20,
        validation_alias=AliasChoices("SAM_POINT_SPACING", "SAMPOINTSPACING"),
    )
    SAM_PRED_IOU_THRESHOLD: float = Field(
        default=0.85,
        validation_alias=AliasChoices("SAM_PRED_IOU_THRESHOLD", "SAMPREDIOUTHRESH"),
    )
    S1_ENABLED: bool = Field(
        default=False,
        validation_alias=AliasChoices("S1_ENABLED", "S1ENABLED"),
    )
    S1_ACQUISITION_MODE: str = Field(
        default="IW",
        validation_alias=AliasChoices("S1_ACQUISITION_MODE", "S1ACQUISITIONMODE"),
    )
    S1_POLARIZATION: str = Field(
        default="DV",
        validation_alias=AliasChoices("S1_POLARIZATION", "S1POLARIZATION"),
    )
    S1_LEE_FILTER_ENABLE: bool = Field(
        default=True,
        validation_alias=AliasChoices("S1_LEE_FILTER_ENABLE", "S1LEEFILTERENABLE"),
    )
    S1_LEE_WINDOW_SIZE: int = Field(
        default=5,
        validation_alias=AliasChoices("S1_LEE_WINDOW_SIZE", "S1LEEWINDOWSIZE"),
    )
    SNIC_REFINE_ENABLED: bool = Field(
        default=False,
        validation_alias=AliasChoices("SNIC_REFINE_ENABLED", "SNICREFINEENABLED"),
    )
    SNIC_N_SEGMENTS: int = Field(
        default=1000,
        validation_alias=AliasChoices("SNIC_N_SEGMENTS", "SNICNSEGMENTS"),
    )
    SNIC_COMPACTNESS: float = Field(
        default=0.01,
        validation_alias=AliasChoices("SNIC_COMPACTNESS", "SNICCOMPACTNESS"),
    )
    SNIC_MERGE_NDVI_THRESH: float = Field(
        default=0.05,
        validation_alias=AliasChoices("SNIC_MERGE_NDVI_THRESH", "SNICMERGENDVITHRESH"),
    )
    SAM_FIELD_DET: bool = True
    SAM_ENABLED: bool = False
    SAM_MODEL_TYPE: str = "vit_b"
    SAM_CHECKPOINT_PATH: str = "checkpoints/sam_vit_b_01ec64.pth"
    SAM_POINTS_PER_SIDE: int = 16
    SAM_PRED_IOU_THRESH: float = 0.88
    SAM_STABILITY_SCORE: float = 0.95
    SAM_BOX_NMS_THRESH: float = 0.70
    SAM_MIN_MASK_REGION_AREA: int = 500
    SAM_OUTPUT_DIR: str = "debug_runs/{run_id}/sam"

    HYBRID_MERGE_MIN_IOU: float = 0.20
    HYBRID_SAM_MIN_CROP_RATIO: float = 0.30
    HYBRID_SAM_MAX_FOREST_RATIO: float = 0.40
    HYBRID_SAM_MAX_WATER_RATIO: float = 0.20
    HYBRID_SNAP_TOLERANCE_M: float = 15.0
    SAM_RUNTIME_POLICY: str = Field(
        default="safe_optional",
        validation_alias=AliasChoices("SAM_RUNTIME_POLICY", "SAMRUNTIMEPOLICY"),
    )
    SAM_MAX_TILE_PIXELS: int = 600_000
    SAM_MAX_COMPONENTS: int = 64
    SAM_MAX_CANDIDATE_COVERAGE_PCT: float = 18.0
    SAM_MAX_EST_MEMORY_MB: int = 2200
    SAM_TIMEOUT_S: int = 45
    SAM_USE_CROP_BOXES: bool = True
    SAM_CROP_BOX_PADDING_PX: int = 24
    SAM_SUBPROCESS_ISOLATION: bool = True
    SAM_SUBPROCESS_KILL_ON_LIMIT: bool = True

    # Hydro-aware boundary handling
    HYDRO_BOUNDARY_PROFILE: str = Field(
        default="water_aware",
        validation_alias=AliasChoices("HYDRO_BOUNDARY_PROFILE", "HYDROBOUNDARYPROFILE"),
    )
    HYDRO_OPEN_WATER_NDWI: float = 0.14
    HYDRO_OPEN_WATER_MNDWI: float = 0.08
    HYDRO_SEASONAL_WET_NDWI: float = 0.06
    HYDRO_SEASONAL_WET_MNDWI: float = 0.02
    HYDRO_RIPARIAN_BUFFER_PX: int = 2
    HYDRO_RIPARIAN_SOFT_MODE: bool = True
    HYDRO_RIPARIAN_HARD_EXCLUSION_RATIO_MAX: float = 0.65
    HYDRO_FIELD_NEAR_WATER_RESCUE_ENABLED: bool = True
    HYDRO_FIELD_NEAR_WATER_MIN_NDVI_MAX: float = 0.40
    HYDRO_FIELD_NEAR_WATER_MIN_NDVI_STD: float = 0.12

    # Boundary geometry guardrails
    BOUNDARY_OUTER_EXPAND_WATER_AWARE: bool = True
    BOUNDARY_OUTER_EXPAND_MAX_PX: int = 4
    BOUNDARY_OUTER_EXPAND_NEAR_WATER_MAX_PX: int = 2
    BOUNDARY_OUTER_EXPAND_NEAR_ROAD_MAX_PX: int = 1
    BOUNDARY_VECTOR_SNAP_GRID_M: float = 1.0
    BOUNDARY_KEEP_HOLES_MIN_AREA_M2: float = 300.0
    BOUNDARY_FILL_HOLES_MAX_AREA_M2: float = 800.0

    # Vector smoothing
    ALPHA_SHAPE_ALPHA: float | None = None
    ALPHA_SHAPE_COVERAGE: float = 0.90
    ALPHA_SHAPE_DOWNSAMPLE: int = 3
    ALPHA_MIN_FIELD_PX: int = 10
    POST_MERGE_SMOOTH: bool = True
    VECTORIZE_SIMPLIFY_TOL_M: float = 0.8
    BOUNDARY_SMOOTH_SIMPLIFY_TOL_M: float = 1.2
    POST_SIMPLIFY_TOLERANCE: float = 4.0
    POST_BUFFER_SMOOTH_M: float = 2.0
    TOPOLOGY_SIMPLIFY_ENABLED: bool = False
    TOPOLOGY_SIMPLIFY_TOL_M: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _normalize_validation_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for field_name, field in cls.model_fields.items():
            validation_alias = field.validation_alias
            aliases: list[str] = []
            if isinstance(validation_alias, AliasChoices):
                aliases.extend(choice for choice in validation_alias.choices if isinstance(choice, str))
            elif isinstance(validation_alias, str):
                aliases.append(validation_alias)
            for alias in aliases:
                if alias != field_name and alias in normalized and field_name not in normalized:
                    normalized[field_name] = normalized.pop(alias)
        return normalized

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any):
        return super().model_validate(cls._normalize_validation_aliases(obj), *args, **kwargs)

    @field_validator("ALLOW_SYNTHETIC_DATA", mode="before")
    @classmethod
    def _validate_allow_synthetic_data(cls, value: Any) -> bool:
        return parse_env_bool(value, field_name="ALLOW_SYNTHETIC_DATA")

    @field_validator(
        "WEATHER_ALLOW_STALE_ON_FAILURE",
        "AUTH_REQUIRED",
        "AUTH_BOOTSTRAP_ENABLED",
        "MLFLOW_ENABLED",
        mode="before",
    )
    @classmethod
    def _validate_new_booleans(cls, value: Any, info) -> bool:
        return parse_env_bool(value, field_name=info.field_name)

    @field_validator("GEOMETRY_REFINE_PROFILE")
    @classmethod
    def _validate_geometry_refine_profile(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"strict", "balanced", "recall_first"}
        if token not in allowed:
            raise ValueError(f"GEOMETRY_REFINE_PROFILE must be one of {sorted(allowed)}")
        return token

    @field_validator("BOUNDARY_QUALITY_PROFILE")
    @classmethod
    def _validate_boundary_quality_profile(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"balanced", "quality_first"}
        if token not in allowed:
            raise ValueError(f"BOUNDARY_QUALITY_PROFILE must be one of {sorted(allowed)}")
        return token

    @field_validator("DATE_SELECTION_PROFILE")
    @classmethod
    def _validate_date_selection_profile(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"coverage_only", "adaptive_region"}
        if token not in allowed:
            raise ValueError(f"DATE_SELECTION_PROFILE must be one of {sorted(allowed)}")
        return token

    @field_validator("REGION_PROFILE_MODE")
    @classmethod
    def _validate_region_profile_mode(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"auto_only"}
        if token not in allowed:
            raise ValueError(f"REGION_PROFILE_MODE must be one of {sorted(allowed)}")
        return token

    @field_validator("MODE")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"production", "research"}
        if token not in allowed:
            raise ValueError(f"MODE must be one of {sorted(allowed)}")
        return token

    @field_validator("ML_FEATURE_PROFILE")
    @classmethod
    def _validate_ml_feature_profile(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"v1_18ch", "v2_16ch"}
        if token not in allowed:
            raise ValueError(f"ML_FEATURE_PROFILE must be one of {sorted(allowed)}")
        return token

    @field_validator("SNAKE_REFINE_MODE")
    @classmethod
    def _validate_snake_refine_mode(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"off", "guarded", "aggressive"}
        if token not in allowed:
            raise ValueError(f"SNAKE_REFINE_MODE must be one of {sorted(allowed)}")
        return token

    @field_validator("SAM_RUNTIME_POLICY")
    @classmethod
    def _validate_sam_runtime_policy(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"disabled", "safe_optional", "always_on"}
        if token not in allowed:
            raise ValueError(f"SAM_RUNTIME_POLICY must be one of {sorted(allowed)}")
        return token

    @field_validator("HYDRO_BOUNDARY_PROFILE")
    @classmethod
    def _validate_hydro_boundary_profile(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"off", "balanced", "water_aware"}
        if token not in allowed:
            raise ValueError(f"HYDRO_BOUNDARY_PROFILE must be one of {sorted(allowed)}")
        return token

    @field_validator("ROAD_CLASS_PROFILE")
    @classmethod
    def _validate_road_class_profile(cls, value: str) -> str:
        token = str(value).strip().lower()
        allowed = {"legacy", "major_minor"}
        if token not in allowed:
            raise ValueError(f"ROAD_CLASS_PROFILE must be one of {sorted(allowed)}")
        return token


def get_bool_env_alias_map() -> dict[str, tuple[str, ...]]:
    """Return boolean Settings fields mapped to accepted env aliases."""
    alias_map: dict[str, tuple[str, ...]] = {}
    for name, field in Settings.model_fields.items():
        if field.annotation is not bool:
            continue
        aliases: list[str] = [name]
        validation_alias = field.validation_alias
        if isinstance(validation_alias, AliasChoices):
            for choice in validation_alias.choices:
                if isinstance(choice, str):
                    aliases.append(choice)
        elif isinstance(validation_alias, str):
            aliases.append(validation_alias)
        alias_map[name] = tuple(dict.fromkeys(aliases))
    return alias_map



@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_px_area_m2(resolution_m: float) -> float:
    """Return area of one pixel in square meters for a given raster resolution."""
    return float(resolution_m) ** 2


def get_adaptive_season_window(lat: float, settings: Settings) -> tuple[str, str]:
    """Return region-aware Sentinel-2 season boundaries as MM-DD strings.

    Southern AOI get an earlier start to preserve winter crops / early spring.
    Northern AOI keep later summer / autumn availability because green-up is delayed.
    """
    from core.region import resolve_region_band

    region_band = resolve_region_band(
        float(lat),
        south_max=float(getattr(settings, "REGION_LAT_SOUTH_MAX", 48.0)),
        north_min=float(getattr(settings, "REGION_LAT_NORTH_MIN", 57.0)),
    )
    if region_band == "south":
        return "03-01", "11-01"
    if region_band == "north":
        return "04-25", "10-25"
    if float(lat) >= 53.0:
        return "04-05", "10-20"
    return str(settings.S2_SEASON_START), str(settings.S2_SEASON_END)


def get_adaptive_pheno_thresholds(lat: float, settings: Settings) -> dict[str, float]:
    """Return phenological threshold overrides based on AOI centroid latitude.

    Northern regions have lower peak NDVI and less seasonal contrast,
    so thresholds must be relaxed to detect agricultural fields.
    """
    from core.region import resolve_region_band

    overrides: dict[str, float] = {}
    region_band = resolve_region_band(
        float(lat),
        south_max=float(getattr(settings, "REGION_LAT_SOUTH_MAX", 48.0)),
        north_min=float(getattr(settings, "REGION_LAT_NORTH_MIN", 57.0)),
    )
    if region_band == "north":  # Northern Russia (LO, Pskov, Tver, Perm)
        overrides["PHENO_NDVI_CROP_MAX"] = 0.50
        overrides["PHENO_DELTA_CROP"] = 0.20
        overrides["PHENO_NDVI_CROP_MIN"] = 0.15
        overrides["PHENO_NDVI_FOREST_MIN"] = 0.40
        overrides["PHENO_NDVI_GRASS_MEAN"] = 0.25
        overrides["PHENO_DELTA_GRASS"] = 0.12
        overrides["PHENO_FIELD_MAX_NDVI_MIN"] = 0.35
        overrides["PHENO_FIELD_MAX_NDVI_MAX"] = 0.55
    elif region_band == "central" and float(lat) > 53.0:  # Upper central belt
        overrides["PHENO_NDVI_CROP_MAX"] = 0.58
        overrides["PHENO_DELTA_CROP"] = 0.25
    elif region_band == "south":  # Southern Russia (Krasnodar, Stavropol, Rostov)
        overrides["PHENO_NDVI_CROP_MAX"] = 0.70
        overrides["PHENO_DELTA_CROP"] = 0.35
    return overrides
