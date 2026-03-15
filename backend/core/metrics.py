from prometheus_client import Counter, Gauge, Histogram

DETECT_DURATION = Histogram(
    "autodetect_total_duration_seconds",
    "Total duration of field detection pipeline",
    ["aoi_run_id"],
)

STEP_DURATION = Histogram(
    "autodetect_step_duration_seconds",
    "Duration of each pipeline step",
    ["step"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

DETECT_REQUESTS = Counter(
    "autodetect_requests_total",
    "Total number of detect requests",
)

TILES_PROCESSED = Counter(
    "autodetect_tiles_processed_total",
    "Total tiles processed",
)

CACHE_HITS = Counter(
    "sentinel_cache_hits_total",
    "Sentinel Hub cache hits",
)

CACHE_MISSES = Counter(
    "sentinel_cache_misses_total",
    "Sentinel Hub cache misses",
)

ACTIVE_RUNS = Gauge(
    "autodetect_active_runs",
    "Number of currently running detections",
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path", "status"],
)

UNET_INFERENCE_TIME = Histogram(
    "unet_inference_seconds",
    "U-Net edge prediction duration",
)

SAM2_INFERENCE_TIME = Histogram(
    "sam2_inference_seconds",
    "SAM2 primary segmentation duration",
)

S1_FETCH_TIME = Histogram(
    "s1_fetch_seconds",
    "Sentinel-1 fetch duration",
)

GPU_MEMORY_USAGE = Gauge(
    "gpu_memory_mb",
    "Approximate GPU memory usage in MB",
)
