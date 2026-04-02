from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class AoiType(str, Enum):
    point_radius = "point_radius"
    bbox = "bbox"
    polygon = "polygon"


class AoiInput(BaseModel):
    type: AoiType = AoiType.point_radius
    lat: float = Field(58.689077, ge=-90, le=90)
    lon: float = Field(29.892103, ge=-180, le=180)
    radius_km: float = Field(15.0, gt=0, le=40)
    bbox: list[float] | None = None
    polygon: list[list[float]] | None = None

    @field_validator("radius_km")
    @classmethod
    def validate_radius(cls, v: float) -> float:
        if v > 40:
            raise ValueError("radius_km must be <= 40")
        return v

    @model_validator(mode="after")
    def validate_aoi_payload(self):
        if self.type == AoiType.point_radius:
            return self

        if self.type == AoiType.bbox:
            if self.bbox is None:
                raise ValueError("bbox must be provided when aoi.type='bbox'")
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain 4 values: [min_lon, min_lat, max_lon, max_lat]")
            minx, miny, maxx, maxy = self.bbox
            if minx >= maxx or miny >= maxy:
                raise ValueError("bbox must satisfy min < max for both axes")
            return self

        if self.type == AoiType.polygon:
            if not self.polygon:
                raise ValueError("polygon must be provided when aoi.type='polygon'")
            if len(self.polygon) < 3:
                raise ValueError("polygon must contain at least 3 coordinate pairs")
            return self

        raise ValueError(f"unsupported aoi.type: {self.type}")


class TimeRange(BaseModel):
    start_date: date = Field(default_factory=lambda: date(2025, 5, 1))
    end_date: date = Field(default_factory=lambda: date(2025, 8, 31))

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


class DetectRequest(BaseModel):
    aoi: AoiInput = Field(default_factory=AoiInput)
    time_range: TimeRange = Field(default_factory=TimeRange)
    resolution_m: int = Field(10, ge=10, le=60)
    max_cloud_pct: int = Field(40, ge=0, le=100)
    target_dates: int = Field(7, ge=1, le=12)
    min_field_area_ha: float = Field(0.5, ge=0.1, le=10)
    seed_mode: str = Field("auto", pattern="^(auto|grid|custom|edges|distance)$")
    seed_points: list[list[float]] | None = None
    debug: bool = False
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_seed_mode(self):
        if self.seed_mode == "custom":
            if not self.seed_points:
                raise ValueError("seed_points must be provided when seed_mode='custom'")
            if len(self.seed_points) == 0:
                raise ValueError("seed_points must contain at least one point")
            for point in self.seed_points:
                if len(point) != 2:
                    raise ValueError("each seed point must contain [lon, lat]")
        return self


class DetectResponse(BaseModel):
    aoi_run_id: UUID
    status: str


class DetectPreflightResponse(BaseModel):
    preset: str | None = None
    budget_ok: bool
    hard_block: bool = False
    estimated_tiles: int
    estimated_ram_mb: int | None = None
    estimated_runtime_class: str
    pipeline_profile: str | None = None
    preview_only: bool = False
    output_mode: str | None = None
    operational_eligible: bool = True
    max_radius_km: int | None = None
    recommended_radius_km: int | None = None
    enabled_stages: list[str] = Field(default_factory=list)
    expected_dates_ok: bool = True
    regional_profile: str | None = None
    s1_planned: bool = False
    tta_mode: str | None = None
    budget_reason: str | None = None
    recommended_preset: str | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    season_window: dict[str, str] = Field(default_factory=dict)
    region_band: str | None = None
    launch_tier: str | None = None
    review_required: bool = False
    review_reason: str | None = None
    review_reason_code: str | None = None
    review_reason_params: dict[str, Any] = Field(default_factory=dict)


class FreshnessMeta(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str | None = None
    fetched_at: str | None = None
    cache_written_at: str | None = None
    freshness_state: str = "unknown"
    source_published_at: str | None = None
    model_version: str | None = None
    dataset_version: str | None = None


class BuildInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    app_version: str | None = None
    model_version: str | None = None
    train_data_version: str | None = None
    yield_model_version: str | None = None
    feature_stack_version: str | None = None


class ModelTruthInfo(BaseModel):
    head_count: int = 3
    heads: list[str] = Field(default_factory=list)
    tta_standard: str | None = None
    tta_quality: str | None = None
    retrain_description: str | None = None


class GeometrySummary(BaseModel):
    head_count: int = 3
    heads: list[str] = Field(default_factory=list)
    tta_standard: str | None = None
    tta_quality: str | None = None
    retrain_description: str | None = None
    geometry_confidence: float | None = None
    tta_consensus: float | None = None
    boundary_uncertainty: float | None = None
    tta_extent_disagreement: float | None = None
    tta_boundary_disagreement: float | None = None
    uncertainty_source: str | None = None
    watershed_applied: bool | None = None
    watershed_skipped_reason: str | None = None
    watershed_rollback_reason: str | None = None
    components_after_grow: int | None = None
    components_after_gap_close: int | None = None
    components_after_infill: int | None = None
    components_after_merge: int | None = None
    components_after_watershed: int | None = None
    split_score_p50: float | None = None
    split_score_p90: float | None = None
    tiles_summarized: int = 0


class BootstrapAuthResponse(BaseModel):
    enabled: bool = False
    bootstrap_admin_email: str | None = None
    bootstrap_org_slug: str | None = None
    bootstrap_org_name: str | None = None


class RunStatus(BaseModel):
    aoi_run_id: UUID
    status: str
    progress: int
    progress_pct: float = 0.0
    error_msg: str | None = None
    stage_code: str | None = None
    stage_label: str | None = None
    stage_detail: str | None = None
    stage_detail_code: str | None = None
    stage_detail_params: dict[str, Any] = Field(default_factory=dict)
    stage_progress_pct: float | None = None
    tile_progress_pct: float | None = None
    started_at: str | None = None
    updated_at: str | None = None
    last_heartbeat_ts: str | None = None
    stale_running: bool = False
    queue_ahead: int = 0
    blocking_run_id: UUID | None = None
    blocking_status: str | None = None
    elapsed_s: int | None = None
    estimated_remaining_s: int | None = None
    qc_mode: str | None = None
    processing_profile: str | None = None
    pipeline_profile: str | None = None
    preview_only: bool = False
    output_mode: str | None = None
    operational_eligible: bool = True
    max_radius_km: int | None = None
    recommended_radius_km: int | None = None
    enabled_stages: list[str] = Field(default_factory=list)
    candidate_branch_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    candidate_reject_summary: dict[str, int] = Field(default_factory=dict)
    candidates_total: int = 0
    candidates_kept: int = 0
    geometry_summary: GeometrySummary | None = None
    runtime: dict[str, Any] | None = None


class RunResult(BaseModel):
    aoi_run_id: UUID
    status: str
    progress: int
    progress_pct: float = 0.0
    error_msg: str | None = None
    stage_code: str | None = None
    stage_label: str | None = None
    stage_detail: str | None = None
    stage_detail_code: str | None = None
    stage_detail_params: dict[str, Any] = Field(default_factory=dict)
    stage_progress_pct: float | None = None
    tile_progress_pct: float | None = None
    started_at: str | None = None
    updated_at: str | None = None
    last_heartbeat_ts: str | None = None
    stale_running: bool = False
    queue_ahead: int = 0
    blocking_run_id: UUID | None = None
    blocking_status: str | None = None
    elapsed_s: int | None = None
    estimated_remaining_s: int | None = None
    qc_mode: str | None = None
    processing_profile: str | None = None
    pipeline_profile: str | None = None
    preview_only: bool = False
    output_mode: str | None = None
    operational_eligible: bool = True
    max_radius_km: int | None = None
    recommended_radius_km: int | None = None
    enabled_stages: list[str] = Field(default_factory=list)
    candidate_branch_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    candidate_reject_summary: dict[str, int] = Field(default_factory=dict)
    candidates_total: int = 0
    candidates_kept: int = 0
    geometry_summary: GeometrySummary | None = None
    runtime: dict[str, Any] | None = None
    geojson: dict[str, Any] | None = None


class RunSummary(BaseModel):
    id: UUID
    status: str
    progress: int
    created_at: str | None = None
    preset: str | None = None
    aoi: dict[str, Any] | None = None
    use_sam: bool = False
    resolution_m: int | None = None
    target_dates: int | None = None
    qc_mode: str | None = None
    processing_profile: str | None = None
    candidates_total: int = 0
    candidates_kept: int = 0


class RunListResponse(BaseModel):
    runs: list[RunSummary]


class DetectionCandidateInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    tile_diagnostic_id: int | None = None
    tile_index: int | None = None
    tile_id: str | None = None
    field_id: UUID | None = None
    branch: str
    area_m2: float | None = None
    score: float
    rank: int | None = None
    kept: bool
    reject_reason: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    model_version: str | None = None
    created_at: str | None = None
    geometry: dict[str, Any] | None = None


class DetectionCandidatesResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    run_id: UUID
    total: int
    candidates: list[DetectionCandidateInfo]


class LayerInfo(BaseModel):
    id: str
    name: str
    unit: str | None
    range_min: float | None
    range_max: float | None
    source: str
    description: str | None
    freshness: FreshnessMeta | None = None


class LayersListResponse(BaseModel):
    layers: list[LayerInfo]


class FieldSummary(BaseModel):
    id: UUID
    aoi_run_id: UUID
    area_m2: float
    perimeter_m: float
    quality_score: float | None = None
    source: str
    created_at: datetime | None = None


class FieldsListResponse(BaseModel):
    fields: list[FieldSummary]


class ManualFieldCreateRequest(BaseModel):
    geometry: dict[str, Any]
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ManualFieldResponse(BaseModel):
    field: FieldSummary


class FieldDeleteResponse(BaseModel):
    field_id: UUID
    aoi_run_id: UUID
    deleted: bool = True


class WeatherCurrentResponse(BaseModel):
    latitude: float
    longitude: float
    observed_at: str
    provider: str
    cached: bool
    error: str | None = None
    temperature_c: float | None = None
    apparent_temperature_c: float | None = None
    precipitation_mm: float | None = None
    wind_speed_m_s: float | None = None
    u_wind_10m: float | None = None
    v_wind_10m: float | None = None
    wind_direction_deg: float | None = None
    humidity_pct: float | None = None
    cloud_cover_pct: float | None = None
    pressure_hpa: float | None = None
    soil_moisture: float | None = None
    freshness: FreshnessMeta | None = None


class WeatherForecastDay(BaseModel):
    date: str
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    temp_mean_c: float | None = None
    precipitation_mm: float | None = None
    wind_speed_m_s: float | None = None
    cloud_cover_pct: float | None = None


class WeatherForecastResponse(BaseModel):
    latitude: float
    longitude: float
    provider: str
    days: int
    forecast: list[WeatherForecastDay]
    error: str | None = None
    freshness: FreshnessMeta | None = None


class ComponentStatus(BaseModel):
    status: str
    detail: Any


class SystemStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    timestamp: str
    components: dict[str, ComponentStatus]
    runs: dict[str, int]
    build: BuildInfo | None = None
    model_truth: ModelTruthInfo | None = None
    freshness: FreshnessMeta | None = None


class BootstrapResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    timestamp: str
    components: dict[str, ComponentStatus]
    build: BuildInfo
    auth: BootstrapAuthResponse
    model_truth: ModelTruthInfo | None = None
    freshness: FreshnessMeta | None = None


class StorageAuthPromptResponse(BaseModel):
    provider_label: str
    title: str
    description: str
    steps: list[str] = Field(default_factory=list)
    login_url: str | None = None
    suggested_command: str | None = None


class StorageConfigRequest(BaseModel):
    storage_mode: str = Field("local", pattern="^(local|cloud)$")
    cloud_url: str | None = None

    @model_validator(mode="after")
    def validate_cloud_mode(self):
        if self.storage_mode == "cloud" and not str(self.cloud_url or "").strip():
            raise ValueError("cloud_url must be provided when storage_mode='cloud'")
        return self


class StorageConfigResponse(BaseModel):
    storage_mode: str
    cloud_url: str | None = None
    provider: str | None = None
    provider_label: str | None = None
    status: str
    message: str | None = None
    auth_state: str = "not_required"
    auth_required: bool = False
    rclone_available: bool = False
    remote_name: str | None = None
    hierarchy_ready: bool = False
    workspace_root: str | None = None
    workspace_folders: list[str] = Field(default_factory=list)
    auth_prompt: StorageAuthPromptResponse | None = None
    updated_at: str | None = None


class SatelliteBrowseResponse(BaseModel):
    bbox: list[float]
    width: int
    height: int
    status: str = "ready"
    requested_date: str | None = None
    resolved_date: str | None = None
    requested_window: dict[str, str] | None = None
    provider: str
    provider_account: str | None = None
    failover_level: int = 0
    cloud_cover_pct: float | None = None
    valid_coverage_pct: float | None = None
    image_base64: str | None = None
    freshness: FreshnessMeta | None = None


class AuthLoginRequest(BaseModel):
    email: str
    password: str
    organization_slug: str | None = None


class AuthRefreshRequest(BaseModel):
    refresh_token: str


class AuthUserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    organization_id: str
    organization_slug: str
    organization_name: str
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUserResponse


class CropInfoResponse(BaseModel):
    id: int
    code: str
    name: str
    category: str
    yield_baseline_kg_ha: float
    ndvi_target: float
    base_temp_c: float
    description: str | None = None


class CropListResponse(BaseModel):
    crops: list[CropInfoResponse]


class YieldPredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    field_id: str
    crop: dict[str, Any]
    prediction_date: str
    estimated_yield_kg_ha: float
    confidence: float
    confidence_tier: str | None = None
    model_version: str
    details: dict[str, Any]
    input_features: dict[str, Any] = Field(default_factory=dict)
    explanation: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    prediction_interval: dict[str, float | None] = Field(default_factory=dict)
    model_applicability: dict[str, Any] = Field(default_factory=dict)
    training_domain: dict[str, Any] = Field(default_factory=dict)
    feature_coverage: dict[str, Any] = Field(default_factory=dict)
    crop_suitability: dict[str, Any] = Field(default_factory=dict)
    crop_hint: dict[str, Any] = Field(default_factory=dict)
    seasonal_series: dict[str, Any] = Field(default_factory=dict)
    phenology: dict[str, Any] = Field(default_factory=dict)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    water_balance: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    history_trend: dict[str, Any] = Field(default_factory=dict)
    forecast_curve: dict[str, Any] = Field(default_factory=dict)
    management_zone_summary: dict[str, Any] = Field(default_factory=dict)
    driver_breakdown: list[dict[str, Any]] = Field(default_factory=list)
    geometry_quality_impact: dict[str, Any] = Field(default_factory=dict)
    support_reason: str | None = None
    support_reason_code: str | None = None
    support_reason_params: dict[str, Any] = Field(default_factory=dict)
    operational_tier: str | None = None
    review_required: bool = False
    review_reason: str | None = None
    review_reason_code: str | None = None
    review_reason_params: dict[str, Any] = Field(default_factory=dict)
    freshness: FreshnessMeta | None = None


class AsyncJobSubmitResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    progress_pct: float = 0.0
    stage_code: str | None = None
    stage_label: str | None = None
    stage_detail: str | None = None
    stage_detail_code: str | None = None
    stage_detail_params: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    updated_at: str | None = None
    elapsed_s: int | None = None
    estimated_remaining_s: int | None = None
    logs: list[str] = Field(default_factory=list)
    error_msg: str | None = None


class AsyncJobStatusResponse(BaseModel):
    task_id: str
    job_type: str
    status: str
    progress: int = 0
    progress_pct: float = 0.0
    stage_code: str | None = None
    stage_label: str | None = None
    stage_detail: str | None = None
    stage_detail_code: str | None = None
    stage_detail_params: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    updated_at: str | None = None
    elapsed_s: int | None = None
    estimated_remaining_s: int | None = None
    logs: list[str] = Field(default_factory=list)
    error_msg: str | None = None
    result_ready: bool = False


class AsyncJobResultResponse(AsyncJobStatusResponse):
    result: dict[str, Any] | None = None


class WeeklyAmountEventRequest(BaseModel):
    week: int = Field(..., ge=1, le=53)
    amount_mm: float | None = Field(None, ge=0.0)
    n_kg_ha: float | None = Field(None, ge=0.0)


class WeeklyFeatureRowResponse(BaseModel):
    week_number: int
    week_start: str | None = None
    ndvi_mean: float | None = None
    ndvi_max: float | None = None
    ndre_mean: float | None = None
    ndmi_mean: float | None = None
    ndwi_mean: float | None = None
    bsi_mean: float | None = None
    tmean_c: float | None = None
    tmax_c: float | None = None
    tmin_c: float | None = None
    precipitation_mm: float | None = None
    vpd_kpa: float | None = None
    solar_radiation_mj: float | None = None
    soil_moisture: float | None = None
    wind_speed_m_s: float | None = None
    gdd: float | None = None
    irrigation_mm: float | None = None
    n_applied_kg_ha: float | None = None
    previous_crop_code: str | None = None
    geometry_confidence: float | None = None
    tta_consensus: float | None = None
    boundary_uncertainty: float | None = None
    stage: int | None = None
    canopy_cover: float | None = None
    water_stress: float | None = None
    heat_stress: float | None = None
    nutrient_stress: float | None = None
    biomass_proxy: float | None = None
    satellite_coverage: float | None = None
    weather_coverage: float | None = None
    source: str | None = None
    feature_schema_version: str | None = None


class WeeklyProfileResponse(BaseModel):
    field_id: str
    season_year: int
    weeks_count: int
    feature_schema_version: str
    geometry_quality_summary: dict[str, Any] = Field(default_factory=dict)
    crop_hint: dict[str, Any] = Field(default_factory=dict)
    rows: list[WeeklyFeatureRowResponse] = Field(default_factory=list)


class ModelingRequest(BaseModel):
    field_id: UUID
    crop_code: str | None = None
    scenario_name: str | None = None
    irrigation_pct: float = Field(0.0, ge=-100.0, le=100.0)
    fertilizer_pct: float = Field(0.0, ge=-100.0, le=100.0)
    expected_rain_mm: float = Field(0.0, ge=0.0, le=500.0)
    temperature_delta_c: float = Field(0.0, ge=-10.0, le=10.0)
    precipitation_factor: float | None = Field(None, ge=0.0, le=5.0)
    planting_density_pct: float = Field(0.0, ge=-80.0, le=100.0)
    sowing_shift_days: int | None = Field(None, ge=-30, le=30)
    tillage_type: int | None = Field(None, ge=0, le=3)
    pest_pressure: int | None = Field(None, ge=0, le=3)
    soil_compaction: float | None = Field(None, ge=0.0, le=1.0)
    cloud_cover_factor: float = Field(1.0, ge=0.1, le=3.0,
        description="Multiplier on seasonal solar radiation (ERA5). "
                    "1.0 = baseline, <1 = overcast season, >1 = sunnier than average. "
                    "Affects Priestley-Taylor ET and vegetation response.")
    irrigation_events: list[WeeklyAmountEventRequest] = Field(default_factory=list)
    fertilizer_events: list[WeeklyAmountEventRequest] = Field(default_factory=list)


class ModelingResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    field_id: str
    baseline_yield_kg_ha: float
    scenario_yield_kg_ha: float
    predicted_yield_change_pct: float
    factors: dict[str, float]
    scenario_name: str | None = None
    model_version: str | None = None
    engine_mode: str | None = None
    confidence_tier: str | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    risk_summary: dict[str, Any] = Field(default_factory=dict)
    saved_scenario_id: int | None = None
    supported: bool = False
    model_applicability: dict[str, Any] = Field(default_factory=dict)
    training_domain: dict[str, Any] = Field(default_factory=dict)
    feature_coverage: dict[str, Any] = Field(default_factory=dict)
    crop_suitability: dict[str, Any] = Field(default_factory=dict)
    crop_hint: dict[str, Any] = Field(default_factory=dict)
    observed_range_guardrails: dict[str, Any] = Field(default_factory=dict)
    counterfactual_feature_diff: dict[str, Any] = Field(default_factory=dict)
    scenario_time_series: dict[str, Any] = Field(default_factory=dict)
    forecast_curve: dict[str, Any] = Field(default_factory=dict)
    scenario_water_balance: dict[str, Any] = Field(default_factory=dict)
    scenario_risk_projection: dict[str, Any] = Field(default_factory=dict)
    baseline_trace: list[dict[str, Any]] = Field(default_factory=list)
    scenario_trace: list[dict[str, Any]] = Field(default_factory=list)
    trace_supported: bool = False
    engine_version: str | None = None
    weeks_simulated: int | None = None
    driver_breakdown: list[dict[str, Any]] = Field(default_factory=list)
    geometry_quality_impact: dict[str, Any] = Field(default_factory=dict)
    constraint_warnings: list[str] = Field(default_factory=list)
    support_reason: str | None = None
    support_reason_code: str | None = None
    support_reason_params: dict[str, Any] = Field(default_factory=dict)
    operational_tier: str | None = None
    review_required: bool = False
    review_reason: str | None = None
    review_reason_code: str | None = None
    review_reason_params: dict[str, Any] = Field(default_factory=dict)
    freshness: FreshnessMeta | None = None


class SensitivitySweepRequest(BaseModel):
    field_id: UUID
    crop_code: str | None = None
    sweep_param: str = Field(..., description="Parameter to sweep: irrigation_pct, fertilizer_pct, expected_rain_mm, etc.")
    sweep_min: float = Field(-80.0)
    sweep_max: float = Field(80.0)
    sweep_steps: int = Field(9, ge=3, le=20)
    base_adjustments: dict[str, float] = Field(default_factory=dict)


class SensitivityPointResponse(BaseModel):
    param_value: float
    yield_kg_ha: float


class SensitivitySweepResponse(BaseModel):
    field_id: str
    sweep_param: str
    baseline_yield_kg_ha: float
    points: list[SensitivityPointResponse]


class ScenarioRunResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    field_id: str
    scenario_name: str
    model_version: str
    created_at: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    baseline_snapshot: dict[str, Any] = Field(default_factory=dict)
    result_snapshot: dict[str, Any] = Field(default_factory=dict)
    delta_pct: float | None = None
    freshness: FreshnessMeta | None = None


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioRunResponse]


class ArchiveCreateRequest(BaseModel):
    field_id: UUID
    date_from: datetime
    date_to: datetime
    layers: list[str] = Field(default_factory=lambda: ["ndvi", "ndwi", "weather"])


class ArchiveEntryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    field_id: str
    date_from: str
    date_to: str
    layers: list[str]
    file_path: str
    status: str
    expires_at: str
    created_at: str | None = None
    meta: dict[str, Any]
    field_snapshot: dict[str, Any] = Field(default_factory=dict)
    prediction_snapshot: dict[str, Any] = Field(default_factory=dict)
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    weather_snapshot: dict[str, Any] = Field(default_factory=dict)
    scenario_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_meta: dict[str, Any] = Field(default_factory=dict)
    freshness: FreshnessMeta | None = None


class ArchiveListResponse(BaseModel):
    archives: list[ArchiveEntryResponse]


class FieldDashboardResponse(BaseModel):
    mode: str
    field: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    kpis: dict[str, Any] = Field(default_factory=dict)
    current_metrics: dict[str, Any] = Field(default_factory=dict)
    series: dict[str, Any] = Field(default_factory=dict)
    histograms: dict[str, Any] = Field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    analytics_summary: dict[str, Any] = Field(default_factory=dict)
    supported_sections: dict[str, Any] = Field(default_factory=dict)
    zones_summary: dict[str, Any] = Field(default_factory=dict)
    archives: list[dict[str, Any]] = Field(default_factory=list)
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    fields: list[dict[str, Any]] = Field(default_factory=list)


class ArchiveSnapshotResponse(BaseModel):
    archive: ArchiveEntryResponse
    snapshot: dict[str, Any]


class FieldGroupDashboardRequest(BaseModel):
    field_ids: list[UUID] = Field(min_length=1)


class FieldMergeRequest(BaseModel):
    field_ids: list[UUID] = Field(min_length=2)


class FieldSplitRequest(BaseModel):
    field_id: UUID
    geometry: dict[str, Any]


class LabelTaskCreateRequest(BaseModel):
    aoi_run_id: UUID | None = None
    field_id: UUID | None = None
    title: str
    source: str = "manual"
    queue_name: str = "default"
    priority_score: float = Field(0.0, ge=0.0, le=1.0)
    geometry: dict[str, Any] | None = None
    task_payload: dict[str, Any] = Field(default_factory=dict)


class LabelVersionCreateRequest(BaseModel):
    geometry: dict[str, Any]
    notes: str | None = None
    quality_tier: str = "draft"


class LabelReviewDecisionRequest(BaseModel):
    notes: str | None = None


class LabelTaskResponse(BaseModel):
    id: int
    aoi_run_id: str | None = None
    field_id: str | None = None
    title: str
    status: str
    source: str
    queue_name: str
    priority_score: float
    task_payload: dict[str, Any] = Field(default_factory=dict)
    claimed_by_user_id: str | None = None
    latest_version: dict[str, Any] | None = None
    latest_review: dict[str, Any] | None = None
    created_at: str | None = None


class LabelTaskListResponse(BaseModel):
    tasks: list[LabelTaskResponse]


class DatasetManifestResponse(BaseModel):
    dataset_version: str
    checksum: str
    manifest: dict[str, Any]


class DataImportPreviewResponse(BaseModel):
    id: int
    import_type: str
    status: str
    source_filename: str
    preview_summary: dict[str, Any] = Field(default_factory=dict)
    commit_summary: dict[str, Any] = Field(default_factory=dict)
    error_count: int = 0
    created_at: str | None = None


class DataImportListResponse(BaseModel):
    jobs: list[DataImportPreviewResponse]


class DataImportCreateRequest(BaseModel):
    import_type: str
    source_filename: str
    content_base64: str


class DataImportErrorResponse(BaseModel):
    id: int
    row_number: int | None = None
    error_code: str
    error_message: str
    raw_record: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class DataImportErrorsListResponse(BaseModel):
    errors: list[DataImportErrorResponse]


class MlDatasetVersionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    dataset_version: str
    checksum: str
    code_sha: str
    status: str
    manifest_json: dict[str, Any] = Field(default_factory=dict)
    split_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str | None = None
    created_at: str | None = None


class MlDatasetListResponse(BaseModel):
    datasets: list[MlDatasetVersionResponse]


class MlBenchmarkResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    dataset_version_id: int
    benchmark_name: str
    model_version: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    gates_passed: bool
    created_at: str | None = None


class MlBenchmarkListResponse(BaseModel):
    benchmarks: list[MlBenchmarkResponse]


class MlDeploymentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: int
    deployment_name: str
    model_version: str
    dataset_version_id: int | None = None
    benchmark_id: int | None = None
    model_uri: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    code_sha: str
    status: str
    created_at: str | None = None
    rolled_back_at: str | None = None


class MlDeploymentListResponse(BaseModel):
    deployments: list[MlDeploymentResponse]


class MlModelRegistryEntryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_version: str
    latest_deployment_id: int | None = None
    deployment_name: str | None = None
    dataset_version_id: int | None = None
    benchmark_id: int | None = None
    model_uri: str | None = None
    status: str | None = None
    created_at: str | None = None


class MlModelRegistryListResponse(BaseModel):
    models: list[MlModelRegistryEntryResponse]


class MlPromotionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    deployment_name: str
    model_version: str
    benchmark_id: int
    dataset_version_id: int
    model_uri: str | None = None
    mlflow_run_id: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    code_sha: str


class MlRollbackRequest(BaseModel):
    deployment_id: int


class MlDatasetRegisterRequest(BaseModel):
    dataset_version: str
    checksum: str
    code_sha: str
    manifest_json: dict[str, Any]
    split_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str | None = None


class MlBenchmarkRegisterRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    dataset_version_id: int
    benchmark_name: str
    model_version: str
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Management Events
# ---------------------------------------------------------------------------

class ManagementEventCreate(BaseModel):
    season_year: int = Field(ge=2000, le=2100)
    event_date: datetime
    event_type: str = Field(min_length=1, max_length=128)
    amount: float | None = None
    unit: str | None = Field(None, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class ManagementEventUpdate(BaseModel):
    event_date: datetime | None = None
    event_type: str | None = Field(None, min_length=1, max_length=128)
    amount: float | None = None
    unit: str | None = Field(None, max_length=64)
    payload: dict[str, Any] | None = None


class ManagementEventResponse(BaseModel):
    id: int
    field_season_id: int
    season_year: int
    event_date: datetime
    event_type: str
    amount: float | None
    unit: str | None
    source: str
    payload: dict[str, Any]


class ManagementEventsListResponse(BaseModel):
    events: list[ManagementEventResponse]
    total: int
