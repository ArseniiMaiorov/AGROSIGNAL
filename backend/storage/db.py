from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    delete,
    Float,
    ForeignKey,
    insert,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from core.settings import get_settings


class Base(DeclarativeBase):
    pass


role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permission_id", ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
)


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(String(128), nullable=False, unique=True, index=True)
    name = Column(String(256), nullable=False, unique=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    memberships = relationship("Membership", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(256), nullable=False, unique=True, index=True)
    full_name = Column(String(256), nullable=False, default="Administrator")
    password_hash = Column(String(512), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    memberships = relationship("Membership", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, unique=True, index=True)
    description = Column(Text)
    is_system = Column(Boolean, nullable=False, default=True)

    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    memberships = relationship("Membership", back_populates="role")


class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(128), nullable=False, unique=True, index=True)
    description = Column(Text)

    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_membership_org_user"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    organization = relationship("Organization", back_populates="memberships")
    user = relationship("User", back_populates="memberships")
    role = relationship("Role", back_populates="memberships")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(256), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    revoked_at = Column(DateTime(timezone=True))
    user_agent = Column(String(512))
    ip_address = Column(String(128))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="refresh_tokens")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    actor_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(128), nullable=False, index=True)
    resource_type = Column(String(128), nullable=False)
    resource_id = Column(String(256))
    payload = Column(JSONB, nullable=False, default=dict)
    ip_address = Column(String(128))
    user_agent = Column(String(512))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


class AoiRun(Base):
    __tablename__ = "aoi_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    aoi = Column(Geometry("GEOMETRY", srid=4326), nullable=False)
    time_start = Column(DateTime, nullable=False)
    time_end = Column(DateTime, nullable=False)
    params = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    error_msg = Column(Text)
    progress = Column(Integer, default=0)
    log_ref = Column(Text)

    fields = relationship("Field", back_populates="aoi_run", cascade="all, delete-orphan", passive_deletes=True)
    grid_cells = relationship("GridCell", back_populates="aoi_run", cascade="all, delete-orphan", passive_deletes=True)


class Field(Base):
    __tablename__ = "fields"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    geom = Column(Geometry("MULTIPOLYGON", srid=4326), nullable=False)
    area_m2 = Column(Float, nullable=False)
    perimeter_m = Column(Float, nullable=False)
    quality_score = Column(Float)
    source = Column(String(50), nullable=False, default="autodetect")
    external_field_id = Column(String(128), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    aoi_run = relationship("AoiRun", back_populates="fields")
    predictions = relationship("YieldPrediction", back_populates="field", cascade="all, delete-orphan", passive_deletes=True)
    archives = relationship("ArchiveEntry", back_populates="field", cascade="all, delete-orphan", passive_deletes=True)
    metric_series = relationship("FieldMetricSeries", back_populates="field", cascade="all, delete-orphan", passive_deletes=True)
    scenarios = relationship("ScenarioRun", back_populates="field", cascade="all, delete-orphan", passive_deletes=True)
    seasons = relationship("FieldSeason", back_populates="field", cascade="all, delete-orphan", passive_deletes=True)


class GridCell(Base):
    __tablename__ = "grid_cells"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    geom = Column(Geometry("POLYGON", srid=4326), nullable=False)
    zoom_level = Column(Integer, nullable=False, index=True)
    row = Column(Integer, nullable=False)
    col = Column(Integer, nullable=False)
    field_coverage = Column(Float)
    ndvi_mean = Column(Float)
    ndwi_mean = Column(Float)
    ndmi_mean = Column(Float)
    bsi_mean = Column(Float)
    precipitation_mm = Column(Float)
    wind_speed_m_s = Column(Float)
    u_wind_10m = Column(Float)
    v_wind_10m = Column(Float)
    wind_direction_deg = Column(Float)
    gdd_sum = Column(Float)
    vpd_mean = Column(Float)
    soil_moist = Column(Float)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    aoi_run = relationship("AoiRun", back_populates="grid_cells")


class Layer(Base):
    __tablename__ = "layers"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    unit = Column(String)
    range_min = Column(Float)
    range_max = Column(Float)
    source = Column(String, nullable=False)
    description = Column(Text)


class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    provider = Column(String(32), nullable=False)
    temperature_c = Column(Float)
    apparent_temperature_c = Column(Float)
    precipitation_mm = Column(Float)
    wind_speed_m_s = Column(Float)
    humidity_pct = Column(Float)
    cloud_cover_pct = Column(Float)
    pressure_hpa = Column(Float)
    soil_moisture = Column(Float)
    payload = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Crop(Base):
    __tablename__ = "crops"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(32), nullable=False, unique=True, index=True)
    name = Column(String(128), nullable=False)
    category = Column(String(64), nullable=False, default="grain")
    yield_baseline_kg_ha = Column(Float, nullable=False, default=4000.0)
    ndvi_target = Column(Float, nullable=False, default=0.68)
    base_temp_c = Column(Float, nullable=False, default=5.0)
    description = Column(Text)

    predictions = relationship("YieldPrediction", back_populates="crop")


class YieldPrediction(Base):
    __tablename__ = "yield_predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    crop_id = Column(Integer, ForeignKey("crops.id", ondelete="SET NULL"), index=True)
    prediction_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    estimated_yield_kg_ha = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False, default=0.65)
    model_version = Column(String(64), nullable=False, default="unsupported_v2")
    details = Column(JSONB, nullable=False, default=dict)
    input_features = Column(JSONB, nullable=False, default=dict)
    explanation = Column(JSONB, nullable=False, default=dict)
    data_quality = Column(JSONB, nullable=False, default=dict)

    field = relationship("Field", back_populates="predictions")
    crop = relationship("Crop", back_populates="predictions")
    scenarios = relationship("ScenarioRun", back_populates="baseline_prediction")


class FieldMetricSeries(Base):
    __tablename__ = "field_metric_series"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="SET NULL"), index=True)
    archive_entry_id = Column(BigInteger, ForeignKey("archive_entries.id", ondelete="SET NULL"), index=True)
    metric = Column(String(64), nullable=False, index=True)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    value_mean = Column(Float)
    value_min = Column(Float)
    value_max = Column(Float)
    value_median = Column(Float)
    value_p25 = Column(Float)
    value_p75 = Column(Float)
    coverage = Column(Float)
    source = Column(String(64), nullable=False, default="run_snapshot")
    meta = Column(JSONB, nullable=False, default=dict)

    field = relationship("Field", back_populates="metric_series")
    archive_entry = relationship("ArchiveEntry", back_populates="metric_series")


class ScenarioRun(Base):
    __tablename__ = "scenario_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    crop_id = Column(Integer, ForeignKey("crops.id", ondelete="SET NULL"), index=True)
    baseline_prediction_id = Column(BigInteger, ForeignKey("yield_predictions.id", ondelete="SET NULL"), index=True)
    scenario_name = Column(String(128), nullable=False, default="Сценарий")
    model_version = Column(String(64), nullable=False, default="unsupported_scenario_v2")
    parameters = Column(JSONB, nullable=False, default=dict)
    baseline_snapshot = Column(JSONB, nullable=False, default=dict)
    result_snapshot = Column(JSONB, nullable=False, default=dict)
    delta_pct = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    field = relationship("Field", back_populates="scenarios")
    crop = relationship("Crop")
    baseline_prediction = relationship("YieldPrediction", back_populates="scenarios")


class ArchiveEntry(Base):
    __tablename__ = "archive_entries"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    date_from = Column(DateTime(timezone=True), nullable=False)
    date_to = Column(DateTime(timezone=True), nullable=False)
    layers = Column(JSONB, nullable=False, default=list)
    file_path = Column(String, nullable=False)
    status = Column(String(32), nullable=False, default="ready")
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    meta = Column(JSONB, nullable=False, default=dict)
    field_snapshot = Column(JSONB, nullable=False, default=dict)
    prediction_snapshot = Column(JSONB, nullable=False, default=dict)
    metrics_snapshot = Column(JSONB, nullable=False, default=dict)
    weather_snapshot = Column(JSONB, nullable=False, default=dict)
    scenario_snapshot = Column(JSONB, nullable=False, default=dict)
    model_meta = Column(JSONB, nullable=False, default=dict)

    field = relationship("Field", back_populates="archives")
    metric_series = relationship("FieldMetricSeries", back_populates="archive_entry")


class LabelTask(Base):
    __tablename__ = "label_tasks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="SET NULL"), index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="SET NULL"), index=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    claimed_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    title = Column(String(256), nullable=False)
    status = Column(String(32), nullable=False, default="queued", index=True)
    source = Column(String(64), nullable=False, default="active_learning")
    queue_name = Column(String(64), nullable=False, default="default")
    priority_score = Column(Float, nullable=False, default=0.0, index=True)
    task_payload = Column(JSONB, nullable=False, default=dict)
    claimed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LabelVersion(Base):
    __tablename__ = "label_versions"
    __table_args__ = (UniqueConstraint("label_task_id", "version_no", name="uq_label_version_task_version"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    label_task_id = Column(BigInteger, ForeignKey("label_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    version_no = Column(Integer, nullable=False)
    geometry_geojson = Column(JSONB, nullable=False, default=dict)
    mask_artifact_uri = Column(String(512))
    quality_tier = Column(String(64), nullable=False, default="draft")
    source = Column(String(64), nullable=False, default="manual")
    checksum = Column(String(128))
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class LabelReview(Base):
    __tablename__ = "label_reviews"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    label_task_id = Column(BigInteger, ForeignKey("label_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    label_version_id = Column(BigInteger, ForeignKey("label_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    decision = Column(String(32), nullable=False, index=True)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ActiveLearningCandidate(Base):
    __tablename__ = "active_learning_candidates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="SET NULL"), index=True)
    priority_score = Column(Float, nullable=False, index=True)
    uncertainty_score = Column(Float, nullable=False, default=0.0)
    rule_ml_disagreement = Column(Float, nullable=False, default=0.0)
    region_rarity = Column(Float, nullable=False, default=0.0)
    error_mode_quota = Column(Float, nullable=False, default=0.0)
    candidate_payload = Column(JSONB, nullable=False, default=dict)
    status = Column(String(32), nullable=False, default="queued", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class MlDatasetVersion(Base):
    __tablename__ = "ml_dataset_versions"
    __table_args__ = (UniqueConstraint("organization_id", "dataset_version", name="uq_ml_dataset_org_version"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version = Column(String(128), nullable=False, index=True)
    checksum = Column(String(128), nullable=False)
    code_sha = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False, default="ready")
    manifest_json = Column(JSONB, nullable=False, default=dict)
    split_summary = Column(JSONB, nullable=False, default=dict)
    artifact_uri = Column(String(512))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class MlBenchmark(Base):
    __tablename__ = "ml_benchmarks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version_id = Column(BigInteger, ForeignKey("ml_dataset_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    benchmark_name = Column(String(128), nullable=False)
    model_version = Column(String(128), nullable=False)
    metrics = Column(JSONB, nullable=False, default=dict)
    gates_passed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    dataset_version = relationship("MlDatasetVersion")


class MlDeployment(Base):
    __tablename__ = "ml_deployments"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    deployment_name = Column(String(128), nullable=False)
    model_version = Column(String(128), nullable=False)
    dataset_version_id = Column(BigInteger, ForeignKey("ml_dataset_versions.id", ondelete="SET NULL"), index=True)
    benchmark_id = Column(BigInteger, ForeignKey("ml_benchmarks.id", ondelete="SET NULL"), index=True)
    mlflow_run_id = Column(String(256))
    model_uri = Column(String(512))
    config_snapshot = Column(JSONB, nullable=False, default=dict)
    code_sha = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False, default="promoted", index=True)
    promoted_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    rolled_back_at = Column(DateTime(timezone=True))

    dataset_version = relationship("MlDatasetVersion")
    benchmark = relationship("MlBenchmark")


# ---------------------------------------------------------------------------
# Detection v4: tile diagnostics, candidate tracking, feature store
# ---------------------------------------------------------------------------

class TileDiagnostic(Base):
    """Per-tile quality assessment and runtime diagnostics."""
    __tablename__ = "tile_diagnostics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    tile_index = Column(Integer, nullable=False)
    quality_mode = Column(String(32), nullable=False, default="normal", index=True)  # normal/boundary_recovery/degraded/skip
    coverage_fraction = Column(Float)
    valid_scene_count = Column(Integer)
    edge_strength_mean = Column(Float)
    edge_strength_p90 = Column(Float)
    ndvi_temporal_std = Column(Float)
    cloud_interference = Column(Float)
    selected_dates = Column(JSONB, nullable=False, default=list)
    runtime_flags = Column(JSONB, nullable=False, default=dict)
    artifact_refs = Column(JSONB, nullable=False, default=dict)  # paths to .npz/.gpkg
    candidates_total = Column(Integer, default=0)
    candidates_kept = Column(Integer, default=0)
    processing_time_s = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FieldDetectionCandidate(Base):
    """Individual field candidate from detection branches before final merge."""
    __tablename__ = "field_detection_candidates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    aoi_run_id = Column(UUID(as_uuid=True), ForeignKey("aoi_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    tile_diagnostic_id = Column(BigInteger, ForeignKey("tile_diagnostics.id", ondelete="CASCADE"), index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="SET NULL"), index=True)  # set after merge
    branch = Column(String(32), nullable=False, index=True)  # boundary/crop_region/refine
    geom = Column(Geometry("POLYGON", srid=4326))
    area_m2 = Column(Float)
    score = Column(Float, nullable=False, default=0.0, index=True)
    rank = Column(Integer)
    kept = Column(Boolean, nullable=False, default=False, index=True)
    reject_reason = Column(String(128))
    features = Column(JSONB, nullable=False, default=dict)  # scoring features
    model_version = Column(String(64))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FieldFeatureWeekly(Base):
    """Weekly satellite/weather/water/stage features per field-season.

    Materialized for fast access by yield model and scenario simulator.
    """
    __tablename__ = "field_feature_weekly"
    __table_args__ = (
        UniqueConstraint("organization_id", "field_id", "season_year", "week_number", name="uq_field_feature_week"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    season_year = Column(Integer, nullable=False, index=True)
    week_number = Column(Integer, nullable=False)  # ISO week
    week_start = Column(Date, nullable=False)
    # Satellite
    ndvi_mean = Column(Float)
    ndvi_max = Column(Float)
    ndre_mean = Column(Float)
    ndmi_mean = Column(Float)
    ndwi_mean = Column(Float)
    bsi_mean = Column(Float)
    # Weather
    tmean_c = Column(Float)
    tmax_c = Column(Float)
    tmin_c = Column(Float)
    precipitation_mm = Column(Float)
    vpd_kpa = Column(Float)
    solar_radiation_mj = Column(Float)
    soil_moisture = Column(Float)
    wind_speed_m_s = Column(Float)
    gdd = Column(Float)
    # Management
    irrigation_mm = Column(Float, default=0.0)
    n_applied_kg_ha = Column(Float, default=0.0)
    previous_crop_code = Column(String(32))
    # Geometry quality
    geometry_confidence = Column(Float)
    tta_consensus = Column(Float)
    boundary_uncertainty = Column(Float)
    # Model state
    stage = Column(Integer)
    canopy_cover = Column(Float)
    water_stress = Column(Float)
    heat_stress = Column(Float)
    nutrient_stress = Column(Float)
    biomass_proxy = Column(Float)
    # Coverage
    satellite_coverage = Column(Float)
    weather_coverage = Column(Float)
    source = Column(String(64), default="computed")
    feature_schema_version = Column(String(32), default="weekly_v1")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FieldCropPosterior(Base):
    """Advisory crop posterior distribution for a field-season.

    This does not override the user-selected crop. It is stored as an auxiliary
    hint/warning layer for future benchmarking and UI validation.
    """
    __tablename__ = "field_crop_posteriors"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "field_id",
            "season_year",
            "crop_code",
            "source",
            name="uq_field_crop_posterior",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    season_year = Column(Integer, nullable=False, index=True)
    crop_code = Column(String(32), nullable=False, index=True)
    probability = Column(Float, nullable=False, default=0.0)
    source = Column(String(64), nullable=False, default="manual_selection")
    model_version = Column(String(64))
    payload = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class YieldModel(Base):
    """Registry of yield model versions for audit and reproducibility."""
    __tablename__ = "yield_models"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)
    model_name = Column(String(128), nullable=False, index=True)
    model_version = Column(String(64), nullable=False, unique=True)
    mechanistic_params_version = Column(String(64))
    residual_model_version = Column(String(64))
    tenant_calibration_version = Column(String(64))
    calibration_set_hash = Column(String(128))
    training_summary = Column(JSONB, nullable=False, default=dict)
    metrics = Column(JSONB, nullable=False, default=dict)  # MAE, RMSE, coverage, bias
    config_snapshot = Column(JSONB, nullable=False, default=dict)
    artifact_uri = Column(String(512))
    status = Column(String(32), nullable=False, default="training", index=True)  # training/validated/deployed/retired
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deployed_at = Column(DateTime(timezone=True))


class PredictionRun(Base):
    """Detailed inference history for yield predictions."""
    __tablename__ = "prediction_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    yield_prediction_id = Column(BigInteger, ForeignKey("yield_predictions.id", ondelete="SET NULL"), index=True)
    model_version = Column(String(64), nullable=False)
    engine_version = Column(String(64), nullable=False, default="mechanistic_v1")
    crop_code = Column(String(64), nullable=False)
    # Results
    mechanistic_yield = Column(Float)
    residual_adjustment = Column(Float)
    tenant_adjustment = Column(Float)
    final_yield = Column(Float)
    confidence = Column(Float)
    support_tier = Column(String(32))
    # Trace
    weekly_trace = Column(JSONB, nullable=False, default=list)
    interval_payload = Column(JSONB, nullable=False, default=dict)
    explain_payload = Column(JSONB, nullable=False, default=dict)
    review_required = Column(Boolean, default=False)
    review_reason = Column(String(256))
    # Meta
    feature_snapshot = Column(JSONB, nullable=False, default=dict)
    quality_flags = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


class FieldSeason(Base):
    __tablename__ = "field_seasons"
    __table_args__ = (UniqueConstraint("organization_id", "field_id", "season_year", name="uq_field_season"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    season_year = Column(Integer, nullable=False, index=True)
    label = Column(String(128), nullable=False, default="")
    external_field_id = Column(String(128), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    field = relationship("Field", back_populates="seasons")
    crop_assignments = relationship("CropAssignment", back_populates="field_season", cascade="all, delete-orphan")
    yield_observations = relationship("YieldObservation", back_populates="field_season", cascade="all, delete-orphan")
    management_events = relationship("ManagementEvent", back_populates="field_season", cascade="all, delete-orphan")
    weather_daily = relationship("WeatherDaily", back_populates="field_season", cascade="all, delete-orphan")


class CropAssignment(Base):
    __tablename__ = "crop_assignments"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_season_id = Column(BigInteger, ForeignKey("field_seasons.id", ondelete="CASCADE"), nullable=False, index=True)
    crop_id = Column(Integer, ForeignKey("crops.id", ondelete="SET NULL"))
    crop_code = Column(String(32), nullable=False, index=True)
    source = Column(String(64), nullable=False, default="customer_import")
    payload = Column(JSONB, nullable=False, default=dict)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    field_season = relationship("FieldSeason", back_populates="crop_assignments")
    crop = relationship("Crop")


class YieldObservation(Base):
    __tablename__ = "yield_observations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_season_id = Column(BigInteger, ForeignKey("field_seasons.id", ondelete="CASCADE"), nullable=False, index=True)
    yield_kg_ha = Column(Float, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    source = Column(String(64), nullable=False, default="customer_import")
    payload = Column(JSONB, nullable=False, default=dict)

    field_season = relationship("FieldSeason", back_populates="yield_observations")


class SoilProfile(Base):
    __tablename__ = "soil_profiles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True)
    sampled_at = Column(DateTime(timezone=True), nullable=False, index=True)
    source = Column(String(64), nullable=False, default="customer_import")
    texture_class = Column(String(128))
    organic_matter_pct = Column(Float)
    ph = Column(Float)
    n_ppm = Column(Float)
    p_ppm = Column(Float)
    k_ppm = Column(Float)
    payload = Column(JSONB, nullable=False, default=dict)


class ManagementEvent(Base):
    __tablename__ = "management_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_season_id = Column(BigInteger, ForeignKey("field_seasons.id", ondelete="CASCADE"), nullable=False, index=True)
    event_date = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(128), nullable=False, index=True)
    amount = Column(Float)
    unit = Column(String(64))
    source = Column(String(64), nullable=False, default="customer_import")
    payload = Column(JSONB, nullable=False, default=dict)

    field_season = relationship("FieldSeason", back_populates="management_events")


class WeatherDaily(Base):
    __tablename__ = "weather_daily"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    field_season_id = Column(BigInteger, ForeignKey("field_seasons.id", ondelete="CASCADE"), nullable=False, index=True)
    observed_on = Column(Date, nullable=False, index=True)
    temperature_mean_c = Column(Float)
    precipitation_mm = Column(Float)
    gdd = Column(Float)
    vpd = Column(Float)
    soil_moisture = Column(Float)
    source = Column(String(64), nullable=False, default="enriched")
    payload = Column(JSONB, nullable=False, default=dict)

    field_season = relationship("FieldSeason", back_populates="weather_daily")


class DataImportJob(Base):
    __tablename__ = "data_import_jobs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    import_type = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="previewed", index=True)
    source_filename = Column(String(256), nullable=False)
    source_path = Column(String(512), nullable=False)
    preview_summary = Column(JSONB, nullable=False, default=dict)
    commit_summary = Column(JSONB, nullable=False, default=dict)
    error_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    committed_at = Column(DateTime(timezone=True))

    errors = relationship("DataImportError", back_populates="import_job", cascade="all, delete-orphan")


class DataImportError(Base):
    __tablename__ = "data_import_errors"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    import_job_id = Column(BigInteger, ForeignKey("data_import_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    row_number = Column(Integer)
    error_code = Column(String(128), nullable=False)
    error_message = Column(Text, nullable=False)
    raw_record = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    import_job = relationship("DataImportJob", back_populates="errors")


_engine = None
_session_factory = None

_SCHEMA_GUARDRAIL_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_aoi_runs_aoi_gist ON aoi_runs USING GIST (aoi)",
    "CREATE INDEX IF NOT EXISTS idx_fields_geom_gist ON fields USING GIST (geom)",
    "CREATE INDEX IF NOT EXISTS idx_grid_cells_geom_gist ON grid_cells USING GIST (geom)",
    "ALTER TABLE aoi_runs ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE aoi_runs ADD COLUMN IF NOT EXISTS created_by_user_id UUID",
    "ALTER TABLE fields ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE fields ADD COLUMN IF NOT EXISTS external_field_id VARCHAR(128)",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS ndwi_mean DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS bsi_mean DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS precipitation_mm DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS wind_speed_m_s DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS u_wind_10m DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS v_wind_10m DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS wind_direction_deg DOUBLE PRECISION",
    "ALTER TABLE grid_cells ADD COLUMN IF NOT EXISTS field_coverage DOUBLE PRECISION",
    "ALTER TABLE weather_data ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE yield_predictions ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE yield_predictions ADD COLUMN IF NOT EXISTS input_features JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE yield_predictions ADD COLUMN IF NOT EXISTS explanation JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE yield_predictions ADD COLUMN IF NOT EXISTS data_quality JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE field_metric_series ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE scenario_runs ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS organization_id UUID",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS field_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS prediction_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS metrics_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS weather_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS scenario_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE archive_entries ADD COLUMN IF NOT EXISTS model_meta JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE field_feature_weekly ADD COLUMN IF NOT EXISTS geometry_confidence DOUBLE PRECISION",
    "ALTER TABLE field_feature_weekly ADD COLUMN IF NOT EXISTS tta_consensus DOUBLE PRECISION",
    "ALTER TABLE field_feature_weekly ADD COLUMN IF NOT EXISTS boundary_uncertainty DOUBLE PRECISION",
    "ALTER TABLE field_feature_weekly ADD COLUMN IF NOT EXISTS nutrient_stress DOUBLE PRECISION",
    "ALTER TABLE field_feature_weekly ADD COLUMN IF NOT EXISTS previous_crop_code VARCHAR(32)",
    "ALTER TABLE field_feature_weekly ADD COLUMN IF NOT EXISTS feature_schema_version VARCHAR(32) DEFAULT 'weekly_v1'",
    "CREATE INDEX IF NOT EXISTS idx_field_metric_series_field_id ON field_metric_series (field_id)",
    "CREATE INDEX IF NOT EXISTS idx_field_metric_series_observed_at ON field_metric_series (observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_field_metric_series_metric ON field_metric_series (metric)",
    "CREATE INDEX IF NOT EXISTS idx_scenario_runs_field_id ON scenario_runs (field_id)",
    "CREATE INDEX IF NOT EXISTS idx_scenario_runs_created_at ON scenario_runs (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_fields_aoi_run_id ON fields (aoi_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_yield_predictions_crop_id ON yield_predictions (crop_id)",
    "CREATE INDEX IF NOT EXISTS idx_scenario_runs_crop_id ON scenario_runs (crop_id)",
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'aoi_runs_aoi_valid_chk') THEN
            ALTER TABLE aoi_runs
            ADD CONSTRAINT aoi_runs_aoi_valid_chk CHECK (ST_IsValid(aoi)) NOT VALID;
        END IF;
    END $$;
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fields_geom_valid_chk') THEN
            ALTER TABLE fields
            ADD CONSTRAINT fields_geom_valid_chk CHECK (ST_IsValid(geom)) NOT VALID;
        END IF;
    END $$;
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'grid_cells_geom_valid_chk') THEN
            ALTER TABLE grid_cells
            ADD CONSTRAINT grid_cells_geom_valid_chk CHECK (ST_IsValid(geom)) NOT VALID;
        END IF;
    END $$;
    """,
)


DEFAULT_PERMISSION_SPECS: tuple[tuple[str, str], ...] = (
    ("fields:read", "Read fields, runs and dashboards"),
    ("fields:write", "Create detect runs and manual field edits"),
    ("weather:read", "Read weather and status"),
    ("layers:read", "Read grid layers"),
    ("crops:read", "Read crop catalog"),
    ("predictions:read", "Read yield predictions"),
    ("predictions:write", "Refresh yield predictions"),
    ("scenarios:read", "Read saved scenarios"),
    ("scenarios:write", "Run and save scenarios"),
    ("archive:read", "Read and download archives"),
    ("archive:write", "Create archives"),
    ("labeling:read", "Read labeling queue"),
    ("labeling:write", "Create and update labeling tasks"),
    ("labeling:review", "Approve or reject labeling versions"),
    ("imports:read", "Read import jobs"),
    ("imports:write", "Create and commit import jobs"),
    ("mlops:read", "Read ML datasets, benchmarks and deployments"),
    ("mlops:write", "Promote, rollback and register ML metadata"),
    ("status:read", "Read system health"),
    ("storage:read", "Read storage settings"),
    ("storage:write", "Configure local or cloud storage"),
)

DEFAULT_ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "tenant_admin": tuple(code for code, _ in DEFAULT_PERMISSION_SPECS),
    "agronomist": (
        "fields:read",
        "fields:write",
        "weather:read",
        "layers:read",
        "crops:read",
        "predictions:read",
        "predictions:write",
        "scenarios:read",
        "scenarios:write",
        "archive:read",
        "archive:write",
        "imports:read",
        "imports:write",
        "status:read",
        "storage:read",
        "storage:write",
    ),
    "label_reviewer": (
        "fields:read",
        "labeling:read",
        "labeling:write",
        "labeling:review",
        "mlops:read",
        "status:read",
        "storage:read",
    ),
    "viewer": (
        "fields:read",
        "weather:read",
        "layers:read",
        "crops:read",
        "predictions:read",
        "scenarios:read",
        "archive:read",
        "labeling:read",
        "imports:read",
        "mlops:read",
        "status:read",
        "storage:read",
    ),
}


DEFAULT_LAYERS = [
    Layer(id="ndvi", name="NDVI", unit="index", range_min=-0.2, range_max=0.9,
          source="sentinel2", description="Normalized Difference Vegetation Index"),
    Layer(id="ndmi", name="NDMI", unit="index", range_min=-0.5, range_max=0.7,
          source="sentinel2", description="Normalized Difference Moisture Index"),
    Layer(id="ndwi", name="NDWI", unit="index", range_min=-0.5, range_max=0.5,
          source="sentinel2", description="Normalized Difference Water Index"),
    Layer(id="bsi", name="BSI", unit="index", range_min=-0.5, range_max=0.5,
          source="sentinel2", description="Bare Soil Index"),
    Layer(id="gdd", name="GDD", unit="°C·day", range_min=0, range_max=3000,
          source="era5", description="Growing Degree Days"),
    Layer(id="vpd", name="VPD", unit="kPa", range_min=0, range_max=4.0,
          source="era5", description="Vapor Pressure Deficit"),
    Layer(id="precipitation", name="Precipitation", unit="mm", range_min=0, range_max=200,
          source="era5", description="Total precipitation"),
    Layer(id="wind", name="Wind", unit="m/s", range_min=0, range_max=20,
          source="era5", description="10m wind speed"),
    Layer(id="soil_moisture", name="Soil Moisture", unit="m³/m³", range_min=0, range_max=0.5,
          source="era5", description="Volumetric soil water layer 1"),
]

DEFAULT_CROPS = [
    Crop(code="wheat", name="Пшеница", category="grain", yield_baseline_kg_ha=4200, ndvi_target=0.72, base_temp_c=5.0,
         description="Базовая зерновая культура для умеренной зоны."),
    Crop(code="barley", name="Ячмень", category="grain", yield_baseline_kg_ha=3600, ndvi_target=0.68, base_temp_c=5.0,
         description="Яровой и озимый ячмень."),
    Crop(code="corn", name="Кукуруза", category="grain", yield_baseline_kg_ha=6500, ndvi_target=0.78, base_temp_c=10.0,
         description="Кукуруза на зерно."),
    Crop(code="sunflower", name="Подсолнечник", category="oilseed", yield_baseline_kg_ha=2600, ndvi_target=0.66, base_temp_c=8.0,
         description="Масличная культура для южных регионов."),
    Crop(code="soy", name="Соя", category="legume", yield_baseline_kg_ha=2400, ndvi_target=0.70, base_temp_c=8.0,
         description="Бобовая культура."),
    Crop(code="rapeseed", name="Рапс", category="oilseed", yield_baseline_kg_ha=3000, ndvi_target=0.69, base_temp_c=5.0,
         description="Озимый и яровой рапс."),
]


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_size=max(1, int(settings.DB_POOL_SIZE)),
            max_overflow=max(0, int(settings.DB_MAX_OVERFLOW)),
            pool_timeout=max(1.0, float(settings.DB_POOL_TIMEOUT_S)),
            pool_recycle=max(60, int(settings.DB_POOL_RECYCLE_S)),
            pool_pre_ping=bool(settings.DB_POOL_PRE_PING),
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def get_db():
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS postgis")
        await conn.run_sync(Base.metadata.create_all)
        for statement in _SCHEMA_GUARDRAIL_SQL:
            await conn.exec_driver_sql(statement)


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "default-org"


async def seed_defaults() -> None:
    factory = get_session_factory()
    async with factory() as session:
        for layer in DEFAULT_LAYERS:
            existing = await session.get(Layer, layer.id)
            if existing is None:
                session.add(layer)
        for crop in DEFAULT_CROPS:
            result = await session.execute(select(Crop).where(Crop.code == crop.code))
            if result.scalar_one_or_none() is None:
                session.add(crop)

        permission_map: dict[str, Permission] = {}
        for code, description in DEFAULT_PERMISSION_SPECS:
            result = await session.execute(select(Permission).where(Permission.code == code))
            permission = result.scalar_one_or_none()
            if permission is None:
                permission = Permission(code=code, description=description)
                session.add(permission)
                await session.flush()
            permission_map[code] = permission

        for role_name, permission_codes in DEFAULT_ROLE_PERMISSIONS.items():
            result = await session.execute(select(Role).where(Role.name == role_name))
            role = result.scalar_one_or_none()
            if role is None:
                role = Role(name=role_name, description=role_name.replace("_", " ").title(), is_system=True)
                session.add(role)
                await session.flush()
            desired_permission_ids = {permission_map[code].id for code in permission_codes}
            existing_permission_ids = set(
                (
                    await session.execute(
                        select(role_permissions.c.permission_id).where(role_permissions.c.role_id == role.id)
                    )
                ).scalars()
            )
            missing_permission_ids = desired_permission_ids - existing_permission_ids
            stale_permission_ids = existing_permission_ids - desired_permission_ids
            if stale_permission_ids:
                await session.execute(
                    delete(role_permissions)
                    .where(role_permissions.c.role_id == role.id)
                    .where(role_permissions.c.permission_id.in_(stale_permission_ids))
                )
            if missing_permission_ids:
                await session.execute(
                    insert(role_permissions),
                    [{"role_id": role.id, "permission_id": permission_id} for permission_id in missing_permission_ids],
                )

        settings = get_settings()
        if settings.AUTH_BOOTSTRAP_ENABLED:
            from core.security import hash_password

            org_slug = _slugify(settings.AUTH_BOOTSTRAP_ORG_NAME)
            result = await session.execute(select(Organization).where(Organization.slug == org_slug))
            organization = result.scalar_one_or_none()
            if organization is None:
                organization = Organization(slug=org_slug, name=settings.AUTH_BOOTSTRAP_ORG_NAME)
                session.add(organization)
                await session.flush()

            result = await session.execute(select(User).where(User.email == settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower()))
            user = result.scalar_one_or_none()
            if user is None:
                user = User(
                    email=settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower(),
                    full_name="Bootstrap Admin",
                    password_hash=hash_password(settings.AUTH_BOOTSTRAP_ADMIN_PASSWORD),
                    is_active=True,
                )
                session.add(user)
                await session.flush()

            role_result = await session.execute(select(Role).where(Role.name == "tenant_admin"))
            admin_role = role_result.scalar_one()
            membership_result = await session.execute(
                select(Membership)
                .where(Membership.organization_id == organization.id)
                .where(Membership.user_id == user.id)
            )
            membership = membership_result.scalar_one_or_none()
            if membership is None:
                membership = Membership(
                    organization_id=organization.id,
                    user_id=user.id,
                    role_id=admin_role.id,
                    is_active=True,
                )
                session.add(membership)

        await session.commit()


async def seed_layers() -> None:
    await seed_defaults()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
