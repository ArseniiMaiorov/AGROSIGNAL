"""Production-таблицы погоды, архивов, прогнозов и культур.

Revision ID: 20260307_0002
Revises: 20260306_0001
Create Date: 2026-03-07 02:30:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260307_0002"
down_revision: Union[str, None] = "20260306_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "weather_data",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("apparent_temperature_c", sa.Float(), nullable=True),
        sa.Column("precipitation_mm", sa.Float(), nullable=True),
        sa.Column("wind_speed_m_s", sa.Float(), nullable=True),
        sa.Column("humidity_pct", sa.Float(), nullable=True),
        sa.Column("cloud_cover_pct", sa.Float(), nullable=True),
        sa.Column("pressure_hpa", sa.Float(), nullable=True),
        sa.Column("soil_moisture", sa.Float(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_weather_data_latitude", "weather_data", ["latitude"])
    op.create_index("ix_weather_data_longitude", "weather_data", ["longitude"])
    op.create_index("ix_weather_data_observed_at", "weather_data", ["observed_at"])

    op.create_table(
        "crops",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False, server_default="grain"),
        sa.Column("yield_baseline_kg_ha", sa.Float(), nullable=False, server_default="4000"),
        sa.Column("ndvi_target", sa.Float(), nullable=False, server_default="0.68"),
        sa.Column("base_temp_c", sa.Float(), nullable=False, server_default="5.0"),
        sa.Column("description", sa.Text(), nullable=True),
    )
    op.create_index("ix_crops_code", "crops", ["code"], unique=True)

    op.create_table(
        "yield_predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("field_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("crop_id", sa.Integer(), sa.ForeignKey("crops.id", ondelete="SET NULL"), nullable=True),
        sa.Column("prediction_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("estimated_yield_kg_ha", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.65"),
        sa.Column("model_version", sa.String(length=64), nullable=False, server_default="heuristic_v1"),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_yield_predictions_field_id", "yield_predictions", ["field_id"])
    op.create_index("ix_yield_predictions_prediction_date", "yield_predictions", ["prediction_date"])

    op.create_table(
        "archive_entries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("field_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("date_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("date_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("layers", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ready"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_archive_entries_field_id", "archive_entries", ["field_id"])
    op.create_index("ix_archive_entries_expires_at", "archive_entries", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_archive_entries_expires_at", table_name="archive_entries")
    op.drop_index("ix_archive_entries_field_id", table_name="archive_entries")
    op.drop_table("archive_entries")
    op.drop_index("ix_yield_predictions_prediction_date", table_name="yield_predictions")
    op.drop_index("ix_yield_predictions_field_id", table_name="yield_predictions")
    op.drop_table("yield_predictions")
    op.drop_index("ix_crops_code", table_name="crops")
    op.drop_table("crops")
    op.drop_index("ix_weather_data_observed_at", table_name="weather_data")
    op.drop_index("ix_weather_data_longitude", table_name="weather_data")
    op.drop_index("ix_weather_data_latitude", table_name="weather_data")
    op.drop_table("weather_data")
