"""Серии метрик по полям, сохранённые сценарии и explainability прогноза.

Revision ID: 20260307_0003
Revises: 20260307_0002
Create Date: 2026-03-07 15:40:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260307_0003"
down_revision: Union[str, None] = "20260307_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "yield_predictions",
        sa.Column("input_features", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "yield_predictions",
        sa.Column("explanation", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "yield_predictions",
        sa.Column("data_quality", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )

    op.add_column(
        "archive_entries",
        sa.Column("field_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "archive_entries",
        sa.Column("prediction_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "archive_entries",
        sa.Column("metrics_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "archive_entries",
        sa.Column("weather_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "archive_entries",
        sa.Column("scenario_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "archive_entries",
        sa.Column("model_meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )

    op.create_table(
        "field_metric_series",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("field_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("aoi_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("aoi_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("archive_entry_id", sa.BigInteger(), sa.ForeignKey("archive_entries.id", ondelete="SET NULL"), nullable=True),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value_mean", sa.Float(), nullable=True),
        sa.Column("value_min", sa.Float(), nullable=True),
        sa.Column("value_max", sa.Float(), nullable=True),
        sa.Column("value_median", sa.Float(), nullable=True),
        sa.Column("value_p25", sa.Float(), nullable=True),
        sa.Column("value_p75", sa.Float(), nullable=True),
        sa.Column("coverage", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="run_snapshot"),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_field_metric_series_field_id", "field_metric_series", ["field_id"])
    op.create_index("ix_field_metric_series_metric", "field_metric_series", ["metric"])
    op.create_index("ix_field_metric_series_observed_at", "field_metric_series", ["observed_at"])

    op.create_table(
        "scenario_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("field_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("crop_id", sa.Integer(), sa.ForeignKey("crops.id", ondelete="SET NULL"), nullable=True),
        sa.Column("baseline_prediction_id", sa.BigInteger(), sa.ForeignKey("yield_predictions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("scenario_name", sa.String(length=128), nullable=False, server_default="Сценарий"),
        sa.Column("model_version", sa.String(length=64), nullable=False, server_default="heuristic_v1"),
        sa.Column("parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("baseline_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("result_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("delta_pct", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_scenario_runs_field_id", "scenario_runs", ["field_id"])
    op.create_index("ix_scenario_runs_created_at", "scenario_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_scenario_runs_created_at", table_name="scenario_runs")
    op.drop_index("ix_scenario_runs_field_id", table_name="scenario_runs")
    op.drop_table("scenario_runs")

    op.drop_index("ix_field_metric_series_observed_at", table_name="field_metric_series")
    op.drop_index("ix_field_metric_series_metric", table_name="field_metric_series")
    op.drop_index("ix_field_metric_series_field_id", table_name="field_metric_series")
    op.drop_table("field_metric_series")

    op.drop_column("archive_entries", "model_meta")
    op.drop_column("archive_entries", "scenario_snapshot")
    op.drop_column("archive_entries", "weather_snapshot")
    op.drop_column("archive_entries", "metrics_snapshot")
    op.drop_column("archive_entries", "prediction_snapshot")
    op.drop_column("archive_entries", "field_snapshot")

    op.drop_column("yield_predictions", "data_quality")
    op.drop_column("yield_predictions", "explanation")
    op.drop_column("yield_predictions", "input_features")
