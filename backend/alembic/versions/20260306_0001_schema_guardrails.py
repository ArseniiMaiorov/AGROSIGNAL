"""Схемные guardrails для PostGIS-таблиц.

Revision ID: 20260306_0001
Revises:
Create Date: 2026-03-06 00:00:00
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20260306_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_UPGRADE_SQL: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS postgis",
    "CREATE INDEX IF NOT EXISTS idx_aoi_runs_aoi_gist ON aoi_runs USING GIST (aoi)",
    "CREATE INDEX IF NOT EXISTS idx_fields_geom_gist ON fields USING GIST (geom)",
    "CREATE INDEX IF NOT EXISTS idx_grid_cells_geom_gist ON grid_cells USING GIST (geom)",
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

_DOWNGRADE_SQL: tuple[str, ...] = (
    "ALTER TABLE grid_cells DROP CONSTRAINT IF EXISTS grid_cells_geom_valid_chk",
    "ALTER TABLE fields DROP CONSTRAINT IF EXISTS fields_geom_valid_chk",
    "ALTER TABLE aoi_runs DROP CONSTRAINT IF EXISTS aoi_runs_aoi_valid_chk",
    "DROP INDEX IF EXISTS idx_grid_cells_geom_gist",
    "DROP INDEX IF EXISTS idx_fields_geom_gist",
    "DROP INDEX IF EXISTS idx_aoi_runs_aoi_gist",
)


def upgrade() -> None:
    for statement in _UPGRADE_SQL:
        op.execute(statement)


def downgrade() -> None:
    for statement in _DOWNGRADE_SQL:
        op.execute(statement)
