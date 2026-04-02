from __future__ import annotations

import uuid
from uuid import UUID

import geopandas as gpd
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import box, mapping
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.db import Field, GridCell


class GridRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def insert_grid_cells(self, run_id: uuid.UUID, cells_gdf: gpd.GeoDataFrame, *, organization_id: UUID) -> int:
        count = 0
        for _, row in cells_gdf.iterrows():
            cell = GridCell(
                organization_id=organization_id,
                aoi_run_id=run_id,
                geom=from_shape(row.geometry, srid=4326),
                zoom_level=int(row.get("zoom_level", 0)),
                row=int(row.get("row", 0)),
                col=int(row.get("col", 0)),
                field_coverage=row.get("field_coverage"),
                ndvi_mean=row.get("ndvi_mean"),
                ndwi_mean=row.get("ndwi_mean"),
                ndmi_mean=row.get("ndmi_mean"),
                bsi_mean=row.get("bsi_mean"),
                precipitation_mm=row.get("precipitation_mm"),
                wind_speed_m_s=row.get("wind_speed_m_s"),
                u_wind_10m=row.get("u_wind_10m"),
                v_wind_10m=row.get("v_wind_10m"),
                wind_direction_deg=row.get("wind_direction_deg"),
                gdd_sum=row.get("gdd_sum"),
                vpd_mean=row.get("vpd_mean"),
                soil_moist=row.get("soil_moist"),
            )
            self.session.add(cell)
            count += 1
        await self.session.flush()
        return count

    async def get_grid_cells(
        self,
        *,
        organization_id: UUID,
        run_id: uuid.UUID,
        zoom: int,
        bbox: tuple[float, float, float, float],
        final_fields_only: bool = False,
    ) -> dict:
        bbox_geom = box(*bbox)
        stmt = (
            select(GridCell)
            .where(GridCell.organization_id == organization_id)
            .where(GridCell.aoi_run_id == run_id)
            .where(GridCell.zoom_level == zoom)
            .where(GridCell.geom.ST_Intersects(from_shape(bbox_geom, srid=4326)))
        )
        if final_fields_only:
            field_intersection_exists = (
                select(Field.id)
                .where(Field.organization_id == organization_id)
                .where(Field.aoi_run_id == run_id)
                .where(GridCell.geom.ST_Intersects(Field.geom))
                .exists()
            )
            stmt = stmt.where(field_intersection_exists)
        result = await self.session.execute(stmt)
        cells = result.scalars().all()

        features = []
        for c in cells:
            geom = to_shape(c.geom)
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "id": c.id,
                    "zoom_level": c.zoom_level,
                    "row": c.row,
                    "col": c.col,
                    "field_coverage": c.field_coverage,
                    "ndvi_mean": c.ndvi_mean,
                    "ndwi_mean": c.ndwi_mean,
                    "ndmi_mean": c.ndmi_mean,
                    "bsi_mean": c.bsi_mean,
                    "precipitation_mm": c.precipitation_mm,
                    "wind_speed_m_s": c.wind_speed_m_s,
                    "u_wind_10m": c.u_wind_10m,
                    "v_wind_10m": c.v_wind_10m,
                    "wind_direction_deg": c.wind_direction_deg,
                    "gdd_sum": c.gdd_sum,
                    "vpd_mean": c.vpd_mean,
                    "soil_moist": c.soil_moist,
                },
            })
        return {"type": "FeatureCollection", "features": features}
