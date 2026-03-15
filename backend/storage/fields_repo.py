from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import geopandas as gpd
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from sqlalchemy import desc, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from services.field_quality import describe_field_quality, extract_runtime_geometry_quality
from storage.db import AoiRun, ArchiveEntry, Field, FieldMetricSeries, ScenarioRun, YieldPrediction


class FieldsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_run(
        self,
        aoi_wkt: str,
        time_start: date,
        time_end: date,
        params: dict[str, Any],
        *,
        organization_id: UUID,
        created_by_user_id: UUID | None = None,
    ) -> AoiRun:
        run = AoiRun(
            id=uuid.uuid4(),
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            aoi=f"SRID=4326;{aoi_wkt}",
            time_start=time_start,
            time_end=time_end,
            params=params,
            status="queued",
            progress=0,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def update_run_status(
        self,
        run_id: uuid.UUID,
        status: str,
        error_msg: str | None = None,
        progress: int | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if error_msg is not None:
            values["error_msg"] = error_msg
        if progress is not None:
            values["progress"] = progress
        stmt = update(AoiRun).where(AoiRun.id == run_id).values(**values)
        await self.session.execute(stmt)
        await self.session.commit()

    async def get_run(self, run_id: uuid.UUID, *, organization_id: UUID | None = None) -> AoiRun | None:
        stmt = select(AoiRun).where(AoiRun.id == run_id)
        if organization_id is not None:
            stmt = stmt.where(AoiRun.organization_id == organization_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_runs(self, *, organization_id: UUID, limit: int = 20) -> list[AoiRun]:
        stmt = (
            select(AoiRun)
            .where(AoiRun.organization_id == organization_id)
            .order_by(desc(AoiRun.created_at))
            .limit(max(1, min(int(limit), 100)))
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_fields(self, *, organization_id: UUID, run_id: uuid.UUID | None = None) -> list[Field]:
        stmt = (
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.source != "merged_hidden")
        )
        if run_id is not None:
            stmt = stmt.where(Field.aoi_run_id == run_id)
        stmt = stmt.order_by(Field.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_field(self, field_id: uuid.UUID, *, organization_id: UUID) -> Field | None:
        stmt = select(Field).where(Field.id == field_id).where(Field.organization_id == organization_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    def _normalize_polygonal_geometry(geom: Any) -> MultiPolygon:
        if geom.is_empty:
            raise ValueError("Геометрия пуста")
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            raise ValueError("Не удалось исправить геометрию")
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        if isinstance(geom, MultiPolygon):
            return geom
        if isinstance(geom, GeometryCollection):
            polygons: list[Polygon] = []
            for item in geom.geoms:
                if isinstance(item, Polygon) and not item.is_empty:
                    polygons.append(item)
                elif isinstance(item, MultiPolygon) and not item.is_empty:
                    polygons.extend(list(item.geoms))
            if polygons:
                return MultiPolygon(polygons)
        raise ValueError("Геометрия не содержит полигональных объектов")

    async def _load_archive_flags(self, field_ids: list[uuid.UUID], *, organization_id: UUID) -> dict[uuid.UUID, bool]:
        if not field_ids:
            return {}
        result = await self.session.execute(
            select(ArchiveEntry.field_id)
            .where(ArchiveEntry.organization_id == organization_id)
            .where(ArchiveEntry.field_id.in_(field_ids))
            .distinct()
        )
        archived_ids = {row[0] for row in result.all()}
        return {field_id: field_id in archived_ids for field_id in field_ids}

    async def _load_scenario_flags(self, field_ids: list[uuid.UUID], *, organization_id: UUID) -> dict[uuid.UUID, bool]:
        if not field_ids:
            return {}
        result = await self.session.execute(
            select(ScenarioRun.field_id)
            .where(ScenarioRun.organization_id == organization_id)
            .where(ScenarioRun.field_id.in_(field_ids))
            .distinct()
        )
        scenario_ids = {row[0] for row in result.all()}
        return {field_id: field_id in scenario_ids for field_id in field_ids}

    async def _load_run_runtime_map(
        self,
        run_ids: list[uuid.UUID],
        *,
        organization_id: UUID,
    ) -> dict[uuid.UUID, dict[str, Any]]:
        unique_ids = [run_id for run_id in dict.fromkeys(run_ids) if run_id is not None]
        if not unique_ids:
            return {}
        result = await self.session.execute(
            select(AoiRun.id, AoiRun.params)
            .where(AoiRun.organization_id == organization_id)
            .where(AoiRun.id.in_(unique_ids))
        )
        runtime_map: dict[uuid.UUID, dict[str, Any]] = {}
        for run_id, params in result.all():
            payload = dict(params or {})
            runtime_map[run_id] = dict(payload.get("runtime") or {})
        return runtime_map

    async def _build_geojson_for_fields(self, fields: list[Field], *, organization_id: UUID) -> dict[str, Any]:
        archive_flags = await self._load_archive_flags([f.id for f in fields], organization_id=organization_id)
        scenario_flags = await self._load_scenario_flags([f.id for f in fields], organization_id=organization_id)
        runtime_map = await self._load_run_runtime_map(
            [f.aoi_run_id for f in fields],
            organization_id=organization_id,
        )
        features = []
        for f in fields:
            geom = to_shape(f.geom)
            centroid = geom.centroid
            runtime_quality = extract_runtime_geometry_quality(
                runtime_map.get(f.aoi_run_id),
                lon=float(centroid.x),
                lat=float(centroid.y),
            )
            quality_meta = describe_field_quality(
                f.quality_score,
                f.source,
                geometry_confidence=runtime_quality.get("geometry_confidence"),
                tta_consensus=runtime_quality.get("tta_consensus"),
                boundary_uncertainty=runtime_quality.get("boundary_uncertainty"),
                uncertainty_source=runtime_quality.get("uncertainty_source"),
            )
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "field_id": str(f.id),
                    "aoi_run_id": str(f.aoi_run_id),
                    "area_m2": f.area_m2,
                    "perimeter_m": f.perimeter_m,
                    "quality_score": f.quality_score,
                    "quality_confidence": quality_meta["confidence"],
                    "geometry_confidence": quality_meta.get("geometry_confidence"),
                    "tta_consensus": quality_meta.get("tta_consensus"),
                    "boundary_uncertainty": quality_meta.get("boundary_uncertainty"),
                    "quality_band": quality_meta["band"],
                    "quality_label": quality_meta["label"],
                    "quality_reason": quality_meta["reason"],
                    "quality_reason_code": quality_meta.get("reason_code"),
                    "quality_reason_params": quality_meta.get("reason_params") or {},
                    "operational_tier": quality_meta.get("operational_tier"),
                    "review_required": bool(quality_meta.get("review_required")),
                    "review_reason": quality_meta.get("review_reason"),
                    "review_reason_code": quality_meta.get("review_reason_code"),
                    "review_reason_params": quality_meta.get("review_reason_params") or {},
                    "source": f.source,
                    "external_field_id": f.external_field_id,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                    "has_archive": bool(archive_flags.get(f.id, False)),
                    "has_scenarios": bool(scenario_flags.get(f.id, False)),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    async def create_manual_field(
        self,
        geometry_geojson: dict[str, Any],
        *,
        organization_id: UUID,
        created_by_user_id: UUID | None = None,
        quality_score: float | None = None,
        external_field_id: str | None = None,
    ) -> Field:
        geom = shape(geometry_geojson)
        if geom.is_empty:
            raise ValueError("Пустая геометрия ручного поля")
        geom = self._normalize_polygonal_geometry(geom)
        # `aoi_runs.time_start/time_end` are stored as naive UTC timestamps.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        run = AoiRun(
            id=uuid.uuid4(),
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            aoi=f"SRID=4326;{geom.envelope.wkt}",
            time_start=now - timedelta(days=1),
            time_end=now,
            params={"source": "manual_markup", "runtime": {"created_manual": True}},
            status="done",
            progress=100,
        )
        self.session.add(run)
        await self.session.flush()

        field = Field(
            id=uuid.uuid4(),
            organization_id=organization_id,
            aoi_run_id=run.id,
            geom=from_shape(geom, srid=4326),
            area_m2=0.0,
            perimeter_m=0.0,
            quality_score=quality_score,
            source="manual",
            external_field_id=external_field_id,
        )
        self.session.add(field)
        await self.session.flush()
        await self.session.execute(
            text(
                """
                UPDATE fields
                SET area_m2 = ST_Area(geom::geography),
                    perimeter_m = ST_Perimeter(geom::geography)
                WHERE id = :field_id
                """
            ),
            {"field_id": str(field.id)},
        )
        await self.session.flush()
        await self.session.commit()
        await self.session.refresh(field)
        return field

    async def get_manual_fields_geojson(self, *, organization_id: UUID) -> dict[str, Any]:
        stmt = (
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.source == "manual")
            .order_by(Field.created_at.desc())
        )
        result = await self.session.execute(stmt)
        fields = result.scalars().all()
        return await self._build_geojson_for_fields(list(fields), organization_id=organization_id)

    async def get_all_fields_geojson(self, *, organization_id: UUID, run_id: uuid.UUID | None = None) -> dict[str, Any]:
        if run_id is None:
            latest_run_result = await self.session.execute(
                select(Field.aoi_run_id)
                .where(Field.organization_id == organization_id)
                .where(Field.source != "manual")
                .where(Field.source != "merged_hidden")
                .order_by(Field.created_at.desc())
                .limit(1)
            )
            run_id = latest_run_result.scalar_one_or_none()

        stmt = (
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.source != "manual")
            .where(Field.source != "merged_hidden")
        )
        if run_id is not None:
            stmt = stmt.where(Field.aoi_run_id == run_id)
        stmt = stmt.order_by(Field.created_at.desc())
        result = await self.session.execute(stmt)
        fields = result.scalars().all()
        return await self._build_geojson_for_fields(list(fields), organization_id=organization_id)

    async def insert_fields(self, run_id: uuid.UUID, fields_gdf: gpd.GeoDataFrame, *, organization_id: UUID) -> int:
        count = 0
        for _, row in fields_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == "Polygon":
                geom = MultiPolygon([geom])
            field = Field(
                id=uuid.uuid4(),
                organization_id=organization_id,
                aoi_run_id=run_id,
                geom=from_shape(geom, srid=4326),
                area_m2=row.get("area_m2", 0.0),
                perimeter_m=row.get("perimeter_m", 0.0),
                quality_score=row.get("quality_score"),
                source="autodetect",
            )
            self.session.add(field)
            count += 1
        await self.session.flush()
        return count

    async def get_fields_geojson(self, run_id: uuid.UUID, *, organization_id: UUID) -> dict[str, Any]:
        stmt = (
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.aoi_run_id == run_id)
            .where(Field.source != "merged_hidden")
        )
        result = await self.session.execute(stmt)
        fields = result.scalars().all()
        return await self._build_geojson_for_fields(list(fields), organization_id=organization_id)

    async def merge_fields(self, field_ids: list[uuid.UUID], *, organization_id: UUID) -> Field:
        ordered_ids = list(dict.fromkeys(field_ids))
        if len(ordered_ids) < 2:
            raise ValueError("Для объединения нужно выбрать минимум два поля")

        result = await self.session.execute(
            select(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.id.in_(ordered_ids))
            .where(Field.source != "merged_hidden")
        )
        found = {field.id: field for field in result.scalars().all()}
        fields = [found[field_id] for field_id in ordered_ids if field_id in found]
        if len(fields) < 2:
            raise ValueError("Не удалось найти все выбранные поля для объединения")

        merged_geom = self._normalize_polygonal_geometry(unary_union([to_shape(field.geom) for field in fields]))
        target = fields[0]
        others = fields[1:]
        other_ids = [field.id for field in others]

        target.geom = from_shape(merged_geom, srid=4326)
        target.source = "manual"
        await self.session.flush()
        await self.session.execute(
            update(ArchiveEntry)
            .where(ArchiveEntry.organization_id == organization_id)
            .where(ArchiveEntry.field_id.in_(other_ids))
            .values(field_id=target.id)
        )
        await self.session.execute(
            update(YieldPrediction)
            .where(YieldPrediction.organization_id == organization_id)
            .where(YieldPrediction.field_id.in_(other_ids))
            .values(field_id=target.id)
        )
        await self.session.execute(
            update(ScenarioRun)
            .where(ScenarioRun.organization_id == organization_id)
            .where(ScenarioRun.field_id.in_(other_ids))
            .values(field_id=target.id)
        )
        await self.session.execute(
            update(FieldMetricSeries)
            .where(FieldMetricSeries.organization_id == organization_id)
            .where(FieldMetricSeries.field_id.in_(other_ids))
            .values(field_id=target.id)
        )
        await self.session.execute(
            update(Field)
            .where(Field.organization_id == organization_id)
            .where(Field.id.in_(other_ids))
            .values(source="merged_hidden")
        )
        await self.session.execute(
            text(
                """
                UPDATE fields
                SET area_m2 = ST_Area(geom::geography),
                    perimeter_m = ST_Perimeter(geom::geography)
                WHERE id = :field_id
                """
            ),
            {"field_id": str(target.id)},
        )
        await self.session.commit()
        await self.session.refresh(target)
        return target

    async def split_field(self, field_id: uuid.UUID, line_geojson: dict[str, Any], *, organization_id: UUID) -> list[Field]:
        field = await self.get_field(field_id, organization_id=organization_id)
        if field is None or field.source == "merged_hidden":
            raise ValueError("Поле для разделения не найдено")

        splitter = shape(line_geojson)
        if splitter.is_empty:
            raise ValueError("Линия разделения пуста")
        if not isinstance(splitter, (LineString, MultiLineString)):
            raise ValueError("Разделение поддерживает только линию")

        geom = to_shape(field.geom)
        parts_geom = geom.difference(splitter.buffer(1e-7))
        parts = []
        if isinstance(parts_geom, Polygon):
            parts = [parts_geom]
        elif isinstance(parts_geom, MultiPolygon):
            parts = list(parts_geom.geoms)
        elif isinstance(parts_geom, GeometryCollection):
            parts = [part for part in parts_geom.geoms if isinstance(part, Polygon) and not part.is_empty]
        parts = [part for part in parts if not part.is_empty]
        if len(parts) < 2:
            raise ValueError("Линия не разделила поле на несколько частей")

        parts.sort(key=lambda item: item.area, reverse=True)
        updated_fields: list[Field] = []
        primary_geom = self._normalize_polygonal_geometry(parts[0])
        field.geom = from_shape(primary_geom, srid=4326)
        field.source = "manual"
        updated_fields.append(field)

        for part in parts[1:]:
            normalized = self._normalize_polygonal_geometry(part)
            new_field = Field(
                id=uuid.uuid4(),
                organization_id=organization_id,
                aoi_run_id=field.aoi_run_id,
                geom=from_shape(normalized, srid=4326),
                area_m2=0.0,
                perimeter_m=0.0,
                quality_score=field.quality_score,
                source="manual",
                external_field_id=field.external_field_id,
            )
            self.session.add(new_field)
            updated_fields.append(new_field)

        await self.session.flush()
        for item in updated_fields:
            await self.session.execute(
                text(
                    """
                    UPDATE fields
                    SET area_m2 = ST_Area(geom::geography),
                        perimeter_m = ST_Perimeter(geom::geography)
                    WHERE id = :field_id
                    """
                ),
                {"field_id": str(item.id)},
            )
        await self.session.commit()
        for item in updated_fields:
            await self.session.refresh(item)
        return updated_fields

    async def delete_field(self, field_id: uuid.UUID, *, organization_id: UUID) -> dict[str, Any]:
        field = await self.get_field(field_id, organization_id=organization_id)
        if field is None or field.source == "merged_hidden":
            raise ValueError("Поле для удаления не найдено")

        payload = {
            "field_id": field.id,
            "aoi_run_id": field.aoi_run_id,
            "deleted": True,
        }
        run = await self.get_run(field.aoi_run_id, organization_id=organization_id)
        should_delete_parent_run = False
        if run is not None and isinstance(run.params, dict):
            run_source = str(run.params.get("source") or "").strip().lower()
            if run_source == "manual_markup":
                remaining_count = int(
                    (
                        await self.session.execute(
                            select(func.count())
                            .select_from(Field)
                            .where(Field.organization_id == organization_id)
                            .where(Field.aoi_run_id == field.aoi_run_id)
                            .where(Field.source != "merged_hidden")
                        )
                    ).scalar_one()
                    or 0
                )
                should_delete_parent_run = remaining_count <= 1

        # Always delete the field first
        await self.session.delete(field)
        # If it was the last field in a manual_markup run, also clean up the run
        if should_delete_parent_run and run is not None:
            await self.session.flush()
            await self.session.delete(run)

        await self.session.commit()
        return payload

    async def topology_cleanup(self, run_id: uuid.UUID, *, organization_id: uuid.UUID | None = None) -> None:
        params: dict[str, Any] = {"run_id": str(run_id)}
        org_clause = ""
        if organization_id is not None:
            org_clause = " AND organization_id = :organization_id"
            params["organization_id"] = str(organization_id)

        await self.session.execute(text(f"""
            UPDATE fields
            SET geom = ST_Multi(ST_MakeValid(geom))
            WHERE aoi_run_id = :run_id AND NOT ST_IsValid(geom){org_clause}
        """), params)

        await self.session.execute(text(f"""
            UPDATE fields f1
            SET geom = ST_Multi(ST_CollectionExtract(
                ST_Difference(f1.geom, (
                    SELECT ST_Union(f2.geom)
                    FROM fields f2
                    WHERE f2.aoi_run_id = :run_id
                      AND f2.id != f1.id
                      AND f2.area_m2 > f1.area_m2
                      AND ST_Intersects(f1.geom, f2.geom)
                )), 3))
            WHERE f1.aoi_run_id = :run_id
              AND EXISTS (
                  SELECT 1 FROM fields f2
                  WHERE f2.aoi_run_id = :run_id
                    AND f2.id != f1.id
                    AND f2.area_m2 > f1.area_m2
                    AND ST_Intersects(f1.geom, f2.geom)
              ){org_clause}
        """), params)

        await self.session.execute(text(f"""
            DELETE FROM fields
            WHERE aoi_run_id = :run_id
              AND ST_Area(geom::geography) < 200{org_clause}
        """), params)

        await self.session.execute(text(f"""
            UPDATE fields
            SET geom = ST_Multi(ST_SimplifyPreserveTopology(geom, 0.000025))
            WHERE aoi_run_id = :run_id{org_clause}
        """), params)

        await self.session.execute(text(f"""
            UPDATE fields
            SET area_m2 = ST_Area(geom::geography),
                perimeter_m = ST_Perimeter(geom::geography)
            WHERE aoi_run_id = :run_id{org_clause}
        """), params)

        await self.session.commit()
