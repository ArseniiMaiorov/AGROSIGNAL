from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import LayerInfo, LayersListResponse
from services.payload_meta import build_freshness
from storage.db import AoiRun, GridCell, Layer, get_db
from storage.grid_repo import GridRepository

router = APIRouter(prefix="/layers", tags=["layers"])


@router.get("", response_model=LayersListResponse)
async def list_layers(
    _ctx: RequestContext = Depends(require_permissions("layers:read")),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Layer))
    layers = result.scalars().all()
    fetched_at = datetime.now(timezone.utc)
    return LayersListResponse(
        layers=[
            LayerInfo(
                id=l.id,
                name=l.name,
                unit=l.unit,
                range_min=l.range_min,
                range_max=l.range_max,
                source=l.source,
                description=l.description,
                freshness=build_freshness(
                    provider="layer_catalog",
                    fetched_at=fetched_at,
                    cache_written_at=fetched_at,
                ),
            )
            for l in layers
        ]
    )


async def _find_fallback_run(
    db: AsyncSession,
    *,
    organization_id: UUID,
    bbox: tuple[float, float, float, float],
    zoom: int,
) -> UUID | None:
    """Find the most recent completed run that has grid data intersecting the bbox."""
    from geoalchemy2.shape import from_shape
    from shapely.geometry import box
    bbox_geom = from_shape(box(*bbox), srid=4326)
    result = await db.execute(
        select(GridCell.aoi_run_id)
        .where(GridCell.organization_id == organization_id)
        .where(GridCell.zoom_level == zoom)
        .where(GridCell.geom.ST_Intersects(bbox_geom))
        .join(AoiRun, AoiRun.id == GridCell.aoi_run_id)
        .where(AoiRun.organization_id == organization_id)
        .where(AoiRun.status == "done")
        .order_by(desc(AoiRun.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.get("/{layer_id}/grid")
async def get_layer_grid(
    layer_id: str,
    aoi_run_id: UUID = Query(...),
    zoom: int = Query(2, ge=0, le=4),
    allow_run_fallback: bool = Query(False),
    final_fields_only: bool = Query(False),
    minx: float = Query(...),
    miny: float = Query(...),
    maxx: float = Query(...),
    maxy: float = Query(...),
    ctx: RequestContext = Depends(require_permissions("layers:read")),
    db: AsyncSession = Depends(get_db),
):
    layer = await db.get(Layer, layer_id)
    if layer is None:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found")

    repo = GridRepository(db)
    bbox = (minx, miny, maxx, maxy)
    geojson = await repo.get_grid_cells(
        organization_id=ctx.organization_id,
        run_id=aoi_run_id,
        zoom=zoom,
        bbox=bbox,
        final_fields_only=final_fields_only,
    )

    used_fallback = False
    if allow_run_fallback and not geojson.get("features"):
        fallback_run_id = await _find_fallback_run(db, organization_id=ctx.organization_id, bbox=bbox, zoom=zoom)
        if fallback_run_id and fallback_run_id != aoi_run_id:
            geojson = await repo.get_grid_cells(
                organization_id=ctx.organization_id,
                run_id=fallback_run_id,
                zoom=zoom,
                bbox=bbox,
                final_fields_only=final_fields_only,
            )
            used_fallback = bool(geojson.get("features"))

    if not geojson.get("features"):
        geojson["_meta"] = {
            "status": "no_data",
            "reason": f"Нет данных сетки для слоя '{layer_id}' при zoom={zoom}. "
                      "Запустите автодетекцию для заполнения сетки данных.",
            "layer_id": layer_id,
            "aoi_run_id": str(aoi_run_id),
            "zoom": zoom,
        }
    elif used_fallback:
        geojson["_meta"] = {
            "status": "fallback",
            "reason": "Использованы данные предыдущего запуска автодетекции.",
            "fallback_run_id": str(fallback_run_id),
        }
    return geojson
