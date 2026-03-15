from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, require_permissions
from api.schemas import (
    MlBenchmarkListResponse,
    MlBenchmarkRegisterRequest,
    MlBenchmarkResponse,
    MlDatasetListResponse,
    MlDatasetRegisterRequest,
    MlDatasetVersionResponse,
    MlDeploymentListResponse,
    MlDeploymentResponse,
    MlModelRegistryEntryResponse,
    MlModelRegistryListResponse,
    MlPromotionRequest,
    MlRollbackRequest,
)
from services.mlops_service import MlOpsService
from storage.db import get_db

router = APIRouter(prefix="/admin/ml", tags=["admin-ml"])


@router.get("/datasets", response_model=MlDatasetListResponse)
async def list_datasets(
    ctx: RequestContext = Depends(require_permissions("mlops:read")),
    db: AsyncSession = Depends(get_db),
) -> MlDatasetListResponse:
    service = MlOpsService(db)
    items = await service.list_datasets(organization_id=ctx.organization_id)
    return MlDatasetListResponse(datasets=[MlDatasetVersionResponse(**item) for item in items])


@router.post("/datasets", response_model=MlDatasetVersionResponse)
async def register_dataset(
    payload: MlDatasetRegisterRequest,
    ctx: RequestContext = Depends(require_permissions("mlops:write")),
    db: AsyncSession = Depends(get_db),
) -> MlDatasetVersionResponse:
    service = MlOpsService(db)
    try:
        item = await service.register_dataset(
            organization_id=ctx.organization_id,
            actor_user_id=ctx.user_id,
            dataset_version=payload.dataset_version,
            checksum=payload.checksum,
            code_sha=payload.code_sha,
            manifest_json=payload.manifest_json,
            split_summary=payload.split_summary,
            artifact_uri=payload.artifact_uri,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MlDatasetVersionResponse(**item)


@router.get("/benchmarks", response_model=MlBenchmarkListResponse)
async def list_benchmarks(
    ctx: RequestContext = Depends(require_permissions("mlops:read")),
    db: AsyncSession = Depends(get_db),
) -> MlBenchmarkListResponse:
    service = MlOpsService(db)
    items = await service.list_benchmarks(organization_id=ctx.organization_id)
    return MlBenchmarkListResponse(benchmarks=[MlBenchmarkResponse(**item) for item in items])


@router.post("/benchmarks", response_model=MlBenchmarkResponse)
async def register_benchmark(
    payload: MlBenchmarkRegisterRequest,
    ctx: RequestContext = Depends(require_permissions("mlops:write")),
    db: AsyncSession = Depends(get_db),
) -> MlBenchmarkResponse:
    service = MlOpsService(db)
    try:
        item = await service.register_benchmark(
            organization_id=ctx.organization_id,
            actor_user_id=ctx.user_id,
            dataset_version_id=payload.dataset_version_id,
            benchmark_name=payload.benchmark_name,
            model_version=payload.model_version,
            metrics=payload.metrics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MlBenchmarkResponse(**item)


@router.get("/models", response_model=MlModelRegistryListResponse)
async def list_models(
    ctx: RequestContext = Depends(require_permissions("mlops:read")),
    db: AsyncSession = Depends(get_db),
) -> MlModelRegistryListResponse:
    service = MlOpsService(db)
    items = await service.list_models(organization_id=ctx.organization_id)
    return MlModelRegistryListResponse(models=[MlModelRegistryEntryResponse(**item) for item in items])


@router.get("/deployments", response_model=MlDeploymentListResponse)
async def list_deployments(
    ctx: RequestContext = Depends(require_permissions("mlops:read")),
    db: AsyncSession = Depends(get_db),
) -> MlDeploymentListResponse:
    service = MlOpsService(db)
    items = await service.list_deployments(organization_id=ctx.organization_id)
    return MlDeploymentListResponse(deployments=[MlDeploymentResponse(**item) for item in items])


@router.post("/promote", response_model=MlDeploymentResponse)
async def promote_model(
    payload: MlPromotionRequest,
    ctx: RequestContext = Depends(require_permissions("mlops:write")),
    db: AsyncSession = Depends(get_db),
) -> MlDeploymentResponse:
    service = MlOpsService(db)
    try:
        item = await service.promote(
            organization_id=ctx.organization_id,
            actor_user_id=ctx.user_id,
            deployment_name=payload.deployment_name,
            model_version=payload.model_version,
            benchmark_id=payload.benchmark_id,
            dataset_version_id=payload.dataset_version_id,
            model_uri=payload.model_uri,
            mlflow_run_id=payload.mlflow_run_id,
            config_snapshot=payload.config_snapshot,
            code_sha=payload.code_sha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MlDeploymentResponse(**item)


@router.post("/rollback", response_model=MlDeploymentResponse)
async def rollback_model(
    payload: MlRollbackRequest,
    ctx: RequestContext = Depends(require_permissions("mlops:write")),
    db: AsyncSession = Depends(get_db),
) -> MlDeploymentResponse:
    service = MlOpsService(db)
    try:
        item = await service.rollback(
            organization_id=ctx.organization_id,
            actor_user_id=ctx.user_id,
            deployment_id=payload.deployment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MlDeploymentResponse(**item)
