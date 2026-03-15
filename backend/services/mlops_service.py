from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.settings import get_settings
from services.audit_service import record_audit_event
from storage.db import MlBenchmark, MlDatasetVersion, MlDeployment

logger = get_logger(__name__)


class MlOpsService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def list_datasets(self, *, organization_id: UUID) -> list[dict[str, Any]]:
        stmt = (
            select(MlDatasetVersion)
            .where(MlDatasetVersion.organization_id == organization_id)
            .order_by(MlDatasetVersion.created_at.desc(), MlDatasetVersion.id.desc())
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return [self._dataset_to_dict(row) for row in rows]

    async def register_dataset(
        self,
        *,
        organization_id: UUID,
        actor_user_id: UUID,
        dataset_version: str,
        checksum: str,
        code_sha: str,
        manifest_json: dict[str, Any],
        split_summary: dict[str, Any],
        artifact_uri: str | None,
    ) -> dict[str, Any]:
        self._validate_manifest(manifest_json)
        existing = (
            await self.db.execute(
                select(MlDatasetVersion)
                .where(MlDatasetVersion.organization_id == organization_id)
                .where(MlDatasetVersion.dataset_version == dataset_version)
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = MlDatasetVersion(
                organization_id=organization_id,
                dataset_version=dataset_version,
                checksum=checksum,
                code_sha=code_sha,
                status="ready",
                manifest_json=manifest_json,
                split_summary=split_summary,
                artifact_uri=artifact_uri,
            )
            self.db.add(existing)
        else:
            existing.checksum = checksum
            existing.code_sha = code_sha
            existing.manifest_json = manifest_json
            existing.split_summary = split_summary
            existing.artifact_uri = artifact_uri
            existing.status = "ready"
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="mlops.dataset.register",
            resource_type="ml_dataset_version",
            resource_id=str(existing.id),
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            payload={"dataset_version": dataset_version, "checksum": checksum},
        )
        return self._dataset_to_dict(existing)

    async def list_benchmarks(self, *, organization_id: UUID) -> list[dict[str, Any]]:
        stmt = (
            select(MlBenchmark)
            .where(MlBenchmark.organization_id == organization_id)
            .order_by(MlBenchmark.created_at.desc(), MlBenchmark.id.desc())
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return [self._benchmark_to_dict(row) for row in rows]

    async def register_benchmark(
        self,
        *,
        organization_id: UUID,
        actor_user_id: UUID,
        dataset_version_id: int,
        benchmark_name: str,
        model_version: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        dataset = await self._dataset(dataset_version_id, organization_id=organization_id)
        gates = self._evaluate_promotion_gates(metrics)
        benchmark = MlBenchmark(
            organization_id=organization_id,
            dataset_version_id=dataset.id,
            benchmark_name=benchmark_name,
            model_version=model_version,
            metrics={**metrics, "promotion_gate_report": gates},
            gates_passed=bool(gates["passed"]),
        )
        self.db.add(benchmark)
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="mlops.benchmark.register",
            resource_type="ml_benchmark",
            resource_id=str(benchmark.id),
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            payload={"dataset_version_id": dataset_version_id, "benchmark_name": benchmark_name, "gates": gates},
        )
        return self._benchmark_to_dict(benchmark)

    async def list_deployments(self, *, organization_id: UUID) -> list[dict[str, Any]]:
        stmt = (
            select(MlDeployment)
            .where(MlDeployment.organization_id == organization_id)
            .order_by(MlDeployment.created_at.desc(), MlDeployment.id.desc())
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return [self._deployment_to_dict(row) for row in rows]

    async def list_models(self, *, organization_id: UUID) -> list[dict[str, Any]]:
        deployments = await self.list_deployments(organization_id=organization_id)
        latest_by_model: dict[str, dict[str, Any]] = {}
        for item in deployments:
            latest_by_model.setdefault(item["model_version"], item)
        return [
            {
                "model_version": model_version,
                "latest_deployment_id": payload["id"],
                "deployment_name": payload["deployment_name"],
                "dataset_version_id": payload["dataset_version_id"],
                "benchmark_id": payload["benchmark_id"],
                "model_uri": payload["model_uri"],
                "status": payload["status"],
                "created_at": payload["created_at"],
            }
            for model_version, payload in latest_by_model.items()
        ]

    async def promote(
        self,
        *,
        organization_id: UUID,
        actor_user_id: UUID,
        deployment_name: str,
        model_version: str,
        benchmark_id: int,
        dataset_version_id: int,
        model_uri: str | None,
        mlflow_run_id: str | None,
        config_snapshot: dict[str, Any],
        code_sha: str,
    ) -> dict[str, Any]:
        dataset = await self._dataset(dataset_version_id, organization_id=organization_id)
        benchmark = await self._benchmark(benchmark_id, organization_id=organization_id)
        gate_report = dict((benchmark.metrics or {}).get("promotion_gate_report") or {})
        if benchmark.dataset_version_id != dataset.id:
            raise ValueError("Benchmark and dataset_version_id mismatch")
        if not benchmark.gates_passed or not gate_report.get("passed", False):
            reason = "; ".join(gate_report.get("reasons") or ["Promotion gates failed"])
            raise ValueError(reason)

        active_stmt = (
            select(MlDeployment)
            .where(MlDeployment.organization_id == organization_id)
            .where(MlDeployment.deployment_name == deployment_name)
            .where(MlDeployment.status == "promoted")
        )
        active_rows = (await self.db.execute(active_stmt)).scalars().all()
        for row in active_rows:
            row.status = "superseded"

        deployment = MlDeployment(
            organization_id=organization_id,
            deployment_name=deployment_name,
            model_version=model_version,
            dataset_version_id=dataset.id,
            benchmark_id=benchmark.id,
            mlflow_run_id=mlflow_run_id,
            model_uri=model_uri,
            config_snapshot={
                **config_snapshot,
                "mlflow_tracking_uri": self.settings.MLFLOW_TRACKING_URI,
                "artifact_bucket": self.settings.MINIO_BUCKET,
            },
            code_sha=code_sha,
            status="promoted",
            promoted_by_user_id=actor_user_id,
        )
        self.db.add(deployment)
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="mlops.promote",
            resource_type="ml_deployment",
            resource_id=str(deployment.id),
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            payload={
                "deployment_name": deployment_name,
                "model_version": model_version,
                "benchmark_id": benchmark_id,
                "dataset_version_id": dataset_version_id,
            },
        )
        logger.info(
            "ml_deployment_promoted",
            organization_id=str(organization_id),
            deployment_id=deployment.id,
            model_version=model_version,
            benchmark_id=benchmark_id,
        )
        return self._deployment_to_dict(deployment)

    async def rollback(self, *, organization_id: UUID, actor_user_id: UUID, deployment_id: int) -> dict[str, Any]:
        deployment = await self._deployment(deployment_id, organization_id=organization_id)
        deployment.status = "rolled_back"
        deployment.rolled_back_at = datetime.now(timezone.utc)

        previous_stmt = (
            select(MlDeployment)
            .where(MlDeployment.organization_id == organization_id)
            .where(MlDeployment.deployment_name == deployment.deployment_name)
            .where(MlDeployment.id != deployment.id)
            .order_by(MlDeployment.created_at.desc(), MlDeployment.id.desc())
        )
        previous = (await self.db.execute(previous_stmt)).scalars().first()
        if previous is not None and previous.status in {"superseded", "rolled_back"}:
            previous.status = "promoted"
            previous.rolled_back_at = None

        await self.db.flush()
        await record_audit_event(
            self.db,
            action="mlops.rollback",
            resource_type="ml_deployment",
            resource_id=str(deployment.id),
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            payload={"restored_previous_id": previous.id if previous is not None else None},
        )
        return self._deployment_to_dict(deployment)

    def _evaluate_promotion_gates(self, metrics: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        manual_holdout_size = int(metrics.get("manual_holdout_size") or metrics.get("holdout_size") or 0)
        if manual_holdout_size < 300:
            reasons.append("manual holdout must contain at least 300 AOI")

        geo_iou = self._maybe_float(metrics.get("iou_geo"))
        baseline_geo_iou = self._maybe_float(metrics.get("baseline_iou_geo"))
        if geo_iou is None:
            reasons.append("iou_geo is required")
        elif baseline_geo_iou is not None and geo_iou < baseline_geo_iou - 0.02:
            reasons.append("global geo IoU dropped by more than 0.02")

        hd95_p90 = self._maybe_float(metrics.get("hd95_m_p90") or metrics.get("hd95_m"))
        baseline_hd95_p90 = self._maybe_float(metrics.get("baseline_hd95_m_p90"))
        if hd95_p90 is None:
            reasons.append("hd95_m_p90 is required")
        elif baseline_hd95_p90 is not None and baseline_hd95_p90 > 0 and hd95_p90 > baseline_hd95_p90 * 1.10:
            reasons.append("hd95_m p90 worsened by more than 10%")

        regional_report = self._evaluate_regional_targets(metrics)
        if not regional_report["passed"]:
            reasons.extend(regional_report["reasons"])

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
            "manual_holdout_size": manual_holdout_size,
            "regional_targets": regional_report,
            "reference_plan": str(Path("backend/training/REGIONAL_RETRAIN_PLAN.md")),
        }

    def _evaluate_regional_targets(self, metrics: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        checks = {
            "field_recall_south": (">=", 0.92),
            "missed_fields_rate_south": ("<=", 0.08),
            "oversegmented_fields_rate_south": ("<=", 0.12),
            "mean_components_per_gt_field_south": ("<=", 1.20),
            "boundary_iou_south_median": (">=", 0.78),
            "boundary_iou_north_median": (">=", 0.82),
            "centroid_shift_m_north_p90": ("<=", 5.0),
            "north_inward_shrink_obvious_rate": ("<=", 0.10),
        }
        for key, (operator, threshold) in checks.items():
            value = self._maybe_float(metrics.get(key))
            if value is None:
                reasons.append(f"{key} is required")
                continue
            if operator == ">=" and value < threshold:
                reasons.append(f"{key} must be >= {threshold}")
            if operator == "<=" and value > threshold:
                reasons.append(f"{key} must be <= {threshold}")

        contour_shrink = self._maybe_float(metrics.get("contour_shrink_ratio_north_median"))
        if contour_shrink is None:
            reasons.append("contour_shrink_ratio_north_median is required")
        elif not (0.96 <= contour_shrink <= 1.04):
            reasons.append("contour_shrink_ratio_north_median must be in 0.96-1.04")

        return {"passed": len(reasons) == 0, "reasons": reasons}

    def _validate_manifest(self, manifest_json: dict[str, Any]) -> None:
        items = list(manifest_json.get("items") or [])
        if not items:
            raise ValueError("manifest_json.items must not be empty")
        split_names = {str(item.get("split") or "").strip().lower() for item in items}
        if not {"train", "calibration", "holdout"}.issubset(split_names):
            raise ValueError("manifest_json must include train, calibration and holdout splits")

        split_aoi_ids: dict[str, set[str]] = {"train": set(), "calibration": set(), "holdout": set()}
        for item in items:
            split = str(item.get("split") or "").strip().lower()
            if split not in split_aoi_ids:
                continue
            aoi_run_id = item.get("aoi_run_id")
            if aoi_run_id:
                if aoi_run_id in split_aoi_ids[split]:
                    continue
                for other_split, values in split_aoi_ids.items():
                    if other_split != split and aoi_run_id in values:
                        raise ValueError("dataset manifest leakage: aoi_run_id appears in multiple splits")
                split_aoi_ids[split].add(str(aoi_run_id))

    async def _dataset(self, dataset_id: int, *, organization_id: UUID) -> MlDatasetVersion:
        dataset = (
            await self.db.execute(
                select(MlDatasetVersion)
                .where(MlDatasetVersion.organization_id == organization_id)
                .where(MlDatasetVersion.id == dataset_id)
            )
        ).scalar_one_or_none()
        if dataset is None:
            raise ValueError("Dataset version not found")
        return dataset

    async def _benchmark(self, benchmark_id: int, *, organization_id: UUID) -> MlBenchmark:
        benchmark = (
            await self.db.execute(
                select(MlBenchmark)
                .where(MlBenchmark.organization_id == organization_id)
                .where(MlBenchmark.id == benchmark_id)
            )
        ).scalar_one_or_none()
        if benchmark is None:
            raise ValueError("Benchmark not found")
        return benchmark

    async def _deployment(self, deployment_id: int, *, organization_id: UUID) -> MlDeployment:
        deployment = (
            await self.db.execute(
                select(MlDeployment)
                .where(MlDeployment.organization_id == organization_id)
                .where(MlDeployment.id == deployment_id)
            )
        ).scalar_one_or_none()
        if deployment is None:
            raise ValueError("Deployment not found")
        return deployment

    @staticmethod
    def _dataset_to_dict(row: MlDatasetVersion) -> dict[str, Any]:
        return {
            "id": int(row.id),
            "dataset_version": row.dataset_version,
            "checksum": row.checksum,
            "code_sha": row.code_sha,
            "status": row.status,
            "manifest_json": dict(row.manifest_json or {}),
            "split_summary": dict(row.split_summary or {}),
            "artifact_uri": row.artifact_uri,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def _benchmark_to_dict(row: MlBenchmark) -> dict[str, Any]:
        return {
            "id": int(row.id),
            "dataset_version_id": int(row.dataset_version_id),
            "benchmark_name": row.benchmark_name,
            "model_version": row.model_version,
            "metrics": dict(row.metrics or {}),
            "gates_passed": bool(row.gates_passed),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def _deployment_to_dict(row: MlDeployment) -> dict[str, Any]:
        return {
            "id": int(row.id),
            "deployment_name": row.deployment_name,
            "model_version": row.model_version,
            "dataset_version_id": int(row.dataset_version_id) if row.dataset_version_id is not None else None,
            "benchmark_id": int(row.benchmark_id) if row.benchmark_id is not None else None,
            "model_uri": row.model_uri,
            "config_snapshot": dict(row.config_snapshot or {}),
            "code_sha": row.code_sha,
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "rolled_back_at": row.rolled_back_at.isoformat() if row.rolled_back_at else None,
        }

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
