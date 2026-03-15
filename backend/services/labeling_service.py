from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.audit_service import record_audit_event
from storage.db import LabelReview, LabelTask, LabelVersion, MlDatasetVersion


class LabelingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_tasks(self, *, organization_id: UUID, status: str | None = None) -> list[dict[str, Any]]:
        stmt = (
            select(LabelTask)
            .where(LabelTask.organization_id == organization_id)
            .order_by(LabelTask.priority_score.desc(), LabelTask.created_at.desc())
        )
        if status:
            stmt = stmt.where(LabelTask.status == status)
        tasks = (await self.db.execute(stmt)).scalars().all()
        return [await self._task_to_dict(task) for task in tasks]

    async def get_task(self, task_id: int, *, organization_id: UUID) -> dict[str, Any]:
        task = (
            await self.db.execute(
                select(LabelTask).where(LabelTask.organization_id == organization_id).where(LabelTask.id == task_id)
            )
        ).scalar_one_or_none()
        if task is None:
            raise ValueError("Label task not found")
        return await self._task_to_dict(task)

    async def create_task(
        self,
        *,
        organization_id: UUID,
        created_by_user_id: UUID,
        aoi_run_id: UUID | None,
        field_id: UUID | None,
        title: str,
        source: str,
        queue_name: str,
        priority_score: float,
        task_payload: dict[str, Any],
        geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task = LabelTask(
            organization_id=organization_id,
            aoi_run_id=aoi_run_id,
            field_id=field_id,
            created_by_user_id=created_by_user_id,
            title=title,
            source=source,
            queue_name=queue_name,
            priority_score=priority_score,
            task_payload=dict(task_payload or {}),
            status="queued",
        )
        self.db.add(task)
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="labeling.task.create",
            resource_type="label_task",
            resource_id=str(task.id),
            organization_id=organization_id,
            actor_user_id=created_by_user_id,
            payload={"title": title, "source": source, "queue_name": queue_name},
        )
        if geometry is not None:
            await self.add_version(
                task_id=int(task.id),
                organization_id=organization_id,
                created_by_user_id=created_by_user_id,
                geometry=geometry,
                notes="Initial label",
                quality_tier="draft",
            )
        return await self._task_to_dict(task)

    async def claim_task(self, task_id: int, *, organization_id: UUID, user_id: UUID) -> dict[str, Any]:
        task = await self._task(task_id, organization_id=organization_id)
        task.claimed_by_user_id = user_id
        task.claimed_at = func.now()
        task.status = "in_review"
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="labeling.task.claim",
            resource_type="label_task",
            resource_id=str(task.id),
            organization_id=organization_id,
            actor_user_id=user_id,
            payload={"status": task.status},
        )
        return await self._task_to_dict(task)

    async def add_version(
        self,
        *,
        task_id: int,
        organization_id: UUID,
        created_by_user_id: UUID,
        geometry: dict[str, Any],
        notes: str | None,
        quality_tier: str,
    ) -> dict[str, Any]:
        task = await self._task(task_id, organization_id=organization_id)
        version_no = int(
            (
                await self.db.execute(
                    select(func.coalesce(func.max(LabelVersion.version_no), 0)).where(
                        LabelVersion.organization_id == organization_id,
                        LabelVersion.label_task_id == task.id,
                    )
                )
            ).scalar_one()
        ) + 1
        checksum = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode("utf-8")).hexdigest()
        version = LabelVersion(
            organization_id=organization_id,
            label_task_id=task.id,
            created_by_user_id=created_by_user_id,
            version_no=version_no,
            geometry_geojson=geometry,
            quality_tier=quality_tier,
            checksum=checksum,
            notes=notes,
        )
        self.db.add(version)
        await self.db.flush()
        review = LabelReview(
            organization_id=organization_id,
            label_task_id=task.id,
            label_version_id=version.id,
            decision="pending",
        )
        self.db.add(review)
        task.status = "pending_review"
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="labeling.version.create",
            resource_type="label_version",
            resource_id=str(version.id),
            organization_id=organization_id,
            actor_user_id=created_by_user_id,
            payload={"label_task_id": task.id, "version_no": version_no, "quality_tier": quality_tier},
        )
        return await self._task_to_dict(task)

    async def approve_review(
        self,
        review_id: int,
        *,
        organization_id: UUID,
        reviewer_user_id: UUID,
        notes: str | None,
    ) -> dict[str, Any]:
        review = await self._review(review_id, organization_id=organization_id)
        review.decision = "approved"
        review.reviewer_user_id = reviewer_user_id
        review.notes = notes
        task = await self._task(review.label_task_id, organization_id=organization_id)
        task.status = "approved"
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="labeling.review.approve",
            resource_type="label_review",
            resource_id=str(review.id),
            organization_id=organization_id,
            actor_user_id=reviewer_user_id,
            payload={"label_task_id": task.id, "label_version_id": review.label_version_id},
        )
        return await self._task_to_dict(task)

    async def reject_review(
        self,
        review_id: int,
        *,
        organization_id: UUID,
        reviewer_user_id: UUID,
        notes: str | None,
    ) -> dict[str, Any]:
        review = await self._review(review_id, organization_id=organization_id)
        review.decision = "rejected"
        review.reviewer_user_id = reviewer_user_id
        review.notes = notes
        task = await self._task(review.label_task_id, organization_id=organization_id)
        task.status = "changes_requested"
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="labeling.review.reject",
            resource_type="label_review",
            resource_id=str(review.id),
            organization_id=organization_id,
            actor_user_id=reviewer_user_id,
            payload={"label_task_id": task.id, "label_version_id": review.label_version_id},
        )
        return await self._task_to_dict(task)

    async def export_manifest(self, *, organization_id: UUID, dataset_version: str) -> dict[str, Any]:
        tasks = (
            await self.db.execute(
                select(LabelTask)
                .where(LabelTask.organization_id == organization_id)
                .where(LabelTask.status == "approved")
                .order_by(LabelTask.priority_score.desc(), LabelTask.created_at.asc())
            )
        ).scalars().all()
        items = []
        for task in tasks:
            task_payload = await self._task_to_dict(task)
            latest_version = task_payload.get("latest_version") or {}
            items.append(
                {
                    "task_id": task.id,
                    "aoi_run_id": str(task.aoi_run_id) if task.aoi_run_id else None,
                    "field_id": str(task.field_id) if task.field_id else None,
                    "queue_name": task.queue_name,
                    "priority_score": task.priority_score,
                    "geometry": latest_version.get("geometry_geojson"),
                    "quality_tier": latest_version.get("quality_tier"),
                    "checksum": latest_version.get("checksum"),
                    "source": task.source,
                }
            )
        manifest = {
            "dataset_version": dataset_version,
            "items": items,
            "summary": {
                "approved_tasks": len(items),
            },
        }
        checksum = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        existing = (
            await self.db.execute(
                select(MlDatasetVersion)
                .where(MlDatasetVersion.organization_id == organization_id)
                .where(MlDatasetVersion.dataset_version == dataset_version)
            )
        ).scalar_one_or_none()
        if existing is None:
            dataset = MlDatasetVersion(
                organization_id=organization_id,
                dataset_version=dataset_version,
                checksum=checksum,
                code_sha="manual_gt_export",
                manifest_json=manifest,
                split_summary=manifest["summary"],
                artifact_uri=f"labeling://{organization_id}/{dataset_version}",
            )
            self.db.add(dataset)
            await self.db.flush()
        await record_audit_event(
            self.db,
            action="labeling.export_manifest",
            resource_type="ml_dataset_version",
            resource_id=dataset_version,
            organization_id=organization_id,
            payload={"approved_tasks": len(items), "checksum": checksum},
        )
        return {"dataset_version": dataset_version, "checksum": checksum, "manifest": manifest}

    async def _task(self, task_id: int, *, organization_id: UUID) -> LabelTask:
        task = (
            await self.db.execute(
                select(LabelTask).where(LabelTask.organization_id == organization_id).where(LabelTask.id == task_id)
            )
        ).scalar_one_or_none()
        if task is None:
            raise ValueError("Label task not found")
        return task

    async def _review(self, review_id: int, *, organization_id: UUID) -> LabelReview:
        review = (
            await self.db.execute(
                select(LabelReview).where(LabelReview.organization_id == organization_id).where(LabelReview.id == review_id)
            )
        ).scalar_one_or_none()
        if review is None:
            raise ValueError("Label review not found")
        return review

    async def _task_to_dict(self, task: LabelTask) -> dict[str, Any]:
        latest_version = (
            await self.db.execute(
                select(LabelVersion)
                .where(LabelVersion.label_task_id == task.id)
                .where(LabelVersion.organization_id == task.organization_id)
                .order_by(desc(LabelVersion.version_no))
                .limit(1)
            )
        ).scalar_one_or_none()
        latest_review = (
            await self.db.execute(
                select(LabelReview)
                .where(LabelReview.label_task_id == task.id)
                .where(LabelReview.organization_id == task.organization_id)
                .order_by(desc(LabelReview.created_at))
                .limit(1)
            )
        ).scalar_one_or_none()
        return {
            "id": int(task.id),
            "aoi_run_id": str(task.aoi_run_id) if task.aoi_run_id else None,
            "field_id": str(task.field_id) if task.field_id else None,
            "title": task.title,
            "status": task.status,
            "source": task.source,
            "queue_name": task.queue_name,
            "priority_score": float(task.priority_score),
            "task_payload": dict(task.task_payload or {}),
            "claimed_by_user_id": str(task.claimed_by_user_id) if task.claimed_by_user_id else None,
            "latest_version": None if latest_version is None else {
                "id": int(latest_version.id),
                "version_no": int(latest_version.version_no),
                "geometry_geojson": dict(latest_version.geometry_geojson or {}),
                "quality_tier": latest_version.quality_tier,
                "checksum": latest_version.checksum,
                "notes": latest_version.notes,
                "created_at": latest_version.created_at.isoformat() if latest_version.created_at else None,
            },
            "latest_review": None if latest_review is None else {
                "id": int(latest_review.id),
                "decision": latest_review.decision,
                "notes": latest_review.notes,
                "reviewer_user_id": str(latest_review.reviewer_user_id) if latest_review.reviewer_user_id else None,
                "created_at": latest_review.created_at.isoformat() if latest_review.created_at else None,
            },
            "created_at": task.created_at.isoformat() if task.created_at else None,
        }
