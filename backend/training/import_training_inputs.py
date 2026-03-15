#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from core.settings import get_settings
from services.data_import_service import DataImportService
from storage.db import (
    CropAssignment,
    Field,
    FieldSeason,
    ManagementEvent,
    Organization,
    SoilProfile,
    User,
    WeatherDaily,
    YieldObservation,
    _slugify,
    get_session_factory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "training_inputs"
DEFAULT_REPORT = PROJECT_ROOT / "backend" / "debug" / "runs" / "training_inputs_import_report.json"


@dataclass(frozen=True, slots=True)
class ImportSpec:
    import_type: str
    filenames: tuple[str, ...]
    required_when_empty: bool = False


IMPORT_SPECS: tuple[ImportSpec, ...] = (
    ImportSpec("field_boundaries", ("field_boundaries.geojson", "field_boundaries.gpkg", "field_boundaries.json"), required_when_empty=False),
    ImportSpec("yield_history", ("yield_history.csv",), required_when_empty=True),
    ImportSpec("crop_plan", ("crop_plan.csv",), required_when_empty=True),
    ImportSpec("soil_samples", ("soil_samples.csv",), required_when_empty=False),
    ImportSpec("management_events", ("management_events.csv",), required_when_empty=False),
    ImportSpec("weather_daily", ("weather_daily.csv",), required_when_empty=False),
)


def _required_input_hints(missing_import_types: list[str]) -> list[str]:
    hints: list[str] = []
    for import_type in missing_import_types:
        spec = next((item for item in IMPORT_SPECS if item.import_type == import_type), None)
        if spec is None:
            hints.append(import_type)
            continue
        hints.extend(spec.filenames)
    return hints


async def _existing_counts() -> dict[str, int]:
    factory = get_session_factory()
    async with factory() as session:
        async def scalar_count(model: Any) -> int:
            return int((await session.execute(select(func.count()).select_from(model))).scalar_one() or 0)

        return {
            "fields": await scalar_count(Field),
            "field_seasons": await scalar_count(FieldSeason),
            "crop_assignments": await scalar_count(CropAssignment),
            "yield_observations": await scalar_count(YieldObservation),
            "soil_profiles": await scalar_count(SoilProfile),
            "management_events": await scalar_count(ManagementEvent),
            "weather_daily": await scalar_count(WeatherDaily),
        }


async def _bootstrap_actor() -> tuple[Any, Any]:
    settings = get_settings()
    factory = get_session_factory()
    async with factory() as session:
        org_slug = _slugify(settings.AUTH_BOOTSTRAP_ORG_NAME)
        organization = (await session.execute(select(Organization).where(Organization.slug == org_slug))).scalar_one_or_none()
        user = (await session.execute(select(User).where(User.email == settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower()))).scalar_one_or_none()
        if organization is None or user is None:
            raise RuntimeError("Bootstrap organization/user not found. Start the stack first so defaults can be seeded.")
        return organization, user


def _discover_files(input_dir: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    found: dict[str, Path] = {}
    manifest: list[dict[str, Any]] = []
    for spec in IMPORT_SPECS:
        file_path = next((input_dir / name for name in spec.filenames if (input_dir / name).exists()), None)
        if file_path is not None:
            found[spec.import_type] = file_path
        manifest.append(
            {
                "import_type": spec.import_type,
                "found": file_path is not None,
                "path": str(file_path.resolve()) if file_path is not None else None,
                "candidates": list(spec.filenames),
                "required_when_empty": spec.required_when_empty,
            }
        )
    return found, manifest


async def _run_imports(input_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    settings = get_settings()
    existing = await _existing_counts()
    found, manifest = _discover_files(input_dir)

    missing_required = [
        spec.import_type
        for spec in IMPORT_SPECS
        if spec.required_when_empty and spec.import_type not in found and existing.get(spec.import_type if spec.import_type != "yield_history" else "yield_observations", 0) == 0
    ]

    if "field_boundaries" not in found and existing.get("fields", 0) == 0:
        missing_required.insert(0, "field_boundaries")

    if not found and existing.get("yield_observations", 0) > 0 and existing.get("crop_assignments", 0) > 0:
        return {
            "status": "skipped_existing_data",
            "message": "No input files found, but agronomy tables already contain data. Import stage skipped.",
            "input_dir": str(input_dir.resolve()),
            "existing_counts": existing,
            "discovered_files": manifest,
            "imported_jobs": [],
        }

    if missing_required and dry_run:
        return {
            "status": "missing_inputs",
            "message": "Missing required training inputs.",
            "input_dir": str(input_dir.resolve()),
            "existing_counts": existing,
            "discovered_files": manifest,
            "missing_required": missing_required,
            "expected_filenames": _required_input_hints(missing_required),
            "imported_jobs": [],
        }

    if missing_required:
        expected = ", ".join(_required_input_hints(missing_required))
        raise RuntimeError(
            "Missing required training inputs. Place files into "
            f"{input_dir.resolve()}: {expected}"
        )

    organization, user = await _bootstrap_actor()
    if dry_run:
        return {
            "status": "dry_run",
            "message": "Import plan prepared.",
            "input_dir": str(input_dir.resolve()),
            "organization_slug": organization.slug,
            "user_email": user.email,
            "existing_counts": existing,
            "discovered_files": manifest,
            "imported_jobs": [],
        }

    factory = get_session_factory()
    imported_jobs: list[dict[str, Any]] = []
    async with factory() as session:
        service = DataImportService(session)
        for spec in IMPORT_SPECS:
            file_path = found.get(spec.import_type)
            if file_path is None:
                continue
            created = await service.create_import(
                organization_id=organization.id,
                created_by_user_id=user.id,
                import_type=spec.import_type,
                source_filename=file_path.name,
                content=file_path.read_bytes(),
            )
            committed = await service.commit_job(
                int(created["id"]),
                organization_id=organization.id,
                actor_user_id=user.id,
            )
            imported_jobs.append(
                {
                    "import_type": spec.import_type,
                    "job_id": int(created["id"]),
                    "source_filename": file_path.name,
                    "status": committed.get("status"),
                    "preview_summary": created.get("preview_summary"),
                    "commit_summary": committed.get("commit_summary"),
                }
            )
        await session.commit()

    refreshed = await _existing_counts()
    return {
        "status": "done",
        "message": "Training inputs imported.",
        "input_dir": str(input_dir.resolve()),
        "organization_slug": organization.slug,
        "user_email": user.email,
        "existing_counts_before": existing,
        "existing_counts_after": refreshed,
        "discovered_files": manifest,
        "imported_jobs": imported_jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import training CSV/GeoJSON inputs into the app DB before training")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = __import__("asyncio").run(_run_imports(args.input_dir.resolve(), dry_run=args.dry_run))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    args.report.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.report.resolve())
    print(json.dumps({"status": payload["status"], "message": payload["message"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
