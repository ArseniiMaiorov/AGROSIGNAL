"""Celery-задачи архивирования и очистки архивов."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from core.celery_app import celery
from core.logging import get_logger
from services.archive_service import ArchiveService
from storage.db import get_session_factory

logger = get_logger(__name__)


@celery.task(name="tasks.archive.cleanup_expired")
def cleanup_expired_archives() -> dict[str, int]:
    """Удалить просроченные архивы и их записи из базы."""

    async def _run() -> dict[str, int]:
        factory = get_session_factory()
        async with factory() as session:
            service = ArchiveService(session)
            result = await service.cleanup_expired()
            await session.commit()
            return result

    result = asyncio.run(_run())
    logger.info("archive_cleanup_task_finished", removed_rows=result["removed_rows"], removed_files=result["removed_files"])
    return result
