from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext
from storage.db import AuditEvent


async def record_audit_event(
    db: AsyncSession,
    *,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    payload: dict[str, Any] | None = None,
    ctx: RequestContext | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    organization_id: UUID | None = None,
    actor_user_id: UUID | None = None,
) -> None:
    event = AuditEvent(
        organization_id=organization_id or (ctx.organization_id if ctx else None),
        actor_user_id=actor_user_id or (ctx.user_id if ctx else None),
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        payload=dict(payload or {}),
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(event)
    await db.flush()
