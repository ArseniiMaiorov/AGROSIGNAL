from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import TokenIdentity, decode_jwt
from core.settings import get_settings
from storage.db import Membership, Permission, RefreshToken, Role, User, get_db, role_permissions


@dataclass(slots=True)
class RequestContext:
    organization_id: UUID
    user_id: UUID
    email: str
    role_names: tuple[str, ...]
    permissions: frozenset[str]


async def _release_db_connection(db: AsyncSession) -> None:
    """Release the checked-out connection after auth lookups.

    Request handlers may spend a long time in external APIs after auth succeeds.
    Rolling back the read-only auth transaction lets SQLAlchemy return the
    connection to the pool until the route actually needs the session again.
    """
    try:
        if db.in_transaction():
            await db.rollback()
    except Exception:
        # Best-effort pool pressure relief. Route code can still proceed even if
        # SQLAlchemy already released the transaction underneath.
        return


async def _build_context(
    *,
    db: AsyncSession,
    user_id: UUID,
    organization_id: UUID,
    email: str,
) -> RequestContext:
    membership_stmt = (
        select(Membership, Role)
        .join(Role, Membership.role_id == Role.id)
        .where(Membership.user_id == user_id)
        .where(Membership.organization_id == organization_id)
        .where(Membership.is_active.is_(True))
    )
    membership_result = await db.execute(membership_stmt)
    membership_rows = membership_result.all()
    if not membership_rows:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Membership not found")

    role_ids = [row[1].id for row in membership_rows]
    permission_stmt = (
        select(Permission.code)
        .select_from(role_permissions.join(Permission, role_permissions.c.permission_id == Permission.id))
        .where(role_permissions.c.role_id.in_(role_ids))
    )
    permission_result = await db.execute(permission_stmt)
    permissions = frozenset(code for code in permission_result.scalars().all())
    role_names = tuple(sorted({row[1].name for row in membership_rows}))
    return RequestContext(
        organization_id=organization_id,
        user_id=user_id,
        email=email,
        role_names=role_names,
        permissions=permissions,
    )


async def get_current_context(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> RequestContext:
    cached = getattr(request.state, "auth_context", None)
    if isinstance(cached, RequestContext):
        return cached

    settings = get_settings()
    if not settings.AUTH_REQUIRED:
        stmt = select(User).where(User.email == settings.AUTH_BOOTSTRAP_ADMIN_EMAIL.lower())
        user = (await db.execute(stmt)).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Bootstrap auth user missing")
        payload = TokenIdentity(
            user_id=user.id,
            organization_id=(await _resolve_default_org_id(db, user.id)),
            email=user.email,
            roles=("tenant_admin",),
            permissions=tuple(),
        )
        context = await _build_context(
            db=db,
            user_id=payload.user_id,
            organization_id=payload.organization_id,
            email=payload.email,
        )
        request.state.auth_context = context
        await _release_db_connection(db)
        return context

    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = auth_header[7:].strip()
    try:
        payload = decode_jwt(token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")

    user_id = UUID(str(payload["sub"]))
    organization_id = UUID(str(payload["org"]))
    user_stmt = select(User).where(User.id == user_id).where(User.is_active.is_(True))
    user = (await db.execute(user_stmt)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    context = await _build_context(
        db=db,
        user_id=user_id,
        organization_id=organization_id,
        email=str(payload.get("email") or user.email),
    )
    request.state.auth_context = context
    await _release_db_connection(db)
    return context


async def _resolve_default_org_id(db: AsyncSession, user_id: UUID) -> UUID:
    stmt = (
        select(Membership.organization_id)
        .where(Membership.user_id == user_id)
        .where(Membership.is_active.is_(True))
        .limit(1)
    )
    result = await db.execute(stmt)
    org_id = result.scalar_one_or_none()
    if org_id is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No active organization membership")
    return org_id


def require_permissions(*required: str) -> Callable[[RequestContext], RequestContext]:
    async def _dependency(
        ctx: RequestContext = Depends(get_current_context),
    ) -> RequestContext:
        missing = [perm for perm in required if perm not in ctx.permissions]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(sorted(missing))}",
            )
        return ctx

    return _dependency


def token_is_active(refresh_token: RefreshToken) -> bool:
    return refresh_token.revoked_at is None
