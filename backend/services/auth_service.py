from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext
from core.security import (
    TokenIdentity,
    hash_refresh_token,
    issue_access_token,
    new_refresh_token,
    verify_password,
)
from core.settings import get_settings
from services.audit_service import record_audit_event
from storage.db import Membership, Organization, Permission, RefreshToken, Role, User, role_permissions, utcnow


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def login(
        self,
        *,
        email: str,
        password: str,
        organization_slug: str | None,
        ip_address: str | None,
        user_agent: str | None,
    ) -> dict[str, Any]:
        stmt = select(User).where(User.email == email.lower()).where(User.is_active.is_(True))
        user = (await self.db.execute(stmt)).scalar_one_or_none()
        if user is None or not verify_password(password, user.password_hash):
            raise ValueError("Invalid credentials")

        membership, organization, roles, permissions = await self._resolve_membership(
            user_id=user.id,
            organization_slug=organization_slug,
        )
        access_token = issue_access_token(
            TokenIdentity(
                user_id=user.id,
                organization_id=organization.id,
                email=user.email,
                roles=tuple(roles),
                permissions=tuple(permissions),
            )
        )
        refresh_token = new_refresh_token()
        self.db.add(
            RefreshToken(
                organization_id=organization.id,
                user_id=user.id,
                token_hash=hash_refresh_token(refresh_token),
                expires_at=utcnow() + timedelta(days=int(self.settings.AUTH_REFRESH_TTL_DAYS)),
                ip_address=ip_address,
                user_agent=user_agent,
            )
        )
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="auth.login",
            resource_type="session",
            resource_id=str(user.id),
            organization_id=organization.id,
            actor_user_id=user.id,
            payload={"organization_slug": organization.slug},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self.settings.AUTH_ACCESS_TTL_MINUTES) * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "organization_id": str(organization.id),
                "organization_slug": organization.slug,
                "organization_name": organization.name,
                "roles": list(roles),
                "permissions": list(permissions),
            },
        }

    async def refresh(
        self,
        *,
        refresh_token: str,
        ip_address: str | None,
        user_agent: str | None,
    ) -> dict[str, Any]:
        hashed = hash_refresh_token(refresh_token)
        stmt = (
            select(RefreshToken, User, Organization)
            .join(User, RefreshToken.user_id == User.id)
            .join(Organization, RefreshToken.organization_id == Organization.id)
            .where(RefreshToken.token_hash == hashed)
        )
        row = (await self.db.execute(stmt)).first()
        if row is None:
            raise ValueError("Refresh token not found")
        stored, user, organization = row
        if stored.revoked_at is not None or stored.expires_at <= utcnow():
            raise ValueError("Refresh token expired")

        membership, _organization, roles, permissions = await self._resolve_membership(
            user_id=user.id,
            organization_slug=organization.slug,
        )
        stored.revoked_at = utcnow()
        next_refresh = new_refresh_token()
        self.db.add(
            RefreshToken(
                organization_id=organization.id,
                user_id=user.id,
                token_hash=hash_refresh_token(next_refresh),
                expires_at=utcnow() + timedelta(days=int(self.settings.AUTH_REFRESH_TTL_DAYS)),
                ip_address=ip_address,
                user_agent=user_agent,
            )
        )
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="auth.refresh",
            resource_type="session",
            resource_id=str(stored.id),
            organization_id=organization.id,
            actor_user_id=user.id,
            payload={"organization_slug": organization.slug},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        access_token = issue_access_token(
            TokenIdentity(
                user_id=user.id,
                organization_id=organization.id,
                email=user.email,
                roles=tuple(roles),
                permissions=tuple(permissions),
            )
        )
        return {
            "access_token": access_token,
            "refresh_token": next_refresh,
            "token_type": "bearer",
            "expires_in": int(self.settings.AUTH_ACCESS_TTL_MINUTES) * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "organization_id": str(organization.id),
                "organization_slug": organization.slug,
                "organization_name": organization.name,
                "roles": list(roles),
                "permissions": list(permissions),
            },
        }

    async def logout(
        self,
        *,
        refresh_token: str,
        ctx: RequestContext | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        hashed = hash_refresh_token(refresh_token)
        stmt = select(RefreshToken).where(RefreshToken.token_hash == hashed)
        token_row = (await self.db.execute(stmt)).scalar_one_or_none()
        if token_row is None:
            return
        token_row.revoked_at = utcnow()
        await self.db.flush()
        await record_audit_event(
            self.db,
            action="auth.logout",
            resource_type="session",
            resource_id=str(token_row.id),
            ctx=ctx,
            organization_id=token_row.organization_id,
            actor_user_id=token_row.user_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    async def me(self, ctx: RequestContext) -> dict[str, Any]:
        stmt = select(User, Organization).join(Organization, Organization.id == ctx.organization_id).where(User.id == ctx.user_id)
        user, organization = (await self.db.execute(stmt)).one()
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "organization_id": str(organization.id),
            "organization_slug": organization.slug,
            "organization_name": organization.name,
            "roles": list(ctx.role_names),
            "permissions": sorted(ctx.permissions),
        }

    async def _resolve_membership(
        self,
        *,
        user_id: UUID,
        organization_slug: str | None,
    ) -> tuple[Membership, Organization, tuple[str, ...], tuple[str, ...]]:
        stmt = (
            select(Membership, Organization, Role)
            .join(Organization, Membership.organization_id == Organization.id)
            .join(Role, Membership.role_id == Role.id)
            .where(Membership.user_id == user_id)
            .where(Membership.is_active.is_(True))
            .where(Organization.is_active.is_(True))
        )
        if organization_slug:
            stmt = stmt.where(Organization.slug == organization_slug)
        rows = (await self.db.execute(stmt)).all()
        if not rows:
            raise ValueError("Organization membership not found")
        membership, organization, _role = rows[0]
        role_ids = [row[2].id for row in rows]
        role_names = tuple(sorted({row[2].name for row in rows}))
        perm_stmt = (
            select(Permission.code)
            .select_from(role_permissions.join(Permission, role_permissions.c.permission_id == Permission.id))
            .where(role_permissions.c.role_id.in_(role_ids))
        )
        permissions = tuple(sorted(set((await self.db.execute(perm_stmt)).scalars().all())))
        return membership, organization, role_names, permissions
