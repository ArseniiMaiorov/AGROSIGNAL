"""API аутентификации и tenant bootstrap."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import RequestContext, get_current_context
from api.schemas import AuthLoginRequest, AuthRefreshRequest, AuthTokenResponse, AuthUserResponse
from services.auth_service import AuthService
from storage.db import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=AuthTokenResponse)
async def login(
    payload: AuthLoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    service = AuthService(db)
    try:
        item = await service.login(
            email=payload.email,
            password=payload.password,
            organization_slug=payload.organization_slug,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthTokenResponse(**item)


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh(
    payload: AuthRefreshRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    service = AuthService(db)
    try:
        item = await service.refresh(
            refresh_token=payload.refresh_token,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthTokenResponse(**item)


@router.post("/logout")
async def logout(
    payload: AuthRefreshRequest,
    request: Request,
    ctx: RequestContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_db),
) -> dict[str, bool]:
    service = AuthService(db)
    await service.logout(
        refresh_token=payload.refresh_token,
        ctx=ctx,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
    )
    return {"ok": True}


@router.get("/me", response_model=AuthUserResponse)
async def me(
    ctx: RequestContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_db),
) -> AuthUserResponse:
    service = AuthService(db)
    return AuthUserResponse(**(await service.me(ctx)))
