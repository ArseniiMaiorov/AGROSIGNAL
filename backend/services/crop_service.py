"""Сервис справочника культур."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.db import Crop


class CropService:
    """Операции со справочником культур."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_crops(self) -> list[Crop]:
        result = await self.db.execute(select(Crop).order_by(Crop.name.asc()))
        return list(result.scalars().all())

    async def get_crop_by_code(self, code: str) -> Crop | None:
        result = await self.db.execute(select(Crop).where(Crop.code == code))
        return result.scalar_one_or_none()
