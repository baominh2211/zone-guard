"""Async database engine and session."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from zone_guard.db.models import Base

logger = logging.getLogger(__name__)
_engine: AsyncEngine | None = None
_sf: async_sessionmaker[AsyncSession] | None = None


async def init_db(url: str):
    global _engine, _sf
    _engine = create_async_engine(url, pool_size=5, pool_pre_ping=True)
    _sf = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def close_db():
    global _engine
    if _engine:
        await _engine.dispose()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _sf is None:
        raise RuntimeError("DB not initialized")
    async with _sf() as s:
        try:
            yield s
            await s.commit()
        except:
            await s.rollback()
            raise
