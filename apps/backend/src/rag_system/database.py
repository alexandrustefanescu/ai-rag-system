"""SQLAlchemy async engine and session setup."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

DATABASE_URL = "sqlite+aiosqlite:///./rag_users.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
        await session.commit()


async def create_db_and_tables() -> None:
    from rag_system.models.chat import Conversation, Message

    async with engine.begin() as conn:
        await conn.run_sync(
            Conversation.metadata.create_all,  # type: ignore[attr-defined]
        )
        await conn.run_sync(
            Message.metadata.create_all,  # type: ignore[attr-defined]
        )
        await conn.commit()
    await engine.dispose()
