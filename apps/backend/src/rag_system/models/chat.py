"""Chat history models for multi-user conversations."""

from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_system.database import Base


class Conversation(Base):
    """Stored conversation for a user."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        primary_key=True, default=lambda: __import__("uuid").uuid4().hex
    )
    user_id: Mapped[str] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """Single message within a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        primary_key=True, default=lambda: __import__("uuid").uuid4().hex
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
