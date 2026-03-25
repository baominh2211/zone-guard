"""SQLAlchemy ORM models."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class EventModel(Base):
    __tablename__ = "events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(30), nullable=False)
    camera_id = Column(String(50), nullable=False)
    zone_id = Column(String(50), nullable=False)
    zone_name = Column(String(200))
    track_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox = Column(JSONB)
    snapshot_path = Column(String(500))
    occupancy_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    feedback = Column(String(20))
    feedback_note = Column(Text)
    feedback_at = Column(DateTime(timezone=True))
    model_version = Column(String(100))
    __table_args__ = (
        Index("idx_events_created", "created_at"),
        Index("idx_events_feedback", "feedback"),
    )
