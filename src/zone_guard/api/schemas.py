"""Pydantic API schemas."""
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class FeedbackType(str, Enum):
    CORRECT = "correct"
    FALSE_POSITIVE = "false_positive"
    MISSED = "missed"


class EventResponse(BaseModel):
    id: str
    event_type: str
    camera_id: str
    zone_id: str
    zone_name: str
    track_id: int
    confidence: float
    snapshot_url: str = ""
    created_at: datetime
    resolved_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    occupancy_count: int = 0
    feedback: Optional[str] = None
    model_version: str = ""


class EventListResponse(BaseModel):
    events: list[EventResponse]
    total: int
    page: int
    page_size: int


class FeedbackRequest(BaseModel):
    feedback: FeedbackType
    note: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    status: str
    components: dict[str, Any]
    version: str
