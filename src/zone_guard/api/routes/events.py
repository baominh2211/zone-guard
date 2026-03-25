"""Event CRUD + feedback."""
import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from zone_guard.api.routes.auth import get_current_user
from zone_guard.api.schemas import EventListResponse, EventResponse, FeedbackRequest
from zone_guard.db.database import get_session
from zone_guard.db.models import EventModel

router = APIRouter()


@router.get("", response_model=EventListResponse)
async def list_events(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                       event_type: Optional[str] = None, feedback: Optional[str] = None,
                       user: dict = Depends(get_current_user)):
    async with get_session() as s:
        q = select(EventModel).order_by(desc(EventModel.created_at))
        cq = select(func.count(EventModel.id))
        if event_type:
            q = q.where(EventModel.event_type == event_type)
            cq = cq.where(EventModel.event_type == event_type)
        if feedback:
            q = q.where(EventModel.feedback == feedback)
            cq = cq.where(EventModel.feedback == feedback)
        q = q.offset((page-1)*page_size).limit(page_size)
        events = (await s.execute(q)).scalars().all()
        total = (await s.execute(cq)).scalar() or 0
    return EventListResponse(events=[_resp(e) for e in events], total=total, page=page, page_size=page_size)


@router.get("/stats")
async def stats(user: dict = Depends(get_current_user)):
    async with get_session() as s:
        r = await s.execute(select(EventModel.event_type, func.count(EventModel.id)).group_by(EventModel.event_type))
        by_type = {row[0]: row[1] for row in r}
        r2 = await s.execute(select(EventModel.feedback, func.count(EventModel.id)).where(
            EventModel.feedback.is_not(None)).group_by(EventModel.feedback))
        fb = {row[0]: row[1] for row in r2}
    total_rev = sum(fb.values())
    return {"total_events": sum(by_type.values()), "events_by_type": by_type,
            "feedback": fb, "false_positive_rate": fb.get("false_positive",0)/max(total_rev,1)}


@router.post("/{event_id}/feedback")
async def submit_feedback(event_id: str, req: FeedbackRequest, user: dict = Depends(get_current_user)):
    async with get_session() as s:
        r = await s.execute(select(EventModel).where(EventModel.id == event_id))
        ev = r.scalar_one_or_none()
        if not ev:
            raise HTTPException(404, "Not found")
        ev.feedback = req.feedback.value
        ev.feedback_note = req.note
        ev.feedback_at = datetime.now(timezone.utc)
    return {"status": "ok", "feedback": req.feedback.value}


def _resp(e):
    snap = f"/static/snapshots/{os.path.basename(e.snapshot_path)}" if e.snapshot_path else ""
    return EventResponse(id=str(e.id), event_type=e.event_type, camera_id=e.camera_id,
        zone_id=e.zone_id, zone_name=e.zone_name or "", track_id=e.track_id,
        confidence=e.confidence, snapshot_url=snap, created_at=e.created_at,
        resolved_at=e.resolved_at, duration_seconds=e.duration_seconds,
        occupancy_count=e.occupancy_count or 0, feedback=e.feedback,
        model_version=e.model_version or "")
