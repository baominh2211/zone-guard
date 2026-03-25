"""Event manager — create events, capture snapshots, send alerts."""
import os
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Event:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    camera_id: str = ""
    zone_id: str = ""
    zone_name: str = ""
    track_id: int = 0
    confidence: float = 0.0
    bbox: list = field(default_factory=list)
    snapshot_path: str = ""
    occupancy_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    model_version: str = ""


class EventManager:
    def __init__(self, snapshot_dir="data/snapshots", snapshot_quality=85, on_event=None):
        self._snapshot_dir = snapshot_dir
        self._quality = snapshot_quality
        self._on_event = on_event
        self._active: dict[tuple[int, str], Event] = {}
        self._history: list[Event] = []
        os.makedirs(snapshot_dir, exist_ok=True)

    async def handle_transition(self, transition, camera_id, zone_id, zone_name,
                                 zone_type, track_id, confidence, bbox, frame,
                                 model_version="", occupancy=0):
        from zone_guard.zones.state_machine import ZoneTransition
        if transition == ZoneTransition.INTRUSION_START:
            return await self._create(
                "intrusion_start" if zone_type == "restricted" else "occupancy_exceeded",
                camera_id, zone_id, zone_name, track_id, confidence, bbox, frame,
                model_version, occupancy)
        elif transition == ZoneTransition.INTRUSION_END:
            return self._resolve(track_id, zone_id)
        return None

    async def _create(self, event_type, camera_id, zone_id, zone_name, track_id,
                       confidence, bbox, frame, model_version, occupancy):
        snap = self._save_snapshot(frame, bbox, camera_id)
        event = Event(event_type=event_type, camera_id=camera_id, zone_id=zone_id,
                      zone_name=zone_name, track_id=track_id, confidence=confidence,
                      bbox=bbox, snapshot_path=snap, occupancy_count=occupancy,
                      model_version=model_version)
        self._active[(track_id, zone_id)] = event
        self._history.append(event)
        logger.info("EVENT: %s | cam=%s zone=%s track=#%d conf=%.0f%%",
                     event_type, camera_id, zone_name, track_id, confidence*100)
        if self._on_event:
            await self._on_event(event)
        return event

    def _resolve(self, track_id, zone_id):
        key = (track_id, zone_id)
        event = self._active.pop(key, None)
        if event:
            event.resolved_at = datetime.now(timezone.utc)
            event.duration_seconds = (event.resolved_at - event.created_at).total_seconds()
            logger.info("RESOLVED: track=#%d zone=%s dur=%.1fs",
                         track_id, zone_id, event.duration_seconds)
        return event

    def _save_snapshot(self, frame, bbox, camera_id):
        img = frame.copy()
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(img, "INTRUSION", (x1, y1-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self._snapshot_dir, f"{camera_id}_{ts}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return path

    def get_active(self):
        return list(self._active.values())

    def get_recent(self, limit=50):
        return self._history[-limit:]
