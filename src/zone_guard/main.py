"""ZoneGuard main orchestrator — connects all modules.

Pipeline: Camera -> ByteTrack (detect+track inside zone) -> State Machine -> Events -> Alerts

Key design: zone polygon = ROI. Only detect/track inside the drawn zone.
Everyone detected inside zone = intrusion alert.
"""
import asyncio
import logging
import signal
import threading
import time

import cv2
import numpy as np
import uvicorn

from zone_guard.config import Settings
from zone_guard.ingestion.camera import CameraManager
from zone_guard.tracking.byte_tracker import ByteTracker
from zone_guard.zones.zone_checker import ZoneChecker
from zone_guard.zones.state_machine import TrackZoneStateManager, ZoneTransition
from zone_guard.events.event_manager import EventManager
from zone_guard.alerts.dispatcher import AlertDispatcher
from zone_guard.db.database import init_db, close_db, get_session
from zone_guard.db.models import EventModel
from zone_guard.api.app import create_api
from zone_guard.api.routes.auth import configure_auth

logger = logging.getLogger(__name__)


class ZoneGuardApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._shutdown = asyncio.Event()

        # Camera
        self.camera_mgr = CameraManager(settings.cameras)

        # One tracker per zone (each zone = its own ROI crop)
        self._trackers: dict[str, ByteTracker] = {}
        for zone_cfg in settings.zones:
            self._trackers[zone_cfg.id] = ByteTracker(
                model_path=settings.model_path,
                confidence=settings.confidence_threshold,
                device=settings.resolve_device(),
                zone_polygon=zone_cfg.polygon,    # zone = ROI
            )

        # Fallback tracker if no zones configured (full frame detect)
        if not settings.zones:
            self._trackers["__default__"] = ByteTracker(
                model_path=settings.model_path,
                confidence=settings.confidence_threshold,
                device=settings.resolve_device(),
            )

        # Zone checkers (for foot-point validation after ROI crop)
        self._zone_checkers: dict[str, ZoneChecker] = {}
        for zone_cfg in settings.zones:
            self._zone_checkers[zone_cfg.id] = ZoneChecker(zone_cfg.polygon)

        # State machines
        self.state_mgr = TrackZoneStateManager(
            dwell_frames=settings.events.dwell_frames,
            cooldown_seconds=settings.events.cooldown_seconds,
        )

        # Events + Alerts
        self.event_mgr = EventManager(
            snapshot_dir=f"{settings.storage.local_base_path}/snapshots",
            snapshot_quality=settings.events.snapshot_quality,
            on_event=self._on_new_event,
        )
        self.alert_dispatcher = AlertDispatcher(settings.alerts)

        # Annotated frames for MJPEG
        self._frames: dict[str, np.ndarray] = {}
        self._frame_lock = threading.Lock()

    async def start(self):
        logger.info("=" * 50)
        logger.info("  ZoneGuard v1.0.0 starting")
        logger.info("  Zones: %d | Cameras: %d", len(self.settings.zones), len(self.settings.cameras))
        logger.info("=" * 50)

        await init_db(self.settings.database_url)
        configure_auth(self.settings.jwt_secret, self.settings.admin_username, self.settings.admin_password)
        self.camera_mgr.start_all()

        api = create_api(self)
        config = uvicorn.Config(api, host="0.0.0.0", port=self.settings.api_port, log_level="warning")
        server = uvicorn.Server(config)

        await asyncio.gather(server.serve(), self._inference_loop())

    async def _inference_loop(self):
        """Core pipeline: grab frame -> detect+track per zone -> state machine -> events."""
        logger.info("Inference loop started")

        while not self._shutdown.is_set():
            frames = self.camera_mgr.get_latest_frames()
            if not frames:
                await asyncio.sleep(0.01)
                continue

            for cam_id, frame_data in frames.items():
                frame = frame_data.frame
                h, w = frame.shape[:2]

                all_tracks_for_display = []

                # For each zone assigned to this camera
                zones_for_cam = [z for z in self.settings.zones if z.camera_id == cam_id]

                if not zones_for_cam:
                    # No zones for this camera - just detect full frame
                    tracker = self._trackers.get("__default__")
                    if tracker:
                        tracks = tracker.update(frame)
                        all_tracks_for_display.extend(tracks)
                else:
                    for zone_cfg in zones_for_cam:
                        tracker = self._trackers.get(zone_cfg.id)
                        if not tracker:
                            continue

                        # Detect+track inside this zone's ROI
                        tracks = tracker.update(frame)
                        all_tracks_for_display.extend(tracks)

                        # Every person detected inside zone ROI = potential intrusion
                        # Double-check with foot-point inside zone polygon
                        checker = self._zone_checkers.get(zone_cfg.id)
                        active_ids = set()

                        for t in tracks:
                            active_ids.add(t.track_id)
                            fx, fy = t.foot_point_normalised(w, h)
                            inside = checker.is_inside(fx, fy) if checker else True
                            transition = self.state_mgr.update(t.track_id, zone_cfg.id, inside)

                            if transition:
                                occ = self.state_mgr.get_occupancy(zone_cfg.id)
                                await self.event_mgr.handle_transition(
                                    transition=transition,
                                    camera_id=cam_id,
                                    zone_id=zone_cfg.id,
                                    zone_name=zone_cfg.name,
                                    zone_type=zone_cfg.zone_type,
                                    track_id=t.track_id,
                                    confidence=t.confidence,
                                    bbox=t.bbox,
                                    frame=frame,
                                    occupancy=occ,
                                )

                        # Cleanup stale tracks for this zone
                        self.state_mgr.cleanup_stale(active_ids)

                # Draw annotations
                annotated = self._draw(frame, all_tracks_for_display, cam_id)
                with self._frame_lock:
                    self._frames[cam_id] = annotated

            await asyncio.sleep(0.001)

    def _draw(self, frame, tracks, camera_id):
        """Draw zone polygons (dimmed outside), bboxes, occupancy."""
        out = frame.copy()
        h, w = out.shape[:2]

        zones_for_cam = [z for z in self.settings.zones if z.camera_id == camera_id]

        # Dim outside all zones
        if zones_for_cam:
            mask = np.zeros((h, w), dtype=np.uint8)
            for z in zones_for_cam:
                pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in z.polygon], np.int32)
                cv2.fillPoly(mask, [pts], 255)
            outside = mask == 0
            out[outside] = (out[outside] * 0.3).astype(np.uint8)

        # Draw zone borders + labels
        for z in zones_for_cam:
            pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in z.polygon], np.int32)
            color = (0, 0, 255) if z.zone_type == "restricted" else (0, 255, 0)
            cv2.polylines(out, [pts], True, color, 2)
            occ = self.state_mgr.get_occupancy(z.id)
            cx = int(np.mean([p[0] for p in z.polygon]) * w)
            cy = int(np.mean([p[1] for p in z.polygon]) * h)
            cv2.putText(out, f"{z.name}: {occ}", (cx-40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw tracks
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(out, (x1,y1), (x2,y2), (255,200,0), 2)
            cv2.putText(out, f"#{t.track_id} {t.confidence:.0%}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

        return out

    def get_annotated_frame(self, camera_id):
        with self._frame_lock:
            return self._frames.get(camera_id)

    async def _on_new_event(self, event):
        """Persist to DB + dispatch alerts."""
        try:
            async with get_session() as s:
                s.add(EventModel(
                    event_type=event.event_type, camera_id=event.camera_id,
                    zone_id=event.zone_id, zone_name=event.zone_name,
                    track_id=event.track_id, confidence=event.confidence,
                    bbox=event.bbox, snapshot_path=event.snapshot_path,
                    occupancy_count=event.occupancy_count, model_version=event.model_version,
                ))
        except Exception as e:
            logger.error("DB save failed: %s", e)

        try:
            zone_cfg = next((z for z in self.settings.zones if z.id == event.zone_id), None)
            if zone_cfg and zone_cfg.alert_channels:
                await self.alert_dispatcher.dispatch(event, zone_cfg.alert_channels)
        except Exception as e:
            logger.error("Alert failed: %s", e)

        try:
            from zone_guard.api.routes.stream import ws_manager
            await ws_manager.broadcast({
                "type": "event", "event_type": event.event_type,
                "zone_name": event.zone_name, "confidence": event.confidence,
                "created_at": event.created_at.isoformat(),
            })
        except:
            pass


def main():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    settings = Settings()
    app = ZoneGuardApp(settings)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: app._shutdown.set())
        except NotImplementedError:
            pass
    try:
        loop.run_until_complete(app.start())
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        loop.run_until_complete(close_db())
        loop.close()


if __name__ == "__main__":
    main()
