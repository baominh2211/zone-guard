"""Test: Draw zone on webcam -> only detect+track inside zone -> show intrusion alerts.

This is the complete visual test for the zone-as-ROI pipeline.
Zone polygon = detection ROI + alert zone (one polygon, two purposes).

Controls:
    LMB         : Add point to zone polygon
    RMB         : Undo last point
    S / Enter   : Save zone (lock it, start detecting)
    E           : Edit zone (unlock, stop detecting)
    C           : Clear zone completely
    Q           : Quit
"""
import cv2
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from zone_guard.tracking.byte_tracker import ByteTracker
from zone_guard.zones.zone_checker import ZoneChecker
from zone_guard.zones.state_machine import TrackZoneStateManager, ZoneTransition

WINDOW = "ZoneGuard - Draw Zone then Detect | Q:quit S:save E:edit C:clear"
ZONE_ID = "zone_01"

# State
draw_points = []
zone_polygon_norm = None  # normalised polygon after save
zone_checker = None
zone_locked = False

# Tracker starts without zone (will be set after drawing)
tracker = ByteTracker("models/yolov13n.pt", confidence=0.4, device="cpu")
state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)


def mouse_cb(event, x, y, flags, param):
    global draw_points
    if zone_locked:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and draw_points:
        draw_points.pop()


cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, mouse_cb)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    # --- Detect+Track (only if zone is locked) ---
    tracks = []
    if zone_locked and zone_polygon_norm:
        tracks = tracker.update(frame)

    # --- Draw zone polygon ---
    if draw_points:
        pts_px = np.array(draw_points, np.int32)
        if zone_locked and len(draw_points) >= 3:
            # Locked: dim outside, fill zone
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_px], 255)
            frame[mask == 0] = (frame[mask == 0] * 0.3).astype(np.uint8)
            cv2.polylines(frame, [pts_px], True, (0, 0, 255), 2)
            cx = int(np.mean([p[0] for p in draw_points]))
            cy = int(min(p[1] for p in draw_points)) - 10
            cv2.putText(frame, "RESTRICTED ZONE", (max(10, cx-80), max(20, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Drawing: show points and lines
            for pt in draw_points:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            if len(draw_points) >= 2:
                cv2.polylines(frame, [pts_px], False, (0, 255, 255), 2)

    # --- Check tracks against zone + draw ---
    active_ids = set()
    for t in tracks:
        active_ids.add(t.track_id)
        x1, y1, x2, y2 = [int(v) for v in t.bbox]

        inside = True  # everything inside ROI crop is "inside zone"
        if zone_checker:
            fx, fy = ZoneChecker.bbox_to_foot_point(t.bbox, w, h)
            inside = zone_checker.is_inside(fx, fy)

        transition = state_mgr.update(t.track_id, ZONE_ID, inside)
        color = (0, 0, 255) if inside else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{t.track_id} {t.confidence:.0%}",
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Foot point
        fpx, fpy = int((x1+x2)/2), y2
        cv2.circle(frame, (fpx, fpy), 4, color, -1)

        if transition == ZoneTransition.INTRUSION_START:
            print(f"🚨 INTRUSION! Track #{t.track_id}")
        elif transition == ZoneTransition.INTRUSION_END:
            print(f"✅ Track #{t.track_id} left zone")

    state_mgr.cleanup_stale(active_ids)

    # --- HUD ---
    occ = state_mgr.get_occupancy(ZONE_ID) if zone_locked else 0
    cv2.putText(frame, f"Occupancy: {occ}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    status = "DETECTING" if zone_locked else "DRAW ZONE (click points, press S)"
    cv2.putText(frame, status, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key in (ord('s'), 13):  # Save/Enter
        if len(draw_points) >= 3:
            zone_polygon_norm = [[x/w, y/h] for x, y in draw_points]
            zone_checker = ZoneChecker(zone_polygon_norm)
            tracker.set_zone(zone_polygon_norm)  # Set zone as ROI
            zone_locked = True
            state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)
            print(f"[Zone] Saved: {len(draw_points)} points — now detecting inside zone only")

    elif key == ord('e'):  # Edit
        zone_locked = False
        zone_checker = None
        zone_polygon_norm = None
        tracker.set_zone(None)  # Disable ROI
        state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)
        print("[Zone] Edit mode — detection paused")

    elif key == ord('c'):  # Clear
        draw_points = []
        zone_locked = False
        zone_checker = None
        zone_polygon_norm = None
        tracker.set_zone(None)
        state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)
        print("[Zone] Cleared")

cap.release()
cv2.destroyAllWindows()
