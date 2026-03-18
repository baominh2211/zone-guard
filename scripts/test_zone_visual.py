import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from zone_guard.tracking.byte_tracker import ByteTracker
from zone_guard.zones.zone_checker import ZoneChecker
from zone_guard.zones.state_machine import TrackZoneStateManager, ZoneTransition


WINDOW_NAME = "Zone Test - Draw with mouse | LMB:add RMB:undo S/Enter:save E:edit C:clear Q:quit"
ZONE_ID = "zone_01"

# State
draw_points = []          # điểm vẽ bằng chuột, đơn vị pixel
zone_checker = None       # ZoneChecker sau khi normalize
zone_locked = False       # đã chốt zone hay chưa
frame_w = None
frame_h = None

tracker = ByteTracker("models/yolov13n.pt", confidence=0.4, device="cpu")
state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)


def reset_zone_state():
    global draw_points, zone_checker, zone_locked, state_mgr
    draw_points = []
    zone_checker = None
    zone_locked = False
    state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)


def build_zone_checker(points_px, w, h):
    """Chuyển polygon pixel -> normalized rồi tạo ZoneChecker."""
    normalized = [[x / w, y / h] for (x, y) in points_px]
    return ZoneChecker(normalized)


def mouse_callback(event, x, y, flags, param):
    global draw_points, zone_locked

    if zone_locked:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if draw_points:
            draw_points.pop()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Không mở được webcam")

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Track trực tiếp từ frame
    tracks = tracker.update(frame)

    # ---- Vẽ polygon đang chỉnh / đã chốt ----
    if len(draw_points) > 0:
        # Vẽ các điểm
        for i, (px, py) in enumerate(draw_points):
            cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(i + 1),
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        # Vẽ các cạnh tạm
        if len(draw_points) >= 2:
            pts_np = np.array(draw_points, np.int32)
            cv2.polylines(frame, [pts_np], False, (0, 255, 255), 2)

        # Nếu đã khóa zone thì vẽ polygon kín màu đỏ
        if zone_locked and len(draw_points) >= 3:
            pts_np = np.array(draw_points, np.int32)
            cv2.polylines(frame, [pts_np], True, (0, 0, 255), 2)

            x_text = min(p[0] for p in draw_points)
            y_text = max(20, min(p[1] for p in draw_points) - 10)
            cv2.putText(
                frame,
                "RESTRICTED",
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    # ---- Check từng track với zone ----
    active_ids = set()

    for t in tracks:
        active_ids.add(t.track_id)

        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = (0, 255, 0)
        inside = False
        transition = None

        if zone_checker is not None:
            fx, fy = ZoneChecker.bbox_to_foot_point(t.bbox, frame_w, frame_h)
            inside = zone_checker.is_inside(fx, fy)
            transition = state_mgr.update(t.track_id, ZONE_ID, inside)
            color = (0, 0, 255) if inside else (0, 255, 0)

            # Vẽ foot point
            foot_x = int((x1 + x2) / 2)
            foot_y = y2
            cv2.circle(frame, (foot_x, foot_y), 4, color, -1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID:{t.track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        if transition == ZoneTransition.INTRUSION_START:
            print(f"🚨 INTRUSION! Track #{t.track_id}")
        elif transition == ZoneTransition.INTRUSION_END:
            print(f"✅ Track #{t.track_id} left zone")

    # Cleanup stale track states
    state_mgr.cleanup_stale(active_ids)

    # Occupancy
    occ = state_mgr.get_occupancy(ZONE_ID) if zone_checker is not None else 0
    cv2.putText(
        frame,
        f"Zone Occupancy: {occ}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Hướng dẫn
    info_lines = [
        "LMB:add point  RMB:undo",
        "S or Enter: save zone",
        "E: edit zone   C: clear   Q: quit",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(
            frame,
            line,
            (10, frame_h - 50 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if zone_checker is None:
        cv2.putText(
            frame,
            "Zone not set",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key in (ord("s"), 13):  # 13 = Enter
        if len(draw_points) >= 3 and frame_w and frame_h:
            zone_checker = build_zone_checker(draw_points, frame_w, frame_h)
            zone_locked = True
            state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)
            print("[Zone] Saved zone.")

    elif key == ord("e"):
        if len(draw_points) >= 3:
            zone_locked = False
            zone_checker = None
            state_mgr = TrackZoneStateManager(dwell_frames=3, cooldown_seconds=5)
            print("[Zone] Edit mode enabled.")

    elif key == ord("c"):
        reset_zone_state()
        print("[Zone] Cleared.")

cap.release()
cv2.destroyAllWindows()