"""Draw zone polygons — each zone is BOTH the detection ROI and alert zone.

Usage:
    python scripts/generate_zone.py --camera 0
    python scripts/generate_zone.py --image reference_frame.jpg

Controls:
    Click       : Add polygon point
    C           : Close current polygon (then press R=restricted or M=monitored)
    U           : Undo last point
    X           : Reset current polygon
    D           : Delete last completed zone
    S           : Save all zones to configs/zones.yaml
    Q           : Quit

Each zone you draw = YOLO only detects inside that polygon + alerts when person enters.
"""
import argparse, sys, os
import cv2
import numpy as np
import yaml

points = []
zones = []          # list of {"polygon": [(x,y),...], "type": "restricted"|"monitored"}
frame_clean = None
frame = None


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()


def redraw():
    global frame
    frame = frame_clean.copy()
    h, w = frame.shape[:2]

    # Dim outside all zones
    if zones:
        mask = np.zeros((h, w), dtype=np.uint8)
        for z in zones:
            pts = np.array(z["polygon"], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        outside = mask == 0
        frame[outside] = (frame[outside] * 0.3).astype(np.uint8)

    # Draw completed zones
    for i, z in enumerate(zones):
        pts = np.array(z["polygon"], np.int32)
        color = (0, 0, 255) if z["type"] == "restricted" else (0, 255, 0)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        cv2.polylines(frame, [pts], True, color, 2)
        cx = int(np.mean([p[0] for p in z["polygon"]]))
        cy = int(np.mean([p[1] for p in z["polygon"]]))
        cv2.putText(frame, f"Zone {i+1} ({z['type']})", (cx-50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw current points
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, np.int32)], False, (0, 255, 255), 2)

    # Status bar
    cv2.rectangle(frame, (0, h-40), (w, h), (30,30,30), -1)
    cv2.putText(frame, f"Zones: {len(zones)} | Drawing: {len(points)} pts | "
                f"C:close U:undo X:reset D:delete S:save Q:quit",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)


def main():
    global frame, frame_clean, points

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default="configs/zones.yaml")
    args = parser.parse_args()

    if args.image:
        frame_clean = cv2.imread(args.image)
    else:
        src = int(args.camera) if args.camera.isdigit() else args.camera
        cap = cv2.VideoCapture(src)
        ret, frame_clean = cap.read()
        cap.release()
        if not ret:
            print("Cannot grab frame"); sys.exit(1)

    frame = frame_clean.copy()
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")
    print("Draw zone polygon, press C to close, then R=restricted M=monitored")

    cv2.namedWindow("ZoneGuard - Draw Zones")
    cv2.setMouseCallback("ZoneGuard - Draw Zones", mouse_cb)
    redraw()

    while True:
        cv2.imshow("ZoneGuard - Draw Zones", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and len(points) >= 3:
            print("Zone type? R=restricted, M=monitored")
            while True:
                zk = cv2.waitKey(0) & 0xFF
                if zk == ord('r'):
                    zones.append({"polygon": list(points), "type": "restricted"})
                    print(f"Zone {len(zones)} = RESTRICTED ({len(points)} pts)")
                    break
                elif zk == ord('m'):
                    zones.append({"polygon": list(points), "type": "monitored"})
                    print(f"Zone {len(zones)} = MONITORED ({len(points)} pts)")
                    break
            points = []
            redraw()
        elif key == ord('u') and points:
            points.pop()
            redraw()
        elif key == ord('x'):
            points = []
            redraw()
        elif key == ord('d') and zones:
            removed = zones.pop()
            print(f"Deleted zone ({removed['type']})")
            redraw()
        elif key == ord('s'):
            save(zones, w, h, args.output)
            print(f"Saved {len(zones)} zones -> {args.output}")

    cv2.destroyAllWindows()


def save(zones_data, w, h, path):
    out = []
    for i, z in enumerate(zones_data):
        norm = [[round(x/w, 4), round(y/h, 4)] for x, y in z["polygon"]]
        out.append({
            "id": f"zone_{i+1:02d}",
            "name": f"Zone {i+1}",
            "camera_id": "cam_01",
            "zone_type": z["type"],
            "polygon": norm,
            "dwell_time_seconds": 2.0,
            "cooldown_seconds": 60,
            "max_occupancy": 0 if z["type"] == "restricted" else 10,
            "alert_channels": ["telegram"],
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"zones": out}, f, default_flow_style=False)


if __name__ == "__main__":
    main()
