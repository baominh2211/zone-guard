"""Webcam + ByteTrack real tracking — each person gets a persistent ID."""
import cv2
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from zone_guard.tracking.byte_tracker import ByteTracker

# Single tracker handles both detection and tracking
tracker = ByteTracker(r"D:\topic_make_money\zone-guard\src\models\yolov13n.pt", confidence=0.4, device="cpu")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # One call: detect + ByteTrack
    tracks = tracker.update(frame)

    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]

        # Different color per track ID
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
        color = colors[t.track_id % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{t.track_id} {t.confidence:.0%}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw foot point
        fx, fy = t.foot_point
        cv2.circle(frame, (int(fx), int(fy)), 5, color, -1)

    cv2.putText(frame, f"ByteTrack: {len(tracks)} persons", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("ByteTrack Test - Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()