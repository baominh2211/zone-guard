"""Webcam + ByteTrack — full frame detect (no zone ROI)."""
import cv2, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from zone_guard.tracking.byte_tracker import ByteTracker

tracker = ByteTracker("models/yolov13n.pt", confidence=0.4, device="cpu")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    tracks = tracker.update(frame)
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
        c = colors[t.track_id % len(colors)]
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
        cv2.putText(frame, f"ID:{t.track_id} {t.confidence:.0%}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    cv2.putText(frame, f"ByteTrack: {len(tracks)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.imshow("ByteTrack - Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
