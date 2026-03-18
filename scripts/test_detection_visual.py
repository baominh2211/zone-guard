"""Mở webcam, detect người, vẽ bbox — nhấn Q thoát."""
import cv2
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from zone_guard.detection.detector import DetectionEngine, post_process

detector = DetectionEngine("models/yolov13n.pt", device="cpu")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = detector.predict(frame)
    h, w = frame.shape[:2]
    detections = post_process(detections, h, w)
    
    # Vẽ bbox
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    cv2.putText(frame, f"Persons: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Detection Test - Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()