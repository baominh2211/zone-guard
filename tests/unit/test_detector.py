"""Test detector trên webcam frame."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from zone_guard.ingestion.camera import CameraStream
from zone_guard.detection.detector import DetectionEngine, post_process


def test_detection():
    # Grab 1 frame từ webcam
    cam = CameraStream("test", "0", fps_cap=15)
    cam.start()
    time.sleep(2)
    frame_data = cam.get_latest()
    cam.stop()
    assert frame_data is not None

    # Detect
    detector = DetectionEngine("models/yolov13n.pt", confidence=0.3, device="cpu")
    detections = detector.predict(frame_data.frame)
    
    print(f"Raw detections: {len(detections)}")
    for d in detections:
        print(f"  {d.class_name} conf={d.confidence:.2f} bbox={[int(x) for x in d.bbox]}")

    # Post-process
    h, w = frame_data.frame.shape[:2]
    filtered = post_process(detections, h, w)
    print(f"After filter: {len(filtered)}")
    
    print("✅ Detection OK!")


if __name__ == "__main__":
    test_detection()