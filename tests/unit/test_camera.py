"""
    Test whether camera can capture frames or not
"""
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from zone_guard.ingestion.camera import CameraStream

def test_webcam_capture():
    """Mở webcam, grab 10 frame, kiểm tra."""
    cam = CameraStream(camera_id="test", url="0", fps_cap=15)
    cam.start()
    time.sleep(2)  # Chờ 2 giây để buffer có frame

    frame = cam.get_latest()
    cam.stop()

    assert frame is not None, "Không lấy được frame từ webcam!"
    assert frame.frame.shape[2] == 3, "Frame phải có 3 channels (BGR)"
    assert frame.camera_id == "test"
    print(f"✅ Captured frame #{frame.frame_number}, shape={frame.frame.shape}")

if __name__=='__main__':
    test_webcam_capture()