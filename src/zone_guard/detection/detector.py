"""
    Person detection with yolov13n
"""
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Detection:
    """Dectection Result"""
    bbox: list #  [x1, y1, x2, y2] - pixel 
    confidence: float
    class_id: int
    class_name: str = "person"

class DetectionEngine: 
    PERSON_CLASS = 0 # COCO class ID
    
    def __init__(self, model_path: str = "models/yolov13n.pt", 
                 confidence: float = 0.4, device: str = 'cpu'): 
        from ultralytics import YOLO

        self._confidence = confidence
        self._device = device
        self._model_version = Path(model_path).stem

        print(f"[Detector] Loading {model_path} on {device}...")
        self._model = YOLO(model_path)
        # Warm up: chạy 1 lần dummy để load hết vào memory
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, device=device, verbose=False)
        print(f"[Detector] Ready! Model: {self._model_version}")

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect người trong 1 frame.
        
        Args:
            frame: BGR image (numpy array từ OpenCV)
        
        Returns:
            List of Detection objects
        """
        results = self._model.predict(
            frame,
            conf=self._confidence,
            classes=[self.PERSON_CLASS],  # Chỉ detect person
            device=self._device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, "unknown")
                
                detections.append(Detection(
                    bbox=xyxy,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        return detections

    @property
    def model_version(self) -> str:
        return self._model_version


def post_process(detections: list[Detection], frame_h: int, frame_w: int) -> list[Detection]:
    """
    Lọc bỏ detection bất hợp lý.
    
    Rules:
    - Bbox quá nhỏ (< 800 px²) → nhiễu
    - Bbox quá lớn (> 90% frame) → lỗi detect
    - Aspect ratio không giống người (h/w nên từ 1.0 đến 5.0)
    """
    filtered = []
    frame_area = frame_h * frame_w

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Quá nhỏ
        if area < 800:
            continue
        # Quá lớn
        if area > frame_area * 0.9:
            continue
        # Aspect ratio bất thường (người đứng: h > w)
        aspect = h / max(w, 1)
        if aspect < 0.8 or aspect > 5.0:
            continue

        filtered.append(det)

    return filtered

