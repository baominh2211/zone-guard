"""ByteTrack real tracker using Ultralytics built-in model.track()."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class TrackedObject:
    """A tracked person with persistent ID across frames."""
    track_id: int
    bbox: list          # [x1, y1, x2, y2] in pixels
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of bbox — used for zone intersection check."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    def foot_point_normalised(self, frame_w: int, frame_h: int) -> tuple[float, float]:
        """Foot-point normalised to [0, 1] range."""
        fx, fy = self.foot_point
        return (fx / frame_w, fy / frame_h)


class ByteTracker:
    """
    Real ByteTrack tracker using Ultralytics model.track().

    How it works:
    1. Receives raw frame (not detections)
    2. Internally runs YOLO detection + ByteTrack matching
    3. Returns TrackedObject list with persistent IDs

    Why model.track() instead of manual IoU:
    - Uses Hungarian algorithm (optimal global matching, not greedy)
    - Handles low-confidence detections (ByteTrack's key innovation)
    - Built-in Kalman filter for motion prediction through occlusion
    - Much more stable track IDs in practice
    """

    def __init__(self, model_path: str = "models/yolov13n.pt",
                 confidence: float = 0.4, device: str = "cpu"):
        from ultralytics import YOLO

        self._confidence = confidence
        self._device = device
        self._model = YOLO(model_path)

        # Warm up — first inference is always slow due to memory allocation
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, device=device, verbose=False)
        print(f"[ByteTracker] Ready: {Path(model_path).stem} on {device}")

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        """
        Run detection + ByteTrack on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of TrackedObject with persistent track IDs.

        Notes:
            - persist=True keeps tracker state between frames (required)
            - tracker="bytetrack.yaml" selects ByteTrack algorithm
            - classes=[0] filters to person class only (COCO)
        """
        results = self._model.track(
            frame,
            conf=self._confidence,
            classes=[0],                   # COCO class 0 = person
            device=self._device,
            tracker="bytetrack.yaml",
            persist=True,                  # Keep state across frames
            verbose=False,
        )

        tracked = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # box.id is None when ByteTrack hasn't assigned an ID yet
                if box.id is None:
                    continue

                tracked.append(TrackedObject(
                    track_id=int(box.id[0].cpu().numpy()),
                    bbox=box.xyxy[0].cpu().numpy().tolist(),
                    confidence=float(box.conf[0].cpu().numpy()),
                ))

        return tracked

    def reset(self):
        """Reset tracker state (e.g. when switching camera source)."""
        self._model.predictor = None