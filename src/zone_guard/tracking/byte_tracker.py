"""ByteTrack tracker with zone-as-ROI — only detect+track inside zone polygon."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackedObject:
    """A tracked person with persistent ID."""
    track_id: int
    bbox: list          # [x1, y1, x2, y2] in ORIGINAL frame coordinates
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    def foot_point_normalised(self, frame_w: int, frame_h: int) -> tuple[float, float]:
        fx, fy = self.foot_point
        return (fx / frame_w, fy / frame_h)


class ByteTracker:
    """
    ByteTrack with zone-as-ROI.
    
    The zone polygon serves double duty:
    1. ROI: crop frame before YOLO (only detect inside zone)
    2. Zone: anyone detected here triggers intrusion alert
    
    Draw one polygon = detection region AND alert zone.
    """

    def __init__(self, model_path="models/yolov13n.pt", confidence=0.4,
                 device="cpu", zone_polygon=None):
        from ultralytics import YOLO
        self._confidence = confidence
        self._device = device
        self._model = YOLO(model_path)
        self._zone_polygon_norm = zone_polygon
        self._zone_enabled = zone_polygon is not None and len(zone_polygon) >= 3

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, device=device, verbose=False)
        status = "enabled" if self._zone_enabled else "disabled (full frame)"
        print(f"[ByteTracker] Ready: {Path(model_path).stem} | Zone-ROI: {status}")

    def set_zone(self, polygon):
        """Set/update zone polygon at runtime. Resets tracker."""
        self._zone_polygon_norm = polygon
        self._zone_enabled = polygon is not None and len(polygon) >= 3
        self.reset()

    def update(self, frame):
        if self._zone_enabled:
            return self._update_with_zone(frame)
        return self._update_full(frame)

    def _update_full(self, frame):
        results = self._model.track(frame, conf=self._confidence, classes=[0],
                                     device=self._device, tracker="bytetrack.yaml",
                                     persist=True, verbose=False)
        return self._parse_results(results, 0, 0)

    def _update_with_zone(self, frame):
        h, w = frame.shape[:2]
        zone_pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in self._zone_polygon_norm], np.int32)
        pad = 20
        rx, ry, rw, rh = cv2.boundingRect(zone_pts)
        x1, y1 = max(0, rx-pad), max(0, ry-pad)
        x2, y2 = min(w, rx+rw+pad), min(h, ry+rh+pad)

        crop = frame[y1:y2, x1:x2].copy()
        shifted = zone_pts - np.array([x1, y1])
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], 255)
        crop_masked = cv2.bitwise_and(crop, crop, mask=mask)

        results = self._model.track(crop_masked, conf=self._confidence, classes=[0],
                                     device=self._device, tracker="bytetrack.yaml",
                                     persist=True, verbose=False)
        return self._parse_results(results, x1, y1)

    def _parse_results(self, results, offset_x, offset_y):
        tracked = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()
                tracked.append(TrackedObject(
                    track_id=int(box.id[0].cpu().numpy()),
                    bbox=[bx1+offset_x, by1+offset_y, bx2+offset_x, by2+offset_y],
                    confidence=float(box.conf[0].cpu().numpy()),
                ))
        return tracked

    @property
    def zone_enabled(self):
        return self._zone_enabled

    @property
    def zone_polygon(self):
        return self._zone_polygon_norm

    def reset(self):
        self._model.predictor = None
