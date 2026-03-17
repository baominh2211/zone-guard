"""
    This module captures frames from a camera source in a 
    dedicated background thread and stores them in a ring buffer
"""

import cv2
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FrameData:
    """One frame with metadata"""
    frame:np.ndarray
    timestamp:float
    camera_id:str
    frame_number:int

class CameraStream:
    def __init__(self, camera_id: str, url, fps_cap: int=30, buffer_size: int = 30):
        self.camera_id = camera_id
        self.url = url # 0: webcam, or "rtsp://..."
        self.fps_cap = fps_cap

        # Ring buffer: automactically delete frames if full
        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock() # Thread safety
        self._stop_event = threading.Event() # Signal for stopping
        self._thread = None
        self._frame_count = 0
        self._is_connected = False

    def start(self):
        """Start capturing background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera {self.camera_id}] Thread started")

    def stop(self):
        """Stop capturing."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print(f"[Camera {self.camera_id}] Stopped")

    def get_latest(self) -> Optional[FrameData]:
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None
        
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def _capture_loop(self):
        # Open camera
        src = int(self.url) if str(self.url).isdigit() else self.url
        cap = cv2.VideoCapture(src)

        # Stop if the camera not available
        if not cap.isOpened():
            print(f"[Camera {self.camera_id}] Failed to open: {self.url}") 
            return
        
        self._is_connected = True
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera {self.camera_id}] Connected: {w}x{h}")

        min_interval = 1.0/self.fps_cap # Time to cut between 2 frames

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            
            ret, frame = cap.read() # ret = true, read successfully
            if not ret:
                print(f"[Camera {self.camera_id}] Frame grab failed")
                self._is_connected = False
                break

            self._frame_count += 1
            frame_data = FrameData(
                frame=frame,
                timestamp=time.time(),
                camera_id=self.camera_id,
                frame_number=self._frame_count,
            )

            # push into buffer
            with self._lock:
                self._buffer.append(frame_data)

            elapsed = time.monotonic() - t0
            sleep_time = min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release() # Close

        

