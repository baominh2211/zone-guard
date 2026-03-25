"""MJPEG live stream + WebSocket."""
import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.get("/{camera_id}")
async def mjpeg(camera_id: str, request: Request):
    app = getattr(request.app.state, "app_instance", None)
    async def gen():
        while True:
            frame = app.get_annotated_frame(camera_id) if app else None
            if frame is None:
                frame = np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(frame, f"No feed: {camera_id}", (50,240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            await asyncio.sleep(1/15)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


class WSManager:
    def __init__(self):
        self.conns: list[WebSocket] = []
    async def connect(self, ws):
        await ws.accept()
        self.conns.append(ws)
    def disconnect(self, ws):
        self.conns.remove(ws)
    async def broadcast(self, data):
        dead = []
        for ws in self.conns:
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead:
            self.conns.remove(ws)

ws_manager = WSManager()

@router.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
