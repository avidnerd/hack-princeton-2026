"""FastAPI app: pages, MJPEG camera stream with overlay, WebSocket state push at app.ws_hz."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..perception.face import draw_overlay
from ..system import System
from .api import router

STATIC = Path(__file__).parent / "static"


def snapshot(sys: System) -> dict:
    f = sys.face.latest()
    fsm = sys.fsm.status()
    return {
        "t": time.time(),
        "fsm": fsm,
        "face": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.scalars().items()},
        "arm": sys.arm.status(),
        "pace": sys.pace.status(),
        "profile": sys.profiles.current.__dict__,
        "profiles": sys.profiles.names(),
        "camera": {"fps": round(sys.camera.fps, 1), "face_fps": round(sys.face.fps, 1), "error": sys.camera.error,
                   "face_error": sys.face.error, "w": sys.camera.width, "h": sys.camera.height},
        "esp32": sys.esp32.status(),
        "voice": sys.voice.status(),
        "food": sys.food.status(),
        "teach": sys.recorder.live() if sys.recorder.recording else {"recording": False},
        "sim": {"backend": sys.is_sim, "face": sys.face.sim_enabled, "face_state": sys.face.sim},
        "summary_outbox": sys.summary_outbox[-1:] if sys.summary_outbox else [],
        "calib": {"samples": [{"label": s["label"], "px": s["px"]} for s in sys.calib_samples],
                  "map": bool(sys.offset_map)},
        "window": sys.cfg.face.delivery_window_px.model_dump(),
    }


def create_app(sys: System) -> FastAPI:
    app = FastAPI(title="Spoonful")
    app.state.sys = sys
    app.state.timeline = []
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def on_log(kind: str, row: dict) -> None:
        tl = app.state.timeline
        item = {k: v for k, v in row.items() if k not in ("detail", "kind")}
        detail = dict(row.get("detail") or {})
        if kind == "fault":
            detail = {"kind": row.get("kind"), **detail}
        tl.append({**item, "kind": kind, "detail": detail})
        del tl[:-40]
    sys.log.on_event(on_log)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/caregiver")
    def caregiver():
        return FileResponse(STATIC / "caregiver.html")

    def frames():
        boundary = b"--frame\r\n"
        import cv2
        period = 1.0 / sys.cfg.app.stream_fps
        last_seq = -1
        while True:
            t0 = time.monotonic()
            fr = sys.camera.latest()
            if fr is None or fr.seq == last_seq:
                time.sleep(0.02)
                continue
            last_seq = fr.seq
            st = sys.fsm.status()
            img = draw_overlay(fr.image, sys.cfg, sys.face.latest(), st["gate"]["ok"], st["gate"]["reason"],
                               st["state"], st["countdown"])
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, sys.cfg.camera.jpeg_quality])
            if ok:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            sys.food.maybe_check(fr.image, time.time()) if sys.fsm.state.value == "LISTENING" else None
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    @app.get("/stream")
    def stream():
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        period = 1.0 / sys.cfg.app.ws_hz
        try:
            while True:
                snap = snapshot(sys)
                snap["timeline"] = app.state.timeline[-20:]
                await sock.send_text(json.dumps(snap, default=str))
                await asyncio.sleep(period)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
