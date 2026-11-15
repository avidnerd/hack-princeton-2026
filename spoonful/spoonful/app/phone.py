"""Phone link: take the overhead photo on a phone and have it appear on the laptop.

The laptop serves one small page to the phone (same Wi-Fi). The page opens the phone's camera, shrinks the photo
and posts it back; the Setup screen shows it and maps the table straight away.

Opening the server to the local network must not open the robot to it. LocalOnly lets other devices reach exactly
two URLs, both carrying a token that is new each run and only shown on the laptop (as a QR code): the phone page and
the photo upload. Everything else (controls, camera stream, WebSocket) answers only to the laptop itself.
"""
from __future__ import annotations

import io
import secrets
import socket
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from ..brain import photo_setup

LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
PAGE = Path(__file__).parent / "static" / "phone.html"


def lan_ip() -> str:
    """This machine's address on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class PhoneLink:
    def __init__(self, port: int):
        self.token = secrets.token_hex(4)
        self.port = port
        self.count = 0
        self.received_at: float | None = None
        self.status = "waiting"            # waiting | mapping | done | problem
        self.message = ""
        self._ip = (0.0, "127.0.0.1")

    @property
    def url(self) -> str:
        if time.time() - self._ip[0] > 10:
            self._ip = (time.time(), lan_ip())
        return f"http://{self._ip[1]}:{self.port}/p/{self.token}"

    def allows(self, path: str) -> bool:
        return path in (f"/p/{self.token}", f"/api/phone/photo/{self.token}")

    def state(self) -> dict:
        return {"url": self.url, "count": self.count, "received_at": self.received_at, "status": self.status,
                "message": self.message}


class LocalOnly:
    """ASGI middleware: requests from other machines are refused unless `allows(path)`."""

    def __init__(self, app, allows):
        self.app, self.allows = app, allows

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            host = (scope.get("client") or ("",))[0]
            if host not in LOCAL and not self.allows(scope["path"]):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 403,
                                "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
                    await send({"type": "http.response.body", "body": b"Not available from this device."})
                return
        await self.app(scope, receive, send)


def handle_photo(sys, link: PhoneLink, img) -> None:
    """Save the photo for the Setup screen, then map the table in the background."""
    import cv2
    cv2.imwrite(str(sys.cfg.path("data/phone_photo.jpg")), img)
    link.count += 1
    link.received_at = time.time()
    link.status, link.message = "mapping", "Finding the bowls"

    def run():
        try:
            r = photo_setup.map_table(sys, img)
            n = len(r["bowls"])
            link.status, link.message = "done", f"{n} bowl{'s' if n != 1 else ''} mapped"
        except photo_setup.SetupProblem as e:
            link.status, link.message = "problem", str(e)
        except Exception as e:
            sys.log.event("photo_map_failed", detail_error=repr(e)[:200])
            link.status, link.message = "problem", "Mapping failed. Retake the photo."
    threading.Thread(target=run, name="map-table", daemon=True).start()


def make_router(sys, link: PhoneLink) -> APIRouter:
    r = APIRouter()

    @r.get("/p/{token}", response_class=HTMLResponse)
    def phone_page(token: str):
        if not secrets.compare_digest(token, link.token):
            raise HTTPException(404)
        return PAGE.read_text(encoding="utf-8").replace("{{TOKEN}}", link.token)

    @r.post("/api/phone/photo/{token}")
    async def phone_photo(token: str, request: Request):
        import cv2
        import numpy as np
        if not secrets.compare_digest(token, link.token):
            raise HTTPException(404)
        data = await request.body()
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Photo too large")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Not a photo")
        handle_photo(sys, link, img)
        return {"ok": True}

    @r.get("/api/phone/link")
    def phone_link():
        return link.state()

    @r.get("/api/phone/qr.svg")
    def qr():
        import segno
        buf = io.BytesIO()
        segno.make(link.url, error="m").save(buf, kind="svg", scale=5, border=2, dark="#14212b", light="#ffffff")
        return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @r.get("/api/phone/photo.jpg")
    def latest(mapped: int = 0):
        p = sys.cfg.path("data/photo_map.jpg" if mapped else "data/phone_photo.jpg")
        if not p.exists():
            raise HTTPException(404)
        return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return r
