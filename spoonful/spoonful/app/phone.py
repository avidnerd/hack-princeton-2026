"""Phone link: the phone as the robot's camera for the table.

The laptop serves one small page to the phone (same Wi-Fi). From it the phone can
  - take the overhead table photo, which appears on the Setup screen and is mapped straight away, and
  - stream its camera live: the laptop shows the view and uses it to find the spoon and the bowls (aim check, the
    look before every scoop). The laptop's own webcam stays the face-safety camera: it is wired, so a Wi-Fi hiccup
    can't stall or mislead the check that guards the diner's face.

A phone browser only allows live camera access on a secure page, so the phone page is served over HTTPS on its own
port with a certificate made on this machine (the phone shows a warning once; the link is still encrypted).

Opening a port to the local network must not open the robot to it. LocalOnly lets other devices reach exactly the
phone page and its two uploads, each carrying a token that is new every run and only shown on the laptop (as a QR
code). Everything else (controls, camera stream, WebSocket) answers only to the laptop itself.
"""
from __future__ import annotations

import io
import secrets
import socket
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

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


class Chosen(BaseModel):
    ids: list[int]


class Aim(BaseModel):
    bowl: str
    further_mm: float = 0
    left_mm: float = 0


class PhoneCamera:
    """The latest frame streamed from the phone."""

    FRESH_S = 3.0               # Wi-Fi stalls for a second now and then
    GONE_S = 20.0               # after this long without a frame the phone is treated as put away

    def __init__(self):
        self.frame = None                   # BGR image
        self.at = 0.0                       # time.monotonic() when it arrived
        self.count = 0
        self.fps = 0.0
        self._window = (time.monotonic(), 0)

    def put(self, img) -> None:
        now = time.monotonic()
        self.frame, self.at = img, now
        self.count += 1
        t0, n0 = self._window
        if now - t0 >= 1.0:
            self.fps = (self.count - n0) / (now - t0)
            self._window = (now, self.count)

    @property
    def live(self) -> bool:
        return self.frame is not None and time.monotonic() - self.at <= self.FRESH_S

    @property
    def in_use(self) -> bool:
        """The phone is the table camera: it is sending, or stopped only moments ago."""
        return self.frame is not None and time.monotonic() - self.at <= self.GONE_S

    def image(self):
        return self.frame if self.live else None


def ensure_cert(directory: Path) -> tuple[Path, Path]:
    """A self-signed certificate for the phone page, made once and reused."""
    cert, key = directory / "phone_cert.pem", directory / "phone_key.pem"
    if cert.exists() and key.exists():
        return cert, key
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    k = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Spoonful")])
    now = datetime.datetime.now(datetime.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=365))
         .add_extension(x509.SubjectAlternativeName([x509.DNSName("spoonful.local")]), critical=False)
         .sign(k, hashes.SHA256()))
    directory.mkdir(parents=True, exist_ok=True)
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    return cert, key


class PhoneLink:
    def __init__(self, port: int):
        self.token = secrets.token_hex(4)
        self.port = port
        self.count = 0
        self.received_at: float | None = None
        self.status = "waiting"            # waiting | finding | choose | done | problem
        self.message = ""
        self._ip = (0.0, "127.0.0.1")

    @property
    def url(self) -> str:
        if time.time() - self._ip[0] > 10:
            self._ip = (time.time(), lan_ip())
        return f"https://{self._ip[1]}:{self.port}/p/{self.token}"

    def allows(self, path: str) -> bool:
        return path in (f"/p/{self.token}", f"/api/phone/photo/{self.token}", f"/api/phone/frame/{self.token}")

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
    """Find the bowls in the photo in the background; the Setup screen then asks which ones to serve from."""
    link.count += 1
    link.received_at = time.time()
    link.status, link.message = "finding", "Finding the bowls"

    def run():
        try:
            found = photo_setup.find_bowls(sys, img)
            n = sum(1 for c in found if c["ok"])
            link.status, link.message = "choose", f"{n} bowl{'s' if n != 1 else ''} in reach. Choose which to serve."
        except photo_setup.SetupProblem as e:
            link.status, link.message = "problem", str(e)
        except Exception as e:
            sys.log.event("photo_bowls_failed", detail_error=repr(e)[:200])
            link.status, link.message = "problem", "Couldn't read the photo. Retake it."
    threading.Thread(target=run, name="find-bowls", daemon=True).start()


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

    @r.post("/api/setup/bowls")
    def choose_bowls(b: Chosen):
        try:
            bowls = photo_setup.use_bowls(sys, b.ids)
        except photo_setup.SetupProblem as e:
            raise HTTPException(409, str(e))
        n = len(bowls)
        link.status, link.message = "done", f"Serving from {n} bowl{'s' if n != 1 else ''}"
        return {"ok": True, "bowls": bowls}

    @r.post("/api/setup/aim")
    def aim(b: Aim):
        """Hover over the bowl; with a nudge, move its scoop first."""
        try:
            if b.further_mm or b.left_mm:
                return {"ok": True, **photo_setup.nudge(sys, b.bowl, b.further_mm, b.left_mm)}
            photo_setup.aim(sys, b.bowl)
        except photo_setup.SetupProblem as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @r.post("/api/setup/aim/camera")
    def aim_camera(b: Aim):
        """Check the aim with the webcam and correct it: one bowl, or every bowl when bowl is "all"."""
        try:
            sys.aim_check.start(list(sys.plate.bowls) if b.bowl == "all" else [b.bowl])
        except photo_setup.SetupProblem as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @r.get("/api/phone/link")
    def phone_link():
        return link.state()

    @r.post("/api/phone/frame/{token}")
    async def phone_frame(token: str, request: Request):
        """One frame of the phone's live camera."""
        import cv2
        import numpy as np
        if not secrets.compare_digest(token, link.token):
            raise HTTPException(404)
        data = await request.body()
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Frame too large")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Not a picture")
        sys.phone_camera.put(img)
        return {"ok": True}

    @r.get("/phone/stream")
    def phone_stream():
        """The phone's view, for the laptop screen."""
        import cv2

        def frames():
            last = -1
            while True:
                cam = sys.phone_camera
                if cam.count == last or cam.frame is None:
                    time.sleep(0.05)
                    if not cam.live and last != -1:
                        return
                    continue
                last = cam.count
                ok, jpg = cv2.imencode(".jpg", cam.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @r.get("/api/phone/qr.svg")
    def qr():
        import segno
        buf = io.BytesIO()
        segno.make(link.url, error="m").save(buf, kind="svg", scale=5, border=2, dark="#14212b", light="#ffffff")
        return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @r.get("/api/phone/photo.jpg")
    def latest():
        p = sys.cfg.path("data/phone_photo.jpg")
        if not p.exists():
            raise HTTPException(404)
        return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return r
