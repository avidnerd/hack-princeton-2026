"""Phone link: other devices can reach the photo page and upload, and nothing else."""
import shutil
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from spoonful.app.server import create_app
from spoonful.arm import kinematics as K
from conftest import wait_state


@pytest.fixture
def app_system(sim_system, tmp_path, monkeypatch):
    s = sim_system
    d = tmp_path / "moves"
    shutil.copytree(s.lib.dir, d)
    s.lib.dir = d
    s.lib.reload()
    cls = type(s.cfg)
    monkeypatch.setattr(cls, "save", lambda self, *a, **k: None)
    orig = cls.path
    monkeypatch.setattr(cls, "path", lambda self, p: tmp_path / p if str(p).startswith("data/ph") else orig(self, p))
    (tmp_path / "data").mkdir()
    wait_state(s, "IDLE")
    app = create_app(s)
    return s, app, app.state.phone


def _jpeg():
    return cv2.imencode(".jpg", np.full((600, 800, 3), 180, np.uint8))[1].tobytes()


def test_other_devices_reach_only_the_phone_page_and_upload(app_system):
    s, app, link = app_system
    phone = TestClient(app, client=("192.168.1.50", 50000))
    assert phone.get(f"/p/{link.token}").status_code == 200
    assert link.token in phone.get(f"/p/{link.token}").text
    for path in ("/", "/api/health", "/stream", "/api/phone/qr.svg", "/p/wrongtoken"):
        assert phone.get(path).status_code == 403, path
    for path in ("/api/stop", "/api/go", "/api/torque", "/api/phone/photo/wrongtoken"):
        assert phone.post(path, json={}).status_code == 403, path
    with pytest.raises(Exception):
        with phone.websocket_connect("/ws"):
            pass
    laptop = TestClient(app)
    assert laptop.get("/").status_code == 200
    assert laptop.get("/api/phone/qr.svg").headers["content-type"].startswith("image/svg")
    assert laptop.get("/p/wrongtoken").status_code == 404


def test_phone_photo_shows_up_and_reports_what_to_fix(app_system):
    s, app, link = app_system
    phone, laptop = TestClient(app, client=("192.168.1.50", 50000)), TestClient(app)
    s.plate.image_generate = lambda jpg, prompt: {}
    assert phone.post(f"/api/phone/photo/{link.token}", content=b"not a photo").status_code == 400
    r = phone.post(f"/api/phone/photo/{link.token}", content=_jpeg(), headers={"content-type": "image/jpeg"})
    assert r.status_code == 200
    t0 = time.time()
    while link.status == "finding" and time.time() - t0 < 5:
        time.sleep(0.05)
    assert link.count == 1 and link.status == "problem"          # the spoon wasn't posed: says so, moves nothing
    assert "Pose for photo" in link.message
    assert laptop.get("/api/phone/photo.jpg").status_code == 200  # and the photo itself is on the laptop


def test_save_mouth_sets_waiting_pose_and_face_zone(app_system):
    s, app, link = app_system
    laptop = TestClient(app)
    kin = s.cfg.kinematics
    s.arm.goto(dict(s.lib.pose("pre_delivery")), "normal", 1.0).wait(10)
    assert laptop.post("/api/setup/mouth/save").status_code == 200
    mouth, pre = s.lib.pose("mouth_nominal"), s.lib.pose("pre_delivery")
    assert kin.face_cyl is not None and K.in_face_zone(kin, mouth) and not K.in_face_zone(kin, pre)
    assert 70 <= K.tip_distance(kin, mouth, pre) <= 110
    assert s.arm.torque_on


def test_nudge_moves_the_scoop_and_shows_the_new_aim(app_system):
    import math
    from spoonful.arm.scoop_style import scoop_anchor
    s, app, link = app_system
    laptop = TestClient(app)
    kin = s.cfg.kinematics
    x0, y0 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    r = laptop.post("/api/setup/aim", json={"bowl": "bowl1", "further_mm": 10, "left_mm": 5})
    assert r.status_code == 200, r.text
    x1, y1 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    rr = math.hypot(x0, y0)
    assert (x1 - x0) * x0 / rr + (y1 - y0) * y0 / rr == pytest.approx(10, abs=1.5)        # further from the base
    assert -(x1 - x0) * y0 / rr + (y1 - y0) * x0 / rr == pytest.approx(5, abs=1.5)        # to the arm's left
    assert s.lib.has("lift_bowl1")
    time.sleep(0.2)
    s.arm._motion and s.arm._motion.wait(10)
    tip = K.fk(kin, s.arm.state.deg)
    assert math.hypot(tip[0] - x1, tip[1] - y1) < 6                                        # hovering over the new spot
    assert tip[2] > K.fk(kin, min(s.lib.items["scoop_bowl1"].poses, key=lambda q: K.fk(kin, q)[2]))[2] + 40
    assert laptop.post("/api/setup/aim", json={"bowl": "bowl1", "further_mm": 80}).status_code == 409   # too far
    assert laptop.post("/api/setup/aim", json={"bowl": "bowl9"}).status_code == 409


def test_phone_live_camera_becomes_the_table_view(app_system):
    import numpy as np
    s, app, link = app_system
    phone, laptop = TestClient(app, client=("192.168.1.50", 50000)), TestClient(app)
    assert link.url.startswith("https://")                                # live camera needs a secure page
    laptop_view = s.table_frame()
    assert not s.phone_camera.live and laptop_view is not None

    s.aim_check.jacobians["bowl1"] = np.eye(3, 2)                         # learned through the laptop's view
    pic = np.full((540, 960, 3), 90, np.uint8)
    pic[:, :480] = 200
    r = phone.post(f"/api/phone/frame/{link.token}", content=cv2.imencode(".jpg", pic)[1].tobytes())
    assert r.status_code == 200 and s.phone_camera.live
    assert phone.post("/api/phone/frame/wrongtoken", content=b"x").status_code == 403
    assert phone.get("/phone/stream").status_code == 403                  # the phone can send, not watch
    view = s.table_frame()
    assert view.shape == (540, 960, 3) and abs(int(view[10, 10, 0]) - 200) < 8      # the phone's picture
    assert s.aim_check.jacobians == {}                                    # a new view: what was learned is dropped

    s.phone_camera.at -= 5                                                # the phone stops sending
    assert not s.phone_camera.live
    assert s.table_frame().shape == laptop_view.shape                     # back to the laptop's camera


def test_certificate_is_made_once(tmp_path):
    from spoonful.app.phone import ensure_cert
    cert, key = ensure_cert(tmp_path)
    first = cert.read_bytes()
    assert b"BEGIN CERTIFICATE" in first and b"PRIVATE KEY" in key.read_bytes()
    assert ensure_cert(tmp_path)[0].read_bytes() == first
