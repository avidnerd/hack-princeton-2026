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
    while link.status == "mapping" and time.time() - t0 < 5:
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
