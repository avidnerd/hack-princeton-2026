"""One overhead photo -> table map in arm coordinates -> scoops built without touching the bowls."""
import base64
import math
import shutil
import time

import cv2
import numpy as np
import pytest

from spoonful.arm import kinematics as K
from spoonful.perception.photo_map import PHOTO_TIP_MM, FoodSurface, analyze, predicted_surface_mm, similarity

W, H = 1200, 900
ROT, MM_PER_PX, ORIGIN = math.radians(35), 0.45, (420.0, 610.0)   # where the base sits in the photo


def mm_to_px(mm):
    """A made-up overhead camera: rotated, scaled, base at ORIGIN; pixel y points down."""
    x, y = mm
    c, s = math.cos(ROT), math.sin(ROT)
    u = (x * c - y * s) / MM_PER_PX
    v = (x * s + y * c) / MM_PER_PX
    return ORIGIN[0] + u, ORIGIN[1] - v


def norm(px):
    return [px[1] / H * 1000, px[0] / W * 1000]


def fake_vision(bowls_mm):
    def gen(jpg):
        out = {"base": norm(mm_to_px((0, 0))), "tip": norm(mm_to_px(PHOTO_TIP_MM[:2])), "bowls": []}
        for (x, y), food, amount in bowls_mm:
            u, v = mm_to_px((x, y))
            r = 40 / MM_PER_PX
            out["bowls"].append({"box_2d": [(v - r) / H * 1000, (u - r) / W * 1000, (v + r) / H * 1000, (u + r) / W * 1000],
                                 "food": food, "amount": amount, "texture": "soft"})
        return out
    return gen


def test_similarity_recovers_positions():
    f = similarity(mm_to_px((0, 0)), mm_to_px(PHOTO_TIP_MM[:2]), PHOTO_TIP_MM[:2])
    for mm in [(150, 120), (230, -60), (90, 260)]:
        x, y = f(mm_to_px(mm))
        assert abs(x - mm[0]) < 0.5 and abs(y - mm[1]) < 0.5
    assert abs(f.mm_per_px - MM_PER_PX) < 1e-6


def test_analyze_maps_every_bowl():
    bowls = [((200, 110), "applesauce", "full"), ((210, -70), "rice", "low")]
    m = analyze(b"x", W, H, fake_vision(bowls))
    got = {b["food"]: b["mm"] for b in m["bowls"]}
    assert set(got) == {"applesauce", "rice"}
    for (x, y), food, _ in bowls:
        assert math.dist(got[food], (x, y)) < 3
    assert [b["name"] for b in m["bowls"]] == ["bowl1", "bowl2"]
    assert m["bowls"][0]["food"] == "applesauce"                      # named left to right as the arm sees them
    assert abs(m["bowls"][0]["diameter_mm"] - 80) < 3


def test_analyze_needs_base_and_tip():
    with pytest.raises(ValueError, match="spoon tip"):
        analyze(b"x", W, H, lambda jpg: {"base": [500, 500], "tip": None, "bowls": []})


def test_food_surface_learns_from_touch():
    fs = FoodSurface()
    assert fs.predict("bowl1", "full") == pytest.approx(predicted_surface_mm("full"))
    errs = [abs(fs.felt("bowl1", 30.0)["error_mm"]) for _ in range(4)]
    assert errs[0] > errs[1] > errs[2] > errs[3]                       # prediction converges on what it feels
    assert abs(fs.est["bowl1"] - 30) < 2


def test_photo_upload_builds_scoops_without_touching(sim_system, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from spoonful.app.server import create_app
    s = sim_system
    d = tmp_path / "moves"
    shutil.copytree(s.lib.dir, d)
    s.lib.dir = d
    s.lib.reload()
    cls = type(s.cfg)
    monkeypatch.setattr(cls, "save", lambda self, *a, **k: None)       # never touch the real config.yaml
    orig_path = cls.path
    monkeypatch.setattr(cls, "path", lambda self, p: tmp_path / p if str(p).startswith("data/photo") else orig_path(self, p))
    kin = s.cfg.kinematics
    dip = min(s.lib.items["scoop_bowl1"].poses, key=lambda q: K.fk(kin, q)[2])
    x1, y1, _ = K.fk(kin, dip)
    bowls = [((x1, y1), "applesauce", "full"), ((x1 + 20, y1 - 60), "rice", "half")]
    s.plate.image_generate = lambda jpg, prompt: fake_vision(bowls)(jpg)
    import spoonful.perception.photo_map as PM                         # the fake camera answers only unrotated
    monkeypatch.setattr(PM, "ROTATIONS", (0,))
    monkeypatch.setattr(PM, "MIN_VOTES", 1)
    monkeypatch.setattr(PM, "UNSURE_VOTES", 1)
    monkeypatch.setattr(PM.analyze_robust, "__defaults__", ((170.0, 0.0), (0,)))
    img = np.full((H, W, 3), 200, np.uint8)
    data = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()
    from conftest import wait_state
    wait_state(s, "IDLE")
    c = TestClient(create_app(s))
    assert c.post("/api/photo/upload", json={"image": data}).status_code == 409     # spoon not posed yet
    shown = s.setup_state["candidates"]                                  # ...but the bowls are still outlined,
    assert len(shown) == 2 and all(not k["ok"] and "spoon" in k["why"] and len(k["poly"]) >= 3 for k in shown)
    assert c.post("/api/setup/bowls", json={"ids": [0, 1]}).status_code == 409   # just not choosable
    assert c.post("/api/photo/pose").status_code == 200
    t0 = time.time()
    while s.arm.busy or time.time() - t0 < 0.5:
        time.sleep(0.05)
    PM_TIP = K.fk(kin, s.arm.state.deg)

    def vision(jpg, prompt):                   # the fake camera sees the spoon tip where the arm really holds it
        d = fake_vision(bowls)(jpg)
        d["tip"] = norm(mm_to_px(PM_TIP[:2]))
        return d
    s.plate.image_generate = vision
    r = c.post("/api/photo/upload", json={"image": data})
    assert r.status_code == 200, r.text
    cands = r.json()["candidates"]
    assert len(cands) == 2 and all(k["ok"] and len(k["poly"]) >= 3 for k in cands)
    assert not s.lib.has("lift_bowl1")                                   # nothing is built until bowls are chosen
    assert c.post("/api/setup/bowls", json={"ids": []}).status_code == 409

    rice = next(k for k in cands if k["food"] == "rice")
    r = c.post("/api/setup/bowls", json={"ids": [rice["id"]]})           # serve only the rice
    assert r.status_code == 200, r.text
    served = r.json()["bowls"]
    assert [b["food"] for b in served] == ["rice"] and served[0]["name"] == "bowl1" and served[0]["strategy"]
    assert list(s.plate.bowls) == ["bowl1"] and s.plate.food_of("bowl1") == "rice"
    assert s.plate.find("some applesauce please") is None               # the bowl that wasn't chosen is unknown to it
    sc = s.lib.items["scoop_bowl1"].poses
    lx, ly, lz = K.fk(kin, min(sc, key=lambda q: K.fk(kin, q)[2]))
    target = bowls[1][0]
    assert math.dist((lx, ly), target) < 35                             # it dips into the bowl that was chosen
    assert s.food_surface.est

    r = c.post("/api/setup/bowls", json={"ids": [k["id"] for k in cands]})   # change of mind: both
    assert [b["food"] for b in r.json()["bowls"]] == ["applesauce", "rice"]  # named left to right
    assert s.lib.has("scoop_bowl2") and s.lib.has("lift_bowl2")


def test_outline_follows_the_bowl_not_the_box():
    from spoonful.brain.photo_setup import outline
    img = np.full((400, 600, 3), 235, np.uint8)
    cv2.circle(img, (300, 200), 80, (60, 120, 200), -1)                 # a round bowl on a pale table
    poly = np.array(outline(img, (210, 110, 390, 290)))
    r = np.hypot(poly[:, 0] - 300, poly[:, 1] - 200)
    assert len(poly) >= 6 and 70 < r.mean() < 92 and r.max() < 100      # hugs the circle, not the square box
    flat = outline(np.full((400, 600, 3), 235, np.uint8), (210, 110, 390, 290))
    assert len(flat) == 24                                               # nothing to separate: the box's ellipse
