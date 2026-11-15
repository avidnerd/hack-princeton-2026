"""One overhead photo -> table map in arm coordinates -> scoops built without touching the bowls."""
import base64
import math
import shutil

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
    c = TestClient(create_app(s))
    r = c.post("/api/photo/upload", json={"image": data})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["bowls"], j
    for name in j["bowls"]:
        assert s.lib.has(f"scoop_{name}") and s.lib.has(f"lift_{name}")
    sc = s.lib.items[f"scoop_{j['bowls'][0]}"].poses
    low = min(sc, key=lambda q: K.fk(kin, q)[2])
    lx, ly, lz = K.fk(kin, low)
    target = [b for b in bowls if b[1] == s.plate.food_of(j["bowls"][0])][0][0]
    assert math.dist((lx, ly), target) < 35                             # it dips into the bowl it saw
    assert s.food_surface.est                                           # and holds a prediction for each bowl



def test_consensus_keeps_what_the_rotated_views_agree_on():
    """Truth in a W x H photo; each view's answer is the truth rotated (plus a little noise); one view invents a
    bowl. Mapped back, the real things agree, the phantom is dropped, a 2-of-4 bowl is 'unsure'."""
    from spoonful.perception.photo_map import consensus
    truth = {"base": (600, 420), "tip": (640, 640), "cup": (500, 760), "cup2": (760, 760), "maybe": (1100, 300)}

    def rotate(x, y, rot):                     # original pixel -> pixel in the image rotated clockwise by rot
        return {0: (x, y), 90: (H - y, x), 180: (W - x, H - y), 270: (y, W - x)}[rot]

    answers = []
    for i, rot in enumerate((0, 90, 180, 270)):
        w, h = (W, H) if rot in (0, 180) else (H, W)
        def n(pt, jitter=0):
            x, y = rotate(pt[0] + jitter, pt[1] - jitter, rot)
            return [y / h * 1000, x / w * 1000]
        def box(pt, r=45):
            (ya, xa), (yb, xb) = n((pt[0] - r, pt[1] - r)), n((pt[0] + r, pt[1] + r))
            return [min(ya, yb), min(xa, xb), max(ya, yb), max(xa, xb)]
        bowls = [{"box_2d": box(truth["cup"]), "food": ["jelly", "green jelly", "jelly", "gelatin"][i]},
                 {"box_2d": box(truth["cup2"]), "food": "applesauce"}]
        if i < 2:
            bowls.append({"box_2d": box(truth["maybe"]), "food": "rice"})
        if i == 3:
            bowls.append({"box_2d": box((60, 300)), "food": "peanut butter"})
        answers.append((rot, {"base": n(truth["base"], 3 * i), "tip": n(truth["tip"], i), "bowls": bowls}, w, h))
    c = consensus(answers, W, H)
    assert math.dist(c["base"]["px"], truth["base"]) < 8 and c["base"]["votes"] == 4
    assert math.dist(c["tip"]["px"], truth["tip"]) < 4
    foods = {b["food"]: b for b in c["bowls"]}
    assert "peanut butter" not in foods                                  # seen once: dropped
    assert foods["rice"]["sure"] is False and foods["rice"]["votes"] == 2
    assert foods["jelly"]["sure"] and math.dist(foods["jelly"]["px"], truth["cup"]) < 2
    assert foods["applesauce"]["votes"] == 4
