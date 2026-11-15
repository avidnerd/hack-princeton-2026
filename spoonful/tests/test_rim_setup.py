"""Setup side of the rim-aware scoop: cups from the photo, confirming a cup with the spoon over it, and the screens.
Sim arm only; the config and the moves are never written outside tmp_path."""
import math
import shutil
import sys as _sys
import time
import types

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from spoonful.app.server import create_app, snapshot
from spoonful.arm import kinematics as K
from spoonful.brain import photo_setup as PS
from spoonful.config import CupCfg
from spoonful.perception.rim import Observation, RimNotFound
from conftest import wait_state


class FakeRim:
    """Stands in for perception.rim.RimObserver: remembers what it was asked."""

    def __init__(self, sees=True):
        self.sees, self.refs, self.last, self.calib, self.calls = sees, {}, {}, None, []

    def set_reference(self, bowl, img, roi, cup_xy, radius_mm):
        self.calls.append(("set_reference", bowl, tuple(roi), tuple(cup_xy), radius_mm, img.shape[:2]))
        if not self.sees:
            raise RimNotFound("partly_hidden")
        self.refs[bowl] = (tuple(roi), tuple(cup_xy), radius_mm)

    def has_reference(self, bowl):
        return bowl in self.refs

    def forget(self, bowl=None):
        self.calls.append(("forget", bowl))
        if bowl is None:
            self.refs, self.last = {}, {}
        else:
            self.refs.pop(bowl, None)
            self.last.pop(bowl, None)

    def observe(self, bowl):
        self.last[bowl] = Observation("ok" if bowl in self.refs else "lost", bowl,
                                      reason="" if bowl in self.refs else "no_reference")
        return self.last[bowl]

    def draw(self, img, bowl=None):
        return img.copy()


@pytest.fixture
def rig(sim_system, tmp_path, monkeypatch):
    s = sim_system
    d = tmp_path / "moves"
    shutil.copytree(s.lib.dir, d)
    s.lib.dir = d
    s.lib.reload()
    cls = type(s.cfg)
    monkeypatch.setattr(cls, "save", lambda self, *a, **k: None)           # never the real config.yaml
    orig = cls.path
    monkeypatch.setattr(cls, "path", lambda self, p: tmp_path / p if str(p).startswith("data/ph") else orig(self, p))
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(PS, "CAMERA_SETTLE_S", 0.0)
    wait_state(s, "IDLE")
    s.rim = s.fsm.rim = FakeRim()
    s.plate.image_generate = lambda jpg, prompt: [{"food": "rice", "box_2d": [400, 300, 700, 600]}]
    return s


def settle(s):
    time.sleep(0.15)
    s.arm._motion and s.arm._motion.wait(15)
    time.sleep(0.4)


def anchor_cup(s, name="bowl1", **kw):
    """Give a hand-taught sim bowl a cup where its scoop dips, as the photo would have."""
    kin = s.cfg.kinematics
    x, y, _ = K.fk(kin, PS._dip(s, name))
    b = next(b for b in s.cfg.plate.bowls if b.name == name)
    b.cup = CupCfg(**{"x_mm": x, "y_mm": y, "radius_mm": 35.0, "uncertainty_mm": 14.0, "source": "photo", **kw})
    return b


def test_use_bowls_stores_the_cup_the_photo_saw(rig, monkeypatch):
    s = rig
    kin = s.cfg.kinematics
    scoop, lift = s.lib.items["scoop_bowl1"].poses, s.lib.items["lift"].poses
    x, y, _ = K.fk(kin, PS._dip(s, "bowl1"))
    monkeypatch.setattr(s.plate, "plan_attack", lambda bowl, crop: {"strategy": "skim", "why": "test"})
    s.setup_state["candidates"] = [{
        "id": 0, "food": "rice", "amount": "half", "texture": "soft", "sure": True, "votes": 3, "views": 3,
        "uncertainty_mm": 14.5, "diameter_mm": 80.0, "at_cm": [round(x / 10), round(y / 10)], "box": [0, 0, 10, 10],
        "poly": [], "ok": True, "why": "", "used": False,
        "_scoop": scoop, "_lift": lift, "_surface": 30.0, "_mm": (x, y), "_crop": b"jpg"}]
    s.rim.refs["bowl1"] = "from the old table"
    out = PS.use_bowls(s, [0])
    assert [b["name"] for b in out] == ["bowl1"] and all(not k.startswith("_") for k in out[0])
    cup = s.cfg.plate.bowls[0].cup
    assert cup is not None and (cup.x_mm, cup.y_mm) == pytest.approx((x, y), abs=0.1)     # the photo's centre
    assert cup.radius_mm == pytest.approx(80 / 2 - cup.wall_mm)                            # inside the walls
    assert cup.uncertainty_mm == 14.5 and cup.source == "photo"                            # the photo's own doubt
    assert cup.confirmed is False and cup.rim_mm is None                                   # nobody has measured it
    assert ("forget", None) in s.rim.calls and not s.rim.refs                              # a new table: start over
    st = PS.status(s)["cups"]["bowl1"]
    assert st["has_cup"] and not st["confirmed"] and not st["ok"] and st["line"].startswith("Not confirmed yet")
    assert st["width_mm"] == pytest.approx(76.0)


def test_show_aim_hovers_over_the_cup_centre_above_its_rim(rig):
    s = rig
    kin = s.cfg.kinematics
    b = anchor_cup(s, rim_mm=45.0)
    b.cup = b.cup.model_copy(update={"x_mm": b.cup.x_mm + 12, "y_mm": b.cup.y_mm - 9})     # not where the scoop dips
    PS.aim(s, "bowl1")
    x, y, z = K.fk(kin, s.setup_state["aimed"]["bowl1"])
    assert math.hypot(x - b.cup.x_mm, y - b.cup.y_mm) < 2.0
    assert z >= 45.0 + s.cfg.plate.rim.hover_above_rim_mm
    settle(s)
    tip = K.fk(kin, s.arm.state.deg)
    assert math.hypot(tip[0] - b.cup.x_mm, tip[1] - b.cup.y_mm) < 6.0


def test_confirm_refuses_unless_the_spoon_hovers_at_the_aim(rig):
    s = rig
    b = anchor_cup(s)
    before = b.cup
    with pytest.raises(PS.SetupProblem, match="isn't hovering"):            # Show aim was never pressed
        PS.confirm_cup(s, "bowl1", 45, 70)
    PS.aim(s, "bowl1")
    with pytest.raises(PS.SetupProblem, match="isn't hovering"):            # still on its way there
        PS.confirm_cup(s, "bowl1", 45, 70)
    settle(s)
    s.arm.goto(dict(s.lib.pose("rest")), "normal", 1.0).wait(15)            # it was sent somewhere else since
    time.sleep(0.3)
    with pytest.raises(PS.SetupProblem, match="isn't hovering"):
        PS.confirm_cup(s, "bowl1", 45, 70)
    PS.aim(s, "bowl2")                                                      # hovering, but over the other bowl
    settle(s)
    with pytest.raises(PS.SetupProblem, match="isn't hovering"):
        PS.confirm_cup(s, "bowl1", 45, 70)
    for rim_mm, width_mm in ((0, 70), (45, -1), (float("nan"), 70), (45, 5000)):
        with pytest.raises(PS.SetupProblem, match="millimetres"):
            PS.confirm_cup(s, "bowl1", rim_mm, width_mm)
    with pytest.raises(PS.SetupProblem):
        PS.confirm_cup(s, "bowl9", 45, 70)
    assert b.cup == before and not s.rim.refs
    assert not any(c[0] == "set_reference" for c in s.rim.calls)


def test_confirm_takes_the_measured_tip_and_teaches_the_camera(rig):
    s = rig
    kin = s.cfg.kinematics
    b = anchor_cup(s)
    PS.aim(s, "bowl1")
    settle(s)
    s.backend.sag(s.cfg.arm.servo_map["elbow"], -12, 0.0)                   # the arm droops and twists: it is not
    s.backend.sag(s.cfg.arm.servo_map["base"], 14, 0.0)                     # where it was told to be
    time.sleep(0.8)
    told = K.fk(kin, s.setup_state["aimed"]["bowl1"])
    measured = K.fk(kin, s.arm.state.deg)
    assert math.dist(told[:2], measured[:2]) > 3.0                          # the test means something
    r = PS.confirm_cup(s, "bowl1", 45, 70)
    cup = b.cup
    assert math.hypot(cup.x_mm - measured[0], cup.y_mm - measured[1]) < 2.0     # the MEASURED tip (encoder noise only)
    assert math.hypot(cup.x_mm - told[0], cup.y_mm - told[1]) > 2.0             # not the commanded point
    assert cup.confirmed and cup.source == "confirmed" and cup.uncertainty_mm == 3.0
    assert cup.rim_mm == 45 and cup.radius_mm == 35
    assert r["confirmed"] and r["camera"] and r["message"] == "Position confirmed. Camera sees the opening."
    # the arm went to rest before the camera looked, and the camera was shown the dish's region in frame pixels
    assert K.tip_distance(kin, s.arm.state.deg, s.lib.pose("rest")) < 25
    call = next(c for c in s.rim.calls if c[0] == "set_reference")
    h, w = call[5]
    x, y, rw, rh = call[2]
    assert call[1] == "bowl1" and call[3] == (cup.x_mm, cup.y_mm) and call[4] == 35
    assert x < 0.3 * w < 0.6 * w < x + rw and y < 0.4 * h < 0.7 * h < y + rh    # the dish box, a little larger
    assert 0 <= x and 0 <= y and x + rw <= w and y + rh <= h
    st = snapshot(s)["setup"]["cups"]["bowl1"]
    assert st["confirmed"] and st["measured"] and st["reference"] and st["seen"] == "ok" and st["ok"]
    assert st["line"] == "Position confirmed. Camera sees the opening." and st["blocked"] is False
    # Show aim now goes back to the pose it was confirmed in, not to a point short of it
    assert PS.aim_pose(s, "bowl1") == s.setup_state["confirmed_aim"]["bowl1"]


@pytest.mark.parametrize("case", ["no_vision", "rim_hidden", "no_picture"])
def test_confirm_says_plainly_when_the_camera_cannot_see_the_cup(rig, case, monkeypatch):
    s = rig
    b = anchor_cup(s)
    if case == "no_vision":
        s.plate.image_generate = None
    elif case == "rim_hidden":
        s.rim.sees = False
    else:
        monkeypatch.setattr(s, "table_frame", lambda: None)
    PS.aim(s, "bowl1")
    settle(s)
    r = PS.confirm_cup(s, "bowl1", 45, 70)
    assert b.cup.confirmed and r["confirmed"] and r["camera"] is False       # the position stands
    assert "can't see the opening" in r["message"] and "paused" in r["message"]
    assert not s.rim.has_reference("bowl1")
    st = PS.status(s)["cups"]["bowl1"]
    assert not st["ok"] and st["line"].startswith("Camera can't see the opening: scoops from this cup are paused.")
    assert K.tip_distance(s.cfg.kinematics, s.arm.state.deg, s.lib.pose("rest")) < 25    # at rest all the same


def test_two_dishes_are_told_apart_by_food_or_not_at_all(rig):
    s = rig
    anchor_cup(s)
    s.plate.bowls["bowl1"]["food"] = "rice"
    two = [{"food": "peas", "box_2d": [400, 50, 700, 250]}, {"food": "rice", "box_2d": [400, 600, 700, 900]}]
    s.plate.image_generate = lambda jpg, prompt: two
    s.plate._point = None                                                    # the spoon tip can't be pointed out
    PS.aim(s, "bowl1")
    settle(s)
    assert PS.confirm_cup(s, "bowl1", 45, 70)["camera"]
    call = next(c for c in s.rim.calls if c[0] == "set_reference")
    assert call[2][0] > 0.5 * call[5][1]                                     # the rice's box, on the right
    s.rim.calls.clear()
    s.plate.bowls["bowl1"]["food"] = "soup"                                  # neither dish is this food: no guess
    PS.aim(s, "bowl1")
    settle(s)
    r = PS.confirm_cup(s, "bowl1", 45, 70)
    assert r["camera"] is False and "which cup" in r["message"]
    assert not any(c[0] == "set_reference" for c in s.rim.calls)


def test_nudge_unconfirms_the_cup_and_drops_what_the_camera_learned(rig):
    s = rig
    b = anchor_cup(s)
    c = TestClient(create_app(s))
    assert c.post("/api/setup/aim", json={"bowl": "bowl1"}).status_code == 200
    settle(s)
    r = c.post("/api/setup/cup/bowl1/confirm", json={"rim_mm": 45, "width_mm": 70})
    assert r.status_code == 200 and r.json()["camera"] is True, r.text
    x0, y0 = b.cup.x_mm, b.cup.y_mm
    assert b.cup.confirmed and s.rim.has_reference("bowl1")
    assert c.post("/api/setup/aim", json={"bowl": "bowl1", "further_mm": 5}).status_code == 200
    cup = b.cup
    assert cup.confirmed is False and cup.source == "photo" and cup.uncertainty_mm >= 25     # unsure again
    assert not s.rim.has_reference("bowl1") and "bowl1" not in s.setup_state["confirmed_aim"]
    rr = math.hypot(x0, y0)
    assert (cup.x_mm - x0) * x0 / rr + (cup.y_mm - y0) * y0 / rr == pytest.approx(5, abs=1.5)   # it moved with the scoop
    assert cup.rim_mm == 45 and cup.radius_mm == 35                          # its measured size is kept
    assert PS.status(s)["cups"]["bowl1"]["line"].startswith("Not confirmed yet")
    settle(s)
    aimed = K.fk(s.cfg.kinematics, s.setup_state["aimed"]["bowl1"])
    assert math.hypot(aimed[0] - cup.x_mm, aimed[1] - cup.y_mm) < 2.0        # and Show aim is over the new centre


def test_endpoints_check_their_input_and_answer_only_the_laptop(rig, monkeypatch):
    s = rig
    anchor_cup(s)
    app = create_app(s)
    laptop, phone = TestClient(app), TestClient(app, client=("192.168.1.50", 50000))
    for body in ({"rim_mm": 0, "width_mm": 70}, {"rim_mm": 45, "width_mm": -5}, {"rim_mm": 45},
                 {"rim_mm": "tall", "width_mm": 70}, {"rim_mm": 45, "width_mm": 900}):
        assert laptop.post("/api/setup/cup/bowl1/confirm", json=body).status_code == 422, body
    r = laptop.post("/api/setup/cup/bowl1/confirm", json={"rim_mm": 45, "width_mm": 70})
    assert r.status_code == 409 and "isn't hovering" in r.json()["detail"]   # sizes fine, arm not over the cup
    assert s.cfg.plate.bowls[0].cup.confirmed is False

    spoon = {"bowl_length_mm": 48, "bowl_width_mm": 30, "bowl_depth_mm": 9, "neck_length_mm": 70, "neck_width_mm": 10,
             "axis_tilt_deg": 4, "measured": True}
    s.cfg.spoon.measured = False                                             # whatever the live config says
    before = s.cfg.spoon.model_dump()
    for k in ("bowl_length_mm", "bowl_width_mm", "bowl_depth_mm", "neck_length_mm", "neck_width_mm"):
        assert laptop.post("/api/setup/spoon", json={**spoon, k: 0}).status_code == 422, k
        assert laptop.post("/api/setup/spoon", json={**spoon, k: -3}).status_code == 422, k
    assert laptop.post("/api/setup/spoon", json={**spoon, "axis_tilt_deg": 80}).status_code == 422
    assert s.cfg.spoon.model_dump() == before                                # nothing was taken from a bad form
    assert laptop.post("/api/setup/spoon", json=spoon).status_code == 200
    assert s.cfg.spoon.measured and s.cfg.spoon.bowl_width_mm == 30 and s.cfg.spoon.axis_tilt_deg == 4
    assert snapshot(s)["setup"]["spoon"]["neck_length_mm"] == 70

    r = laptop.get("/api/rim/view.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is not None

    import spoonful.brain as brain
    monkeypatch.setitem(_sys.modules, "spoonful.brain.rim_dryrun", None)     # the dry-run module is not there
    monkeypatch.delattr(brain, "rim_dryrun", raising=False)
    r = laptop.get("/api/rim/preview/bowl1.png")
    assert r.status_code == 503 and "preview" in r.json()["detail"]
    fake = types.ModuleType("spoonful.brain.rim_dryrun")
    fake.render = lambda cfg, bowl: np.full((40, 60, 3), 200 if bowl == "bowl1" else 0, np.uint8)
    monkeypatch.setitem(_sys.modules, "spoonful.brain.rim_dryrun", fake)
    monkeypatch.setattr(brain, "rim_dryrun", fake, raising=False)
    r = laptop.get("/api/rim/preview/bowl1.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).shape == (40, 60, 3)
    assert laptop.get("/api/rim/preview/bowl9.png").status_code == 404

    for path in ("/api/setup/cup/bowl1/confirm", "/api/setup/spoon"):        # not from the phone, token or not
        assert phone.post(path, json={"rim_mm": 45, "width_mm": 70}).status_code == 403, path
    for path in ("/api/rim/view.jpg", "/api/rim/preview/bowl1.png"):
        assert phone.get(path).status_code == 403, path
    assert not app.state.phone.allows("/api/setup/cup/bowl1/confirm") and not app.state.phone.allows("/api/setup/spoon")


def test_system_hands_one_rim_observer_to_the_fsm(sim_system):
    from spoonful.perception.rim import RimObserver
    s = sim_system
    assert isinstance(s.rim, RimObserver) and s.fsm.rim is s.rim
    assert s.rim.frame_age_fn() < 1e8 or s.table_frame() is None           # a real age for the picture it is given
    s.phone_camera.put(np.full((90, 160, 3), 120, np.uint8))                # the phone takes over as table camera
    s.table_frame()
    assert s.rim.frame_age_fn() < 1.0
    s.rim.refs["bowl1"] = object()
    s.phone_camera.at -= 60                                                 # the phone is put away: a different view
    s.table_frame()
    assert not s.rim.refs                                                   # what was learned in the old view is gone


def test_two_confirms_with_the_real_observer_on_a_drawn_table(sim_system, tmp_path, monkeypatch):
    """The whole setup path with perception.rim itself, on a drawn picture of two cups standing where the sim's
    two bowls are (the drawing and its limits are test_rim's). The second confirm gives the camera its bearings."""
    import test_rim as TR
    s = sim_system
    d = tmp_path / "moves"
    shutil.copytree(s.lib.dir, d)                                            # the nudge at the end rewrites the scoop
    s.lib.dir = d
    s.lib.reload()
    cls = type(s.cfg)
    monkeypatch.setattr(cls, "save", lambda self, *a, **k: None)
    monkeypatch.setattr(PS, "CAMERA_SETTLE_S", 0.0)
    wait_state(s, "IDLE")
    kin = s.cfg.kinematics
    at = {n: K.fk(kin, PS._dip(s, n))[:2] for n in ("bowl1", "bowl2")}
    mid = np.mean(list(at.values()), axis=0)
    cam = TR.Cam((mid[0] - 60, mid[1] - 450, 330), (mid[0], mid[1], 40))
    img = TR.scene(cam, list(at.values()))
    s.phone_camera.put(img)
    s.phone_camera.at = time.monotonic() + 1e6                               # the phone keeps sending all test long
    foods = {"bowl1": "rice", "bowl2": "peas"}

    def vision(jpg, prompt):
        out = []
        for n, (x, y) in at.items():
            bx, by, bw, bh = TR.box(cam, x, y)
            out.append({"food": foods[n], "box_2d": [by / TR.H * 1000, bx / TR.W * 1000, (by + bh) / TR.H * 1000,
                                                     (bx + bw) / TR.W * 1000]})
        return out
    s.plate.image_generate = vision
    c = TestClient(create_app(s))
    for n in ("bowl1", "bowl2"):
        s.plate.bowls[n]["food"] = foods[n]
        anchor_cup(s, n)
        assert c.post("/api/setup/aim", json={"bowl": n}).status_code == 200
        settle(s)
        r = c.post(f"/api/setup/cup/{n}/confirm", json={"rim_mm": TR.RIM, "width_mm": 2 * TR.RAD})
        assert r.status_code == 200 and r.json()["camera"] is True, r.text
        ref = s.rim.refs[n].sighting                                         # the outline is on that cup's rim
        assert np.linalg.norm(np.array([ref.cx, ref.cy]) - TR.rim_px(cam, *at[n])) < 6
    assert s.rim.calib is not None and s.rim.calib.source == "pair"          # two confirmed cups: the view is known
    cups = snapshot(s)["setup"]["cups"]
    assert all(cups[n]["ok"] and cups[n]["line"] == "Position confirmed. Camera sees the opening." for n in at)
    assert c.get("/api/rim/view.jpg").status_code == 200
    assert all(s.rim.last[n].status == "ok" for n in at)
    assert c.post("/api/setup/aim", json={"bowl": "bowl1", "left_mm": 5}).status_code == 200
    assert not s.rim.has_reference("bowl1") and s.rim.has_reference("bowl2")     # only the nudged cup is forgotten
