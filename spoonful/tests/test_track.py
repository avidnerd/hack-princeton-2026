"""Follow the bowl: table calibration, shifting a taught scoop, refusing a bowl moved too far."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import wait_state
from test_fsm import start_listening
from spoonful.arm import kinematics as K
from spoonful.arm.scoop_style import scoop_anchor, shift_scoop
from spoonful.perception.table import TableCalib

SIM = Path(__file__).resolve().parents[1] / "data" / "trajectories_sim"


def _scoop(name="scoop_bowl1"):
    return [w["deg"] for w in json.loads((SIM / f"{name}.json").read_text())["waypoints"]]


def _px_of(mm):            # a made-up camera: 2 px per mm, rotated 30 deg, offset
    a = math.radians(30)
    x, y = mm
    return (640 + 2 * (x * math.cos(a) - y * math.sin(a)), 360 + 2 * (x * math.sin(a) + y * math.cos(a)))


def test_table_calibration_maps_pixels_to_mm(tmp_path):
    t = TableCalib(tmp_path / "t.json")
    for mm in [(100, 200), (250, 200), (250, 50), (100, 50), (180, 120), (140, 90), (210, 160), (130, 170)]:
        t.add(_px_of(mm), mm)
    assert t.ready and t.rms_mm < 1
    x, y = t.to_table(_px_of((170, 140)))
    assert abs(x - 170) < 1 and abs(y - 140) < 1
    assert TableCalib(tmp_path / "t.json").ready                       # saved and reloaded


def test_shift_scoop_moves_the_dip_on_the_table(fast_cfg):
    kin, poses = fast_cfg.kinematics, _scoop()
    ax, ay = scoop_anchor(kin, poses)
    moved = shift_scoop(kin, poses, 30, -20)
    assert moved is not None
    bx, by = scoop_anchor(kin, moved)
    assert abs(bx - ax - 30) < 1.5 and abs(by - ay + 20) < 1.5
    for a, b in zip(poses, moved):                                     # same heights
        assert abs(K.fk(kin, a)[2] - K.fk(kin, b)[2]) < 1
    assert shift_scoop(kin, poses, 0, 0) is not None
    assert shift_scoop(kin, poses, 900, 0) is None                    # out of reach


def _located(s, bowl, dx, dy):
    """Pretend the plate camera found `bowl` dx, dy mm from its taught spot."""
    anchor = s.plate.anchor_fn(bowl)
    s.plate.bowls[bowl].update(pos_mm=(anchor[0] + dx, anchor[1] + dy), pos_px=(0, 0), pos_ts=time.time())


def test_a_moved_bowl_is_followed(sim_system):
    s = sim_system
    s.plate.set_label("bowl1", "applesauce")
    plans = []
    orig = s.arm.run
    s.arm.run = lambda p: plans.append(p) or orig(p)
    _located(s, "bowl1", 25, 30)
    start_listening(s)
    s.on_voice(None, "Spoonful, applesauce please", 0.9)
    wait_state(s, "SCOOPING")
    pts = [seg.path[-1] for seg in plans[-1].segments]
    dip = min((p for seg in plans[-1].segments for p in seg.path), key=lambda q: K.fk(s.cfg.kinematics, q)[2])
    ax, ay = scoop_anchor(s.cfg.kinematics, s.lib.items["scoop_bowl1"].poses)
    x, y, _ = K.fk(s.cfg.kinematics, dip)
    assert abs(x - ax - 25) < 2 and abs(y - ay - 30) < 2
    s.log.flush()
    assert s.log.query("SELECT detail FROM events WHERE type='scoop_shift'")


def test_a_bowl_moved_too_far_is_refused(sim_system):
    s = sim_system
    s.plate.set_label("bowl1", "applesauce")
    _located(s, "bowl1", 90, 0)
    start_listening(s)
    s.on_voice(None, "Spoonful, applesauce please", 0.9)
    time.sleep(0.4)
    assert s.fsm.state.value == "LISTENING"
    s.log.flush()
    why = [r["detail"] for r in s.log.query("SELECT detail FROM events WHERE type='bowl_refused'")]
    assert why and "moved 90 mm" in why[-1]


def test_stale_or_missing_position_uses_the_taught_spot(sim_system):
    s = sim_system
    assert s.plate.shift_for("bowl1") is None
    _located(s, "bowl1", 20, 0)
    s.plate.bowls["bowl1"]["pos_ts"] = time.time() - 600
    assert s.plate.shift_for("bowl1") is None


def test_locate_assigns_detections_to_bowls(fast_cfg, tmp_path):
    from spoonful.perception.plate import PlateWatcher
    fast_cfg.plate.enabled = True
    fast_cfg.plate.track.calib_file = str(tmp_path / "t.json")
    anchors = {"bowl1": (150.0, 200.0), "bowl2": (60.0, 240.0)}
    def box_at(mm, food):       # detection box centred on the pixel of table point mm
        u, v = _px_of(mm)
        cx, cy = u / 1.28, v / 0.72
        return {"food": food, "box_2d": [cy - 20, cx - 20, cy + 20, cx + 20]}
    dets = [box_at((170, 210), "applesauce"), box_at((58, 250), "yogurt")]
    w = PlateWatcher(fast_cfg, lambda: np.zeros((720, 1280, 3), np.uint8), vision_fn=lambda c: [],
                     locate_fn=lambda jpg: dets)
    for mm in [(0, 100), (300, 100), (300, 350), (0, 350), (150, 220), (80, 300), (220, 160), (60, 180)]:
        w.table.add(_px_of(mm), mm)
    w.anchor_fn = anchors.get
    w.set_label("bowl1", "applesauce"); w.set_label("bowl2", "yogurt")
    w.locate(np.zeros((720, 1280, 3), np.uint8))
    s1, s2 = w.shift_for("bowl1"), w.shift_for("bowl2")
    assert abs(s1[0] - 20) < 2 and abs(s1[1] - 10) < 2
    assert abs(s2[0] + 2) < 2 and abs(s2[1] - 10) < 2


def test_arm_calibrates_the_camera_by_itself(sim_system):
    """The arm visits spots; a fake camera 'sees' the spoon tip where kinematics put it; the fit must recover it."""
    s = sim_system
    kin = s.cfg.kinematics
    s.plate.frame_fn = lambda: np.zeros((720, 1280, 3), np.uint8)

    def point(jpg):
        x, y, _ = K.fk(kin, s.arm.state.deg)
        u, v = _px_of((x, y))
        return {"point": [v / 0.72, u / 1.28]}               # [y, x] normalised 0-1000
    s.plate._point = point
    wait_state(s, "IDLE")
    with pytest.raises(RuntimeError, match="measure the spoon"):
        s.table_auto.start()                                 # not before the spoon is measured
    s.cfg.kinematics.spoon_measured = True
    s.table_auto.start()
    t0 = time.time()
    while s.table_auto.running and time.time() - t0 < 120:
        time.sleep(0.1)
    assert not s.table_auto.running
    t = s.plate.table
    assert t.ready and t.rms_mm < 3, s.table_auto.msg
    assert s.table_auto.done_points >= 8
    x, y = t.to_table(_px_of((150, 200)))
    assert abs(x - 150) < 5 and abs(y - 200) < 5


def test_autocal_refuses_outside_idle(sim_system):
    s = sim_system
    start_listening(s)
    with pytest.raises(RuntimeError):
        s.table_auto.start()


def test_spots_fan_left_centre_and_right(sim_system):
    s = sim_system
    spots = s.table_auto._plan_spots()
    ys = [K.fk(s.cfg.kinematics, low)[1] for _, low in spots]
    assert len(spots) >= 8
    assert min(ys) < -50 and max(ys) > 50                     # both sides of the arm
    for high, low in spots:
        assert abs(K.fk(s.cfg.kinematics, low)[2] - 50) < 1      # 5 cm over the table, never lower
        assert not K.in_face_zone(s.cfg.kinematics, low) and not K.in_face_zone(s.cfg.kinematics, high)


def test_spoon_length_is_fitted_from_table_touches(fast_cfg):
    """Make touches with a 'true' spoon, fit with a wrong one in the config: the fit must recover the truth."""
    from spoonful.arm.scoop_style import solve_rz
    kin = fast_cfg.kinematics
    true_kin = kin.model_copy(update={"tool_mm": 245.0, "tool_offset_mm": -12.0})
    touches = []
    for sh, wrist in [(a, w) for a in (-40.0, -10.0, 20.0, 50.0, 75.0) for w in (-60.0, -20.0, 20.0, 60.0, 100.0)]:
        lo, hi = -115.0, 115.0                                # bisect the elbow until the true tip touches z = 0
        q = {"base": kin.zero_deg["base"], "shoulder": sh, "elbow": 0.0, "wrist_pitch": wrist, "wrist_roll": 0.0}
        f = lambda e: K.fk(true_kin, dict(q, elbow=e))[2]
        if f(lo) * f(hi) > 0:
            continue
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        touches.append(dict(q, elbow=(lo + hi) / 2))
    assert len(touches) >= 3
    L, o, rms, spread = K.fit_spoon(kin, touches)
    assert abs(L - 245) < 1 and abs(o + 12) < 1 and rms < 0.5 and spread > 40


def test_bad_vision_points_are_thrown_out(tmp_path):
    t = TableCalib(tmp_path / "t.json")
    good = [(100, 200), (250, 200), (250, 50), (100, 50), (180, 120), (140, 160), (210, 90), (120, 130)]
    for mm in good:
        t.add(_px_of(mm), mm)
    t.add(_px_of((100, 200)), (260, -60))                     # vision pointed at the wrong thing twice
    t.add(_px_of((250, 50)), (-40, 300))
    assert t.ready and t.inliers == len(good) and t.rms_mm < 1


def test_autocal_stays_where_the_camera_can_see(sim_system):
    """A camera that only sees part of the table: calibration must find it, stay in it, and not wander off."""
    s = sim_system
    kin = s.cfg.kinematics
    s.cfg.kinematics.spoon_measured = True
    s.plate.frame_fn = lambda: np.zeros((720, 1280, 3), np.uint8)
    seed = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    centre = (seed[0] + 60, seed[1] - 40)

    def px(mm):                       # 2.2 px/mm around the visible centre; the picture is 1280x720
        return 640 + 2.2 * (mm[0] - centre[0]), 360 - 2.2 * (mm[1] - centre[1])
    visits = []

    def point(jpg):
        x, y, _ = K.fk(kin, s.arm.state.deg)
        u, v = px((x, y))
        visits.append((u, v))
        if not (0 <= u <= 1280 and 0 <= v <= 720):
            return {"point": None}                               # spoon out of the picture
        return {"point": [v / 0.72, u / 1.28]}
    s.plate._point = point
    wait_state(s, "IDLE")
    s.table_auto.start()
    t0 = time.time()
    while s.table_auto.running and time.time() - t0 < 150:
        time.sleep(0.1)
    assert s.plate.table.ready, s.table_auto.msg
    inside = [1 for u, v in visits if 0 <= u <= 1280 and 0 <= v <= 720]
    assert len(inside) >= 8
    assert len(inside) >= 0.7 * len(visits), f"{len(visits) - len(inside)} of {len(visits)} spots were out of the picture"
