"""Camera aim check: the webcam measures how far the spoon is from the bowl and the scoop is corrected."""
import math
import shutil
import time

import numpy as np
import pytest

from conftest import wait_state
from spoonful.arm import kinematics as K
from spoonful.arm.scoop_style import scoop_anchor
from spoonful.brain.aim_check import AimProblem, solve_offset

# a side-on camera: pixels from table mm (x, y, z). Depth (x) is foreshortened, height shows up strongly.
CAM = np.array([[0.35, -1.9, 0.0], [0.55, 0.1, -1.7]])
ORIGIN = np.array([500.0, 600.0])


def project(p):
    return ORIGIN + CAM @ np.asarray(p, float)


def test_solve_offset_recovers_a_known_error():
    tip, bowl = np.array([210.0, 40.0, 95.0]), np.array([222.0, 31.0, 0.0])     # bowl is 12 further, 9 to the right
    yaw = math.atan2(tip[1], tip[0])
    out, left = np.array([math.cos(yaw), math.sin(yaw), 0]), np.array([-math.sin(yaw), math.cos(yaw), 0])
    f, l = solve_offset(project(tip), project(bowl), tip[2], CAM @ out, CAM @ left, CAM @ np.array([0, 0, 1.0]))
    want = bowl - np.array([tip[0], tip[1], 0])
    assert f == pytest.approx(want @ out, abs=0.01) and l == pytest.approx(want @ left, abs=0.01)
    with pytest.raises(AimProblem):                                           # camera looking straight down the arm
        solve_offset([0, 0], [5, 5], 50, [1.0, 0.0], [1.0, 0.001], [0, -1])


@pytest.fixture
def aim_system(sim_system, tmp_path, monkeypatch):
    s = sim_system
    d = tmp_path / "moves"
    shutil.copytree(s.lib.dir, d)
    s.lib.dir = d
    s.lib.reload()
    monkeypatch.setattr(type(s.cfg), "save", lambda self, *a, **k: None)
    wait_state(s, "IDLE")
    s.plate.frame_fn = lambda: np.zeros((720, 1280, 3), np.uint8)
    return s


def _run(s, bowl="bowl1"):
    s.aim_check.start(bowl)
    t0 = time.time()
    while s.aim_check.running and time.time() - t0 < 90:
        time.sleep(0.05)
    assert not s.aim_check.running
    return s.aim_check.results[bowl]


def test_camera_finds_and_corrects_a_miss(aim_system):
    s = aim_system
    kin = s.cfg.kinematics
    x0, y0 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    r = math.hypot(x0, y0)
    truth = np.array([x0 + 14 * x0 / r + 10 * y0 / r, y0 + 14 * y0 / r - 10 * x0 / r, 0.0])   # 14 further, 10 right
    s.plate._point = lambda jpg: (lambda p: {"point": [p[1] / 0.72, p[0] / 1.28]})(project(K.fk(kin, s.arm.state.deg)))
    s.plate.image_generate = lambda jpg, prompt: (lambda p: {"base": [p[1] / 0.72, p[0] / 1.28], "opening": None})(project(truth))
    res = _run(s)
    assert res["state"] == "ok", res
    x1, y1 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    assert math.hypot(x1 - truth[0], y1 - truth[1]) < 6                      # the scoop now dips over the real bowl
    assert math.hypot(x0 - truth[0], y0 - truth[1]) > 15                     # ...which it would have missed


def test_camera_that_cannot_see_changes_nothing(aim_system):
    s = aim_system
    kin = s.cfg.kinematics
    before = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    s.plate._point = lambda jpg: {"point": None}
    s.plate.image_generate = lambda jpg, prompt: {"base": None}
    res = _run(s)
    assert res["state"] == "problem" and "spoon" in res["message"]
    assert scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses) == before


def test_wild_camera_answer_is_not_applied(aim_system):
    s = aim_system
    kin = s.cfg.kinematics
    before = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    s.plate._point = lambda jpg: (lambda p: {"point": [p[1] / 0.72, p[0] / 1.28]})(project(K.fk(kin, s.arm.state.deg)))
    far = np.array([before[0] + 90, before[1] + 60, 0.0])                    # 11 cm away: the photo and camera disagree
    s.plate.image_generate = lambda jpg, prompt: (lambda p: {"base": [p[1] / 0.72, p[0] / 1.28]})(project(far))
    res = _run(s)
    assert res["state"] == "problem" and "Retake" in res["message"]
    assert scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses) == before
