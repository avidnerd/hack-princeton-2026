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


def dish_at(p, food="applesauce"):
    """What the dish detector returns for a dish whose base centre is at table point p: a box (0-1000 coords)."""
    u, v = project(p)
    y1 = v + 8                                        # base centre = 8% of the box height above the box bottom
    return {"food": food, "box_2d": [(y1 - 100) / 0.72, (u - 45) / 1.28, y1 / 0.72, (u + 45) / 1.28]}


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
    s.aim_check.start([bowl])
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
    s.plate.image_generate = lambda jpg, prompt: [dish_at(truth)]
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
    s.plate.image_generate = lambda jpg, prompt: []
    res = _run(s)
    assert res["state"] == "problem" and "spoon" in res["message"]
    assert scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses) == before


def test_wild_camera_answer_is_not_applied(aim_system):
    s = aim_system
    kin = s.cfg.kinematics
    before = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    s.plate._point = lambda jpg: (lambda p: {"point": [p[1] / 0.72, p[0] / 1.28]})(project(K.fk(kin, s.arm.state.deg)))
    far = np.array([before[0] + 90, before[1] + 60, 0.0])                    # 11 cm away: the photo and camera disagree
    s.plate.image_generate = lambda jpg, prompt: [dish_at(far)]
    res = _run(s)
    assert res["state"] == "problem" and "Retake" in res["message"]
    assert scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses) == before


# ---------------------------------------------------------------- before every scoop
def _camera(s, truth):
    kin = s.cfg.kinematics
    s.plate._point = lambda jpg: (lambda p: {"point": [p[1] / 0.72, p[0] / 1.28]})(project(K.fk(kin, s.arm.state.deg)))
    s.plate.image_generate = lambda jpg, prompt: [dish_at(truth())]


def _know_camera(s, bowl="bowl1"):
    """What calibrate() would have measured for this camera, without the test moves."""
    x, y = scoop_anchor(s.cfg.kinematics, s.lib.items[f"scoop_{bowl}"].poses)
    yaw = math.atan2(y, x)
    out, left = np.array([math.cos(yaw), math.sin(yaw), 0]), np.array([-math.sin(yaw), math.cos(yaw), 0])
    s.aim_check.jacobians[bowl] = np.array([CAM @ out, CAM @ left, CAM @ np.array([0, 0, 1.0])])


def _looks(s):
    s.log.flush()
    return [r["detail"] for r in s.log.query("SELECT detail FROM events WHERE type='look' ORDER BY rowid")]


def test_a_bowl_moved_after_setup_is_followed_before_the_scoop(aim_system):
    from test_fsm import start_listening
    s = aim_system
    kin = s.cfg.kinematics
    s.plate.set_label("bowl1", "applesauce")
    x0, y0 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    r = math.hypot(x0, y0)
    moved_to = np.array([x0 + 30 * x0 / r + 20 * y0 / r, y0 + 30 * y0 / r - 20 * x0 / r, 0.0])    # bumped 3.6 cm
    _know_camera(s)
    _camera(s, lambda: moved_to)
    start_listening(s)
    s.fsm.post("bite", source="ui", bowl="bowl1")
    wait_state(s, "LIFTING", timeout=40)
    x1, y1 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    assert math.hypot(x1 - moved_to[0], y1 - moved_to[1]) < 8            # it scooped where the bowl is now
    looks = _looks(s)
    assert len(looks) == 2 and '"followed": true' in looks[0]            # looked, followed, looked again to confirm


def test_no_view_of_the_bowl_scoops_where_it_was(aim_system):
    from test_fsm import start_listening
    s = aim_system
    kin = s.cfg.kinematics
    s.plate.set_label("bowl1", "applesauce")
    before = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    _know_camera(s)
    s.plate._point = lambda jpg: {"point": None}
    s.plate.image_generate = lambda jpg, prompt: []
    start_listening(s)
    s.fsm.post("bite", source="ui", bowl="bowl1")
    wait_state(s, "LIFTING", timeout=40)
    assert scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses) == before
    assert '"seen": false' in _looks(s)[0]


def test_a_bowl_that_is_gone_stops_the_bite(aim_system):
    from test_fsm import start_listening
    s = aim_system
    kin = s.cfg.kinematics
    s.plate.set_label("bowl1", "applesauce")
    x0, y0 = scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses)
    _know_camera(s)
    _camera(s, lambda: np.array([x0 + 120, y0 + 90, 0.0]))                # 15 cm away
    start_listening(s)
    s.fsm.post("bite", source="ui", bowl="bowl1")
    wait_state(s, "LISTENING", timeout=40)                                # it backs off instead of scooping air
    assert s.fsm.session.taken == 0
    assert scoop_anchor(kin, s.lib.items["scoop_bowl1"].poses) == (x0, y0)



def test_the_right_dish_is_picked():
    from spoonful.brain.aim_check import AimCheck
    dishes = [{"food": "green jelly cup", "x": 280.0, "base": np.array([280.0, 500.0])},
              {"food": "applesauce cup", "x": 510.0, "base": np.array([510.0, 540.0])},
              {"food": "rice bowl", "x": 570.0, "base": np.array([570.0, 680.0])}]
    tip = np.array([300.0, 400.0])                                      # hovering over the jelly
    assert AimCheck.pick_dish(dishes, "applesauce", tip)["food"] == "applesauce cup"      # by name, wherever the spoon is
    assert AimCheck.pick_dish(dishes, "honeydew jelly", tip)["food"] == "green jelly cup"
    assert AimCheck.pick_dish(dishes, "yogurt", tip)["food"] == "green jelly cup"         # no such name: under the spoon
    assert AimCheck.pick_dish([], "applesauce", tip) is None
