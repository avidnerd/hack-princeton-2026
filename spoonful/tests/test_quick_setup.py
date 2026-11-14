"""Quick setup: every move built from three touches (bowl dips + mouth), for any table and any seat."""
import pytest

from conftest import wait_state
from test_fsm import start_listening
from spoonful.arm import kinematics as K
from spoonful.arm.scoop_style import _spoon_angle
from spoonful.brain.quick_setup import SetupError, build_all, save_all


def _touches(s):
    kin = s.cfg.kinematics
    dip1 = min(s.lib.items["scoop_bowl1"].poses, key=lambda q: K.fk(kin, q)[2])
    dip2 = min(s.lib.items["scoop_bowl2"].poses, key=lambda q: K.fk(kin, q)[2])
    return {"bowl1": dip1, "bowl2": dip2}, s.lib.pose("mouth_nominal"), s.lib.pose("rest")


def test_builds_every_move_from_touches(sim_system):
    s = sim_system
    dips, mouth, rest = _touches(s)
    b = build_all(s.cfg, dips, mouth, rest)
    kin = b["kin"]
    names = set(b["moves"])
    assert {"scoop_bowl1", "scoop_bowl2", "lift_bowl1", "lift_bowl2", "pre_delivery", "mouth_nominal",
            "scoop", "lift"} <= names
    assert K.in_face_zone(kin, mouth)
    pre = b["moves"]["pre_delivery"][1][0]
    assert not K.in_face_zone(kin, pre)
    assert 70 <= K.tip_distance(kin, pre, mouth) <= 110                     # 8-10 cm in front of the lips
    for bowl, dip in dips.items():
        scoop = b["moves"][f"scoop_{bowl}"][1]
        assert any(K.tip_distance(kin, q, dip) < 1.5 for q in scoop)         # it dips where the spoon was touched
        assert min(K.fk(kin, q)[2] for q in scoop) >= K.fk(kin, dip)[2] - 1  # and never lower than that
        assert K.fk(kin, scoop[0])[2] > K.fk(kin, dip)[2] + 40             # comes in from above
        assert abs(_spoon_angle(kin, scoop[-1])) < abs(_spoon_angle(kin, dip)) or abs(_spoon_angle(kin, dip)) <= 100
        for q in scoop + b["moves"][f"lift_{bowl}"][1][:-1]:
            assert not K.in_face_zone(kin, q)


def test_face_zone_follows_the_diner(sim_system):
    """Sit somewhere else: the zone is built around the new mouth, not a fixed box."""
    s = sim_system
    dips, mouth, rest = _touches(s)
    moved = dict(mouth)
    moved["base"] += 25                                                      # diner sits further round
    b1 = build_all(s.cfg, dips, mouth, rest)
    b2 = build_all(s.cfg, dips, moved, rest)
    assert K.in_face_zone(b2["kin"], moved) and not K.in_face_zone(b2["kin"], mouth)
    assert (b1["face_cyl"].cx, b1["face_cyl"].cy) != (b2["face_cyl"].cx, b2["face_cyl"].cy)


def test_needs_a_bowl(sim_system):
    s = sim_system
    _, mouth, rest = _touches(s)
    with pytest.raises(SetupError):
        build_all(s.cfg, {}, mouth, rest)


def test_built_moves_give_a_full_bite_from_bowl2(sim_system, tmp_path):
    s = sim_system
    dips, mouth, rest = _touches(s)
    b = build_all(s.cfg, dips, mouth, rest)
    import shutil
    from spoonful.arm.trajectory import Library
    d = tmp_path / "moves"                                   # never write into the repo's trajectories_sim/
    shutil.copytree(s.lib.dir, d)
    save_all(s.cfg, d, b)
    s.cfg.kinematics.face_cyl = b["face_cyl"]
    s.lib.dir = d
    s.lib.reload()
    s.plate.set_label("bowl1", "applesauce")
    s.plate.set_label("bowl2", "rice")
    lifts = []
    orig = s.fsm._run_paths
    s.fsm._run_paths = lambda paths, speed_factor=1.0, name="": lifts.append([p for p, _ in paths]) or \
        orig(paths, speed_factor=speed_factor, name=name)
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=True)
    s.on_voice(None, "Spoonful, rice please", 0.9)
    wait_state(s, "DELIVERING", timeout=30)
    assert ["lift_bowl2", "pre_delivery"] in lifts                          # the lift for the bowl it scooped
