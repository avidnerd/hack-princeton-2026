import pytest

from spoonful.arm.teach import downsample, median_filter
from spoonful.arm.trajectory import (LimitViolation, Library, TrajFile, joint_dist, make_waypoints, min_jerk, plan,
                                     plan_goto)
from spoonful.config import ARM_JOINTS


def pose(**kw):
    p = {j: 0.0 for j in ARM_JOINTS}
    p.update(kw)
    return p


def test_min_jerk_endpoints_and_symmetry():
    assert min_jerk(0) == 0 and min_jerk(1) == 1
    assert abs(min_jerk(0.5) - 0.5) < 1e-9
    assert min_jerk(-1) == 0 and min_jerk(2) == 1


def test_plan_respects_speed_cap(cfg):
    a, b = pose(), pose(base=60)
    p = plan_goto(cfg, a, b, "normal")
    dt = 0.002
    t, vmax = 0.0, 0.0
    prev = p.sample(0)
    while t < p.duration:
        t += dt
        cur = p.sample(t)
        vmax = max(vmax, joint_dist(prev, cur) / dt)
        prev = cur
    assert vmax <= cfg.arm.max_speed_deg_s.default * 1.01
    assert p.end["base"] == pytest.approx(60)


def test_near_face_is_slower(cfg):
    a, b = pose(), pose(base=30)
    assert plan_goto(cfg, a, b, "near_face").duration > plan_goto(cfg, a, b, "normal").duration


def test_speed_factor_slows(cfg):
    a, b = pose(), pose(elbow=30)
    assert plan_goto(cfg, a, b, speed_factor=0.5).duration > plan_goto(cfg, a, b).duration


def test_blend_inserted_and_ends_exact(cfg):
    path = [pose(base=10), pose(base=20), pose(base=30)]
    p = plan(cfg, pose(), [(path, "normal", "x")])
    assert p.segments[0].label.startswith("blend")
    assert p.segments[0].duration >= cfg.arm.blend_s
    assert p.sample(p.duration + 1)["base"] == pytest.approx(30)


def test_taught_waypoint_outside_limits_raises(cfg):
    lo, _ = cfg.arm.soft_limits_deg["shoulder"]
    with pytest.raises(LimitViolation):
        plan(cfg, pose(), [([pose(shoulder=lo - 5)], "normal", "bad")])


def test_traj_file_roundtrip_and_library_rejects_invalid(cfg, tmp_path):
    good = TrajFile("lift", "path", make_waypoints(cfg, [pose(), pose(elbow=10)]))
    good.save(tmp_path)
    bad = TrajFile("scoop", "path", make_waypoints(cfg, [pose(base=500)]))
    bad.save(tmp_path)
    lib = Library(cfg, tmp_path)
    assert lib.has("lift") and not lib.has("scoop")
    assert "scoop" in lib.errors
    assert TrajFile.load(tmp_path / "lift.json").poses[1]["elbow"] == pytest.approx(10)


def test_median_filter_removes_spike_and_downsample_keeps_ends():
    s = [pose(base=float(i)) for i in range(20)]
    s[10]["base"] = 90
    f = median_filter(s, 5)
    assert f[10]["base"] < 20
    poses, ts = downsample(f, [i * 0.03 for i in range(20)], 2.0)
    assert poses[0]["base"] == f[0]["base"] and poses[-1]["base"] == f[-1]["base"]
    assert len(poses) < len(f)
