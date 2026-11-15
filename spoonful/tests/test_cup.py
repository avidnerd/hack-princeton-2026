"""arm/cup.py: the spoon and the cup as shapes, and the entry planner. Geometry only: no arm, no files.

Every planner test runs with the live config's kinematics (the measured spoon) and with the sim kinematics that
conftest pins (the original spoon model), because what the arm can reach differs a lot between the two."""
import math
import time
from types import SimpleNamespace

import pytest

from spoonful import config as C
from spoonful.arm import cup as U
from spoonful.arm import kinematics as K
from spoonful.arm.scoop_style import _spoon_angle
from spoonful.arm.trajectory import lerp_pose

SPOON = U.Spoon(bowl_length=45, bowl_width=32, bowl_depth=8, neck_length=60, neck_width=12)
CLEAR, HOVER = 3.0, 25.0


@pytest.fixture(params=["live", "sim"])
def rig(request):
    c = request.getfixturevalue("fast_cfg") if request.param == "sim" else C.load()
    return SimpleNamespace(name=request.param, kin=c.kinematics, limits=c.arm.soft_limits_deg,
                           seed={j: float(v) for j, v in c.arm.home_pose.items()})


def cup_at(x=200.0, y=60.0, radius=35.0, rim=45.0, uncertainty=3.0, **kw):
    return U.Cup(x, y, radius, rim, uncertainty=uncertainty, **kw)


def plan(rig, cup, angle=45.0, drag=20.0, spoon=SPOON, **kw):
    return U.plan_entry(rig.kin, rig.limits, rig.seed, cup, spoon, clearance_mm=CLEAR, hover_above_rim_mm=HOVER,
                        want_angle_deg=angle, want_drag_mm=drag, **kw)


def pose_at(rig, x, y, z, below_deg, in_limits=False):
    """A pose with the tip at (x, y, z) and the spoon below_deg under horizontal (either elbow shape)."""
    base = rig.kin.zero_deg["base"] + math.degrees(math.atan2(y, x)) / rig.kin.sign["base"]
    sols = [q for q in U._ik(rig.kin, {**rig.seed, "base": base}, math.hypot(x, y), z, 90.0 + below_deg)
            if q is not None and (not in_limits or U._within(rig.limits, q))]
    if not sols:
        return None
    q = sols[0]
    assert math.dist(K.fk(rig.kin, q), (x, y, z)) < 0.01 and abs(_spoon_angle(rig.kin, q) - 90 - below_deg) < 0.01
    return q


def reach(rig, specs):
    """(r, poses): a reach where the arm (joint limits aside) can put the tip at height z with the spoon below_deg
    under horizontal, for every (z, below_deg) in specs. The geometry tests put their cup relative to that."""
    for r in range(60, 330, 10):
        qs = [pose_at(rig, float(r), 0.0, z, below) for z, below in specs]
        if all(q is not None for q in qs):
            return float(r), qs
    raise AssertionError(f"no reach for {specs}")


def walk(rig, poses, cup, spoon=SPOON, step_deg=0.5):
    """Smallest clearance along the joint-linear move through poses, one pose at a time with the public scalar
    functions (so it does not share check_path's sampling)."""
    worst = math.inf
    for a, b in zip(poses, poses[1:]):
        n = max(1, int(math.ceil(max(abs(a[j] - b[j]) for j in U.FKJ) / step_deg)))
        for k in range(n + 1):
            worst = min(worst, U.clearance(rig.kin, lerp_pose(a, b, k / n), cup, spoon).mm)
    return worst


def executed(p, step_mm=8.0):
    """The path as it is run: the descent in vertical steps instead of one move."""
    return [p.hover, *p.descent(step_mm), *p.stroke, p.out, p.clear]


# ---------------------------------------------------------------- shapes
def test_cup_and_spoon_from_config():
    c = C.CupCfg(x_mm=200, y_mm=60, radius_mm=35, rim_mm=45, uncertainty_mm=3, confirmed=True, source="confirmed")
    cup = U.Cup.from_cfg(c, 60.0)
    assert (cup.x, cup.y, cup.radius, cup.rim, cup.wall, cup.floor, cup.uncertainty) == (200, 60, 35, 45, 2, 3, 3)
    assert cup.source == "confirmed" and cup.rim_measured
    far = cup.moved(10, -5, uncertainty=9.0, source="followed")
    assert (far.x, far.y, far.uncertainty, far.source, far.rim) == (210, 55, 9.0, "followed", 45)
    assert cup.safe_radius(0, 3) == 35 - 3 - 3 and cup.safe_radius(16, 3) == 13 and cup.safe_radius(40, 3) == 0
    s = U.Spoon.from_cfg(C.SpoonGeomCfg())
    assert (s.bowl_length, s.bowl_width, s.bowl_depth, s.neck_length, s.neck_width, s.axis_tilt_deg) == \
        (45, 32, 8, 60, 12, 0)


def test_fits_counts_clearance_and_uncertainty_on_both_sides():
    assert U.fits(cup_at(radius=35, uncertainty=3), SPOON, 3) == (True, "")
    assert U.fits(cup_at(radius=21, uncertainty=0), SPOON, 3)[0]            # 42 >= 32 + 2 * 3
    ok, why = U.fits(cup_at(radius=21, uncertainty=3), SPOON, 3)            # 42 < 32 + 2 * 6
    assert not ok and "42" in why and "44" in why
    assert U.fits(cup_at(radius=35, uncertainty=17), SPOON, 3)[0] is False  # a wide cup, but where is it?


def test_fk_matches_kinematics(rig):
    import numpy as np
    poses = [rig.seed, {**rig.seed, "base": -40.0, "shoulder": 12.0, "elbow": -66.0, "wrist_pitch": 79.0},
             {**rig.seed, "base": 20.0, "shoulder": -30.0, "elbow": 40.0, "wrist_pitch": -20.0}]
    tips, _, p3, _ = U._fk(rig.kin, U._q(poses))
    for q, tip, p in zip(poses, tips, p3):
        assert np.allclose(tip, K.fk(rig.kin, q), atol=1e-9)
        assert math.degrees(p) == pytest.approx(_spoon_angle(rig.kin, q))
        assert U.spoon_body(rig.kin, q, SPOON)[0][:3] == pytest.approx(K.fk(rig.kin, q))


def test_spoon_body_runs_back_from_the_tip(rig):
    r, (q,) = reach(rig, [(120, 0)])                       # level, pointing along +x
    body = U.spoon_body(rig.kin, q, SPOON)
    assert body[0].part == "tip" and body[0].half_width == 0
    assert {b.part for b in body} == {"tip", "bowl", "neck"}
    assert max(b.half_width for b in body) == pytest.approx(16, abs=0.3)             # widest mid-bowl
    widest = max(body, key=lambda b: b.half_width)
    assert r - widest.x == pytest.approx(22.5, abs=1.5)
    assert min(b.x for b in body) == pytest.approx(r - 105) and max(b.x for b in body) == pytest.approx(r)
    assert all(abs(b.y) < 1e-6 for b in body)
    assert min(b.z for b in body) == pytest.approx(120 - 8, abs=0.1)                 # the underside hangs below
    assert all(b.z == pytest.approx(120) and b.half_width == 6 for b in body if b.part == "neck")
    assert U.spoon_center(rig.kin, q, SPOON) == pytest.approx((r - 22.5, 0, 120))
    # a spoon taped 10 deg nose-down on the tool: from the same tip its handle end is higher
    tilted = U.spoon_body(rig.kin, q, U.Spoon(45, 32, 8, 60, 12, axis_tilt_deg=10))
    assert max(b.z for b in tilted) == pytest.approx(120 + 105 * math.sin(math.radians(10)))
    # nose-down: everything behind the tip is higher than it
    _, (q,) = reach(rig, [(15, 60)])
    steep = U.spoon_body(rig.kin, q, SPOON)
    assert min(b.z for b in steep) == pytest.approx(15)
    assert max(b.z for b in steep) == pytest.approx(15 + 105 * math.sin(math.radians(60)))


def test_tip_centred_but_bowl_on_the_rim(rig):
    """The tip is exactly over the centre and inside the cup, and the spoon is still in the wall."""
    r, (q,) = reach(rig, [(20, 20)])
    c = U.clearance(rig.kin, q, cup_at(r, 0, uncertainty=0), SPOON)         # 35 mm back the line is at z = 33 < 45
    assert c.mm < 0 and c.part == "bowl"
    assert math.hypot(c.point[0] - r, c.point[1]) > 25 and c.point[0] < r and c.point[2] < 45
    r, (shallow, steep) = reach(rig, [(10, 30), (10, 55)])
    wide = cup_at(r, 0, radius=50, uncertainty=0)
    c = U.clearance(rig.kin, shallow, wide, SPOON)                          # 50 mm back: the neck, at z = 39 < 45
    assert c.mm < 0 and c.part == "neck"
    assert U.clearance(rig.kin, steep, wide, SPOON).mm > 3                  # same tip, steeper: neck over the rim
    r, (q,) = reach(rig, [(1, 60)])                                         # the tip pushed below the inside bottom
    c = U.clearance(rig.kin, q, cup_at(r, 0, uncertainty=0), SPOON)
    assert c.mm == pytest.approx(-2) and c.part == "tip"


def test_clearance_above_and_beside_the_cup(rig):
    r, (q,) = reach(rig, [(75, 0)])                                         # level, tip 30 mm above the rim
    over = U.clearance(rig.kin, q, cup_at(r, 0, uncertainty=0), SPOON)
    assert 30 - 8 <= over.mm < 24 and over.part == "bowl"                   # the underside, over the near rim
    r, (q,) = reach(rig, [(20, 60)])                                        # whole spoon outside, on the near side
    beside = U.clearance(rig.kin, q, cup_at(r + 90, 0, uncertainty=0), SPOON)
    assert beside.mm == pytest.approx(90 - 35 - 2, abs=0.01) and beside.part == "tip"


def test_legacy_scoop_is_flagged(rig):
    """The failure being fixed: the saved-style scoop (tip at the centre of a 70 x 45 mm cup, 30 deg below
    horizontal, tip 26 mm above the table) has the bowl or neck on the near rim even with perfect aim."""
    cup = cup_at(195, 82, radius=35, rim=45, uncertainty=0)
    q = pose_at(rig, 195, 82, 26, 30)
    c = U.clearance(rig.kin, q, cup, SPOON)
    assert c.mm < CLEAR and c.part in ("bowl", "neck")
    assert not U.check_path(rig.kin, [q, q], cup, SPOON, CLEAR).ok
    # the point that is too close is on the arm's side of the cup, near the rim
    yaw = math.atan2(82, 195)
    assert (c.point[0] - 195) * math.cos(yaw) + (c.point[1] - 82) * math.sin(yaw) < -25 and c.point[2] > 30


def test_check_path_reports_the_first_violation(rig):
    r, (above, inside) = reach(rig, [(80, 20), (20, 20)])
    cup = cup_at(r, 0, uncertainty=0)
    chk = U.check_path(rig.kin, [above, above, inside], cup, SPOON, CLEAR)
    assert not chk.ok and chk.min_mm < 0 and chk.index == 1 and 0 < chk.s < 1 and chk.part in ("bowl", "neck")
    assert chk.min_mm == pytest.approx(walk(rig, [above, inside], cup), abs=0.5)
    fine = U.check_path(rig.kin, [above, above], cup, SPOON, CLEAR)
    assert fine.ok and fine.min_mm == pytest.approx(U.clearance(rig.kin, above, cup, SPOON).mm)


# ---------------------------------------------------------------- the planner
def test_centred_cup_plan_keeps_the_clearance_everywhere(rig):
    cup = cup_at(200, 60, radius=35, rim=45, uncertainty=3)
    p = plan(rig, cup, start=rig.seed)
    margin = CLEAR + cup.uncertainty
    for q in [*p.approach, *p.poses(), *p.descent(8)]:
        assert all(rig.limits[j][0] <= q[j] <= rig.limits[j][1] for j in U.FKJ)
    # every sampled pose, as one move per waypoint and with the descent in steps
    assert walk(rig, p.poses(), cup) >= margin - 1e-6
    assert walk(rig, executed(p), cup) >= margin - 1e-6
    assert walk(rig, [rig.seed, *p.approach, p.hover], cup) >= margin - 1e-6
    assert U.check_path(rig.kin, p.poses(), cup, SPOON, margin).ok
    assert p.min_clearance_mm >= margin and p.min_clearance_mm == pytest.approx(walk(rig, p.poses(), cup), abs=0.6)
    tip = {n: K.fk(rig.kin, q) for n, q in zip(("hover", "oriented", "dip"), p.poses())}
    # hover: whole spoon well above the rim, tip over the entry point
    assert min(b.z for b in U.spoon_body(rig.kin, p.hover, SPOON)) >= cup.rim + HOVER - 0.5
    assert tip["hover"][:2] == pytest.approx(p.entry_xy, abs=0.6)
    # oriented: same x, y, entry angle, whole spoon still above the rim by the margin
    assert tip["oriented"][:2] == pytest.approx(p.entry_xy, abs=0.6)
    assert _spoon_angle(rig.kin, p.oriented) - 90 == pytest.approx(p.entry_angle_deg, abs=0.2)
    assert min(b.z for b in U.spoon_body(rig.kin, p.oriented, SPOON)) >= cup.rim + margin
    # dip: straight down, same angle, inside the cup, tip inside the safe interior and off the floor
    assert tip["dip"][:2] == pytest.approx(p.entry_xy, abs=0.6) and tip["dip"][2] == pytest.approx(p.dip_z, abs=0.5)
    assert _spoon_angle(rig.kin, p.dip) == pytest.approx(_spoon_angle(rig.kin, p.oriented), abs=0.2)
    assert cup.floor + 2 <= p.dip_z < cup.rim - SPOON.bowl_depth
    assert math.hypot(p.entry_xy[0] - cup.x, p.entry_xy[1] - cup.y) <= cup.safe_radius(0, CLEAR)
    # the drag is toward the arm at the same height; the way out is straight up from where it ends
    end = K.fk(rig.kin, (p.stroke or [p.dip])[-1])
    assert end[2] == pytest.approx(p.dip_z, abs=0.5)
    assert math.hypot(*tip["dip"][:2]) - math.hypot(*end[:2]) == pytest.approx(p.drag_mm, abs=0.6)
    assert K.fk(rig.kin, p.out)[:2] == pytest.approx(end[:2], abs=0.6)
    assert min(b.z for b in U.spoon_body(rig.kin, p.out, SPOON)) >= cup.rim + margin
    assert min(b.z for b in U.spoon_body(rig.kin, p.clear, SPOON)) >= cup.rim + HOVER - 0.5
    assert 25 <= p.entry_angle_deg <= 75 and 0 <= p.drag_mm <= 20


def test_descent_goes_straight_down_in_steps(rig):
    p = plan(rig, cup_at())
    steps = p.descent(8)
    assert steps[0] == p.oriented and steps[-1] == p.dip and len(steps) >= 4
    tips = [K.fk(rig.kin, q) for q in steps]
    for a, b in zip(tips, tips[1:]):
        assert 0 < a[2] - b[2] <= 8.5 and math.dist(a[:2], b[:2]) < 0.8
    assert all(abs(_spoon_angle(rig.kin, q) - 90 - p.entry_angle_deg) < 0.3 for q in steps)
    assert len(p.descent(4)) > len(steps)


def test_shallow_entry_is_rejected_and_a_steeper_one_chosen(rig):
    """A skim at 25 deg cannot go into a 70 x 45 mm cup: with the tip inside, the bowl or neck is on the near
    rim. The planner must say so and go in steeper, not flatten the path to fit the request."""
    cup = cup_at(200, 60)
    p = plan(rig, cup, angle=25.0)
    assert p.entry_angle_deg > 25
    shallow = [r for r in p.rejected if r["stage"] == "geometry" and r["angle_deg"] == 25]
    assert shallow and shallow[0]["why"] == "clearance" and shallow[0]["count"] > 0
    # the same entry point and depth at the angle that was asked for: too close to the cup
    import numpy as np
    tip = np.array([[p.entry_xy[0], p.entry_xy[1], p.dip_z]])
    yaw = np.array([math.atan2(cup.y, cup.x)])
    asked = float(U._clear(tip, np.radians([90.0 + 25.0]), yaw, cup, SPOON)[0].min())
    chosen = float(U._clear(tip, np.radians([90.0 + p.entry_angle_deg]), yaw, cup, SPOON)[0].min())
    assert asked < CLEAR + cup.uncertainty <= chosen
    assert U.check_path(rig.kin, p.poses(), cup, SPOON, CLEAR + cup.uncertainty).ok


def test_opening_narrower_than_the_spoon(rig):
    with pytest.raises(U.NoSafeEntry) as e:
        plan(rig, cup_at(radius=18, uncertainty=0))             # 36 mm opening, 32 mm spoon + 2 x 3 mm
    assert e.value.reason == "too_narrow" and e.value.rejected
    with pytest.raises(U.NoSafeEntry) as e:
        plan(rig, cup_at(radius=14, uncertainty=0))             # narrower than the spoon itself
    assert e.value.reason == "too_narrow"


def test_unreachable_cup(rig):
    for x, y in ((470.0, 60.0), (-150.0, -150.0)):              # beyond the arm's reach; behind the base's limit
        with pytest.raises(U.NoSafeEntry) as e:
            plan(rig, cup_at(x, y))
        assert e.value.reason == "unreachable"


def test_unmeasured_geometry(rig):
    cfg = C.CupCfg(x_mm=200, y_mm=60, radius_mm=45, uncertainty_mm=3)        # rim height never measured
    # with a default: planned against the assumed (high) rim, and the cup says the rim is a guess
    guess = U.Cup.from_cfg(cfg, 60.0)
    assert guess.rim == 60.0 and not guess.rim_measured
    p = plan(rig, guess)
    assert min(b.z for b in U.spoon_body(rig.kin, p.hover, SPOON)) >= 60 + HOVER - 0.5
    assert min(b.z for b in U.spoon_body(rig.kin, p.oriented, SPOON)) >= 60 + CLEAR + 3
    assert U.check_path(rig.kin, p.poses(), guess, SPOON, CLEAR + 3).ok
    # with no default there is nothing to plan against
    for cup in (U.Cup.from_cfg(cfg, None), cup_at(radius=float("nan")), cup_at(radius=0.0), cup_at(rim=0.0)):
        assert U.fits(cup, SPOON, CLEAR) == (False, "unmeasured")
        with pytest.raises(U.NoSafeEntry) as e:
            plan(rig, cup)
        assert e.value.reason == "unmeasured"
    with pytest.raises(U.NoSafeEntry) as e:
        plan(rig, cup_at(), spoon=U.Spoon(0, 0, 0, 0, 0))
    assert e.value.reason == "unmeasured"


def test_uncertainty_shrinks_then_removes_the_feasible_set(rig):
    kept, plans = [], {}
    for u in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 20.0):
        cup = cup_at(uncertainty=u)
        kept.append(len(U._candidates(cup, SPOON, CLEAR + u, HOVER, 45.0, 20.0, None)[0]))
        try:
            plans[u] = plan(rig, cup)
        except U.NoSafeEntry as e:
            plans[u] = e.reason
    assert kept == sorted(kept, reverse=True) and kept[0] > kept[1] > kept[2] and kept[-1] == 0
    assert isinstance(plans[0.0], U.EntryPlan) and isinstance(plans[3.0], U.EntryPlan)
    assert plans[20.0] == "too_narrow"                          # 70 mm opening < 32 + 2 * 23
    seen_fail = False
    for u, p in plans.items():                                  # once it fails it stays failed
        if isinstance(p, str):
            seen_fail = True
            assert p in ("no_clear_path", "too_narrow", "unreachable")
        else:
            assert not seen_fail and p.min_clearance_mm >= CLEAR + u
            assert math.hypot(p.entry_xy[0] - 200, p.entry_xy[1] - 60) <= 35 - CLEAR - u
            assert walk(rig, executed(p), cup_at(uncertainty=u)) >= CLEAR + u - 1e-6
    assert seen_fail


@pytest.mark.parametrize("dx,dy", [(0, 28), (0, -28), (-28, 0), (28, 0)], ids=["left", "right", "nearer", "farther"])
def test_moved_cup_gets_its_own_plan(rig, dx, dy):
    old_cup = cup_at(200, 60)
    old = plan(rig, old_cup)
    cup = old_cup.moved(dx, dy)
    new = plan(rig, cup)
    margin = CLEAR + cup.uncertainty
    assert U.check_path(rig.kin, new.poses(), cup, SPOON, margin).ok and walk(rig, executed(new), cup) >= margin - 1e-6
    # the old plan would hit the cup where it is now
    assert not U.check_path(rig.kin, old.poses(), cup, SPOON, margin).ok
    # the entry point is laid out from the new centre, along the arm's line through it
    yaw = math.atan2(cup.y, cup.x)
    ex, ey = new.entry_xy[0] - cup.x, new.entry_xy[1] - cup.y
    assert abs(ey * math.cos(yaw) - ex * math.sin(yaw)) < 0.01
    assert math.hypot(ex, ey) <= cup.safe_radius(0, CLEAR)
    assert math.dist(K.fk(rig.kin, new.dip)[:2], new.entry_xy) < 0.6
    sx, sy, _ = U.spoon_center(rig.kin, new.dip, SPOON)
    assert math.hypot(sx - cup.x, sy - cup.y) <= cup.safe_radius(SPOON.bowl_width / 2, CLEAR) + 1e-6
    base_yaw = math.radians(rig.kin.sign["base"] * (new.hover["base"] - rig.kin.zero_deg["base"]))
    assert base_yaw == pytest.approx(yaw, abs=1e-6)
    assert new.cup == cup
    if dy:                                                      # sideways: not the old poses slid across
        assert abs(new.hover["base"] - old.hover["base"]) > 5


def test_approach_from_the_far_side_stays_above_the_rim(rig):
    cup = cup_at(160, 40)
    margin = CLEAR + cup.uncertainty
    rc, yaw = math.hypot(cup.x, cup.y), math.atan2(cup.y, cup.x)
    tried = 0
    for below in (35, 40, 45, 50, 60):
        for z in (15, 25, 35):
            r = rc + cup.radius + 50                            # tip beyond the far wall, lower than the rim
            start = pose_at(rig, r * math.cos(yaw), r * math.sin(yaw), z, below, in_limits=True)
            if start is None or U.clearance(rig.kin, start, cup, SPOON).mm < margin:
                continue
            tried += 1
            p = plan(rig, cup, start=start)
            move = [start, *p.approach, p.hover]
            assert U.check_path(rig.kin, move, cup, SPOON, margin).ok and walk(rig, move, cup) >= margin - 1e-6
            # nothing of the spoon is ever below the rim over the cup or its wall
            for a, b in zip(move, move[1:]):
                for k in range(41):
                    for pt in U.spoon_body(rig.kin, lerp_pose(a, b, k / 40), SPOON):
                        over = math.hypot(pt.x - cup.x, pt.y - cup.y) <= cup.radius + cup.wall
                        assert not over or pt.z >= cup.rim + CLEAR
    assert tried >= 2


def test_approach_goes_up_first_when_the_direct_move_would_cut_through(rig):
    """Spoon low beside the cup: turning straight to the hover would sweep it through the wall."""
    cup = cup_at(200, 60)
    margin = CLEAR + cup.uncertainty
    rc, yaw = math.hypot(cup.x, cup.y), math.atan2(cup.y, cup.x) + math.radians(20)
    start = next(q for q in (pose_at(rig, rc * math.cos(yaw), rc * math.sin(yaw), 20, a, in_limits=True)
                             for a in (40, 50, 60, 45, 35, 55)) if q is not None)
    assert U.clearance(rig.kin, start, cup, SPOON).mm >= margin
    p = plan(rig, cup, start=start)
    assert not U.check_path(rig.kin, [start, p.hover], cup, SPOON, margin).ok
    assert p.approach
    # the first waypoint has the whole spoon above the rim, and it is still on the side it started on
    assert min(b.z for b in U.spoon_body(rig.kin, p.approach[0], SPOON)) >= cup.rim + HOVER - 0.5
    assert p.approach[0]["base"] == start["base"]
    move = [start, *p.approach, p.hover]
    assert U.check_path(rig.kin, move, cup, SPOON, margin).ok and walk(rig, move, cup) >= margin - 1e-6
    assert all(rig.limits[j][0] <= q[j] <= rig.limits[j][1] for q in p.approach for j in U.FKJ)
    # a start that is already closer than the margin is refused, not planned around
    close = pose_at(rig, cup.x, cup.y, 26, 30)
    with pytest.raises(U.NoSafeEntry) as e:
        plan(rig, cup, start=close)
    assert e.value.reason == "no_clear_path"
    assert plan(rig, cup).approach == []                        # no start given: nothing to approach from


def test_same_inputs_same_plan(rig):
    cup = cup_at(195, 82)
    a, b = plan(rig, cup, angle=40, drag=15, start=rig.seed), plan(rig, cup, angle=40, drag=15, start=rig.seed)
    assert a.poses() == b.poses() and a.approach == b.approach and a.descent(8) == b.descent(8)
    assert (a.cost, a.min_clearance_mm, a.entry_xy, a.dip_z, a.drag_mm, a.entry_angle_deg, a.rejected) == \
        (b.cost, b.min_clearance_mm, b.entry_xy, b.dip_z, b.drag_mm, b.entry_angle_deg, b.rejected)


def test_wanted_depth_and_drag_are_followed_when_they_fit(rig):
    cup = cup_at(200, 60)
    deep, shallow = plan(rig, cup), plan(rig, cup, want_depth_mm=20.0)
    assert shallow.dip_z == pytest.approx(cup.rim - 20.0, abs=5.0) and shallow.dip_z > deep.dip_z
    assert plan(rig, cup, drag=0.0).stroke == [] and plan(rig, cup, drag=0.0).drag_mm == 0


def test_plan_entry_is_fast(rig):
    """It runs inside one state machine tick. Generous bound here (machines differ); the measured time is printed."""
    cup = cup_at(200, 60)
    plan(rig, cup, start=rig.seed)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        plan(rig, cup, start=rig.seed)
        times.append(time.perf_counter() - t)
    print(f"plan_entry [{rig.name}]: best {min(times) * 1000:.0f} ms, worst {max(times) * 1000:.0f} ms")
    assert min(times) < 0.5
