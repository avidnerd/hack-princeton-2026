"""Arm-level guards for the rim-aware scoop: a probe with no blind window (Plan.probe_settle_s), relative lag as a
contact (Plan.lag_limit_deg), the residual error reported on a finished motion, and the sim hooks they are tested
with (SimBackend.sag / SimBackend.obstruct). Sim only: none of this says anything about the real arm.

Speeds used: descents at 6 mm/s (RimCfg.descent_speed_mm_s), horizontal strokes at 10 mm/s."""
import math
import time
from pathlib import Path

import pytest

from spoonful.arm import kinematics as K
from spoonful.arm.arm import Arm
from spoonful.arm.backend import SimBackend
from spoonful.arm.contact import Contact
from spoonful.arm.scoop_style import dip_index, solve_rz
from spoonful.arm.trajectory import Plan, Segment, TrajFile, plan_goto
from spoonful.config import ARM_JOINTS, JOINTS

DESCENT_MM_S = 6.0
STROKE_MM_S = 10.0
LIFTING = ("shoulder", "elbow", "wrist_pitch")
SAG_UNITS = {"shoulder": 8, "elbow": -6, "wrist_pitch": 5}     # about 2, -1.4 and 1.2 deg of steady sag


def _rz(kin, q):
    x, y, z = K.fk(kin, q)
    return math.hypot(x, y), z


@pytest.fixture
def rig(fast_cfg):
    """An Arm on the sim backend, holding still 10 mm below the sim scoop's dip pose (clear of the reach limit)."""
    cfg, kin = fast_cfg, fast_cfg.kinematics
    poses = TrajFile.load(Path(cfg.path(cfg.arm.sim_trajectories_dir)) / "scoop.json").poses
    dip = poses[dip_index(kin, poses)]
    r, z = _rz(kin, dip)
    start = solve_rz(kin, dip, r, z - 10)
    ids = [cfg.arm.servo_map[j] for j in JOINTS]
    backend = SimBackend(ids, {cfg.arm.servo_map[j]: round(cfg.arm.center_units + start[j] * cfg.arm.units_per_deg)
                               for j in ARM_JOINTS})
    arm = Arm(cfg, backend)
    arm.start()
    arm.torque(True)
    time.sleep(0.3)
    yield arm
    arm.close()


def _sag(arm, lag_s=0.08):
    """Steady sag plus a speed-proportional lag on the lifting joints, then let the hold settle into it."""
    for j, u in SAG_UNITS.items():
        arm.backend.sag(arm.ids[j], u, lag_s)
    time.sleep(0.5)


def _tip(arm):
    """Where the simulated spoon tip really is (the encoder readings carry +-1 unit of noise per joint, which is a
    couple of mm at the tip: too coarse to place an obstruction 2 mm away)."""
    with arm.backend._lock:
        return arm.backend.tip_fn(dict(arm.backend.pos))


def _resting_on(arm, top_mm):
    """The tip stopped at a surface: not through it, and no more than the read noise above it (after a touch the
    arm holds the pose it measured)."""
    return -0.1 <= _tip(arm)[2] - top_mm < 3.0


def _lower(arm, dz_mm, dr_mm=0.0):
    """Pose with the commanded tip dz_mm lower (and dr_mm further out) than the current setpoint."""
    kin = arm.cfg.kinematics
    cur = arm.current()
    r, z = _rz(kin, cur)
    q = solve_rz(kin, cur, r + dr_mm, z - dz_mm)
    assert q is not None
    return cur, {j: q[j] for j in ARM_JOINTS}


def _line(arm, dz_mm, dr_mm, mm_s, **kw):
    cur, end = _lower(arm, dz_mm, dr_mm)
    T = math.hypot(dz_mm, dr_mm) / mm_s
    return Plan([Segment([cur, end], "normal", T, "line", linear=True)], "line", **kw)


def _descent(arm, mm=12.0, **kw):
    return _line(arm, mm, 0.0, DESCENT_MM_S, probe_deg=arm.cfg.plate.rim.stall_deg, **kw)


# ---------------------------------------------------------------- defaults: nothing changes unless asked
def test_new_plan_fields_default_to_off(rig):
    p = _descent(rig)
    assert p.probe_settle_s is None and p.lag_limit_deg is None
    b = rig.backend
    assert b.sags == {} and b.lags == {} and b.obstructions == []


def test_default_probe_still_waits_out_its_settle_time(rig):
    """probe_settle_s=None keeps Arm.PROBE_SETTLE_S: nothing can count as a touch before it."""
    rig.backend.obstruct(_tip(rig)[2] - 2.0)
    t0 = time.monotonic()
    m = rig.run(_descent(rig)).wait(6)
    assert m.ok and m.touched is not None
    assert time.monotonic() - t0 >= Arm.PROBE_SETTLE_S + 0.8 * Arm.PROBE_WINDOW_S


# ---------------------------------------------------------------- probe with no blind window
@pytest.mark.parametrize("sag", [False, True])
def test_probe_from_a_settled_hold_feels_an_obstruction_2mm_down(rig, sag):
    if sag:
        _sag(rig)
    x, y, z = _tip(rig)
    rig.backend.obstruct(z - 2.0, x, y, 0.0, 30.0)          # food / bottom: a disc under the tip
    t0 = time.monotonic()
    m = rig.run(_descent(rig, probe_settle_s=0)).wait(6)
    took = time.monotonic() - t0
    assert m.ok and m.touched is not None, "no touch reported"
    assert _resting_on(rig, z - 2.0)
    assert abs(K.fk(rig.cfg.kinematics, m.touched)[2] - (z - 2.0)) < 4.0     # .touched = the measured pose there
    # The stall window is still needed (a touch is a stall over PROBE_WINDOW_S), but not the 0.8 s settle on top
    # of it. The command got this far past the surface before the stop:
    assert took < 1.3, f"touch took {took:.2f} s"
    assert took * DESCENT_MM_S - 2.0 < 6.0


@pytest.mark.parametrize("sag", [False, True])
def test_probe_from_a_settled_hold_does_not_trip_on_a_free_descent(rig, sag):
    if sag:
        _sag(rig)
    z = _tip(rig)[2]
    m = rig.run(_descent(rig, probe_settle_s=0)).wait(6)     # 12 mm at 6 mm/s
    assert m.ok and m.touched is None
    assert z - _tip(rig)[2] > 10.0


def test_rim_and_bottom_can_both_be_simulated(rig):
    """A ring (the rim) and a lower disc (the bottom) round the same axis: over the ring the tip stops at the rim,
    inside it the tip goes on down to the bottom."""
    x, y, z = _tip(rig)
    r = math.hypot(x, y)
    ux, uy = x / r, y / r
    cx, cy = x - 20 * ux, y - 20 * uy                        # cup centre 20 mm nearer the arm: tip is over its wall
    rig.backend.obstruct(z - 3.0, cx, cy, 18.0, 22.0)        # rim ring
    rig.backend.obstruct(z - 9.0, cx, cy, 0.0, 18.0)         # bottom disc
    m = rig.run(_descent(rig, probe_settle_s=0)).wait(6)
    assert m.touched is not None
    assert _resting_on(rig, z - 3.0)
    rig.goto(_lower(rig, -8.0)[1]).wait(5)                   # up, then in over the opening
    rig.goto(_lower(rig, 0.0, -20.0)[1]).wait(5)
    time.sleep(0.3)
    m = rig.run(_descent(rig, 24.0, probe_settle_s=0)).wait(8)
    assert m.touched is not None
    assert _resting_on(rig, z - 9.0)


# ---------------------------------------------------------------- relative lag = contact
def _stroke(arm, **kw):
    return _line(arm, 0.0, -25.0, STROKE_MM_S, **kw)        # 25 mm toward the arm, level


def _wall(arm, gap_mm=5.0):
    """A cup wall across the stroke, gap_mm from the tip (a ring round a centre 30 mm nearer the arm)."""
    x, y, z = _tip(arm)
    r = math.hypot(x, y)
    arm.backend.obstruct(z + 30.0, x - 30 * x / r, y - 30 * y / r, 30.0 - gap_mm - 3.0, 30.0 - gap_mm)


def _allowance(arm, joint):
    a = arm.cfg.arm
    return a.contact_error_deg_joint.get(joint, a.contact_error_deg)


@pytest.mark.parametrize("sag", [False, True])
def test_lag_limit_stops_a_blocked_stroke_well_before_the_absolute_threshold(rig, sag):
    if sag:
        _sag(rig)
    kin = rig.cfg.kinematics
    _wall(rig)
    r0 = math.hypot(*_tip(rig)[:2])
    m = rig.run(_stroke(rig, lag_limit_deg=rig.cfg.plate.rim.scoop_contact_deg)).wait(6)
    assert isinstance(m.error, Contact) and m.error.grew_deg is not None
    assert m.error.joint in LIFTING
    assert m.error.err_deg < _allowance(rig, m.error.joint) - 2.0            # the absolute limit was not near
    assert rig.last_contact is m.error and rig.mode == "hold"
    assert K.tip_distance(kin, rig.current(), rig.read().deg) < 3.0          # setpoint pulled back to the arm
    assert r0 - math.hypot(*_tip(rig)[:2]) < 5.0 + 0.5                       # and the tip never passed the wall


def test_without_lag_limit_the_same_blocked_stroke_presses_much_further(rig):
    """The comparison: with only the absolute thresholds the command runs on to the end of the stroke (or to a
    much larger error) against the same wall."""
    kin = rig.cfg.kinematics
    _wall(rig)
    t0 = time.monotonic()
    limited = rig.run(_stroke(rig, lag_limit_deg=rig.cfg.plate.rim.scoop_contact_deg)).wait(6)
    t_limited = time.monotonic() - t0
    assert isinstance(limited.error, Contact)
    rig.goto(_lower(rig, 0.0, 6.0)[1]).wait(5)               # back off the wall, same stroke again without the limit
    time.sleep(0.3)
    t0 = time.monotonic()
    plain = rig.run(_stroke(rig))
    plain.wait(t_limited + 0.5)
    pressed = K.tip_distance(kin, rig.current(), rig.read().deg)
    if plain.done.is_set():
        assert isinstance(plain.error, Contact) and plain.error.grew_deg is None
        assert plain.error.err_deg > 2 * limited.error.err_deg
    else:
        assert pressed > 6.0                                 # still pushing, mm of command beyond the wall
    plain.wait(6)


@pytest.mark.parametrize("sag", [False, True])
def test_lag_limit_does_not_trip_on_a_free_move(rig, sag):
    if sag:
        _sag(rig)
    lim = rig.cfg.plate.rim.scoop_contact_deg
    m = rig.run(_stroke(rig, lag_limit_deg=lim)).wait(6)                      # 25 mm at 10 mm/s
    assert m.ok, m.error
    m = rig.run(_line(rig, 12.0, 0.0, DESCENT_MM_S, lag_limit_deg=lim)).wait(6)   # 12 mm down at 6 mm/s
    assert m.ok, m.error
    assert rig.last_contact is None


def test_absolute_thresholds_stay_in_force_under_a_lag_limit(rig):
    """A lag limit can only tighten: with a uselessly large one, a jammed joint is still a contact at the usual
    absolute threshold."""
    cur, end = _lower(rig, 0.0, -25.0)
    rig.backend.jam(rig.ids["shoulder"], True)
    m = rig.run(Plan(plan_goto(rig.cfg, cur, end).segments, "jammed", lag_limit_deg=90.0)).wait(6)
    assert isinstance(m.error, Contact) and m.error.grew_deg is None
    assert m.error.joint == "shoulder" and m.error.err_deg > _allowance(rig, "shoulder")


def test_growth_check_ignores_steady_sag_and_needs_the_hold_time(fast_cfg):
    from spoonful.arm.contact import ContactMonitor
    mon = ContactMonitor(fast_cfg)
    hold = fast_cfg.arm.contact_hold_s
    cmd = {j: 0.0 for j in ARM_JOINTS}
    sagging = dict(cmd, elbow=-4.0)
    base = {"shoulder": 0.0, "elbow": 4.0, "wrist_pitch": 0.0}
    assert mon.check_growth(0.0, cmd, sagging, base, 3.0, 0.0) is None
    assert mon.check_growth(hold + 1, cmd, sagging, base, 3.0, 0.0) is None           # 4 deg of sag, no growth
    pressed = dict(cmd, elbow=-7.5)
    assert mon.check_growth(10.0, cmd, pressed, base, 3.0, 0.0) is None               # grew 3.5: not held yet
    c = mon.check_growth(10.0 + hold, cmd, pressed, base, 3.0, 0.0)
    assert c is not None and c.joint == "elbow" and c.grew_deg == pytest.approx(3.5) and c.err_deg == pytest.approx(7.5)
    mon.reset()
    assert mon.check_growth(20.0 + hold, cmd, pressed, base, 3.0, 10.0) is None       # speed grace covers it
    assert mon.check(30.0, cmd, pressed, 0.0) is None and mon.check(30.0 + hold, cmd, pressed, 0.0) is None


# ---------------------------------------------------------------- finished, or finished while pressing
def test_residual_is_reported_when_a_plan_ends_against_an_obstruction(rig):
    """6 mm of command into a surface 1 mm down leaves the joints short of the command by less than the contact
    threshold: the plan finishes as before, and now says how far short."""
    x, y, z = _tip(rig)
    rig.backend.obstruct(z - 1.0)
    cur, end = _lower(rig, 6.0)
    m = rig.goto(end).wait(6)
    assert m.ok and m.touched is None
    assert m.residual_deg, "no residual reported"
    for j, e in m.residual_deg.items():
        assert Arm.RESIDUAL_DEG < abs(e) <= _allowance(rig, j)
    assert max(abs(v) for v in m.lag_growth_deg.values()) > Arm.RESIDUAL_DEG
    assert _tip(rig)[2] > z - 2.5                            # the arm really is still up at the surface


def test_no_residual_on_a_free_move(rig):
    m = rig.run(_line(rig, 6.0, 0.0, DESCENT_MM_S)).wait(6)
    assert m.ok and m.residual_deg is None
    assert set(m.lag_growth_deg) == set(LIFTING) and max(abs(v) for v in m.lag_growth_deg.values()) < 1.5
