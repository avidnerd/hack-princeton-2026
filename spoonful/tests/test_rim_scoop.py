"""The staged scoop (brain/rim_scoop.py) end to end on the sim backend, with a fake camera observer and the sim's
sag / obstruction hooks. No hardware, no camera, no config.yaml writes. What the sim shows is the logic; none of it
says how the real arm behaves."""
import json
import math
import threading
import time

import pytest

from conftest import wait_state
from test_fsm import bites, start_listening
from spoonful import config as C
from spoonful.arm import cup as U
from spoonful.arm import kinematics as K
from spoonful.brain.rim_scoop import RimScoop
from spoonful.config import ARM_JOINTS
from spoonful.perception.rim import Observation

CUP = dict(x_mm=200.0, y_mm=60.0, radius_mm=35.0, rim_mm=45.0, uncertainty_mm=3.0, confirmed=True, source="confirmed")
ORDER = ["OBSERVE", "SAFE_HOVER", "VISUAL_ALIGNMENT", "ENTRY_ORIENTATION", "GUARDED_DESCENT", "INTERIOR_SCOOP",
         "RETRACT_AND_VERIFY"]
LIFT = ("shoulder", "elbow", "wrist_pitch")
SLOW = dict(descent_step_mm=8.0, descent_settle_s=0.0, press_deg=2.0, descent_speed_mm_s=6.0, view_valid_s=30.0)   # the shipped descent: a step outlasts the probe window


class FakeRim:
    """Stands in for perception.rim.RimObserver: answers (status, dx, dy) in turn; the last one repeats."""

    def __init__(self, *answers):
        self.answers, self.calls, self.accepted = list(answers), 0, []

    def observe(self, bowl):
        a = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        status, dx, dy = (a + (0.0, 0.0))[:3]
        return Observation(status, bowl, dx, dy, confidence=0.0 if status == "lost" else 0.9, stamp=time.time(),
                           reason="no_ellipse" if status == "lost" else "")

    def accept_move(self, bowl, obs):
        self.accepted.append((bowl, obs.dx, obs.dy))


class Recorder(threading.Thread):
    """Samples the commanded setpoint and the simulated arm's true pose (no encoder noise) every few ms."""

    def __init__(self, s):
        super().__init__(daemon=True)
        self.s, self.rows, self._halt = s, [], threading.Event()

    def run(self):
        s = self.s
        while not self._halt.is_set():
            rs = s.fsm._rim_scoop
            true = {j: s.arm.to_deg(j, s.backend.pos[s.arm.ids[j]]) for j in ARM_JOINTS}
            self.rows.append((time.time(), s.fsm.state.value, rs.stage if rs else "", dict(s.arm.setpoint), true))
            time.sleep(0.004)

    def stop(self):
        self._halt.set()
        self.join(timeout=1)

    def tips(self, which, since=0.0):
        k = 3 if which == "cmd" else 4
        return [K.fk(self.s.cfg.kinematics, r[k]) for r in self.rows if r[0] >= since and r[k]]

    def clearance(self, which, cup, states=("SCOOPING", "LIFTING")):
        k = 3 if which == "cmd" else 4
        spoon = U.Spoon.from_cfg(self.s.cfg.spoon)
        return min(U.clearance(self.s.cfg.kinematics, r[k], cup, spoon).mm for r in self.rows if r[1] in states and r[k])


@pytest.fixture
def build(fast_cfg, monkeypatch):
    """build(rim=FakeRim(...), cup={...} or None, probe=False, **plate.rim overrides) -> a running sim System with
    bowl1 = applesauce in a known cup. s.rec records the arm."""
    from spoonful.system import Options, System
    monkeypatch.setattr(C.Config, "save", lambda *a, **k: None)       # never the real config.yaml
    monkeypatch.setattr(RimScoop, "SETTLE_S", 0.25)
    made = []

    def make(rim="default", cup="default", probe=False, **rc):
        c = fast_cfg
        c.spoon = C.SpoonGeomCfg(measured=True, bowl_length_mm=32.8, bowl_width_mm=21.4, bowl_depth_mm=10.3,
                                 neck_length_mm=66.8, neck_width_mm=9.8)
        c.plate.rim = C.RimCfg(descent_step_mm=15.0, descent_settle_s=0.0, press_deg=2.0, descent_speed_mm_s=20.0)   # the rest: the code's defaults
        for k, v in rc.items():
            setattr(c.plate.rim, k, v)
        c.plate.bowls[0].cup = None if cup is None else C.CupCfg(**(CUP if cup == "default" else cup))
        c.plate.probe.enabled = probe
        if probe:
            c.plate.probe.speed_mm_s, c.plate.probe.below_mm = 10, 20
        s = System(c, Options(backend="sim", sim_face=True, camera=False, voice=False, esp32=False))
        s.profiles.update(hold_s=0.6)
        s.start()
        made.append(s)
        s.plate.set_label("bowl1", "applesauce")
        s.fsm.rim = FakeRim(("ok",)) if rim == "default" else rim
        s.rec = Recorder(s)
        s.rec.start()
        start_listening(s)
        return s

    yield make
    for s in made:
        s.rec.stop()
        s.log.flush()                                              # shown by pytest only when the test failed
        for r in s.log.query("SELECT type, detail FROM events WHERE type LIKE 'rim_%' AND type != 'rim_stage' "
                             "ORDER BY rowid"):
            print(r["type"], r["detail"][:300])
        s.stop()


def bite(s, bowl="bowl1"):
    s.n_bites = len(bites(s))
    s.fsm.post("bite", source="test", bowl=bowl)


def events(s, kind):
    s.log.flush()
    return [(r["ts"], json.loads(r["detail"])) for r in
            s.log.query("SELECT ts, detail FROM events WHERE type=? ORDER BY rowid", (kind,))]


def stages(s):
    return [d["detail_stage"] for _, d in events(s, "rim_stage")]


def spoken(s):
    return [d["detail_line"] for _, d in events(s, "speech")]


def cup_of(s, **kw):
    return U.Cup.from_cfg(s.cfg.plate.bowls[0].cup, 60.0).moved(0, 0, **kw)


def refused(s, outcome):
    """The bite ended without a scoop; returns the bite's outcome row."""
    t0 = time.time()
    while len(bites(s)) <= s.n_bites and time.time() - t0 < 30:    # a refusal can be over within one tick
        time.sleep(0.05)
    wait_state(s, "LISTENING", timeout=10)
    assert bites(s)[-1]["outcome"] == outcome
    assert not events(s, "fault")
    return bites(s)[-1]


def never_moved(s):
    first = s.rec.rows[0][3]
    return all(max(abs(r[3][j] - first[j]) for j in ARM_JOINTS) < 0.5 for r in s.rec.rows)


def no_lower_after(s, t_touch, touch_z):
    """Nothing commanded after the touch is lower than what was already commanded when it was noticed (a step can
    end pressing a few mm past the surface before the lag shows), nor lower than the touch once the arm is off it."""
    k = s.cfg.kinematics
    before = [K.fk(k, r[3])[2] for r in s.rec.rows if r[0] < t_touch][-1]
    after = [t[2] for t in s.rec.tips("cmd", since=t_touch)]
    assert min(after) >= min(before, touch_z) - 1.5
    assert before >= touch_z - 8                                    # and the press itself was a few mm, not more


def sag(s, **units):
    for j, u in units.items():
        s.backend.sag(s.cfg.arm.servo_map[j], u, 0.05)
    time.sleep(0.6)


# ---------------------------------------------------------------- the whole scoop
def test_full_scoop_visits_the_stages_in_order_and_stays_clear(build):
    s = build()
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    wait_state(s, "PRE_DELIVERY_HOLD", timeout=20)
    assert stages(s) == ORDER
    cup, rc = cup_of(s), s.cfg.plate.rim
    assert s.rec.clearance("cmd", cup) >= rc.clearance_mm          # every commanded pose
    assert s.rec.clearance("true", cup) >= rc.clearance_mm         # and where the simulated arm really was
    low = min(s.rec.tips("true"), key=lambda t: t[2])
    assert low[2] < cup.rim - 20 and math.hypot(low[0] - cup.x, low[1] - cup.y) < cup.radius   # it did go in
    plan = events(s, "rim_plan")[-1][1]
    assert plan["cup_xy"] == [200.0, 60.0] and plan["min_clearance_mm"] >= rc.clearance_mm + cup.uncertainty
    assert events(s, "unvalidated_trial") and "Trial scoop. Keep clear." in spoken(s)
    # never the old way: no camera-model look, no saved scoop path, no probe scoop
    for old in ("look", "attack_plan", "scoop_style", "scoop_shape", "scoop_probe", "probe_unavailable"):
        assert not events(s, old), old
    assert s.fsm.status()["rim_blocked"] == []


# ---------------------------------------------------------------- measured-tip alignment
def test_sag_is_corrected_on_the_measured_tip_and_the_real_path_stays_clear(build):
    s = build()
    start = s.backend.tip_fn(dict(s.backend.pos))
    sag(s, wrist_pitch=12)                                         # the spoon droops: about 10 mm low at the hover
    assert s.backend.tip_fn(dict(s.backend.pos))[2] < start[2] - 6
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    al = [d for _, d in events(s, "rim_align") if d["detail_stage"] == "VISUAL_ALIGNMENT"]
    tol = s.cfg.plate.rim.align_tol_mm
    assert not al[0]["height_ok"] and al[0]["err"][2] > 6          # found low (err_mm is the error across the table)
    assert al[-1]["err_mm"] <= tol and al[-1]["height_ok"] and len(al) >= 2   # corrected until the measured tip was there
    assert stages(s) == ORDER
    assert s.rec.clearance("true", cup_of(s)) > 0.0
    done = events(s, "rim_done")[-1][1]
    assert math.hypot(*done["fix"]) > 5                            # the command really was moved


def test_sag_beyond_the_allowed_correction_aborts_without_descending(build):
    s = build(align_max_fix_mm=5.0)
    sag(s, wrist_pitch=12)
    bite(s)
    refused(s, "align_failed")
    assert "GUARDED_DESCENT" not in stages(s) and "ENTRY_ORIENTATION" not in stages(s)
    assert s.rec.clearance("true", cup_of(s), states=("SCOOPING", "RETREAT")) > 20    # never came down to the cup


def test_a_correction_that_does_not_help_aborts_as_diverged(build, monkeypatch):
    real = RimScoop._shifted

    def wrong(self, q, fix):                                       # corrects across the table the wrong way
        return real(self, q, (-0.5 * fix[0], -0.5 * fix[1], fix[2])) or dict(q)
    monkeypatch.setattr(RimScoop, "_shifted", wrong)
    s = build()
    # the fixture's short settle reads the hover while the drooping wrist is still catching up (12-17 mm for a
    # steady 10.5); a first reading that far off makes the wrong correction look like a gain
    monkeypatch.setattr(RimScoop, "SETTLE_S", 0.8)
    sag(s, wrist_pitch=12, base=2.5)                               # the spoon sits low and about 10 mm to one side
    bite(s)
    refused(s, "align_diverged")
    al = [d["err_mm"] for _, d in events(s, "rim_align")]
    assert len(al) >= 2 and al[-1] > al[-2] - 1.0
    assert "GUARDED_DESCENT" not in stages(s)


# ---------------------------------------------------------------- the camera's answer
@pytest.mark.parametrize("rim", [FakeRim(("lost",)), None])
def test_cup_not_seen_means_no_motion_at_all(build, rim):
    s = build(rim=rim)
    bite(s)
    refused(s, "cup_not_seen")
    assert never_moved(s)
    assert stages(s) == ["OBSERVE"]
    assert "I can't see the applesauce cup. Not scooping." in spoken(s)
    assert not events(s, "rim_plan")


def test_a_moved_cup_is_planned_where_it_is_now(build):
    rim = FakeRim(("moved", 6.0, -4.0), ("ok",))
    s = build(rim=rim)
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    plan = events(s, "rim_plan")[-1][1]
    assert plan["cup_xy"] == [206.0, 56.0] and plan["source"] == "followed"
    assert plan["uncertainty_mm"] > CUP["uncertainty_mm"]          # a followed position is known less well
    c = s.cfg.plate.bowls[0].cup
    assert (c.x_mm, c.y_mm, c.source) == (206.0, 56.0, "followed")
    assert rim.accepted == [("bowl1", 6.0, -4.0)]
    assert "The applesauce moved. Following it" in spoken(s)
    new = cup_of(s, uncertainty=0.0)
    low = min(s.rec.tips("true"), key=lambda t: t[2])
    assert low[2] < new.rim - 20 and math.hypot(low[0] - 206.0, low[1] - 56.0) < new.radius - 3
    assert s.rec.clearance("true", new) > 0.0


def test_cup_moved_between_two_bites_is_replanned(build):
    rim = FakeRim(("ok",), ("ok",), ("moved", -6.0, 3.0), ("ok",))
    s = build(rim=rim)
    s.face.sim.update(present=False)                               # nobody takes it: the bite times out and retreats
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    wait_state(s, "LISTENING", timeout=30)
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    plans = [d for _, d in events(s, "rim_plan")]
    assert [p["cup_xy"] for p in plans] == [[200.0, 60.0], [194.0, 63.0]]
    assert math.dist(plans[0]["entry_xy"], plans[1]["entry_xy"]) > 4


def test_a_cup_that_moves_while_hovering_is_replanned_from_the_hover(build):
    rim = FakeRim(("ok",), ("moved", 5.0, 5.0), ("ok",))
    s = build(rim=rim)
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    plans = [d for _, d in events(s, "rim_plan")]
    assert [p["cup_xy"] for p in plans] == [[200.0, 60.0], [205.0, 65.0]] and plans[1]["replan"] == 1
    assert stages(s)[:4] == ["OBSERVE", "SAFE_HOVER", "VISUAL_ALIGNMENT", "SAFE_HOVER"]
    assert s.rec.clearance("true", cup_of(s, uncertainty=0.0)) > 0.0


def test_sighting_lost_and_too_old_at_the_hover_aborts_there(build):
    s = build(rim=FakeRim(("ok",), ("lost",)), view_valid_s=0.2)
    bite(s)
    refused(s, "cup_not_seen")
    st = stages(s)
    assert st[:3] == ["OBSERVE", "SAFE_HOVER", "VISUAL_ALIGNMENT"]
    assert "ENTRY_ORIENTATION" not in st and "GUARDED_DESCENT" not in st
    assert s.rec.clearance("true", cup_of(s), states=("SCOOPING", "RETREAT")) > 20
    assert "I lost sight of the applesauce cup. Stopping this bite." in spoken(s)


def test_sighting_that_goes_stale_during_the_descent_stops_it(build):
    s = build(rim=FakeRim(("ok",), ("lost",)), view_valid_s=3.2)   # hidden by the arm: relied on for 3.2 s only
    bite(s)
    refused(s, "cup_not_seen")
    st = stages(s)
    assert "GUARDED_DESCENT" in st and "INTERIOR_SCOOP" not in st and st[-1] == "RETRACT_ABORT"
    assert s.rec.clearance("true", cup_of(s), states=("SCOOPING", "RETREAT")) > 0.0


# ---------------------------------------------------------------- touching things
def test_obstruction_at_rim_height_stops_retracts_straight_up_and_blocks_the_cup(build):
    s = build(rim=FakeRim(("ok",), ("lost",)), **SLOW)             # hidden by the arm at the hover: still valid
    s.backend.obstruct(46.0, 200.0, 60.0, 0.0, 60.0)               # something solid across the opening
    bite(s)
    refused(s, "rim_contact")
    st = stages(s)
    assert st[:5] == ORDER[:5] and st[-1] == "RETRACT_ABORT" and "INTERIOR_SCOOP" not in st
    t_touch, touch = events(s, "rim_touch")[-1]
    assert touch["inside"] is False and abs(touch["tip"][2] - 46) < 4
    ev = events(s, "rim_contact")[-1][1]
    assert ev["detail_stage"] == "GUARDED_DESCENT"
    for k in ("joint_err_deg", "commanded", "measured", "clearance_mm", "sighting", "seen"):
        assert k in ev
    assert set(ev["joint_err_deg"]) >= set(LIFT) and ev["seen"]["status"] == "ok"
    # after the touch: nothing commanded lower, and straight up until well above the rim
    after = s.rec.tips("cmd", since=t_touch)
    no_lower_after(s, t_touch, touch["tip"][2])
    near = [t for t in after if t[2] < CUP["rim_mm"] + 20]
    assert near and max(math.hypot(t[0] - touch["tip"][0], t[1] - touch["tip"][1]) for t in near) < 3.0
    assert max(t[2] for t in after) > CUP["rim_mm"] + 25
    assert "I touched the applesauce cup. Stopping this bite." in spoken(s)
    # the same cup with the same sighting is not tried again
    assert s.fsm.status()["rim_blocked"] == ["bowl1"]
    n = len(s.rec.rows)
    s.fsm.rim = FakeRim(("ok",))
    bite(s)
    refused(s, "blocked")
    first = s.rec.rows[n][3]
    assert all(max(abs(r[3][j] - first[j]) for j in ARM_JOINTS) < 0.5 for r in s.rec.rows[n:])
    s.fsm.rim_unblock("bowl1")                                     # the operator lined it up again
    assert s.fsm.rim_blocked == {}


def test_a_blocked_cup_is_tried_again_once_it_is_seen_somewhere_new(build):
    s = build(rim=FakeRim(("moved", 6.0, 0.0), ("ok",)))
    s.backend.obstruct(None)
    s.fsm.rim_blocked["bowl1"] = RimScoop._key(cup_of(s))
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    assert s.fsm.rim_blocked == {}


def test_a_touch_well_inside_the_cup_is_the_food_and_the_stroke_never_goes_lower(build):
    s = build(**SLOW)
    # a firm lump of food under the entry point, 20 mm below the rim (the sim's solids are rigid, so a whole rigid
    # layer would also catch the drag just above it; that case is test_contact_during_the_stroke_retracts)
    s.backend.obstruct(25.0, 217.0, 65.0, 0.0, 8.0)
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    assert stages(s) == ORDER
    t_touch, touch = events(s, "rim_touch")[-1]
    assert touch["inside"] is True and abs(touch["tip"][2] - 25) < 6
    depth = events(s, "rim_depth")[-1][1]
    assert depth["tip_z"] >= touch["tip"][2] and depth["tip_z"] > depth["planned_z"] + 5
    no_lower_after(s, t_touch, touch["tip"][2])
    assert not events(s, "rim_contact")
    assert s.rec.clearance("true", cup_of(s, floor=0.0)) > 0.0


def test_contact_during_the_stroke_retracts(build):
    s = build()
    bite(s)
    t0 = time.time()
    while not (s.fsm._rim_scoop and s.fsm._rim_scoop.stage == "INTERIOR_SCOOP") and time.time() - t0 < 30:
        time.sleep(0.003)
    for j in LIFT:                                                 # the spoon meets something mid-stroke
        s.backend.jam(s.cfg.arm.servo_map[j], True)
    while s.fsm._rim_scoop and s.fsm._rim_scoop.stage == "INTERIOR_SCOOP" and time.time() - t0 < 40:
        time.sleep(0.003)
    for j in LIFT:
        s.backend.jam(s.cfg.arm.servo_map[j], False)
    refused(s, "rim_contact")
    assert events(s, "rim_contact")[-1][1]["detail_stage"] == "INTERIOR_SCOOP"
    assert stages(s)[-1] == "RETRACT_ABORT" and "RETRACT_AND_VERIFY" not in stages(s)
    assert s.fsm.rim_blocked


def test_stop_during_a_rim_scoop_stops_as_always(build):
    s = build()
    bite(s)
    t0 = time.time()
    while not (s.fsm._rim_scoop and s.fsm._rim_scoop.stage == "GUARDED_DESCENT") and time.time() - t0 < 30:
        time.sleep(0.005)
    s.fsm.post("stop", source="test")
    wait_state(s, "STOPPED")
    time.sleep(0.5)
    n = len(s.rec.rows)
    time.sleep(0.5)
    hold = s.rec.rows[n][3]
    assert all(max(abs(r[3][j] - hold[j]) for j in ARM_JOINTS) < 1.0 for r in s.rec.rows[n:])
    assert bites(s)[-1]["outcome"] == "aborted" and s.fsm.state.value == "STOPPED"


# ---------------------------------------------------------------- what is known about the cup
def test_unconfirmed_cup_is_refused_without_moving(build):
    s = build(cup={**CUP, "confirmed": False})
    bite(s)
    refused(s, "not_confirmed")
    assert never_moved(s) and stages(s) == ["OBSERVE"]
    assert s.fsm.rim.calls == 0                                    # not even looked at


def test_unmeasured_rim_uses_the_high_default_and_says_so(build):
    s = build(cup={**CUP, "rim_mm": None, "radius_mm": 45.0}, default_rim_mm=50.0)
    bite(s)
    wait_state(s, "LIFTING", timeout=40)
    assert events(s, "rim_unmeasured")[-1][1]["assumed_rim_mm"] == 50.0
    plan = events(s, "rim_plan")[-1][1]
    assert plan["rim_mm"] == 50.0 and plan["rim_measured"] is False


def test_unmeasured_rim_with_no_default_is_refused(build):
    s = build(cup={**CUP, "rim_mm": None}, default_rim_mm=0.0)
    bite(s)
    refused(s, "unmeasured")
    assert never_moved(s) and not events(s, "rim_plan")


def test_cup_too_uncertain_for_the_spoon_is_refused(build):
    s = build(cup={**CUP, "uncertainty_mm": 25.0})                  # a position known only from the photo
    bite(s)
    refused(s, "too_narrow")
    assert never_moved(s)
    assert "The spoon doesn't fit the applesauce cup safely. Not scooping." in spoken(s)


# ---------------------------------------------------------------- the old path
@pytest.mark.parametrize("how", ["no_cup", "rim_off"])
def test_bowl_without_a_cup_still_uses_the_taught_scoop(build, how):
    s = build(cup=None) if how == "no_cup" else build(enabled=False)
    bite(s)
    wait_state(s, "LIFTING", timeout=20)
    assert not events(s, "rim_stage") and events(s, "attack_plan") and events(s, "scoop_style")
    assert s.fsm.rim.calls == 0


def test_legacy_probe_touch_above_the_taught_dip_ends_the_bite_instead_of_pressing_down(build):
    s = build(cup=None, probe=True)
    bite(s)
    t0 = time.time()
    while s.fsm.motion_step < 1 and time.time() - t0 < 10:         # the slow descent has begun
        time.sleep(0.02)
    time.sleep(0.8)                                                 # ~8 mm down: still above the taught dip
    for j in LIFT:
        s.backend.jam(s.cfg.arm.servo_map[j], True)
    while s.fsm.motion_step == 1 and time.time() - t0 < 20:
        time.sleep(0.005)
    t_touch = time.time()
    held = K.fk(s.cfg.kinematics, s.rec.rows[-1][3])[2]
    for j in LIFT:
        s.backend.jam(s.cfg.arm.servo_map[j], False)
    refused(s, "touch_above_food")
    ev = [d for _, d in events(s, "scoop_probe")]
    assert "lifting out" in ev[-1]["note"] and ev[-1]["detail_touch_mm"] > 0
    assert not any("scoop_at_mm" in d for d in ev)
    assert min(t[2] for t in s.rec.tips("cmd", since=t_touch)) >= held - 1.5   # nothing lower after the touch
    assert "I felt something above the food. Stopping this bite." in spoken(s)
