"""The staged scoop for a bowl whose cup is known (BowlCfg.cup): one RimScoop per bite, driven by FSM._in_scooping.

  OBSERVE             arm still. Is the cup confirmed, does the camera still see the opening where it was (or
                      moved by a measured amount), is there an entry that keeps the whole spoon clear of the cup?
                      Any "no" ends the bite here. There is no fallback to the saved scoop path.
  SAFE_HOVER          over the entry point, whole spoon above the rim.
  VISUAL_ALIGNMENT    look once more from above (a cup that moved is replanned, at most twice), then move the
                      command until the MEASURED tip (encoders) is over the planned point. Servos sag under load, so
                      the commanded pose alone puts the spoon low and short.
  ENTRY_ORIENTATION   pivot to the entry angle above the rim, check the measured spoon against the cup, line up again.
  GUARDED_DESCENT     short steps, each a probe move. Resistance before the spoon is well inside the cup is the rim:
                      stop. Resistance well inside is the food or the bottom: scoop from there, never lower.
  INTERIOR_SCOOP      the drag, replanned at the depth actually reached, with a tight contact limit.
  RETRACT_AND_VERIFY  straight up and clear, check the measured spoon is above the rim, hand over to LIFTING.
  RETRACT_ABORT       after a contact or a failed check: nothing further down or sideways. Straight up from where
                      the spoon really is, back the way it came, and the FSM retreats.

Every wait is a state plus a time: tick() never sleeps. A Contact inside the face zone, a denied motion or any other
motion error goes to FSM._handle_motion_failure exactly as for the old scoop. Nothing here has been run on the real
arm; the thresholds come from config (plate.rim) and are untested guesses until the empty-cup check is done.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

from ..arm import kinematics as K
from ..arm.arm import Arm
from ..arm.contact import Contact
from ..arm.cup import Cup, NoSafeEntry, Spoon, check_path, clearance, plan_entry, spoon_body
from ..arm.scoop_style import _rz, solve_rz
from ..arm.trajectory import Plan, Segment, check_limits, plan
from ..config import ARM_JOINTS

STAGES = ("OBSERVE", "SAFE_HOVER", "VISUAL_ALIGNMENT", "ENTRY_ORIENTATION", "GUARDED_DESCENT", "INTERIOR_SCOOP",
          "RETRACT_AND_VERIFY", "RETRACT_ABORT")


def rim_bowl(cfg, bowl: Optional[str]):
    """The bowl's config if it is scooped by the staged pipeline (it has a cup and plate.rim is on), else None."""
    if not bowl or not cfg.plate.rim.enabled:
        return None
    return next((b for b in cfg.plate.bowls if b.name == bowl and b.cup is not None), None)


DIG_STRATEGIES = ("dig", "scrape")
# Off: with this spoon (246 mm from the wrist) the elbow would have to fold 20-50 deg past its range to hold the
# spoon level near the table, so a dig ends with the spoon still 30 deg nose-down and loose food slides off as it
# lifts (seen with rice). Every food is dipped and drawn instead, which holds anything that clings to the spoon.
DIG_ENABLED = True          # the dig is now a push along the spoon's own line (arm/cup.py), which needs no levelling
DEPTHS_BELOW_RIM = (45.0, 35.0, 25.0)       # tried in turn when a cup is too tall to be scooped at its bottom
DIG_SHARE = 0.45            # a dig pushes across this share of the opening, at most DIG_MAX_MM
DIG_MAX_MM = 45.0


def want_depth(cup_cfg, rc) -> Optional[float]:
    """How far below the rim the tip should go: under the food's surface when that was measured, else None (as
    deep as the cup allows)."""
    if cup_cfg is None or cup_cfg.food_below_rim_mm is None:
        return None
    return float(cup_cfg.food_below_rim_mm) + float(rc.bite_depth_mm)


def plan_deepest(kin, limits, seed, cup, spoon, depth, **kw):
    """plan_entry at the wanted depth; if the cup cannot be entered that deep (a tall tub: the handle would meet
    the rim), the deepest of a few shallower dips that can. Every plan returned passed the same checks."""
    tries = [depth] + [d for d in DEPTHS_BELOW_RIM if depth is None or d < depth]
    first = None
    for d in tries:
        try:
            return plan_entry(kin, limits, seed, cup, spoon, want_depth_mm=d, **kw)
        except NoSafeEntry as e:
            first = first or e
            if e.reason in ("unmeasured", "too_narrow"):
                break
    raise first


def dig_mm(attack: dict, cup) -> float:
    """How far a digging stroke should push across this cup (0 = this food is not dug: drag as usual)."""
    if not DIG_ENABLED or attack.get("strategy") not in DIG_STRATEGIES:
        return 0.0
    return min(DIG_MAX_MM, DIG_SHARE * 2 * cup.radius)


def _r(v, n: int = 1):
    return [round(float(x), n) for x in v] if isinstance(v, (list, tuple)) else \
        {k: round(float(x), n) for k, x in v.items()}


class RimScoop:
    SETTLE_S = 0.4              # after a move, before the encoders are trusted (readings averaged over the 2nd half)
    RISE_STEP_MM = 10.0         # the straight-up retract is built in steps this long
    BACKOFF_MM = 4.0            # the stroke runs this far above where the food (or the bottom) was felt
    DIG_BACKOFF_MM = 2.0        # ...a dig stays closer to it: it has to get under the food
    Z_GAIN = 0.6                # share of a height error corrected per try while lining up (see _align)
    STEP_GAIN = 0.5             # share of a descent step's measured error that corrects the next command (the
                                # encoders are good to a few mm at the tip; chasing every reading makes it worse)
    MAX_REPLANS = 2
    FOLLOW_ERR_FRAC, FOLLOW_ERR_MM = 0.15, 2.0   # a followed move is known this much worse (perception/rim.py sweeps)

    def __init__(self, fsm, bowl: str):
        self.fsm, self.bowl, self.cfg = fsm, bowl, fsm.cfg
        self.rc, self.kin = fsm.cfg.plate.rim, fsm.cfg.kinematics
        self.bcfg = rim_bowl(fsm.cfg, bowl)
        food = fsm.plate.food_of(bowl) if fsm.plate is not None else None
        self.name = f"{food} cup" if food else "cup"
        self.spoon = Spoon.from_cfg(fsm.cfg.spoon)
        self.stage, self.sub = "", 0
        self.cup: Optional[Cup] = None
        self.plan = None
        self.attack: dict = {}
        self.obs = None                         # last sighting of the opening (Observation, status ok or moved)
        self.seen_at: Optional[float] = None    # when (FSM clock)
        self.fix = (0.0, 0.0, 0.0)              # mm added to every commanded tip so the measured tip is on the plan
        self.moved = False                      # has this scoop commanded the arm at all
        self.replans = 0
        self.trail: list[dict] = []             # poses from the start up to (not including) the hover: the way back
        self.top: dict = {}                     # the pose the pivot aims the measured spoon at (plan.oriented or lower)
        self.steps: list[dict] = []
        self.i = 0
        self.outcome = ""
        self.meas: dict = {}                    # settled, averaged joint readings
        self._pending = False
        self._settle_at: Optional[float] = None
        self._samples: list[dict] = []
        self._m = None                          # the motion just run
        self._target: dict = {}
        self._lag: dict = {}                    # commanded - measured, last tick of the motion (for rim_contact)
        self._cmd: dict = {}
        self._lag0: dict = {}
        self._iters, self._prev_err = 0, None
        self._free_stalls = 0
        self._cup_stalls = 0
        self._frame0 = None
        self._result = None
        self._enter("OBSERVE")

    # ------------------------------------------------------------------ plumbing
    def tick(self, now: float):
        """None while working; "done" (go to LIFTING); ("abort", outcome) (go to RETREAT)."""
        if self._result is not None:
            return self._result
        arm = self.fsm.arm
        if self._pending:
            r = self.fsm._motion_result()
            st = arm.read()
            if r is None:
                self._cmd = dict(arm.setpoint)
                self._lag = {j: self._cmd[j] - st.deg[j] for j in ARM_JOINTS if j in self._cmd and j in st.deg}
                return None
            self._pending = False
            if r != "ok":
                err = self._m.error if self._m is not None else None
                if not isinstance(err, Contact) or err.in_face_zone or self.stage == "RETRACT_ABORT":
                    self.fsm._handle_motion_failure(r, now)     # as for any other motion: fault and park / retreat
                    return None
                self.meas = {j: st.deg[j] for j in ARM_JOINTS}
                self._contact(err, "contact")
                return self._result
            self._settle_at, self._samples = now + self.SETTLE_S, []
            return None
        if self._settle_at is not None:
            if now >= self._settle_at - self.SETTLE_S / 2:
                self._samples.append(arm.read().deg)
            if now < self._settle_at:
                return None
            self._settle_at = None
        s = self._samples or [arm.read().deg]
        self.meas = {j: sum(d[j] for d in s) / len(s) for j in ARM_JOINTS}
        self._samples = []
        getattr(self, "_st_" + self.stage.lower())(now)
        return self._result

    def _enter(self, stage: str) -> None:
        self.stage, self.sub, self._iters, self._prev_err = stage, 0, 0, None
        self.fsm.log.event("rim_stage", detail_stage=stage, bowl=self.bowl)

    def _go(self, p: Plan, target: dict) -> None:
        self._m = self.fsm.motion = self.fsm.arm.run(p)
        self._pending, self.moved, self._target = True, True, dict(target)

    def _run(self, poses: list[dict], name: str, speed: float = 1.0, lag_limit: Optional[float] = None) -> bool:
        try:
            p = plan(self.cfg, self.fsm.arm.current(), [(poses, "normal", name)], speed, name)
        except Exception as e:                      # outside the soft limits
            self._abort("limits", error=str(e)[:160])
            return False
        p.lag_limit_deg = lag_limit
        self._go(p, poses[-1])
        return True

    def _line(self, poses: list[dict], name: str, mm_s: float, **kw) -> bool:
        """A constant-speed move through poses at mm_s (tip speed), like the probe descent."""
        cur = self.fsm.arm.current()
        path = [cur] + [{j: float(q[j]) for j in ARM_JOINTS} for q in poses]
        try:
            for q in path[1:]:
                check_limits(self.cfg, q, name + " ")
        except Exception as e:
            self._abort("limits", error=str(e)[:160])
            return False
        mm = sum(K.tip_distance(self.kin, a, b) for a, b in zip(path, path[1:]))
        self._go(Plan([Segment(path, "normal", max(0.2, mm / max(1.0, mm_s)), name, linear=True)], name, **kw), path[-1])
        return True

    def _shifted(self, q: dict, fix) -> Optional[dict]:
        """q with the tip moved by fix (dx, dy, dz) at the same spoon angle: what to command so that the measured
        tip lands where q's tip is. None when the arm cannot reach it within its limits."""
        if not any(fix):
            return dict(q)
        x, y, z = K.fk(self.kin, q)
        x2, y2 = x + fix[0], y + fix[1]
        turn = math.degrees(math.atan2(y2, x2) - math.atan2(y, x))
        out = solve_rz(self.kin, {**q, "base": q["base"] + turn / self.kin.sign["base"]}, math.hypot(x2, y2),
                       z + fix[2], iters=25)
        try:
            if out is not None:
                check_limits(self.cfg, {j: out[j] for j in ARM_JOINTS})
        except Exception:
            return None
        return out

    def _path_ok(self, path: list[dict]):
        """Is the move from where the spoon really is through `path` clear of the cup by clearance_mm? A start that
        is already closer than that (readings are good to a few mm) must at least not get any closer."""
        here = clearance(self.kin, self.meas, self.cup, self.spoon).mm
        need = max(0.0, min(self.rc.clearance_mm, here - 0.5))
        return check_path(self.kin, [self.meas, *path], self.cup, self.spoon, need)

    def _say(self, text: str) -> None:
        self.fsm.speaker.say_text(text)

    def _tip_err(self, goal: dict) -> tuple[float, float, float]:
        a, b = K.fk(self.kin, goal), K.fk(self.kin, self.meas)
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def _view_ok(self, now: float) -> bool:
        """Is the last sighting of the opening still young enough to act on? (The arm may be hiding the cup.)"""
        return not self.rc.require_view or (self.seen_at is not None and now - self.seen_at <= self.rc.view_valid_s)

    # ------------------------------------------------------------------ ending early
    def _abort(self, reason: str, say: str = "", **detail) -> None:
        """Halt. If the spoon is in or just over the cup, straight up first; then back the way it came."""
        if self.outcome:
            return
        self.outcome = reason
        self.fsm.log.event("rim_abort", detail_reason=reason, bowl=self.bowl, stage=self.stage, **detail)
        if say:
            self._say(say)
        arm = self.fsm.arm
        arm.hold("rim abort")
        self.fsm.motion, self._pending, self._settle_at = None, False, None
        q = {j: arm.read().deg[j] for j in ARM_JOINTS}
        near = self.cup is not None and clearance(self.kin, q, self.cup, self.spoon).mm < 0.8 * self.rc.hover_above_rim_mm
        if near and self._rise(q):
            return
        self._leave(q)

    def _rise(self, q: dict) -> bool:
        """Straight up from the measured pose q (same reach, same spoon angle) until the whole spoon is hover height
        over the rim, or as high as the arm can hold that. False if no move was started."""
        low = min(p.z for p in spoon_body(self.kin, q, self.spoon))
        dz = self.cup.rim + self.rc.hover_above_rim_mm - low
        if not dz > 1.0:
            return False
        r, z = _rz(self.kin, q)
        n, poses, cur = int(math.ceil(dz / self.RISE_STEP_MM)), [], q
        for k in range(1, n + 1):
            up = solve_rz(self.kin, cur, r, z + dz * k / n, iters=25)
            cmd = self._shifted(up, self.fix) if up is not None else None
            if cmd is None:
                break
            poses.append(cmd)
            cur = up
        if not poses:            # the arm cannot go straight up from here at this spoon angle (it is already near the
            self.fsm.log.event("rim_rise_skipped", detail_bowl=self.bowl)     # top of its reach): try the checked way out
            return False
        self._enter("RETRACT_ABORT")
        try:
            p = plan(self.cfg, self.fsm.arm.current(), [(poses, "normal", "rim_retract")], 0.5, "rim_retract")
        except Exception as e:                      # cannot lift straight out: stop here and get the operator
            self.fsm._fault("rim_retract_failed", self.fsm.last_tick, {"error": str(e)[:160]})
            return True
        self.sub = "rise"
        self._go(p, poses[-1])
        return True

    def _leave(self, q: dict) -> None:
        """Back to where the bite started, over the hover and the approach in reverse (a path the planner checked)."""
        if self.plan is not None and self.moved and self.trail:
            path = [self._shifted(self.plan.hover, self.fix) or self.plan.hover, *reversed(self.trail)]
            chk = check_path(self.kin, [q, *path], self.cup, self.spoon, self.rc.clearance_mm)
            if chk.ok:
                try:
                    p = plan(self.cfg, self.fsm.arm.current(), [(path, "normal", "rim_leave")], 1.0, "rim_leave")
                    if self.stage != "RETRACT_ABORT":
                        self._enter("RETRACT_ABORT")
                    self.sub = "leave"
                    self._go(p, path[-1])
                    return
                except Exception as e:
                    self.fsm.log.event("rim_leave_skipped", detail_error=str(e)[:160])
            else:
                self.fsm.log.event("rim_leave_skipped", detail_min_mm=round(chk.min_mm, 1), part=chk.part)
        if self.cup is not None and self.moved and \
                clearance(self.kin, q, self.cup, self.spoon).mm < 0.8 * self.rc.hover_above_rim_mm:
            # still in or just over the cup with no checked way out: the retreat's sweep to rest could take the cup
            # with it. Stop here and get the operator (as for any contact away from the face).
            self.fsm._fault("rim_retract_failed", self.fsm.last_tick, {"reason": self.outcome})
            return
        self._finish()

    def _finish(self) -> None:
        if not self.moved:
            self.fsm._retreat_in_place = True       # refused before any move: the arm stays where it is
        self._result = ("abort", self.outcome)

    def _st_retract_abort(self, now: float) -> None:
        if self.sub == "rise":
            self._leave(self.meas)
        else:
            self._finish()

    def _contact(self, c: Optional[Contact], why: str) -> None:
        """The spoon met something it should not have. Record everything, block this cup, retract."""
        q, cl = self.meas, None
        if self.cup is not None:
            cl = clearance(self.kin, q, self.cup, self.spoon)
        s = self.obs.sighting if self.obs is not None else None
        self.fsm.log.event(
            "rim_contact", detail_stage=self.stage, bowl=self.bowl, why=why,
            joint=c.joint if c else None, err_deg=round(c.err_deg, 1) if c else None,
            grew_deg=round(c.grew_deg, 1) if c and c.grew_deg is not None else None,
            joint_err_deg=_r(self._lag), commanded=_r(self._cmd), target=_r(self._target), measured=_r(q),
            measured_tip=_r(K.fk(self.kin, q)), clearance_mm=round(cl.mm, 1) if cl else None,
            clearance_part=cl.part if cl else None,
            sighting=(dataclasses.asdict(s) if dataclasses.is_dataclass(s) else s),
            seen=None if self.obs is None else {"status": self.obs.status, "dx": self.obs.dx, "dy": self.obs.dy,
                                                "confidence": self.obs.confidence, "stamp": self.obs.stamp})
        self.fsm.rim_blocked[self.bowl] = self._key(self.cup)
        self._abort("rim_contact", say=f"I touched the {self.name}. Stopping this bite.")

    @staticmethod
    def _key(cup: Cup) -> tuple:
        return (round(cup.x, 2), round(cup.y, 2), round(cup.radius, 2), round(cup.rim, 2))

    # ------------------------------------------------------------------ OBSERVE
    def _observe(self, now: float):
        """Ask the camera. Returns the Observation when the opening is seen (ok or moved), else None."""
        rim, obs = self.fsm.rim, None
        if rim is not None:
            try:
                obs = rim.observe(self.bowl)
            except Exception as e:                  # the observer promises not to raise; a broken one is "not seen"
                self.fsm.log.event("rim_view_error", detail_error=repr(e)[:160])
        seen = obs is not None and obs.status in ("ok", "moved") and obs.confidence >= self.rc.min_confidence
        if seen and obs.status == "moved" and math.hypot(obs.dx, obs.dy) > self.rc.max_follow_mm:
            seen = False
        self.fsm.log.event("rim_view", detail_bowl=self.bowl, stage=self.stage, seen=seen,
                           status=getattr(obs, "status", "no_observer"), reason=getattr(obs, "reason", ""),
                           dx=round(getattr(obs, "dx", 0.0), 1), dy=round(getattr(obs, "dy", 0.0), 1))
        if not seen:
            return None
        self.obs, self.seen_at = obs, now
        if not self.moved:                          # the arm has not left its rest: the picture to compare with later
            self._frame0 = self._grab()
        return obs

    def _grab(self):
        try:
            img = self.fsm.rim.frame_fn()
            return None if img is None else img.copy()
        except Exception:
            return None

    def _spoon_seen(self) -> bool:
        """At the hover, lined up by the encoders: does the camera agree the spoon is over this opening? Logged
        always; ends the bite only when plate.rim.spoon_view_required and the camera clearly sees it elsewhere."""
        from ..perception.rim import spoon_over
        s = self.obs.sighting if self.obs is not None else None
        if s is None or self._frame0 is None:
            self.fsm.log.event("rim_spoon_view", detail_bowl=self.bowl, seen=False, why="no picture from before")
            return True
        try:
            v = spoon_over(self._frame0, self._grab(), s)
        except Exception as e:
            v = {"seen": False, "why": repr(e)[:120]}
        self.fsm.log.event("rim_spoon_view", detail_bowl=self.bowl, **v)
        if self.rc.spoon_view_required and v.get("seen") and not v.get("over"):
            self._abort("spoon_not_over", say=f"The camera doesn't see the spoon over the {self.name}. Stopping this bite.",
                        off=v.get("off"))
            return False
        return True

    def _follow(self, cup: Cup, obs) -> Cup:
        d = math.hypot(obs.dx, obs.dy)
        self._say(f"The {self.name.removesuffix(' cup')} moved. Following it")
        return cup.moved(obs.dx, obs.dy, uncertainty=cup.uncertainty + self.FOLLOW_ERR_FRAC * d + self.FOLLOW_ERR_MM,
                         source="followed", stamp=obs.stamp, confidence=obs.confidence)

    def _accept(self, obs) -> None:
        """A followed move is now where the cup is: in the running config (not saved to disk here) and as the
        camera's reference."""
        c, cup = self.bcfg.cup, self.cup
        c.x_mm, c.y_mm, c.uncertainty_mm, c.source = cup.x, cup.y, cup.uncertainty, "followed"
        try:
            self.fsm.rim.accept_move(self.bowl, obs)
        except Exception as e:
            self.fsm.log.event("rim_view_error", detail_error=repr(e)[:160])

    def _make_plan(self, cup: Cup, start: dict) -> bool:
        fsm, rc = self.fsm, self.rc
        attack = fsm.plate.attack_of(self.bowl)
        seed = {**start, **{j: float(v) for j, v in self.cfg.arm.home_pose.items()}}
        try:
            p = plan_deepest(self.kin, self.cfg.arm.soft_limits_deg, seed, cup, self.spoon, want_depth(self.bcfg.cup, rc),
                             clearance_mm=rc.clearance_mm, hover_above_rim_mm=rc.hover_above_rim_mm,
                             want_angle_deg=attack["entry_angle_deg"], want_drag_mm=attack["drag_mm"], start=start,
                             dig_mm=dig_mm(attack, cup), headroom_mm=rc.sag_headroom_mm)
        except NoSafeEntry as e:
            say = {"unmeasured": f"I don't know the size of the {self.name}. Not scooping.",
                   "too_narrow": f"The spoon doesn't fit the {self.name} safely. Not scooping."}.get(
                       e.reason, f"I can't find a safe way into the {self.name}. Not scooping.")
            self._abort(e.reason, say=say, why=e.detail[:160], rejected=len(e.rejected))
            return False
        if any(K.in_face_zone(self.kin, q) for q in [*p.approach, *p.poses()]):
            self._abort("near_face", say=f"The {self.name} is too close to you. Not scooping.")
            return False
        self.cup, self.plan, self.attack = cup, p, attack
        fsm.log.event("rim_plan", detail_bowl=self.bowl, cup_xy=_r((cup.x, cup.y)), radius_mm=cup.radius,
                      rim_mm=cup.rim, rim_measured=cup.rim_measured, uncertainty_mm=round(cup.uncertainty, 1),
                      source=cup.source, spoon_measured=self.cfg.spoon.measured,
                      entry_angle_deg=p.entry_angle_deg, entry_xy=_r(p.entry_xy), dip_z=round(p.dip_z, 1),
                      drag_mm=p.drag_mm, min_clearance_mm=round(p.min_clearance_mm, 1), cost=round(p.cost, 2),
                      approach=len(p.approach), want_angle_deg=attack["entry_angle_deg"],
                      want_drag_mm=attack["drag_mm"], strategy=attack.get("strategy"), replan=self.replans)
        return True

    def _to_hover(self, start: dict) -> None:
        self.trail += [dict(start), *self.plan.approach]
        self._enter("SAFE_HOVER")
        self._run([*self.plan.approach, self.plan.hover], "rim_hover")

    def _st_observe(self, now: float) -> None:
        fsm, rc, c = self.fsm, self.rc, self.bcfg.cup
        self.cup = cup = Cup.from_cfg(c, rc.default_rim_mm)
        if not math.isfinite(cup.rim):
            self.cup = None
        if rc.require_confirmed and not c.confirmed:
            return self._abort("not_confirmed", say=f"The {self.name} isn't lined up yet. Not scooping.")
        obs = self._observe(now)
        if obs is None and rc.require_view:
            return self._abort("cup_not_seen", say=f"I can't see the {self.name}. Not scooping.")
        if obs is not None and obs.status == "moved":
            cup = self._follow(cup, obs)
        if not cup.rim_measured:
            fsm.log.event("rim_unmeasured", detail_bowl=self.bowl, assumed_rim_mm=cup.rim)
        if math.isfinite(cup.rim) and fsm.rim_blocked.get(self.bowl) == self._key(cup):
            return self._abort("blocked", say=f"I touched the {self.name} last time. Please check it and line it up again.")
        fsm.rim_blocked.pop(self.bowl, None)       # the cup is somewhere new, or was confirmed again
        start = fsm.arm.current()
        if not self._make_plan(cup, start):
            return
        if obs is not None and obs.status == "moved":
            self._accept(obs)
        if fsm.bite:
            fsm.bite.texture, fsm.bite.attack = fsm.plate.texture_of(self.bowl)[0], self.attack.get("strategy")
        if not rc.validated_empty:
            fsm.log.event("unvalidated_trial", detail_bowl=self.bowl)
            if fsm._rim_trial_said != ("session", fsm.session.id):       # spoken once per session
                fsm._rim_trial_said = ("session", fsm.session.id)
                self._say("Trial scoop. Keep clear.")
        self._to_hover(start)

    # ------------------------------------------------------------------ SAFE_HOVER, VISUAL_ALIGNMENT
    def _st_safe_hover(self, now: float) -> None:
        self._enter("VISUAL_ALIGNMENT")
        self._st_visual_alignment(now)

    def _st_visual_alignment(self, now: float) -> None:
        if self.sub == 0:                           # look again from above, once
            self.sub = 1
            obs = self._observe(now) if self.fsm.rim is not None else None
            if obs is not None and obs.status == "moved":
                if self.replans >= self.MAX_REPLANS:
                    return self._abort("cup_moving", say=f"The {self.name} keeps moving. Stopping this bite.")
                self.replans += 1
                start = self.fsm.arm.current()
                if self._make_plan(self._follow(self.cup, obs), start):
                    self._accept(obs)
                    self._to_hover(start)
                return
            if obs is None and not self._view_ok(now):
                return self._abort("cup_not_seen", say=f"I lost sight of the {self.name}. Stopping this bite.")
        if self._align(self.plan.hover):
            if not self._spoon_seen():
                return
            self._enter("ENTRY_ORIENTATION")
            cmd = self._pivot()
            if cmd is None:
                return self._abort("align_failed", why="the corrected pivot is out of reach",
                                   say=f"I can't line the spoon up with the {self.name}. Stopping this bite.")
            # The pivot is a large joint move well above the rim. Its lag while moving is not a touch (on the real
            # arm the elbow trailed by 10 deg in free air), so only the arm's standing contact limits apply; slowly.
            self._run([cmd], "rim_orient", 0.35)

    def _pivot(self) -> Optional[dict]:
        """What to command for the pivot to the entry angle. The plan pivots as high as the arm can hold that angle,
        so a command corrected upward for sag can be out of reach. Then pivot lower on the same vertical line, as
        long as the whole spoon is still the full margin above the rim there. Sets self.top (where the measured
        tip should end up)."""
        p, kin, cup = self.plan, self.kin, self.cup
        margin = self.rc.clearance_mm + cup.uncertainty
        r, z = _rz(kin, p.oriented)
        q = p.oriented
        for drop in range(0, 45, 5):
            if drop:
                q = solve_rz(kin, q, r, z - drop, iters=25)
                if q is None or min(b.z for b in spoon_body(kin, q, self.spoon)) < cup.rim + margin \
                        or not check_path(kin, [p.hover, q], cup, self.spoon, margin).ok:
                    return None
            cmd = self._shifted(q, self.fix)
            if cmd is not None:
                if drop:
                    self.fsm.log.event("rim_pivot_lower", detail_bowl=self.bowl, lower_mm=drop)
                self.top = dict(q)
                return cmd
        return None

    def _align(self, goal: dict) -> bool:
        """True when the measured tip is within align_tol_mm of goal's tip. Otherwise one corrected move is started
        (or the scoop is aborted: the correction is not converging, or more of it is needed than allowed)."""
        rc = self.rc
        e = self._tip_err(goal)
        err = math.hypot(e[0], e[1])        # across the table: this is what keeps the spoon off the rim
        z_ok = -rc.align_high_mm <= e[2] <= rc.align_low_mm      # height: e[2] > 0 = the spoon is lower than planned
        self.fsm.log.event("rim_align", detail_stage=self.stage, iter=self._iters, err_mm=round(err, 1), err=_r(e),
                           fix=_r(self.fix), height_ok=z_ok)
        if err <= rc.align_tol_mm and z_ok:
            return True
        say = f"I can't line the spoon up with the {self.name}. Stopping this bite."
        prev = self._prev_err
        # Across the table each try must gain 25% or 1 mm. Height is not held to that: the loaded joints stick and
        # overshoot by several mm (seen on the arm: 18 low, then 6 high, then 7 low), so it is corrected gently
        # (Z_GAIN) and only has to end inside its band.
        if prev is not None and err > rc.align_tol_mm and err > max(0.75 * prev, prev - 1.0):
            self._abort("align_diverged", say=say, err_mm=round(err, 1), before_mm=round(prev, 1))
            return False
        fix = (self.fix[0] + e[0], self.fix[1] + e[1], self.fix[2] + (0.0 if z_ok else self.Z_GAIN * e[2]))
        cmd = self._shifted(goal, fix) if math.hypot(*fix) <= rc.align_max_fix_mm else None
        if self._iters >= rc.align_max_iters or cmd is None:
            self._abort("align_failed", say=say, err_mm=round(err, 1), fix_mm=round(math.hypot(*fix), 1),
                        iters=self._iters)
            return False
        self.fix, self._iters, self._prev_err = fix, self._iters + 1, err
        self._run([cmd], "rim_align", 0.5)
        return False

    # ------------------------------------------------------------------ ENTRY_ORIENTATION
    def _spoon_clear(self, need_mm: float) -> bool:
        cl = clearance(self.kin, self.meas, self.cup, self.spoon)
        if cl.mm >= need_mm:
            return True
        self._abort("clearance", say=f"The spoon is too close to the {self.name}. Stopping this bite.",
                    measured_mm=round(cl.mm, 1), part=cl.part, need_mm=need_mm)
        return False

    def _st_entry_orientation(self, now: float) -> None:
        if not self._spoon_clear(self.rc.clearance_mm) or not self._align(self.top):
            return
        self._enter("GUARDED_DESCENT")
        z_top = _rz(self.kin, self.top)[1]             # the plan's steps, from where the pivot really was
        self.steps = [self.top] + [q for q in self.plan.descent(self.rc.descent_step_mm)[1:]
                                   if _rz(self.kin, q)[1] < z_top - 1.0]
        self.i = 0
        self._lag0 = self._lag_now()
        self._descend(now)

    # ------------------------------------------------------------------ GUARDED_DESCENT
    def _lag_now(self) -> dict:
        sp = self.fsm.arm.setpoint
        return {j: sp[j] - self.meas[j] for j in Arm.PROBE_JOINTS}

    def _food_extra(self) -> float:
        """Extra lag allowed inside the cup for food that has to be pushed through (thick or firm): the
        allowance set for this food (attack contact_extra_deg), never more than 5 deg. Soft food has none."""
        if self.plan is None:
            return 0.0
        return max(0.0, min(5.0, float(self.attack.get("contact_extra_deg", 0.0) or 0.0)))

    def _inside(self, tip) -> bool:
        return math.hypot(tip[0] - self.cup.x, tip[1] - self.cup.y) <= self.cup.radius - self.rc.clearance_mm

    def _descend(self, now: float) -> None:
        rc = self.rc
        if self.i >= len(self.steps) - 1:
            return self._stroke(self.plan.dip_z)       # as low as planned without meeting anything
        if not self._view_ok(now):
            return self._abort("cup_not_seen", say=f"I lost sight of the {self.name}. Stopping this bite.")
        nxt = self.steps[self.i + 1]
        chk = self._path_ok([nxt])
        if not chk.ok:
            return self._abort("clearance", predicted_mm=round(chk.min_mm, 1), part=chk.part, step=self.i + 1)
        cmd = self._shifted(nxt, self.fix)
        if cmd is None:
            return self._abort("limits", step=self.i + 1)
        tip = K.fk(self.kin, self.meas)
        in_food = tip[2] < self.cup.rim - self.spoon.bowl_depth and self._inside(tip)
        self._line([cmd], "rim_descent", rc.descent_speed_mm_s,
                   probe_deg=rc.stall_deg + (self._food_extra() if in_food else 0.0),
                   probe_settle_s=rc.descent_settle_s)

    def _st_guarded_descent(self, now: float) -> None:
        rc, cup = self.rc, self.cup
        lag = self._lag_now()
        grew = max(abs(lag[j] - self._lag0[j]) for j in lag)     # against the lag before the descent: a press that
        tip = K.fk(self.kin, self.meas)                          # builds up over several steps still shows
        if self._m.touched is not None or grew >= rc.press_deg:
            wall = clearance(self.kin, self.meas, dataclasses.replace(cup, floor=-1e3), self.spoon)
            low = min(b.z for b in spoon_body(self.kin, self.meas, self.spoon))
            if self._m.touched is not None and grew < rc.press_deg and low >= cup.rim                     and wall.mm >= rc.free_air_mm and self._free_stalls < 2:
                # The joints move in small jumps on a slow descent and a pause reads as a stall. With the whole
                # spoon still above the rim and well clear of the walls it cannot be the cup: try the step again.
                self._free_stalls += 1
                self.fsm.log.event("rim_free_air_stall", detail_bowl=self.bowl, tip=_r(tip), low_mm=round(low, 1),
                                   wall_mm=round(wall.mm, 1), lag_grew_deg=round(grew, 1), n=self._free_stalls)
                return self._descend(now)
            inside = tip[2] < cup.rim - self.spoon.bowl_depth and self._inside(tip) and wall.mm >= rc.clearance_mm
            self.fsm.log.event("rim_touch", detail_bowl=self.bowl, tip=_r(tip), below_rim_mm=round(cup.rim - tip[2], 1),
                               lag_grew_deg=round(grew, 1), wall_mm=round(wall.mm, 1), inside=inside,
                               stalled=self._m.touched is not None)
            if not inside:
                return self._contact(None, "resistance before the spoon was inside the cup")
            firm = (self.fsm.bite.texture if self.fsm.bite else None) == "firm"
            if self._m.touched is not None and grew < rc.press_deg and not firm                     and self._cup_stalls < rc.in_cup_stall_retries and tip[2] > self.plan.dip_z + 2.0:
                # Inside the cup, clear of the walls, above the planned depth (which the floor margin already
                # bounds), in food that gives way: the slow joints pausing, or the surface of something thick.
                self._cup_stalls += 1
                self.fsm.log.event("rim_in_cup_stall", detail_bowl=self.bowl, tip=_r(tip), n=self._cup_stalls,
                                   lag_grew_deg=round(grew, 1))
                return self._descend(now)
            back = self.DIG_BACKOFF_MM if self.plan.drag_mm < 0 else self.BACKOFF_MM
            return self._stroke(tip[2] + back)                     # the food or the bottom: scoop here, never lower
        self.i += 1
        if not self._spoon_clear(0.0):
            return
        e = self._tip_err(self.steps[self.i])
        fix = tuple(a + self.STEP_GAIN * b for a, b in zip(self.fix, e))
        self.fsm.log.event("rim_step", detail_bowl=self.bowl, step=self.i, of=len(self.steps) - 1, tip=_r(tip),
                           err=_r(e), lag_grew_deg=round(grew, 1))
        if not self._inside(tip) or math.hypot(e[0], e[1]) > 2 * rc.align_tol_mm                 or math.hypot(*fix) > rc.align_max_fix_mm:        # sideways drift ends it; depth is felt, not chased
            return self._abort("off_path", say=f"I can't line the spoon up with the {self.name}. Stopping this bite.",
                               err=_r(e), step=self.i)
        self.fix = fix                                 # the next step is commanded so the measured tip stays on the line
        self._descend(now)

    # ------------------------------------------------------------------ INTERIOR_SCOOP
    def _stroke(self, z: float) -> None:
        """The drag at height z (where the descent ended), if it is clear from here; else straight to the way out."""
        p, kin = self.plan, self.kin
        self._enter("INTERIOR_SCOOP")
        self.fsm.log.event("rim_depth", detail_bowl=self.bowl, tip_z=round(z, 1), planned_z=round(p.dip_z, 1))
        if p.stroke:
            up = max(0.0, z - _rz(kin, p.dip)[1])      # the whole stroke this much higher than planned; each
            pts = []                                   # waypoint keeps its own reach and spoon angle
            for q in [p.dip, *p.stroke]:
                r_q, z_q = _rz(kin, q)
                pts.append(solve_rz(kin, q, r_q, z_q + up, iters=25))
            cmds = [self._shifted(q, self.fix) for q in pts] if None not in pts else [None]
            if None not in cmds and self._path_ok([*pts, p.out]).ok:
                speed = 2 * self.rc.descent_speed_mm_s * float(self.attack.get("speed", 1.0))
                self._line(cmds, "rim_stroke", speed, lag_limit_deg=self.rc.scoop_contact_deg + self._food_extra())
                return
            self.fsm.log.event("rim_stroke_skipped", detail_bowl=self.bowl, reachable=None not in cmds,
                               measured_mm=round(clearance(kin, self.meas, self.cup, self.spoon).mm, 1))
        self._retract()

    def _st_interior_scoop(self, now: float) -> None:
        lag = self._lag_now()                          # settled, against the settled lag before the descent
        grew = max(abs(lag[j] - self._lag0[j]) for j in lag)
        if grew >= self.rc.scoop_contact_deg + self._food_extra():     # the stroke ended pressing on something
            return self._contact(None, "resistance in the stroke")
        self._retract()

    # ------------------------------------------------------------------ RETRACT_AND_VERIFY
    def _retract(self) -> None:
        p, rc = self.plan, self.rc
        self._enter("RETRACT_AND_VERIFY")
        chk = self._path_ok([p.out, p.clear])
        out = self._shifted(p.out, self.fix)
        if not chk.ok or out is None:
            return self._abort("no_clear_path", predicted_mm=round(chk.min_mm, 1), part=chk.part)
        self._run([out], "rim_out", 0.35 * float(self.attack.get("speed", 1.0)), lag_limit=rc.out_contact_deg)

    def _st_retract_and_verify(self, now: float) -> None:
        p, rc, fsm = self.plan, self.rc, self.fsm
        if self.sub == 0:
            self.sub = 1
            self._run([self._shifted(p.clear, self.fix) or p.clear], "rim_clear")
            return
        low = min(b.z for b in spoon_body(self.kin, self.meas, self.spoon))
        if low < self.cup.rim + rc.clearance_mm:
            # Loaded and reaching out, the arm sags more than it did at the hover. Lift by what is missing (and a
            # little nearer the base if it cannot go higher out there), once, before giving the bite up.
            cmd = None
            if self.sub == 1:
                self.sub = 2
                need = self.cup.rim + rc.clearance_mm + 8.0 - low
                r0, z0 = _rz(self.kin, p.clear)
                for back in (0.0, 15.0, 30.0, 45.0):
                    q = solve_rz(self.kin, p.clear, r0 - back, z0, iters=25) if back else p.clear
                    c2 = self._shifted(q, (self.fix[0], self.fix[1], self.fix[2] + need)) if q is not None else None
                    if c2 is not None and self._path_ok([q]).ok:
                        cmd = c2
                        break
                self.fsm.log.event("rim_lift_more", detail_bowl=self.bowl, measured_low_mm=round(low, 1),
                                   need_mm=round(need, 1), found=cmd is not None)
            if cmd is not None:
                self._run([cmd], "rim_clear_up", 0.5)
                return
            return self._abort("not_clear", measured_low_mm=round(low, 1))
        # the lift path was taught for the old scoop: use only the part of it that keeps clear of this cup
        name = f"lift_{self.bowl}" if fsm.lib.has(f"lift_{self.bowl}") else "lift"
        lift, pre = fsm.lib.path(name), fsm.lib.pose("pre_delivery")
        for k in range(len(lift) + 1):
            path = [*lift[k:], pre]
            if check_path(self.kin, [self.meas, *path], self.cup, self.spoon, rc.clearance_mm + self.cup.uncertainty).ok:
                fsm._rim_lift = path
                fsm.log.event("rim_done", detail_bowl=self.bowl, fix=_r(self.fix), lift_from=k)
                self._result = "done"
                return
        self._abort("no_clear_lift")
