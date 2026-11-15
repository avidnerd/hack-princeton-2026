"""How hard to scoop, by food texture: jello gets a deeper, slower scoop with a saw; applesauce the taught one."""
import json
from pathlib import Path

import pytest

from conftest import wait_state
from test_fsm import start_listening
from spoonful.arm import kinematics as K
from spoonful.arm.contact import ContactMonitor
from spoonful.arm.scoop_style import MAX_DEPTH_MM, adapt_scoop, texture_from_name

SIM = Path(__file__).resolve().parents[1] / "data" / "trajectories_sim"


@pytest.mark.parametrize("food,tex", [
    ("applesauce", "soft"), ("apple sauce", "soft"), ("yogurt", "soft"), ("chocolate pudding", "soft"),
    ("mashed potatoes", "thick"), ("oatmeal", "thick"), ("peanut butter", "thick"),
    ("jello", "firm"), ("Jell-O cubes", "firm"), ("banana slices", "firm"), ("tofu", "firm"),
    ("chicken soup", "liquid"), ("tomato soup", "liquid"),
    ("mystery", None), ("", None), (None, None),
])
def test_texture_from_name(food, tex):
    assert texture_from_name(food) == tex


def _scoop(name="scoop_bowl1"):
    return [w["deg"] for w in json.loads((SIM / f"{name}.json").read_text())["waypoints"]]


def test_deeper_scoop_keeps_ends_and_spoon_angle(fast_cfg):
    kin, poses = fast_cfg.kinematics, _scoop()
    out, info = adapt_scoop(kin, poses, depth_mm=10)
    assert info["lowered"] >= 1 and info["depth_mm"] == 10
    assert out[0] == poses[0] and out[-1] == poses[-1]           # joins the rest of the bite unchanged
    z_in = min(K.fk(kin, p)[2] for p in poses[1:-1])
    z_out = min(K.fk(kin, p)[2] for p in out[1:-1])
    assert z_out == pytest.approx(z_in - 10, abs=0.6)
    for a, b in zip(poses, out):                                 # same reach and spoon tilt, just lower
        ra, rb = K.fk(kin, a), K.fk(kin, b)
        assert abs(((ra[0] ** 2 + ra[1] ** 2) ** .5) - ((rb[0] ** 2 + rb[1] ** 2) ** .5)) < 0.6
        sa = sum(kin.sign[j] * (a[j] - kin.zero_deg[j]) for j in ("shoulder", "elbow", "wrist_pitch"))
        sb = sum(kin.sign[j] * (b[j] - kin.zero_deg[j]) for j in ("shoulder", "elbow", "wrist_pitch"))
        assert sa == pytest.approx(sb, abs=0.3)


def test_depth_is_capped_and_saw_stays_level(fast_cfg):
    kin, poses = fast_cfg.kinematics, _scoop()
    out, info = adapt_scoop(kin, poses, depth_mm=80, wiggle=2)
    assert info["depth_mm"] == MAX_DEPTH_MM and info["wiggle"] == 2
    assert len(out) == len(poses) + 5                             # 2 strokes x 2 + return
    z_low = min(K.fk(kin, p)[2] for p in out)
    assert z_low >= min(K.fk(kin, p)[2] for p in poses) - MAX_DEPTH_MM - 0.6
    i = min(range(1, len(poses) - 1), key=lambda k: K.fk(kin, poses[k])[2])
    saw = out[i:i + 6]
    assert max(K.fk(kin, p)[2] for p in saw) - min(K.fk(kin, p)[2] for p in saw) < 0.5   # sideways, not down


def test_soft_food_is_unchanged(fast_cfg):
    poses = _scoop()
    out, info = adapt_scoop(fast_cfg.kinematics, poses, 0, 0)
    assert out == [dict(p) for p in poses] and info["lowered"] == 0


def test_contact_extra_allowance_is_capped(fast_cfg):
    c = ContactMonitor(fast_cfg)
    assert c.allowance(0, 4) == pytest.approx(fast_cfg.arm.contact_error_deg + 4)
    assert c.allowance(0, 50) == pytest.approx(fast_cfg.arm.contact_error_deg + 5)
    assert c.allowance(0, -3) == pytest.approx(fast_cfg.arm.contact_error_deg)


def test_texture_precedence(sim_system):
    p = sim_system.plate
    p.set_label("bowl1", "jello")
    assert p.texture_of("bowl1") == ("firm", "food name")
    p.set_label("bowl1", "", texture="soft")                    # operator picks the texture
    assert p.texture_of("bowl1") == ("soft", "operator") and p.food_of("bowl1") == "jello"
    p.set_label("bowl1", "", texture="")                        # back to automatic
    assert p.texture_of("bowl1") == ("firm", "food name")
    with pytest.raises(ValueError):
        p.set_label("bowl1", "", texture="crunchy")
    p.set_label("bowl2", "")
    p.manual.clear()
    p.vision_name = "gemini"
    p.apply([{"bowl": "bowl2", "food": "mystery dish", "amount": "full", "texture": "thick"}])
    assert p.texture_of("bowl2") == ("thick", "vision")         # unknown food: vision's guess
    p.apply([{"bowl": "bowl2", "food": "green jelly", "amount": "full", "texture": "soft"}])
    assert p.texture_of("bowl2") == ("firm", "food name")       # known food: the name wins over vision


# ---------------------------------------------------------------- end to end in the sim
def _bite(s, food_text):
    start_listening(s)
    s.on_voice(None, food_text, 0.9)
    wait_state(s, "SCOOPING")
    return s.fsm.motion


def test_jello_is_stabbed_and_sawn_applesauce_is_skimmed(sim_system):
    """Strategy by food: jelly -> stab_and_saw (steep entry, saw strokes, slower, more push allowed);
    applesauce -> skim (shallow entry, no saw)."""
    from spoonful.arm.scoop_style import _spoon_angle
    s = sim_system
    s.plate.set_label("bowl1", "applesauce")
    s.plate.set_label("bowl2", "jello")
    orig_run = s.arm.run
    plans = []
    s.arm.run = lambda p: plans.append(p) or orig_run(p)
    kin = s.cfg.kinematics

    def dip_angle(p):
        pts = [q for seg in p.segments if not seg.label.startswith("blend") for q in seg.path]
        low = min(pts, key=lambda q: K.fk(kin, q)[2])
        return abs(_spoon_angle(kin, low)) - 90               # degrees below horizontal at the bottom of the scoop

    _bite(s, "Spoonful, some jello please")
    assert s.fsm.bite.texture == "firm" and s.fsm.bite.attack == "stab_and_saw"
    firm = plans[-1]
    assert firm.contact_extra_deg == 4
    s.log.flush()
    ev = s.log.query("SELECT detail FROM events WHERE type='attack_plan' ORDER BY rowid")
    assert ev and "stab_and_saw" in ev[-1]["detail"]

    s.fsm.post("stop", source="key"); wait_state(s, "STOPPED")
    s.fsm.post("stop", source="key"); s.fsm.post("reset"); wait_state(s, "IDLE")
    _bite(s, "Spoonful, applesauce")
    assert s.fsm.bite.texture == "soft" and s.fsm.bite.attack == "skim"
    soft = plans[-1]
    assert soft.contact_extra_deg == 0
    assert firm.duration > soft.duration                         # slower, plus the saw strokes
    assert dip_angle(firm) > dip_angle(soft) + 10                # steeper entry for jelly


def test_attack_plans_are_clamped_and_low_bowls_are_scraped(fast_cfg):
    from spoonful.arm.scoop_style import default_strategy, verify_strategy
    v = verify_strategy({"strategy": "stab_and_saw", "entry_angle_deg": 95, "depth_mm": 40, "saw_strokes": 9, "speed": 3})
    assert v["entry_angle_deg"] == 70 and v["depth_mm"] == 15 and v["saw_strokes"] == 3 and v["speed"] == 1.0
    assert verify_strategy({"strategy": "fling"}, "firm")["strategy"] == "stab_and_saw"      # unknown -> rules
    assert verify_strategy(None, "liquid")["strategy"] == "ladle"
    assert default_strategy("soft", "low")["strategy"] == "scrape"


def test_vision_plans_the_attack_from_a_close_up(sim_system):
    s = sim_system
    s.plate.set_label("bowl2", "jello")
    seen = []
    s.plate.image_generate = lambda jpg, prompt: seen.append(prompt) or {
        "strategy": "stab_and_saw", "entry_angle_deg": 62, "depth_mm": 9, "saw_strokes": 2, "speed": 0.5,
        "why": "firm cubes need cutting"}
    a = s.plate.plan_attack("bowl2", b"jpeg-bytes")
    assert a["source"] == "vision" and a["entry_angle_deg"] == 62 and "jello" in seen[0]
    assert s.plate.attack_of("bowl2")["why"] == "firm cubes need cutting"
    s.plate.bowls["bowl2"]["amount"] = "low"                     # later the camera sees little left
    assert s.plate.attack_of("bowl2")["strategy"] == "scrape"


def test_per_joint_contact_allowance_for_sagging_joints(fast_cfg):
    fast_cfg.arm.contact_error_deg_joint = {"wrist_pitch": 10}
    c = ContactMonitor(fast_cfg)
    cmd = {j: 0.0 for j in ("base", "shoulder", "elbow", "wrist_pitch", "wrist_roll")}
    sag = dict(cmd, wrist_pitch=-8.0)                          # gravity sag under the gripper: not contact
    assert c.check(0.0, cmd, sag, 0) is None and c.check(1.0, cmd, sag, 0) is None
    blocked = dict(cmd, wrist_pitch=-15.0)                     # a real block still trips
    c.reset(); c.check(0.0, cmd, blocked, 0)
    assert c.check(1.0, cmd, blocked, 0) is not None
    elbow = dict(cmd, elbow=-8.0)                              # other joints keep the tight default
    c.reset(); c.check(0.0, cmd, elbow, 0)
    assert c.check(1.0, cmd, elbow, 0) is not None


# ---------------------------------------------------------------- face gate: jaw score and "looking at the arm"
def _face(**kw):
    from spoonful.perception.face import FaceState
    d = dict(present=True, ts=100.0, yaw_deg=0.0, pitch_deg=0.0, mouth_open_ratio=0.01, jaw_open=0.6,
             mouth_center_px=(640.0, 400.0), frame_w=1280, frame_h=720)
    d.update(kw)
    return FaceState(**d)


def _gate(fast_cfg):
    from spoonful.perception.gate import Gate
    c = fast_cfg.face
    w = c.delivery_window_px
    c.presence_hold_s = c.mouth_open_hold_s = 0.0
    return Gate(c), ((w.x[0] + w.x[1]) / 2, (w.y[0] + w.y[1]) / 2)


def test_jaw_open_counts_even_when_lip_ratio_is_small(fast_cfg):
    g, mid = _gate(fast_cfg)
    assert g.permit(_face(mouth_center_px=mid), "PRE_DELIVERY_HOLD", 100.0).ok        # jaw 0.6, ratio tiny
    g.reset()
    assert g.permit(_face(mouth_center_px=mid, jaw_open=0.05), "PRE_DELIVERY_HOLD", 100.0).reason == "mouth_closed"
    g.reset()                                                                          # no blendshapes: ratio fallback
    assert g.permit(_face(mouth_center_px=mid, jaw_open=-1.0, mouth_open_ratio=0.15), "PRE_DELIVERY_HOLD", 100.0).ok


def test_looking_is_measured_from_the_arm_direction(fast_cfg):
    g, mid = _gate(fast_cfg)
    fast_cfg.face.yaw_center_deg = 35.0                                                # arm is off to the side
    assert g.permit(_face(mouth_center_px=mid, yaw_deg=33.0), "PRE_DELIVERY_HOLD", 100.0).ok
    g.reset()
    assert g.permit(_face(mouth_center_px=mid, yaw_deg=0.0), "PRE_DELIVERY_HOLD", 100.0).reason == "looking_away"
