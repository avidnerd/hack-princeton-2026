"""Feel for the bottom (probe scoop) and the spoon check at pre_delivery."""
import time

import pytest

from conftest import wait_state
from test_fsm import start_listening
from spoonful.arm import kinematics as K


@pytest.fixture
def probe_system(fast_cfg):
    from spoonful.system import Options, System
    fast_cfg.plate.probe.enabled = True
    fast_cfg.plate.probe.speed_mm_s = 10          # 15 above + 20 below = 35 mm in 3.5 s
    fast_cfg.plate.probe.below_mm = 20
    s = System(fast_cfg, Options(backend="sim", sim_face=True, camera=False, voice=False, esp32=False))
    s.profiles.update(hold_s=0.6)
    s.start()
    s.plate.set_label("bowl1", "applesauce")
    yield s
    s.stop()


def _events(s, kind):
    s.log.flush()
    return [r["detail"] for r in s.log.query(f"SELECT detail FROM events WHERE type='{kind}' ORDER BY rowid")]


def _bite_with_jam_after(s, seconds):
    start_listening(s)
    s.on_voice(None, "Spoonful, applesauce", 0.9)
    wait_state(s, "SCOOPING")
    t0 = time.time()
    while s.fsm.motion_step < 1 and time.time() - t0 < 10:      # wait for the slow descent to begin
        time.sleep(0.02)
    time.sleep(seconds)
    for j in ("shoulder", "elbow", "wrist_pitch"):              # the spoon meets the bottom: the arm stops
        s.backend.jam(s.cfg.arm.servo_map[j], True)
    t0 = time.time()
    while s.fsm.motion_step < 2 and time.time() - t0 < 5:
        time.sleep(0.02)
    for j in ("shoulder", "elbow", "wrist_pitch"):
        s.backend.jam(s.cfg.arm.servo_map[j], False)
    assert s.fsm.motion_step == 2, "probe never reported a touch"
    wait_state(s, "LIFTING")
    assert s.fsm.state.value != "FAULT"
    return _events(s, "scoop_probe")[-1]


def test_touch_below_the_taught_dip_sets_the_depth(probe_system):
    # descent: 15 mm above the dip to 20 mm below, at 10 mm/s. Jam ~2.4 s in = ~9 mm below the dip.
    ev = _bite_with_jam_after(probe_system, 2.4)
    assert '"scoop_at_mm"' in ev and "ignored" not in ev
    assert _events(probe_system, "probe_touch")


def test_touch_above_the_taught_dip_ends_the_bite(probe_system):
    # It used to be ignored and the spoon sent on down to the taught depth (pressing on whatever was there).
    # Now the spoon lifts back out and the bite ends. Full check: test_rim_scoop.py (legacy probe test).
    s = probe_system
    start_listening(s)
    s.on_voice(None, "Spoonful, applesauce", 0.9)
    wait_state(s, "SCOOPING")
    t0 = time.time()
    while s.fsm.motion_step < 1 and time.time() - t0 < 10:
        time.sleep(0.02)
    time.sleep(0.8)                                             # ~10 mm down = still above the dip
    for j in ("shoulder", "elbow", "wrist_pitch"):
        s.backend.jam(s.cfg.arm.servo_map[j], True)
    while s.fsm.motion_step == 1 and time.time() - t0 < 15:
        time.sleep(0.01)
    for j in ("shoulder", "elbow", "wrist_pitch"):
        s.backend.jam(s.cfg.arm.servo_map[j], False)
    wait_state(s, "LISTENING")
    assert "lifting out" in _events(s, "scoop_probe")[-1] and '"scoop_at_mm"' not in _events(s, "scoop_probe")[-1]
    s.log.flush()
    assert s.log.query("SELECT outcome FROM bites ORDER BY id")[-1]["outcome"] == "touch_above_food"


def test_probe_without_resistance_still_scoops(probe_system):
    s = probe_system
    start_listening(s)
    s.on_voice(None, "Spoonful, applesauce", 0.9)
    wait_state(s, "LIFTING", timeout=20)
    ev = _events(s, "scoop_probe")
    assert ev and "no resistance" in ev[-1]


def test_empty_spoon_is_rescooped_then_delivered(fast_cfg):
    from spoonful.system import Options, System
    fast_cfg.plate.spoon_check.enabled = True
    s = System(fast_cfg, Options(backend="sim", sim_face=True, camera=False, voice=False, esp32=False))
    answers = [{"spoon_visible": True, "food_on_spoon": False, "amount": "none"},
               {"spoon_visible": True, "food_on_spoon": True, "amount": "good"}]
    s.plate._spoon = lambda jpg: answers.pop(0) if answers else {"spoon_visible": True, "food_on_spoon": True}
    s.plate.spoon_frame_fn = lambda: __import__("numpy").zeros((720, 1280, 3), "uint8")
    s.profiles.update(hold_s=0.6)
    s.start()
    try:
        start_listening(s)
        s.face.sim.update(present=True, mouth_open=True)
        s.fsm.post("bite", source="ui")
        wait_state(s, "DELIVERING", timeout=30)
        assert s.fsm.bite.rescoops == 1
        assert any("re-scoop 1" in d for d in _events(s, "state"))
    finally:
        s.stop()


def test_spoon_check_gives_up_after_max_rescoops(fast_cfg):
    from spoonful.system import Options, System
    fast_cfg.plate.spoon_check.enabled = True
    fast_cfg.plate.spoon_check.max_rescoops = 1
    s = System(fast_cfg, Options(backend="sim", sim_face=True, camera=False, voice=False, esp32=False))
    s.plate._spoon = lambda jpg: {"spoon_visible": True, "food_on_spoon": False, "amount": "none"}
    s.plate.spoon_frame_fn = lambda: __import__("numpy").zeros((720, 1280, 3), "uint8")
    s.profiles.update(hold_s=0.6)
    s.start()
    try:
        start_listening(s)
        s.face.sim.update(present=True, mouth_open=True)
        s.fsm.post("bite", source="ui")
        wait_state(s, "DELIVERING", timeout=30)                 # after one retry it delivers anyway (never loops)
        assert s.fsm.bite.rescoops == 1
    finally:
        s.stop()
