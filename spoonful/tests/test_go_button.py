"""GO button (gate.require_go): the arm approaches only on a fresh press AND a face-gate permit."""
import time

import pytest

from conftest import wait_state
from test_fsm import start_listening


@pytest.fixture
def go_cfg(fast_cfg):
    fast_cfg.gate.require_go = True
    fast_cfg.gate.go_window_s = 1.0
    fast_cfg.esp32.button_role = "go"
    return fast_cfg


@pytest.fixture
def go_system(go_cfg):
    from spoonful.system import Options, System
    s = System(go_cfg, Options(backend="sim", sim_face=True, camera=False, voice=False, esp32=False))
    s.profiles.update(hold_s=0.6)
    s.start()
    yield s
    s.stop()


def stays_in(s, state, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        assert s.fsm.state.value == state, f"left {state} for {s.fsm.state.value}"
        time.sleep(0.02)


def test_permit_alone_does_not_approach(go_system):
    s = go_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=True, yaw=0)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    stays_in(s, "PRE_DELIVERY_HOLD", 1.0)
    assert s.fsm.status()["go_needed"]
    assert "press the button" in s.fsm.explain()
    s.fsm.post("go", source="button")
    wait_state(s, "DELIVERING", timeout=2)


def test_press_alone_does_not_approach(go_system):
    s = go_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=False, yaw=0)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.fsm.post("go", source="button")
    stays_in(s, "PRE_DELIVERY_HOLD", 0.5)


def test_stale_press_expires(go_system):
    s = go_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=False, yaw=0)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.fsm.post("go", source="button")
    time.sleep(s.cfg.gate.go_window_s + 0.3)
    s.face.sim["mouth_open"] = True
    stays_in(s, "PRE_DELIVERY_HOLD", 0.6)


def test_press_before_hold_does_not_count(go_system):
    s = go_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=True, yaw=0)
    s.fsm.post("bite")
    wait_state(s, "SCOOPING")
    s.fsm.post("go", source="button")       # pressed while scooping: ignored
    wait_state(s, "PRE_DELIVERY_HOLD")
    stays_in(s, "PRE_DELIVERY_HOLD", 0.6)


def test_backoff_needs_a_new_press(go_system):
    s = go_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=True, yaw=0)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.fsm.post("go", source="button")
    wait_state(s, "DELIVERING", timeout=2)
    s.face.sim["yaw"] = 40                  # look away: back off
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.face.sim["yaw"] = 0
    stays_in(s, "PRE_DELIVERY_HOLD", 0.6)   # the first press was used up
    s.fsm.post("go", source="button")
    wait_state(s, "DELIVERING", timeout=2)


def test_press_while_listening_requests_a_bite(go_system):
    s = go_system
    start_listening(s)
    s.fsm.post("go", source="button")
    wait_state(s, "SCOOPING", timeout=2)


def test_go_never_stops_and_voice_stop_still_wins(go_system):
    s = go_system
    start_listening(s)
    s.face.sim.update(present=True, mouth_open=True, yaw=0)
    s.fsm.post("bite")
    wait_state(s, "PRE_DELIVERY_HOLD")
    s.fsm.post("go", source="button")
    wait_state(s, "DELIVERING", timeout=2)
    s.fsm.post("stop", source="voice")
    wait_state(s, "STOPPED")
    s.fsm.post("go", source="button")       # GO cannot release a stop
    s.fsm.post("reset")
    time.sleep(0.2)
    assert s.fsm.state.value == "STOPPED"
    assert "spacebar" in s.fsm.explain()
