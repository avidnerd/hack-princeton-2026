"""ESP32 STOP input on GPIO 5 (esp32.stop_switch): rocker = held stop, button = press to stop."""
import time

from conftest import wait_state
from test_fsm import start_listening
from spoonful.io.esp32 import Esp32


def test_rocker_off_stops_and_blocks_release(sim_system):
    s = sim_system
    s.cfg.esp32.stop_switch = "rocker"
    s.on_stop_switch(True)
    start_listening(s)
    s.fsm.post("bite")
    wait_state(s, "SCOOPING")
    s.on_stop_switch(False)                  # flip to O
    wait_state(s, "STOPPED")
    assert "Flip the STOP switch" in s.fsm.explain()
    s.fsm.post("stop", source="key")         # spacebar would arm a release...
    s.fsm.post("reset")                      # ...but not while the switch is off
    time.sleep(0.2)
    assert s.fsm.state.value == "STOPPED"
    s.on_stop_switch(True)                   # back to I: arms the release
    time.sleep(0.1)
    assert s.fsm.status()["stop_release_armed"]
    s.fsm.post("reset")
    wait_state(s, "IDLE")


def test_rocker_off_heartbeats_keep_it_stopped(sim_system):
    s = sim_system
    s.cfg.esp32.stop_switch = "rocker"
    s.on_stop_switch(True)
    wait_state(s, "IDLE")
    s.on_stop_switch(False)
    wait_state(s, "STOPPED")
    for _ in range(3):
        s.on_stop_switch(False)              # heartbeats while still off
    s.fsm.post("reset")
    time.sleep(0.2)
    assert s.fsm.state.value == "STOPPED"


def test_rocker_lost_device_stops(sim_system):
    s = sim_system
    s.cfg.esp32.stop_switch = "rocker"
    s.on_stop_switch(True)
    start_listening(s)
    s.on_stop_switch(None)                   # ESP32 unplugged mid-session
    wait_state(s, "STOPPED")
    s.fsm.post("stop", source="key")
    s.fsm.post("reset")
    time.sleep(0.2)
    assert s.fsm.state.value == "STOPPED"    # can't read the switch: no release


def test_button_mode_press_stops(sim_system):
    s = sim_system
    s.cfg.esp32.stop_switch = "button"
    s.on_stop_switch(False)
    start_listening(s)
    s.on_stop_switch(True)
    wait_state(s, "STOPPED")


def test_none_mode_ignores_switch(sim_system):
    s = sim_system
    s.cfg.esp32.stop_switch = "none"
    start_listening(s)
    s.on_stop_switch(False)
    s.on_stop_switch(None)
    time.sleep(0.2)
    assert s.fsm.state.value == "LISTENING"


def test_serial_parsing(fast_cfg):
    seen = []
    e = Esp32(fast_cfg.esp32, lambda: None, lambda env: None, seen.append)
    e.handle_line('{"sw":0}')
    e.handle_line('{"hb":1,"sw":1}')
    e.handle_line('{"hb":1}')
    assert seen == [False, True] and e.switch_closed is True
