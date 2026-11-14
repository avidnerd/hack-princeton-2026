import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spoonful import config as C  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "hw: needs the real arm (run manually)")


@pytest.fixture
def cfg():
    return C.load()


@pytest.fixture
def fast_cfg(tmp_path):
    """Config tuned so a simulated bite takes a couple of seconds."""
    c = C.load()
    c.log.db = str(tmp_path / "test.db")
    c.profiles.dir = str(tmp_path / "profiles")
    c.speech.backend = "none"
    c.arm.max_speed_deg_s.default = 200
    c.arm.max_speed_deg_s.near_face = 120
    c.arm.max_accel_deg_s2 = 3000
    c.arm.blend_s = 0.05
    c.face.presence_hold_s = 0.1
    c.face.mouth_open_hold_s = 0.05
    c.fsm.pre_delivery_timeout_s = 3
    c.offsets.joint_map = str(tmp_path / "no_map.json")
    c.gate.require_go = False          # GO-button behaviour has its own tests (test_go_button.py)
    c.esp32.button_role = "stop"
    c.esp32.stop_switch = "none"        # STOP-switch behaviour has its own tests (test_stop_switch.py)
    return c


@pytest.fixture
def sim_system(fast_cfg):
    from spoonful.system import Options, System
    s = System(fast_cfg, Options(backend="sim", sim_face=True, camera=False, voice=False, esp32=False))
    s.profiles.update(hold_s=0.6)
    s.start()
    yield s
    s.stop()


def wait_state(sys_, state, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sys_.fsm.state.value == state:
            return True
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {state}; now {sys_.fsm.state.value} "
                         f"fault={sys_.fsm.fault_reason} deny={sys_.arm.last_deny}")
