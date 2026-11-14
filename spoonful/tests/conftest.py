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
    c = C.load()
    c.face.yaw_center_deg, c.face.pitch_offset_deg = 0.0, 0.0   # gate tests use a face looking at the camera
    return c


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
    # plate tests use two unlabelled bowls, independent of the live config (which tracks the real table)
    c.plate.bowls = [C.BowlCfg(name="bowl1", roi=(700, 420, 280, 280)), C.BowlCfg(name="bowl2", roi=(980, 420, 280, 280))]
    c.plate.track.calib_file = str(tmp_path / "table_calib.json")
    c.plate.track.table_min_v_px = 0       # the fake cameras in the tests put the table anywhere in the picture
    c.plate.probe.enabled = False          # feel-for-the-bottom and the spoon check have their own tests (test_probe.py)
    c.plate.spoon_check.enabled = False
    c.face.yaw_center_deg, c.face.pitch_offset_deg = 0.0, 0.0   # the sim face looks straight at the camera
    # the sim trajectories were made with the original spoon model; the live config holds the measured real spoon
    c.kinematics.tool_mm, c.kinematics.tool_offset_mm, c.kinematics.spoon_measured = 191.8, 30.0, False
    c.kinematics.face_zone.x = (235.0, 600.0)
    c.arm.stop_torque_off_after_s = 1.0      # test_fsm checks the torque-off path; the live arm holds instead
    c.voice.requests = True                # voice food requests have tests; the live config may be button-only
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
