"""MANUAL hardware tests. Run with the arm powered, clear area, hand on the rocker:
    SPOONFUL_HW=1 .venv/bin/python -m pytest -m hw tests/hw -s
"""
import os
import time

import pytest

from spoonful.arm.arm import Arm
from spoonful.arm.backend import make_backend
from spoonful.config import ARM_JOINTS, JOINTS

pytestmark = [pytest.mark.hw, pytest.mark.skipif(not os.environ.get("SPOONFUL_HW"), reason="set SPOONFUL_HW=1")]


@pytest.fixture
def arm(cfg):
    b = make_backend(cfg.arm.backend if cfg.arm.backend != "sim" else "xarm", [cfg.arm.servo_map[j] for j in JOINTS])
    a = Arm(cfg, b)
    a.start()
    yield a
    a.hold()
    a.close()


def test_reads_all_joints_and_voltage(arm):
    st = arm.read()
    assert st.ok and set(st.deg) == set(JOINTS)
    assert arm.voltage and 6.5 < arm.voltage < 8.5


def test_soft_limits_clamp_live_targets(arm, cfg):
    """Servo toward a target past the base soft limit: the setpoint must stop at the limit (5 deg inside stops)."""
    input("Base will rotate slowly toward its soft limit. Clear? Enter...")
    lo, hi = cfg.arm.soft_limits_deg["base"]
    tgt = arm.current()
    tgt["base"] = hi + 30
    arm.servo_toward(tgt, "near_face")
    time.sleep(abs(hi - arm.current()["base"]) / cfg.arm.max_speed_deg_s.near_face + 2)
    assert arm.setpoint["base"] <= hi + 1e-6
    arm.goto(cfg.arm.home_pose, "near_face").wait(20)


def test_contact_detected_by_hand(arm, cfg):
    input("Elbow will move 15 deg slowly. HOLD the forearm still with your hand. Enter...")
    tgt = arm.current()
    tgt["elbow"] += 15
    m = arm.goto(tgt, "near_face").wait(10)
    assert m.error is not None and "contact" in str(m.error)


def test_noise_floor(arm):
    """Record pot noise with torque on, arm still: NOTES.md wants the number."""
    vals = {j: [] for j in ARM_JOINTS}
    for _ in range(60):
        st = arm.read()
        for j in ARM_JOINTS:
            vals[j].append(st.units[j])
        time.sleep(1 / 30)
    spread = {j: max(v) - min(v) for j, v in vals.items()}
    print("pot noise (units, median-filtered):", spread)
    assert all(s <= 4 for s in spread.values())
